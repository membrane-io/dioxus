use anyhow::Context;
use itertools::Itertools;
use object::{
    macho::{self},
    read::File,
    write::{MachOBuildVersion, SectionId, StandardSection, Symbol, SymbolId, SymbolSection},
    Endianness, Object, ObjectSection, ObjectSymbol, SymbolFlags, SymbolKind, SymbolScope,
};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::Read,
    ops::Range,
    path::Path,
    path::PathBuf,
    sync::{Arc, RwLock},
};
use subsecond_types::{AddressMap, JumpTable};
use target_lexicon::{Architecture, OperatingSystem, PointerWidth, Triple};
use thiserror::Error;
use walrus::{
    ConstExpr, DataKind, ElementItems, ElementKind, FunctionBuilder, FunctionId, FunctionKind,
    ImportKind, Module, ModuleConfig, TableId, ValType,
};
use wasmparser::{
    BinaryReader, BinaryReaderError, Linking, LinkingSectionReader, Payload, SymbolInfo,
};

type Result<T, E = PatchError> = std::result::Result<T, E>;

#[derive(Debug, Error)]
pub enum PatchError {
    #[error("Failed to read file: {0}")]
    ReadFs(#[from] std::io::Error),

    #[error(
        "No debug symbols in the patch output. Check your profile's `opt-level` and debug symbols config."
    )]
    MissingSymbols,

    #[error("Failed to parse wasm section: {0}")]
    ParseSection(#[from] wasmparser::BinaryReaderError),

    #[error("Failed to parse object file, {0}")]
    ParseObjectFile(#[from] object::read::Error),

    #[error("Failed to write object file: {0}")]
    WriteObjectFIle(#[from] object::write::Error),

    #[error("Failed to emit module: {0}")]
    RuntimeError(#[from] anyhow::Error),

    #[error("Failed to read module's PDB file: {0}")]
    PdbLoadError(#[from] pdb::Error),

    #[error("{0}")]
    InvalidModule(String),

    #[error("Unsupported platform: {0}")]
    UnsupportedPlatform(String),
}

/// A cache for the hotpatching engine that stores the original module's parsed symbol table.
/// For large projects, this can shave up to 50% off the total patching time. Since we compile the base
/// module with every symbol in it, it can be quite large (hundreds of MB), so storing this here lets
/// us avoid re-parsing the module every time we want to patch it.
///
/// On the Dioxus Docsite, it dropped the patch time from 3s to 1.1s (!)
#[derive(Default)]
pub struct HotpatchModuleCache {
    pub path: PathBuf,

    // .... wasm stuff
    pub symbol_ifunc_map: HashMap<String, i32>,
    pub old_exports: HashSet<String>,
    pub old_imports: HashSet<String>,

    /// (wasm) Base data-symbol name → absolute linear-memory offset. Precomputed once so the fast
    /// path can satisfy `GOT.mem` imports without re-parsing the base data section every patch.
    pub data_symbol_offsets: HashMap<String, i32>,

    /// (wasm) Base ifunc-table index → normalized signature. Precomputed once so the fast path can
    /// gate `env` imports and in-place repoints without re-deriving base signatures every patch.
    pub ifunc_sigs: HashMap<i32, SigVec>,

    /// (wasm) The direct callers of each function of the base, by wasm function index:
    /// `callers[callee]` holds the index of every function with a `call callee` instruction.
    /// The functions of a patch link come from this graph, see `patch_functions`.
    pub callers: Vec<Vec<u32>>,

    /// (wasm) The direct callees of each function of the base, by wasm function index.
    pub callees: Vec<Vec<u32>>,

    /// (wasm) Function symbol name → wasm function index, from the linking section of the base.
    pub symbol_func_index: HashMap<String, u32>,

    /// (wasm) The name of each function of the base, by wasm function index, from the `name`
    /// section. An import without a name has an empty string.
    pub func_names: Vec<String>,

    /// (wasm) Whether each function of the base, by wasm function index, has a slot in the table.
    pub in_table: Vec<bool>,

    /// (wasm) The functions that the patches since the fat build defined. The next patch
    /// defines them again, so that the slots of an earlier patch repoint to the newest code,
    /// also when an edit took the code back to the base.
    pub patched_functions: RwLock<HashSet<String>>,

    /// (wasm) The function hashes of the link inputs and of the base files, see
    /// `function_hashes`. A fat build makes a new cache, so the entries live for one fat build.
    pub file_hashes: FileHashCache,

    /// (wasm) Per-build identity read from the base's exported `__subsecond_base_id` global, copied
    /// into every `JumpTable` so the runtime can reject patches built against a different base.
    /// `None` if the base doesn't carry the global (older base, or wasm-bindgen dropped it).
    pub base_id: Option<i32>,

    // ... native stuff
    pub symbol_table: HashMap<String, CachedSymbol>,

    /// Contents of the .tdata section from the original binary (TLS initialization image).
    /// Used to provide correct init data for TLS symbol stubs instead of garbage addresses.
    pub tls_init_data: Vec<u8>,

    /// Map from `$tlv$init` symbol name to (offset_in_tdata, computed_size).
    /// On macOS, Mach-O nlist doesn't carry symbol sizes, so we compute them from
    /// adjacent symbol addresses in the `__thread_data` section. This lets us provide
    /// correctly-sized TLS init data in stubs instead of defaulting to pointer_width.
    pub tls_init_sizes: HashMap<String, (u64, u64)>,
}

pub struct CachedSymbol {
    pub address: u64,
    pub kind: SymbolKind,
    pub is_undefined: bool,
    pub is_weak: bool,
    pub size: u64,
    pub flags: SymbolFlags<SectionId, SymbolId>,
}

impl PartialEq for HotpatchModuleCache {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl std::fmt::Debug for HotpatchModuleCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HotpatchModuleCache")
            .field("_path", &self.path)
            .finish()
    }
}

impl HotpatchModuleCache {
    /// This caching step is crucial for performance on large projects. The original module can be
    /// quite large (hundreds of MB), so this step drastically speeds it up.
    pub fn new(original: &Path, triple: &Triple) -> Result<Self> {
        let cache = match triple.operating_system {
            OperatingSystem::Windows => {
                use pdb::FallibleIterator;

                // due to lifetimes, this code is unfortunately duplicated.
                // the pdb crate doesn't bind the lifetime of the items in the iterator to the symbol table,
                // so we're stuck with local lifetime.s
                let old_pdb_file = original.with_extension("pdb");
                let old_pdb_file_handle = std::fs::File::open(old_pdb_file)?;
                let mut pdb_file = pdb::PDB::open(old_pdb_file_handle)?;
                let global_symbols = pdb_file.global_symbols()?;
                let address_map = pdb_file.address_map()?;
                let mut symbol_table = HashMap::new();
                let mut symbols = global_symbols.iter();
                while let Ok(Some(symbol)) = symbols.next() {
                    match symbol.parse() {
                        Ok(pdb::SymbolData::Public(data)) => {
                            let rva = data.offset.to_rva(&address_map);
                            let is_undefined = rva.is_none();

                            // treat undefined symbols as 0 to match macho/elf
                            let rva = rva.unwrap_or_default();

                            symbol_table.insert(
                                data.name.to_string().to_string(),
                                CachedSymbol {
                                    address: rva.0 as u64,
                                    kind: if data.function {
                                        SymbolKind::Text
                                    } else {
                                        SymbolKind::Data
                                    },
                                    is_undefined,
                                    is_weak: false,
                                    size: 0,
                                    flags: SymbolFlags::None,
                                },
                            );
                        }

                        Ok(pdb::SymbolData::Data(data)) => {
                            let rva = data.offset.to_rva(&address_map);
                            let is_undefined = rva.is_none();

                            // treat undefined symbols as 0 to match macho/elf
                            let rva = rva.unwrap_or_default();

                            symbol_table.insert(
                                data.name.to_string().to_string(),
                                CachedSymbol {
                                    address: rva.0 as u64,
                                    kind: SymbolKind::Data,
                                    is_undefined,
                                    is_weak: false,
                                    size: 0,
                                    flags: SymbolFlags::None,
                                },
                            );
                        }

                        _ => {}
                    }
                }

                HotpatchModuleCache {
                    symbol_table,
                    path: original.to_path_buf(),
                    ..Default::default()
                }
            }

            // We need to load the ifunc table from the original module since that gives us the map
            // of name to address (since ifunc entries are also pointers in wasm - ie 0x30 is the 30th
            // entry in the ifunc table)
            //
            // One detail here is that with high optimization levels, the names of functions in the ifunc
            // table will be smaller than the total number of functions in the module. This is because
            // in high opt-levels, functions are merged. Fortunately, the symbol table remains intact
            // and functions with different names point to the same function index (not to be confused
            // with the function index in the module!).
            //
            // We need to take an extra step to account for merged functions by mapping function index
            // to a set of functions that point to the same index.
            _ if triple.architecture == Architecture::Wasm32 => {
                let bytes = std::fs::read(original)?;
                let ParsedModule {
                    module,
                    symbols,
                    ids,
                } = parse_module_with_ids(&bytes)?;

                if symbols.symbols.is_empty() {
                    return Err(PatchError::MissingSymbols);
                }

                let direct_name_to_ifunc = collect_func_ifuncs(&module);

                // These are the "real" bindings for functions in the module
                // Basically a map between a function's index and its real name
                let func_to_index = module
                    .funcs
                    .par_iter()
                    .filter_map(|f| {
                        let name = f.name.as_deref()?;
                        Some((*symbols.code_symbol_map.get(name)?, name))
                    })
                    .collect::<HashMap<usize, &str>>();

                // Find the corresponding function that shares the same index, but in the ifunc table.
                // This indirection through `code_symbol_map` is what lets us resolve symbols that
                // were merged together at high opt-levels — multiple symbol names can share one
                // wasm function index, so we map symbol-name → function-index → unified-name →
                // ifunc-offset.
                let mut symbol_ifunc_map: HashMap<String, i32> = symbols
                    .code_symbol_map
                    .par_iter()
                    .filter_map(|(name, idx)| {
                        let new_modules_unified_function = func_to_index.get(idx)?;
                        let offset = direct_name_to_ifunc.get(new_modules_unified_function)?;
                        Some((name.to_string(), *offset))
                    })
                    .collect();

                // Also expose any function whose `Function::name` matches an ifunc entry but
                // doesn't appear in the linking section's symbol table. This covers ifunc-table
                // entries we synthesize in `prepare_wasm_base_module` (env-import trap stubs)
                // whose original symbol record in the linking section refers to the (now-deleted)
                // import slot rather than a defined function. Existing entries take precedence —
                // the merged-function indirection above is strictly more informative when it
                // applies.
                for (name, offset) in &direct_name_to_ifunc {
                    symbol_ifunc_map
                        .entry((*name).to_string())
                        .or_insert(*offset);
                }

                // The linking section counts functions in the index space of the module before
                // wasm-bindgen, so its indices do not name the functions of this module. The
                // `name` section does. A symbol that merged into another function maps through
                // `func_to_index` to the name that the `name` section holds.
                let (callers, callees) = collect_direct_calls(&module, &ids);
                let name_to_wasm_index: HashMap<&str, u32> = ids
                    .iter()
                    .enumerate()
                    .filter_map(|(index, id)| {
                        Some((module.funcs.get(*id).name.as_deref()?, index as u32))
                    })
                    .collect();
                let mut symbol_func_index: HashMap<String, u32> = symbols
                    .code_symbol_map
                    .iter()
                    .filter_map(|(name, link_index)| {
                        let unified = func_to_index.get(link_index)?;
                        let index = name_to_wasm_index.get(unified)?;
                        Some((name.to_string(), *index))
                    })
                    .collect();
                for (name, index) in &name_to_wasm_index {
                    symbol_func_index
                        .entry((*name).to_string())
                        .or_insert(*index);
                }
                let func_names: Vec<String> = ids
                    .iter()
                    .map(|id| module.funcs.get(*id).name.clone().unwrap_or_default())
                    .collect();
                let mut in_table = vec![false; ids.len()];
                for (name, index) in &symbol_func_index {
                    if symbol_ifunc_map.contains_key(name) {
                        in_table[*index as usize] = true;
                    }
                }

                let old_exports = module
                    .exports
                    .iter()
                    .map(|e| e.name.to_string())
                    .collect::<HashSet<_>>();

                let old_imports = module
                    .imports
                    .iter()
                    .map(|i| i.name.to_string())
                    .collect::<HashSet<_>>();

                let base_id = read_exported_i32_global(&module, SUBSECOND_BASE_ID_EXPORT);

                // Precompute the base-only inputs the fast path needs per patch, so it never has to
                // re-parse the base data section or re-derive base signatures.
                let data_symbol_offsets = symbols
                    .data_symbol_map
                    .keys()
                    .filter_map(|name| {
                        let offset = resolve_got_mem_offset(name, &symbols, &module).ok()?;
                        Some((name.to_string(), offset))
                    })
                    .collect();
                let ifunc_sigs = collect_ifunc_signatures(&module)
                    .into_iter()
                    .map(|(idx, (params, results))| {
                        (
                            idx,
                            (
                                params.iter().map(walrus_valtype_sig).collect(),
                                results.iter().map(walrus_valtype_sig).collect(),
                            ),
                        )
                    })
                    .collect();

                HotpatchModuleCache {
                    path: original.to_path_buf(),
                    symbol_ifunc_map,
                    old_exports,
                    old_imports,
                    base_id,
                    data_symbol_offsets,
                    ifunc_sigs,
                    callers,
                    callees,
                    symbol_func_index,
                    func_names,
                    in_table,
                    ..Default::default()
                }
            }
            _ => {
                let old_bytes = std::fs::read(original)?;
                let obj = File::parse(&old_bytes as &[u8])?;
                let symbol_table = obj
                    .symbols()
                    .filter_map(|s| {
                        let flags = match s.flags() {
                            SymbolFlags::None => SymbolFlags::None,
                            SymbolFlags::Elf { st_info, st_other } => {
                                SymbolFlags::Elf { st_info, st_other }
                            }
                            SymbolFlags::MachO { n_desc } => SymbolFlags::MachO { n_desc },
                            _ => SymbolFlags::None,
                        };

                        Some((
                            s.name().ok()?.to_string(),
                            CachedSymbol {
                                address: s.address(),
                                is_undefined: s.is_undefined(),
                                is_weak: s.is_weak(),
                                kind: s.kind(),
                                size: s.size(),
                                flags,
                            },
                        ))
                    })
                    .collect::<HashMap<_, _>>();

                // Extract TLS initialization data and section metadata.
                // This is used to correctly initialize TLS symbols in the stub
                // instead of writing bogus absolute addresses into .tdata.
                let tls_section = obj
                    .sections()
                    .find(|s| matches!(s.name(), Ok(".tdata" | "__thread_data")));

                let tls_init_data = tls_section
                    .as_ref()
                    .and_then(|s| s.data().ok())
                    .unwrap_or(&[])
                    .to_vec();

                // Build TLS init size map for macOS. Mach-O nlist doesn't carry symbol
                // sizes, so we compute them from adjacent symbols in __thread_data.
                // LLVM/rustc names init data symbols as `FOO$tlv$init` in __thread_data.
                let tls_data_addr = tls_section.as_ref().map(|s| s.address()).unwrap_or(0);
                let tls_data_size = tls_section.as_ref().map(|s| s.size()).unwrap_or(0);
                let tls_section_index = tls_section.as_ref().map(|s| s.index());

                let mut tls_init_syms: Vec<(u64, String)> = Vec::new();
                for sym in obj.symbols() {
                    if let (Some(section_idx), Ok(sname)) = (sym.section_index(), sym.name()) {
                        if Some(section_idx) == tls_section_index {
                            let offset = sym.address().saturating_sub(tls_data_addr);
                            tls_init_syms.push((offset, sname.to_string()));
                        }
                    }
                }
                tls_init_syms.sort_by_key(|(addr, _)| *addr);
                tls_init_syms.dedup_by_key(|(addr, _)| *addr);

                let mut tls_init_sizes: HashMap<String, (u64, u64)> = HashMap::new();
                for (i, (offset, sname)) in tls_init_syms.iter().enumerate() {
                    let size = if i + 1 < tls_init_syms.len() {
                        tls_init_syms[i + 1].0 - offset
                    } else {
                        tls_data_size.saturating_sub(*offset)
                    };
                    tls_init_sizes.insert(sname.clone(), (*offset, size));
                }

                HotpatchModuleCache {
                    symbol_table,
                    path: original.to_path_buf(),
                    tls_init_data,
                    tls_init_sizes,
                    ..Default::default()
                }
            }
        };

        Ok(cache)
    }
}

pub fn create_windows_jump_table(patch: &Path, cache: &HotpatchModuleCache) -> Result<JumpTable> {
    use pdb::FallibleIterator;
    let old_name_to_addr = &cache.symbol_table;

    let mut new_name_to_addr = HashMap::new();
    let new_pdb_file_handle = std::fs::File::open(patch.with_extension("pdb"))?;
    let mut pdb_file = pdb::PDB::open(new_pdb_file_handle)?;
    let symbol_table = pdb_file.global_symbols()?;
    let address_map = pdb_file.address_map()?;
    let mut symbol_iter = symbol_table.iter();
    while let Ok(Some(symbol)) = symbol_iter.next() {
        if let Ok(pdb::SymbolData::Public(data)) = symbol.parse() {
            let rva = data.offset.to_rva(&address_map);
            if let Some(rva) = rva {
                new_name_to_addr.insert(data.name.to_string(), rva.0 as u64);
            }
        }
    }

    let mut map = AddressMap::default();
    for (new_name, new_addr) in new_name_to_addr.iter() {
        if let Some(old_addr) = old_name_to_addr.get(new_name.as_ref()) {
            map.insert(old_addr.address, *new_addr);
        }
    }

    let new_base_address = new_name_to_addr
        .get("main")
        .cloned()
        .context("failed to find 'main' symbol in patch")?;

    let aslr_reference = old_name_to_addr
        .get("main")
        .map(|s| s.address)
        .context("failed to find '_main' symbol in original module")?;

    Ok(JumpTable {
        lib: patch.to_path_buf(),
        map,
        new_base_address,
        aslr_reference,
        ifunc_count: 0,
        dwarf_sidecar: None,
        ifunc_repoint: Vec::new(),
        previous_patch: None,
        wasm: None,
        base_id: None,
    })
}

/// Assemble a jump table for "nix" architectures. This uses the `object` crate to parse both
/// executable's symbol tables and then creates a mapping between the two. Unlike windows, the symbol
/// tables are stored within the binary itself, so we can use the `object` crate to parse them.
///
/// We use the `_aslr_reference` as a reference point in the base program to calculate the aslr slide
/// both at compile time and at runtime.
///
/// This does not work for WASM since the `object` crate does not support emitting the WASM format,
/// and because WASM requires more logic to handle the wasm-bindgen transformations.
pub fn create_native_jump_table(
    patch: &Path,
    triple: &Triple,
    cache: &HotpatchModuleCache,
) -> Result<JumpTable> {
    let old_name_to_addr = &cache.symbol_table;
    let obj2_bytes = std::fs::read(patch)?;
    let obj2 = File::parse(&obj2_bytes as &[u8])?;
    let mut map = AddressMap::default();
    let new_syms = obj2.symbol_map();

    let new_name_to_addr = new_syms
        .symbols()
        .par_iter()
        .map(|s| (s.name(), s.address()))
        .collect::<HashMap<_, _>>();

    for (new_name, new_addr) in new_name_to_addr.iter() {
        if let Some(old_addr) = old_name_to_addr.get(*new_name) {
            map.insert(old_addr.address, *new_addr);
        }
    }

    let sentinel = main_sentinel(triple);
    let new_base_address = new_name_to_addr
        .get(sentinel)
        .cloned()
        .context("failed to find 'main' symbol in base - are deubg symbols enabled?")?;
    let aslr_reference = old_name_to_addr
        .get(sentinel)
        .map(|s| s.address)
        .context("failed to find 'main' symbol in original module - are debug symbols enabled?")?;

    Ok(JumpTable {
        lib: patch.to_path_buf(),
        map,
        new_base_address,
        aslr_reference,
        ifunc_count: 0,
        dwarf_sidecar: None,
        ifunc_repoint: Vec::new(),
        previous_patch: None,
        wasm: None,
        base_id: None,
    })
}

/// In the web, our patchable functions are actually ifuncs
///
/// We need to line up the ifuncs from the main module to the ifuncs in the patch.
///
/// According to the dylink spec, there will be two sets of entries:
///
/// - got.func: functions in the indirect function table
/// - got.mem: data objects in the data segments
///
/// It doesn't seem like we can compile the base module to export these, sadly, so we're going
/// to satisfy them — but *where* we satisfy them is the whole performance story.
///
/// The historical approach (`create_wasm_jump_table_walrus`) rewrites the patch module to bake the
/// GOT/env imports into local globals and `call_indirect` trampolines, then re-emits the whole module
/// with walrus. That re-emit is the dominant cost of a hot patch (hundreds of ms) because walrus has
/// to re-encode the code section and fix up DWARF to match the new function index space.
///
/// The fast path here avoids all of that: it leaves every dynamic-linking import in place (so the
/// code section and its DWARF stay byte-identical to wasm-ld's output) and ships the values the
/// runtime needs to satisfy those imports at instantiate time in the `JumpTable`. The only mutation
/// to the served bytes is a cheap byte-level pass that strips custom sections, drops the start
/// section, and adds the one export the runtime needs. We fall back to the walrus path only when the
/// patch contains `wbg_cast` function *bodies* that must be rewritten — those are local functions, so
/// the import object can't help and a real code-section edit is unavoidable.
///
/// <https://github.com/WebAssembly/tool-conventions/blob/main/DynamicLinking.md>
/// The functions that a patch put into the shared table: the name and the signature of each
/// slot of its region. dx keeps this for the last patch of a session, so that the next patch
/// can repoint the region of the last patch too. See `JumpTable::previous_patch`.
#[derive(Debug, Clone)]
pub struct PatchIfuncs {
    pub lib: PathBuf,
    pub name_to_ifunc: HashMap<String, i32>,
    pub ifunc_sigs: HashMap<i32, SigVec>,
}

/// Build the jump table of a wasm patch. Returns the table and, on the fast path, the slots
/// of the patch for the next jump table.
pub fn create_wasm_jump_table(
    patch: &Path,
    cache: &HotpatchModuleCache,
    keep_names: bool,
    dwarf_sidecar: bool,
    previous: Option<&PatchIfuncs>,
) -> Result<(JumpTable, Option<PatchIfuncs>)> {
    let t_start = std::time::Instant::now();
    let linked = std::fs::read(patch).context("Could not read patch file")?;

    // Every defined function of the patch gets a slot in the shared table, so that the jump
    // table can repoint the base slot of every function that a changed crate holds. The
    // linker fills the table with the functions whose address the patch takes, and nothing
    // else. The walrus path reads the patch from disk, so the rewrite goes to disk.
    let new_bytes = extend_element_segment(&linked)?;
    std::fs::write(patch, &new_bytes).context("Could not write the patch file")?;

    // With `--dwarf-sidecar`, the served patch carries no DWARF. The linker output, DWARF and
    // all, goes to `<patch>.dwarf.wasm` next to it, and the runtime attaches that file to the
    // patch module. The linker does not garbage-collect DWARF, so the DWARF of every linked
    // object is in the patch: about two thirds of the bytes of a large patch.
    let sidecar = dwarf_sidecar.then(|| patch.with_extension("dwarf.wasm"));
    if let Some(sidecar) = &sidecar {
        std::fs::write(sidecar, &new_bytes).context("Could not write the DWARF sidecar")?;
    }

    // Analyze the patch with a single `wasmparser` pass that skips the code section. The fast path
    // never re-emits the module, so we only need the import/type/function/element/name sections —
    // decoding ~30 MB of function bodies into walrus IR (what `Module::from_buffer` does) is pure
    // overhead here and was the bulk of the old "parse" cost.
    let analysis = analyze_patch_wasm(&new_bytes)?;
    let t_parsed = t_start.elapsed();

    // wbg_cast bodies are local functions that point at `breaks_if_inline` no-ops and must be
    // rewritten to `call_indirect` the original module's cast. An import object can't fix a local
    // function body, so if any are present we have to take the walrus path that re-encodes the code.
    if analysis.needs_body_rewrite {
        tracing::debug!("Patch needs wbg_cast body rewrite; using walrus jump-table path");
        return Ok((
            create_wasm_jump_table_walrus(patch, cache, keep_names, sidecar)?,
            None,
        ));
    }

    create_wasm_jump_table_fast(
        patch, &new_bytes, analysis, cache, t_start, t_parsed, keep_names, sidecar, previous,
    )
}

/// A normalized wasm value type, comparable across the `walrus` (base/cache) and `wasmparser`
/// (patch) parsers so signatures from both can be checked for equality.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WasmSig {
    I32,
    I64,
    F32,
    F64,
    V128,
    FuncRef,
    ExternRef,
    OtherRef,
}

pub type SigVec = (Vec<WasmSig>, Vec<WasmSig>);

fn walrus_valtype_sig(t: &ValType) -> WasmSig {
    match t {
        ValType::I32 => WasmSig::I32,
        ValType::I64 => WasmSig::I64,
        ValType::F32 => WasmSig::F32,
        ValType::F64 => WasmSig::F64,
        ValType::V128 => WasmSig::V128,
        ValType::Ref(walrus::RefType::Funcref) => WasmSig::FuncRef,
        ValType::Ref(walrus::RefType::Externref) => WasmSig::ExternRef,
        ValType::Ref(_) => WasmSig::OtherRef,
    }
}

fn wasmparser_valtype_sig(t: wasmparser::ValType) -> WasmSig {
    match t {
        wasmparser::ValType::I32 => WasmSig::I32,
        wasmparser::ValType::I64 => WasmSig::I64,
        wasmparser::ValType::F32 => WasmSig::F32,
        wasmparser::ValType::F64 => WasmSig::F64,
        wasmparser::ValType::V128 => WasmSig::V128,
        wasmparser::ValType::Ref(r) if r == wasmparser::RefType::FUNCREF => WasmSig::FuncRef,
        wasmparser::ValType::Ref(r) if r == wasmparser::RefType::EXTERNREF => WasmSig::ExternRef,
        wasmparser::ValType::Ref(_) => WasmSig::OtherRef,
    }
}

/// Everything the fast path needs from the patch module, extracted in one `wasmparser` pass that
/// never decodes function bodies.
struct PatchWasmAnalysis {
    /// `GOT.func.<name>` import names, in module order.
    got_func: Vec<String>,
    /// `GOT.mem.<name>` import names, in module order.
    got_mem: Vec<String>,
    /// Mutability of the `GOT.*` imported globals (wasm-ld emits them mutable). `None` if there are
    /// no GOT imports at all.
    got_mutable: Option<bool>,
    /// `env.<name>` function imports paired with their signature.
    env_funcs: Vec<(String, SigVec)>,
    /// Function name → ifunc-table index, from the active element segments.
    name_to_ifunc: HashMap<String, i32>,
    /// Number of table slots the patch's element segments occupy (highest `offset + len` across
    /// segments). This is what the table must grow by — NOT `name_to_ifunc.len()`, which undercounts
    /// whenever two element items share a mangled name (common for generic `fmt`/drop-glue impls).
    ifunc_slots: u64,
    /// ifunc-table index → signature.
    ifunc_sigs: HashMap<i32, SigVec>,
    /// True when a `wbg_cast` function *body* is present and must be rewritten (forces walrus path).
    needs_body_rewrite: bool,
}

/// Parse the patch with `wasmparser`, reading only the sections the fast path needs and skipping the
/// (huge) code section. This is the replacement for `walrus::Module::from_buffer` in the fast path.
fn analyze_patch_wasm(bytes: &[u8]) -> Result<PatchWasmAnalysis> {
    // Type index → signature.
    let mut types: Vec<SigVec> = Vec::new();
    // Function index (imports first, then defined) → type index.
    let mut func_type_idx: Vec<u32> = Vec::new();
    let mut got_func: Vec<String> = Vec::new();
    let mut got_mem: Vec<String> = Vec::new();
    let mut got_mutable: Option<bool> = None;
    // (name, type index) — resolved to a signature after the full pass.
    let mut env_func_imports: Vec<(String, u32)> = Vec::new();
    // (base offset, function indices) — resolved to names/signatures after the full pass.
    let mut elements: Vec<(i32, Vec<u32>)> = Vec::new();
    let mut func_names: HashMap<u32, String> = HashMap::new();

    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        match payload? {
            Payload::TypeSection(reader) => {
                for rec_group in reader {
                    for sub in rec_group?.into_types() {
                        match sub.composite_type.inner {
                            wasmparser::CompositeInnerType::Func(ft) => {
                                let params =
                                    ft.params().iter().copied().map(wasmparser_valtype_sig).collect();
                                let results = ft
                                    .results()
                                    .iter()
                                    .copied()
                                    .map(wasmparser_valtype_sig)
                                    .collect();
                                types.push((params, results));
                            }
                            // Non-function types still consume a type index; push a placeholder so
                            // later type-index lookups stay aligned. They're never used as a func sig.
                            _ => types.push((Vec::new(), Vec::new())),
                        }
                    }
                }
            }
            Payload::ImportSection(reader) => {
                for import in reader {
                    let import = import?;
                    match import.ty {
                        wasmparser::TypeRef::Func(tyidx) => {
                            func_type_idx.push(tyidx);
                            if import.module == "env" {
                                env_func_imports.push((import.name.to_string(), tyidx));
                            }
                        }
                        wasmparser::TypeRef::Global(gt) => match import.module {
                            "GOT.func" => {
                                got_func.push(import.name.to_string());
                                got_mutable.get_or_insert(gt.mutable);
                            }
                            "GOT.mem" => {
                                got_mem.push(import.name.to_string());
                                got_mutable.get_or_insert(gt.mutable);
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for tyidx in reader {
                    func_type_idx.push(tyidx?);
                }
            }
            Payload::ElementSection(reader) => {
                for element in reader {
                    let element = element?;
                    let offset = match &element.kind {
                        wasmparser::ElementKind::Active { offset_expr, .. } => {
                            match offset_expr.get_operators_reader().read()? {
                                wasmparser::Operator::I32Const { value } => value,
                                wasmparser::Operator::I64Const { value } => value as i32,
                                // The ifunc table is offset by an imported global, so the explicit
                                // offset contribution is 0 (matches the walrus path).
                                wasmparser::Operator::GlobalGet { .. } => 0,
                                _ => continue,
                            }
                        }
                        _ => continue,
                    };
                    if let wasmparser::ElementItems::Functions(funcs) = element.items {
                        let ids = funcs
                            .into_iter()
                            .collect::<std::result::Result<Vec<u32>, _>>()?;
                        elements.push((offset, ids));
                    }
                }
            }
            Payload::CustomSection(section) if section.name() == "name" => {
                let reader = wasmparser::NameSectionReader::new(BinaryReader::new(section.data(), 0));
                for subsection in reader {
                    let Ok(wasmparser::Name::Function(map)) = subsection else {
                        continue;
                    };
                    for naming in map {
                        let naming = naming?;
                        func_names.insert(naming.index, naming.name.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    let env_funcs = env_func_imports
        .into_iter()
        .map(|(name, tyidx)| (name, types.get(tyidx as usize).cloned().unwrap_or_default()))
        .collect();

    let mut name_to_ifunc = HashMap::new();
    let mut ifunc_sigs = HashMap::new();
    for (offset, ids) in &elements {
        for (i, &func_idx) in ids.iter().enumerate() {
            let ifunc_idx = offset + i as i32;
            if let Some(name) = func_names.get(&func_idx) {
                name_to_ifunc.insert(name.clone(), ifunc_idx);
            }
            if let Some(&tyidx) = func_type_idx.get(func_idx as usize) {
                if let Some(sig) = types.get(tyidx as usize) {
                    ifunc_sigs.insert(ifunc_idx, sig.clone());
                }
            }
        }
    }

    // Slots needed = highest table index any element segment reaches. The runtime places each
    // segment at `__table_base + offset` and writes `ids.len()` consecutive entries, so the table
    // must grow by this much or `WebAssembly.instantiate` throws "table index is out of bounds".
    let ifunc_slots = elements
        .iter()
        .map(|(offset, ids)| (*offset).max(0) as u64 + ids.len() as u64)
        .max()
        .unwrap_or(0);

    let needs_body_rewrite = func_names
        .values()
        .any(|n| n.contains("wasm_bindgen4__rt8wbg_cast") && !n.contains("breaks_if_inline"));

    Ok(PatchWasmAnalysis {
        got_func,
        got_mem,
        got_mutable,
        env_funcs,
        name_to_ifunc,
        ifunc_slots,
        ifunc_sigs,
        needs_body_rewrite,
    })
}

/// Fast path: emit the patch with its dynamic-linking imports intact and hand the runtime the values
/// it needs to satisfy them. See [`create_wasm_jump_table`] for why this is dramatically cheaper than
/// the walrus round-trip.
fn create_wasm_jump_table_fast(
    patch: &Path,
    new_bytes: &[u8],
    analysis: PatchWasmAnalysis,
    cache: &HotpatchModuleCache,
    t_start: std::time::Instant,
    t_parsed: std::time::Duration,
    keep_names: bool,
    dwarf_sidecar: Option<PathBuf>,
    previous: Option<&PatchIfuncs>,
) -> Result<(JumpTable, Option<PatchIfuncs>)> {
    use subsecond_types::{PreviousPatch, WasmFixups};

    let name_to_ifunc_old = &cache.symbol_ifunc_map;
    // Base-derived signatures, precomputed at cache-build time (normalized for cross-parser
    // comparison against the patch's wasmparser-derived signatures).
    let old_sigs = &cache.ifunc_sigs;

    let mut got_func: Vec<(String, i32)> = Vec::new();
    let mut got_mem: Vec<(String, i32)> = Vec::new();
    let mut env_ifunc: Vec<(String, i32)> = Vec::new();
    let mut env_skipped_sig = 0usize;

    for name in &analysis.got_func {
        let Some(entry) = name_to_ifunc_old.get(name.as_str()).cloned() else {
            return Err(PatchError::InvalidModule(format!(
                "Expected to find GOT.func entry in ifunc table: {name}"
            )));
        };
        got_func.push((name.clone(), entry));
    }

    for name in &analysis.got_mem {
        let offset = *cache
            .data_symbol_offsets
            .get(name)
            .with_context(|| format!("Failed to find GOT.mem import by its name: {name}"))?;
        got_mem.push((name.clone(), offset));
    }

    for (name, sig) in &analysis.env_funcs {
        // Base-exported (or base-imported) functions are satisfied by the host exports the runtime
        // already copies into `env`; nothing to ship for those.
        if cache.old_exports.contains(name) || cache.old_imports.contains(name) {
            continue;
        }
        // Resolve through the shared ifunc table, but only when the signature matches the base slot.
        // A name-matched-but-mismatched pair would fail instantiation with a LinkError; leaving it
        // out makes the runtime install a trapping stub instead, which mirrors the old
        // `call_indirect` behavior (it would only trap if actually called).
        if let Some(&idx) = name_to_ifunc_old.get(name.as_str()) {
            if old_sigs.get(&idx) == Some(sig) {
                env_ifunc.push((name.clone(), idx));
            } else {
                env_skipped_sig += 1;
            }
        }
    }

    let got_mutable = analysis.got_mutable.unwrap_or(true);
    let n_got_func = got_func.len();
    let n_got_mem = got_mem.len();
    let n_env_ifunc = env_ifunc.len();

    // Build the address map (old ifunc index → new ifunc index) and the in-place repoint set exactly
    // as the walrus path does — these are pure analysis over the unmodified module.
    //
    // `ifunc_count` is how much the runtime grows the shared table; it must equal the element
    // segment's slot count, not `name_to_ifunc.len()` — the latter collapses element items that
    // share a mangled name (generic `fmt`/`Write`/drop-glue impls), so it undercounts and the patch
    // then instantiates with "table index is out of bounds".
    let ifunc_count = analysis.ifunc_slots;
    let mut map = AddressMap::default();
    for (name, idx) in analysis.name_to_ifunc.iter() {
        if let Some(old_idx) = name_to_ifunc_old.get(name.as_str()) {
            map.insert(*old_idx as u64, *idx as u64);
        }
    }

    let mut ifunc_repoint = Vec::new();
    for (&old_idx, &new_idx) in map.iter() {
        if let (Some(old_sig), Some(new_sig)) = (
            old_sigs.get(&(old_idx as i32)),
            analysis.ifunc_sigs.get(&(new_idx as i32)),
        ) {
            if old_sig == new_sig {
                ifunc_repoint.push((old_idx, new_idx));
            }
        }
    }

    // The same pairs from the region of the previous patch, so that the runtime repoints the
    // slots that the vtables and the function pointers of the previous patch use. The previous
    // module then holds no slot of the table, and the browser can free it.
    let previous_patch = previous.map(|prev| {
        let mut repoint = Vec::new();
        for (name, &prev_idx) in prev.name_to_ifunc.iter() {
            let Some(&new_idx) = analysis.name_to_ifunc.get(name.as_str()) else {
                continue;
            };
            if prev.ifunc_sigs.get(&prev_idx) == analysis.ifunc_sigs.get(&new_idx) {
                repoint.push((prev_idx as u64, new_idx as u64));
            }
        }
        repoint.sort_unstable();
        PreviousPatch {
            lib: prev.lib.clone(),
            repoint,
        }
    });
    let t_analyzed = t_start.elapsed();

    // Find the function index (in the *served* module's index space, which we don't change) of the
    // global-relocs thunk so we can export it. wasm-ld refuses to export this synthetic function, but
    // the runtime must call it. We read the index from the linking symbol table (falling back to the
    // name section) so it matches the real module layout.
    // wasm-ld only synthesizes `__wasm_apply_global_relocs` when the patch has `GOT.func.internal`
    // globals to rebase by `__table_base`. When it's absent there's simply nothing to relocate, so a
    // missing export is expected, not an error. When present we must export it (wasm-ld won't) so the
    // runtime can call it.
    let reloc_export = find_patch_func_index(new_bytes, "__wasm_apply_global_relocs");
    if reloc_export.is_none() {
        tracing::debug!(
            "patch has no __wasm_apply_global_relocs (no internal global relocs needed)"
        );
    }

    // Produce the served bytes from the *original* linker output: strip custom sections we don't
    // serve, drop the start section, and add the relocs export. The code/data/import sections are
    // copied verbatim, so DWARF stays valid and there's no re-encode.
    let lib = patch.to_path_buf();
    let bytes = finalize_patch_wasm(new_bytes, reloc_export, keep_names, dwarf_sidecar.is_some())?;
    std::fs::write(&lib, bytes)?;
    let t_emitted = t_start.elapsed();

    tracing::info!(
        "Jump table (fast): parse={}ms analyze={}ms emit={}ms total={}ms | map={} repoint={} previous={} ifunc_count={ifunc_count} GOT.func={n_got_func} GOT.mem={n_got_mem} env_ifunc={n_env_ifunc} (sig-skipped {env_skipped_sig})",
        t_parsed.as_millis(),
        t_analyzed.saturating_sub(t_parsed).as_millis(),
        t_emitted.saturating_sub(t_analyzed).as_millis(),
        t_emitted.as_millis(),
        map.len(),
        ifunc_repoint.len(),
        previous_patch.as_ref().map_or(0, |p| p.repoint.len()),
    );

    if map.is_empty() {
        tracing::warn!(
            "Jump table (fast): map is EMPTY — no old→new ifunc redirects, so the patch will apply but old code keeps running. \
             This means the patch's defined-function names didn't intersect the base cache's symbol→ifunc map \
             (e.g. the patch was read after its `name` section was stripped, or the running base doesn't match this cache)."
        );
    }

    let ifuncs = PatchIfuncs {
        lib: lib.clone(),
        name_to_ifunc: analysis.name_to_ifunc,
        ifunc_sigs: analysis.ifunc_sigs,
    };
    let table = JumpTable {
        map,
        lib,
        ifunc_count,
        dwarf_sidecar,
        aslr_reference: 0,
        new_base_address: 0,
        ifunc_repoint,
        previous_patch,
        wasm: Some(WasmFixups {
            got_func,
            got_mem,
            env_ifunc,
            got_mutable,
        }),
        base_id: cache.base_id,
    };
    Ok((table, Some(ifuncs)))
}

/// Resolve a `GOT.mem` import's value: the absolute offset of the named data symbol in the base
/// module's linear memory. Factored out so both the fast path and the walrus path share it.
fn resolve_got_mem_offset(
    name: &str,
    old_symbols: &RawDataSection<'_>,
    old: &Module,
) -> Result<i32> {
    let data_symbol_idx = *old_symbols
        .data_symbol_map
        .get(name)
        .with_context(|| format!("Failed to find GOT.mem import by its name: {name}"))?;
    let data_symbol = old_symbols
        .data_symbols
        .get(&data_symbol_idx)
        .context("Failed to find data symbol by its index")?;
    let data = old
        .data
        .iter()
        .nth(data_symbol.which_data_segment)
        .context("Missing data segment in the main module")?;
    let offset = match data.kind {
        DataKind::Active {
            offset: ConstExpr::Value(walrus::ir::Value::I32(idx)),
            ..
        } => idx,
        DataKind::Active {
            offset: ConstExpr::Value(walrus::ir::Value::I64(idx)),
            ..
        } => idx as i32,
        _ => {
            return Err(PatchError::InvalidModule(format!(
                "Data segment of invalid table: {:?}",
                data.kind
            )));
        }
    };
    Ok(offset + data_symbol.segment_offset as i32)
}

/// Walrus-based jump table generation (fallback path).
///
/// We need to line up the ifuncs from the main module to the ifuncs in the patch.
///
/// According to the dylink spec, there will be two sets of entries:
///
/// - got.func: functions in the indirect function table
/// - got.mem: data objects in the data segments
///
/// It doesn't seem like we can compile the base module to export these, sadly, so we're going
/// to manually satisfy them here, removing their need to be imported.
///
/// <https://github.com/WebAssembly/tool-conventions/blob/main/DynamicLinking.md>
fn create_wasm_jump_table_walrus(
    patch: &Path,
    cache: &HotpatchModuleCache,
    keep_names: bool,
    dwarf_sidecar: Option<PathBuf>,
) -> Result<JumpTable> {
    let t_start = std::time::Instant::now();
    let name_to_ifunc_old = &cache.symbol_ifunc_map;
    let new_bytes = std::fs::read(patch).context("Could not read patch file")?;
    let t_read = t_start.elapsed();

    // Preserve DWARF custom sections through walrus so source-level stack traces
    // continue to resolve in hot-patched code. With the default `ModuleConfig`,
    // walrus drops every `.debug_*` section at emit time.
    let mut config = ModuleConfig::new();
    config.generate_dwarf(true);
    let mut new = Module::from_buffer_with_config(&new_bytes, &config)?;
    let t_parsed = t_start.elapsed();
    let mut got_mems = vec![];
    let mut got_funcs = vec![];
    let mut wbg_funcs = vec![];
    let mut env_funcs = vec![];

    // Collect all the GOT entries from the new module.
    // The GOT imports come from the wasm-ld implementation of the dynamic linking spec
    //
    // https://github.com/WebAssembly/tool-conventions/blob/main/DynamicLinking.md#imports
    //
    // Normally, the base module would synthesize these as exports, but we're not compiling the base
    // module with `--pie` (nor does wasm-bindgen support it yet), so we need to manually satisfy them.
    //
    // One thing to watch out for here is that GOT.func entries have no visibility to any de-duplication
    // or merging, so we need to take great care in the base module to export *every* symbol even if
    // they point to the same function.
    //
    // The other thing to watch out for here is the __wbindgen_placeholder__ entries. These are meant
    // to be satisfied by wasm-bindgen via manual code generation, but we can't run wasm-bindgen on the
    // patch, so we need to do it ourselves. This involves preventing their elimination in the base module
    // by prefixing them with `__saved_wbg_`. When handling the imports here, we need modify the imported
    // name to match the prefixed export name in the base module.
    for import in new.imports.iter() {
        match import.module.as_str() {
            "GOT.func" => {
                let Some(entry) = name_to_ifunc_old.get(import.name.as_str()).cloned() else {
                    return Err(PatchError::InvalidModule(format!(
                        "Expected to find GOT.func entry in ifunc table: {}",
                        import.name.as_str()
                    )));
                };
                got_funcs.push((import.id(), entry));
            }
            "GOT.mem" => got_mems.push(import.id()),
            "env" => env_funcs.push(import.id()),
            "__wbindgen_placeholder__" => wbg_funcs.push(import.id()),
            m => tracing::trace!("Unknown import: {m}:{}", import.name),
        }
    }

    let n_got_funcs = got_funcs.len();
    let n_got_mems = got_mems.len();
    let n_env_funcs = env_funcs.len();
    let n_wbg_funcs = wbg_funcs.len();

    // We need to satisfy the GOT.func imports of this side module. The GOT imports come from the wasm-ld
    // implementation of the dynamic linking spec
    //
    // https://github.com/WebAssembly/tool-conventions/blob/main/DynamicLinking.md#imports
    //
    // Most importantly, these functions are functions meant to be called indirectly. In normal wasm
    // code generation, only functions that Rust code references via pointers are given a slot in
    // the indirection function table. The optimization here traditionally meaning that if a function
    // can be called directly, then it doesn't need to be referenced indirectly and potentially inlined
    // or dissolved during LTO.
    //
    // In our "fat build" setup, we aggregated all symbols from dependencies into a `dependencies.ar` file.
    // By promoting these functions to the dynamic scope, we also prevent their inlining because the
    // linker can still expect some form of interposition to happen, requiring the symbol *actually*
    // exists.
    //
    // Our technique here takes advantage of that and the [`prepare_wasm_base_module`] function promotes
    // every possible function to the indirect function table. This means that the GOT imports that
    // `relocation-model=pic` synthesizes can reference the functions via the indirect function table
    // even if they are not normally synthesized in regular wasm code generation.
    //
    // Normally, the dynamic linker setup would resolve GOT.func against the same GOT.func export in
    // the main module, but we don't have that. Instead, we simply re-parse the main module, aggregate
    // its ifunc table, and then resolve directly to the index in that table.
    for (import_id, ifunc_index) in got_funcs {
        let import = new.imports.get(import_id);
        let ImportKind::Global(id) = import.kind else {
            return Err(PatchError::InvalidModule(format!(
                "Expected GOT.func import to be a global: {}",
                import.name
            )));
        };

        // "satisfying" the import means removing it from the import table and replacing its target
        // value with a local global.
        new.imports.delete(import_id);
        new.globals.get_mut(id).kind =
            walrus::GlobalKind::Local(ConstExpr::Value(walrus::ir::Value::I32(ifunc_index)));
    }

    // We need to satisfy the GOT.mem imports of this side module. The GOT.mem imports come from the wasm-ld
    // implementation of the dynamic linking spec
    //
    // https://github.com/WebAssembly/tool-conventions/blob/main/DynamicLinking.md#imports
    //
    // Unlike the ifunc table, the GOT.mem imports do not need any additional post-processing of the
    // base module to satisfy. Since our patching approach works but leveraging the experimental dynamic
    // PIC support in rustc[wasm] and wasm-ld, we are using the GOT.mem imports as a way of identifying
    // data segments that are present in the base module.
    //
    // Normally, the dynamic linker would synthesize corresponding GOT.mem exports in the main module,
    // but since we're patching on-the-fly, this table will always be out-of-date.
    //
    // Instead, we use the symbol table from the base module to find the corresponding data symbols
    // and then resolve the offset of the data segment in the main module. Using the symbol table
    // can be somewhat finicky if the user compiled the code with a high-enough opt level that nukes
    // the names of the data segments, but otherwise this system works well.
    //
    // We simply use the name of the import as a key into the symbol table and then its offset into
    // its data segment as the value within the global. The cache holds the offset of every data
    // symbol of the base, see `data_symbol_offsets`.
    for mem in got_mems {
        let import = new.imports.get(mem);
        let offset = *cache
            .data_symbol_offsets
            .get(import.name.as_str())
            .with_context(|| {
                format!("Failed to find GOT.mem import by its name: {}", import.name)
            })?;

        let ImportKind::Global(global_id) = import.kind else {
            return Err(PatchError::InvalidModule(
                "Expected GOT.mem import to be a global".to_string(),
            ));
        };

        // "satisfying" the import means removing it from the import table and replacing its target
        // value with a local global.
        new.imports.delete(mem);
        new.globals.get_mut(global_id).kind =
            walrus::GlobalKind::Local(ConstExpr::Value(walrus::ir::Value::I32(offset)));
    }

    // wasm-bindgen has a limit on the number of exports a module can have, so we need to call the main
    // module's functions indirectly. This is done by dropping the env import and replacing it with a
    // local function that calls the indirect function from the table.
    //
    // https://github.com/emscripten-core/emscripten/issues/22863
    let ifunc_table_initializer = new
        .elements
        .iter()
        .find_map(|e| match e.kind {
            ElementKind::Active { table, .. } => Some(table),
            _ => None,
        })
        .context("Missing ifunc table")?;
    for env_func_import in env_funcs {
        let import = new.imports.get(env_func_import);
        let ImportKind::Function(func_id) = import.kind else {
            continue;
        };

        if cache.old_exports.contains(import.name.as_str())
            || cache.old_imports.contains(import.name.as_str())
        {
            continue;
        }
        let name = import.name.as_str().to_string();

        if let Some(table_idx) = name_to_ifunc_old.get(import.name.as_str()) {
            new.imports.delete(env_func_import);
            convert_func_to_ifunc_call(
                &mut new,
                ifunc_table_initializer,
                func_id,
                *table_idx,
                name.clone(),
            );
            continue;
        }

        if name_is_bindgen_symbol(&name) {
            new.imports.delete(env_func_import);
            convert_func_to_ifunc_call(&mut new, ifunc_table_initializer, func_id, 0, name);
            continue;
        }

        tracing::warn!("[hotpatching]: Symbol slipped through the cracks: {}", name);
    }

    // Wire up the preserved intrinsic functions that we saved before running wasm-bindgen to the expected
    // imports from the patch.
    for import_id in wbg_funcs {
        let import = new.imports.get_mut(import_id);
        let ImportKind::Function(func_id) = import.kind else {
            continue;
        };

        import.module = "env".into();
        import.name = format!("__saved_wbg_{}", import.name);

        if name_is_bindgen_symbol(&import.name) {
            let name = import.name.as_str().to_string();
            new.imports.delete(import_id);
            convert_func_to_ifunc_call(&mut new, ifunc_table_initializer, func_id, 0, name);
        }
    }

    // Rewrite the wbg_cast functions to call the indirect functions from the original module.
    // This is necessary because wasm-bindgen uses these calls to perform dynamic type casting through
    // the JS layer. If we don't rewrite these, they end up as calls to `breaks_if_inlined` functions
    // which are no-ops and get rewritten by the wbindgen post-processing step.
    //
    // Here, we find the corresponding wbg_cast function in the old module by name and then rewrite
    // the patch module's cast function to call the indirect function from the original module.
    //
    // See the wbg_cast implementation in wasm-bindgen for more details:
    // <https://github.com/wasm-bindgen/wasm-bindgen/blob/f61a588f674304964a2062b2307edb304aed4d16/src/rt/mod.rs#L30>
    let new_func_ids = new.funcs.iter().map(|f| f.id()).collect::<Vec<_>>();
    let mut wbg_cast_named = 0usize;
    let mut wbg_cast_rewritten = 0usize;
    for func_id in new_func_ids {
        let Some(name) = new.funcs.get(func_id).name.as_deref() else {
            continue;
        };

        if name.contains("wbg_cast") {
            wbg_cast_named += 1;
        }

        if name_is_wbg_cast_symbol(name) {
            let name = name.to_string();
            let old_idx = name_to_ifunc_old
                    .get(&name)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("Could not find matching wbg_cast function for [{name}] - must generate new JS bindings."))?;

            convert_func_to_ifunc_call(&mut new, ifunc_table_initializer, func_id, old_idx, name);
            wbg_cast_rewritten += 1;
        }
    }

    tracing::info!(
        "Patch edits: GOT.func={n_got_funcs} GOT.mem={n_got_mems} env_funcs={n_env_funcs} wbg_funcs={n_wbg_funcs} wbg_cast(named={wbg_cast_named}, rewritten={wbg_cast_rewritten})"
    );

    // Wipe away the unnecessary sections
    let customs = new.customs.iter().map(|f| f.0).collect::<Vec<_>>();
    for custom_id in customs {
        if let Some(custom) = new.customs.get_mut(custom_id) {
            if custom.name().contains("manganis") || custom.name().contains("__wasm_bindgen") {
                new.customs.delete(custom_id);
            }
        }
    }

    // Clear the start function from the patch - we don't want any code automatically running!
    new.start = None;

    // Export __wasm_apply_global_relocs if it exists. wasm-ld generates this synthetic
    // function to relocate GOT.func.internal globals by __table_base, but refuses to
    // export it via --export or --export-if-defined since it's not a linker symbol.
    // Without this export, the runtime can't call it, leaving GOT.func.internal globals
    // unrelocated — they contain element-segment-relative offsets instead of absolute
    // table indices, causing call_indirect type mismatches in PIC-compiled workspace code.
    const APPLY_RELOCS: &str = "__wasm_apply_global_relocs";
    if let Some(func) = new
        .funcs
        .iter()
        .find(|f| f.name.as_deref() == Some(APPLY_RELOCS))
    {
        new.exports.add(APPLY_RELOCS, func.id());
    }

    let t_rewritten = t_start.elapsed();

    // Update the wasm module on the filesystem to use the newly lifted version.
    // Strip the wasm-ld linker sidecars (`linking` + `reloc.*`) from the emitted bytes -
    // the browser does not read them and they are commonly 30-50% of the patch size.
    let lib = patch.to_path_buf();
    let bytes = new.emit_wasm();
    let bytes = strip_linker_sidecars(&bytes, keep_names, dwarf_sidecar.is_some());
    std::fs::write(&lib, bytes)?;
    let t_emitted = t_start.elapsed();

    // And now assemble the jump table by mapping the old ifunc table to the new one, by name
    //
    // The ifunc_count will be passed to the dynamic loader so it can allocate the right amount of space
    // in the indirect function table when loading the patch.
    let name_to_ifunc_new = collect_func_ifuncs(&new);
    // Must be the element segment's slot count, not `name_to_ifunc_new.len()`: the latter collapses
    // element items sharing a mangled name, so it undercounts and the runtime grows the shared table
    // too little → "table index is out of bounds" at instantiate.
    let ifunc_count = count_ifunc_slots(&new);
    let mut map = AddressMap::default();
    for (name, idx) in name_to_ifunc_new.iter() {
        // Find the corresponding ifunc in the old module by name
        if let Some(old_idx) = name_to_ifunc_old.get(*name) {
            map.insert(*old_idx as u64, *idx as u64);
            continue;
        }
    }
    let t_mapped = t_start.elapsed();

    // Determine which old table slots are safe to overwrite in place with the patched function.
    //
    // Repointing an old slot to the new function makes type-erased values that were created before
    // the patch (trait-object vtables, `drop_in_place`/`type_id` glue) dispatch into patched code
    // instead of the stale original. Because the patch's `GOT.func` entries resolve to *old* ifunc
    // indices, both pre- and post-patch vtables reference those old slots, so this also preserves
    // function-pointer identity across the boundary.
    //
    // We only repoint slots whose old and new functions have identical signatures. The `map` is
    // matched by name and, at high opt levels, several differently-typed symbols can collapse onto
    // one ifunc index, so a name-matched pair may otherwise install a wrong-signature funcref and
    // corrupt unrelated `call_indirect` sites.
    let old_sigs = &cache.ifunc_sigs;
    let new_sigs: HashMap<i32, SigVec> = collect_ifunc_signatures(&new)
        .into_iter()
        .map(|(idx, (params, results))| {
            (
                idx,
                (
                    params.iter().map(walrus_valtype_sig).collect(),
                    results.iter().map(walrus_valtype_sig).collect(),
                ),
            )
        })
        .collect();
    let mut ifunc_repoint = Vec::new();
    for (&old_idx, &new_idx) in map.iter() {
        if let (Some(old_sig), Some(new_sig)) = (
            old_sigs.get(&(old_idx as i32)),
            new_sigs.get(&(new_idx as i32)),
        ) {
            if old_sig == new_sig {
                ifunc_repoint.push((old_idx, new_idx));
            }
        }
    }
    let t_repointed = t_start.elapsed();

    tracing::info!(
        "Jump table (walrus): read={}ms parse={}ms rewrite={}ms emit={}ms map={}ms repoint={}ms total={}ms | map={} repoint={} ifunc_count={ifunc_count}",
        t_read.as_millis(),
        t_parsed.saturating_sub(t_read).as_millis(),
        t_rewritten.saturating_sub(t_parsed).as_millis(),
        t_emitted.saturating_sub(t_rewritten).as_millis(),
        t_mapped.saturating_sub(t_emitted).as_millis(),
        t_repointed.saturating_sub(t_mapped).as_millis(),
        t_repointed.as_millis(),
        map.len(),
        ifunc_repoint.len(),
    );

    if map.is_empty() {
        tracing::warn!(
            "Jump table (walrus): map is EMPTY — the patch will apply but old code keeps running \
             (patch defined-function names didn't intersect the base cache's symbol→ifunc map)."
        );
    }

    Ok(JumpTable {
        map,
        lib,
        ifunc_count,
        dwarf_sidecar,
        aslr_reference: 0,
        new_base_address: 0,
        ifunc_repoint,
        previous_patch: None,
        wasm: None,
        base_id: cache.base_id,
    })
}

fn convert_func_to_ifunc_call(
    new: &mut Module,
    ifunc_table_initializer: TableId,
    func_id: FunctionId,
    table_idx: i32,
    name: String,
) {
    use walrus::ir;

    let func = new.funcs.get_mut(func_id);
    let ty_id = func.ty();

    // Convert the import function to a local function that calls the indirect function from the table
    let ty = new.types.get(ty_id);
    let params = ty.params().to_vec();
    let results = ty.results().to_vec();
    let locals: Vec<_> = params.iter().map(|ty| new.locals.add(*ty)).collect();

    // New function that calls the indirect function
    let mut builder = FunctionBuilder::new(&mut new.types, &params, &results);
    let mut body = builder.name(name).func_body();

    // Push the params onto the stack
    for arg in locals.iter() {
        body.local_get(*arg);
    }

    // And then the address of the indirect function
    body.instr(ir::Instr::Const(ir::Const {
        value: ir::Value::I32(table_idx),
    }));

    // And call it
    body.instr(ir::Instr::CallIndirect(ir::CallIndirect {
        ty: ty_id,
        table: ifunc_table_initializer,
    }));

    new.funcs.get_mut(func_id).kind = FunctionKind::Local(builder.local_func(locals));
}

/// Number of table slots the module's active element segments occupy (highest `offset + len`).
/// This is what the runtime must grow the shared table by; unlike `collect_func_ifuncs().len()` it
/// counts every element item, including ones whose function shares a mangled name with another.
fn count_ifunc_slots(m: &Module) -> u64 {
    let mut slots = 0u64;
    for el in m.elements.iter() {
        let ElementKind::Active { offset, .. } = &el.kind else {
            continue;
        };
        let offset = match offset {
            ConstExpr::Value(walrus::ir::Value::I32(idx)) => *idx as i64,
            ConstExpr::Value(walrus::ir::Value::I64(idx)) => *idx,
            // Global-relative (imported `__table_base`): explicit offset contributes 0.
            ConstExpr::Global(_) => 0,
            _ => continue,
        };
        let len = match &el.items {
            ElementItems::Functions(ids) => ids.len() as u64,
            ElementItems::Expressions(_, exprs) => exprs.len() as u64,
        };
        slots = slots.max(offset.max(0) as u64 + len);
    }
    slots
}

/// The functions that a patch must define. See `HotpatchModuleCache::patch_functions`.
pub struct PatchFunctions<'a> {
    /// The functions of `needed` that have a slot in the base table. They are the roots of the
    /// patch link.
    pub roots: Vec<&'a str>,
    /// Every function that the patch must define: the changed functions, every function that
    /// reaches one of them through direct calls, and every callee of those that the patch
    /// cannot import from the base.
    pub needed: HashSet<&'a str>,
}

impl HotpatchModuleCache {
    /// The functions that a patch must define, from the names of the functions whose code
    /// changed.
    ///
    /// The base module calls a changed function either through a table slot, which the jump
    /// table repoints, or through a direct call from another function. The patch must hold a
    /// new copy of every function on such a direct call chain, up to the table slot that starts
    /// the chain. Those table functions are the roots of the patch link. Every other call from
    /// patch code goes to the base through an import, which the jump table resolves from the
    /// table or from the exports of the base. A callee that is in neither must also be in the
    /// patch, with its own callees in turn.
    pub fn patch_functions<'a>(
        &'a self,
        seeds: impl IntoIterator<Item = &'a str>,
    ) -> PatchFunctions<'a> {
        let mut reached = vec![false; self.callers.len()];
        let mut to_visit: Vec<u32> = seeds
            .into_iter()
            .filter_map(|name| self.symbol_func_index.get(name).copied())
            .collect();
        let mut order = Vec::new();
        while let Some(index) = to_visit.pop() {
            let slot = &mut reached[index as usize];
            if *slot {
                continue;
            }
            *slot = true;
            order.push(index);
            to_visit.extend(self.callers[index as usize].iter().copied());
        }
        let mut to_visit = order;
        while let Some(index) = to_visit.pop() {
            for callee in &self.callees[index as usize] {
                let callee = *callee as usize;
                if reached[callee]
                    || self.in_table[callee]
                    || self.old_exports.contains(&self.func_names[callee])
                {
                    continue;
                }
                reached[callee] = true;
                to_visit.push(callee as u32);
            }
        }
        let needed: HashSet<&str> = self
            .symbol_func_index
            .iter()
            .filter(|(_, index)| reached[**index as usize])
            .map(|(name, _)| name.as_str())
            .collect();
        let roots = needed
            .iter()
            .copied()
            .filter(|name| self.symbol_ifunc_map.contains_key(*name))
            .collect();
        PatchFunctions { roots, needed }
    }
}

/// The direct callers and the direct callees of every function of `module`, by wasm function
/// index. `ids` maps a wasm function index to its walrus id.
fn collect_direct_calls(module: &Module, ids: &[FunctionId]) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    struct Calls {
        callees: Vec<FunctionId>,
    }
    impl<'a> walrus::ir::Visitor<'a> for Calls {
        fn visit_instr(&mut self, instr: &'a walrus::ir::Instr, _loc: &'a walrus::ir::InstrLocId) {
            if let walrus::ir::Instr::Call(call) = instr {
                self.callees.push(call.func);
            }
        }
    }

    let id_to_index: HashMap<FunctionId, u32> = ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index as u32))
        .collect();
    let edges: Vec<(u32, u32)> = module
        .funcs
        .par_iter_local()
        .flat_map_iter(|(id, local)| {
            let mut calls = Calls {
                callees: Vec::new(),
            };
            walrus::ir::dfs_in_order(&mut calls, local, local.entry_block());
            calls.callees.sort_unstable();
            calls.callees.dedup();
            let caller = id_to_index[&id];
            let id_to_index = &id_to_index;
            calls
                .callees
                .into_iter()
                .map(move |callee| (caller, id_to_index[&callee]))
                .collect::<Vec<_>>()
        })
        .collect();
    let mut callers = vec![Vec::new(); ids.len()];
    let mut callees = vec![Vec::new(); ids.len()];
    for (caller, callee) in edges {
        callers[callee as usize].push(caller);
        callees[caller as usize].push(callee);
    }
    (callers, callees)
}

/// A function of an rlib or of an object. See `function_hashes`.
#[derive(Clone, Debug)]
pub struct FunctionOrigin {
    /// The canonical hash of the code of the function.
    pub hash: u64,
    /// The index of the object that defines the function, in `ObjectIndex::objects`.
    pub object: usize,
}

/// One wasm object of a link input: an object file, or one member of an rlib.
#[derive(Clone, Debug)]
pub struct PatchObject {
    /// The index of the path in the `paths` of the call.
    pub source: usize,
    /// The byte range of the archive member, or `None` for an object file.
    pub member: Option<Range<usize>>,
    /// The hidden functions that the object does not define but takes the address of, in its
    /// code or in its data. wasm-ld fails on a table entry for an undefined hidden function,
    /// so the object that defines each one must be in the same link.
    pub hidden_address_refs: Vec<String>,
}

/// The functions and the objects of a set of link inputs. See `function_hashes`.
#[derive(Default, Debug)]
pub struct ObjectIndex {
    pub functions: HashMap<String, FunctionOrigin>,
    pub objects: Vec<PatchObject>,
}

/// The object index of each link input that an earlier call of `function_hashes` hashed, by
/// path. The size and the modification time of the file must not change, else the entry is
/// stale. The base files and the rlibs of the crates that did not replay stay the same for
/// the life of a fat build, so a patch hashes only the files that its compile wrote.
#[derive(Default)]
pub struct FileHashCache {
    files: std::sync::Mutex<HashMap<PathBuf, CachedFileHashes>>,
}

struct CachedFileHashes {
    len: u64,
    modified: std::time::SystemTime,
    /// The index of this file alone. Each `PatchObject::source` is 0.
    index: Arc<ObjectIndex>,
}

impl FileHashCache {
    /// The index of the file at `path`, from the cache when the file did not change. Returns
    /// true as the second value when the function hashed the file.
    fn file_index(&self, path: &Path) -> anyhow::Result<(Arc<ObjectIndex>, bool)> {
        let metadata = std::fs::metadata(path)
            .with_context(|| format!("Could not read the metadata of {}", path.display()))?;
        let modified = metadata.modified()?;
        let mut files = self.files.lock().unwrap();
        if let Some(cached) = files.get(path) {
            if cached.len == metadata.len() && cached.modified == modified {
                return Ok((cached.index.clone(), false));
            }
        }
        let index = Arc::new(hash_file_functions(path)?);
        files.insert(
            path.to_path_buf(),
            CachedFileHashes {
                len: metadata.len(),
                modified,
                index: index.clone(),
            },
        );
        Ok((index, true))
    }

    /// Keep the entry of a file that dx moved from `from` to `to`. A move keeps the size and
    /// the modification time, so the entry stays valid.
    pub fn file_moved(&self, from: &Path, to: &Path) {
        let mut files = self.files.lock().unwrap();
        if let Some(cached) = files.remove(from) {
            files.insert(to.to_path_buf(), cached);
        }
    }

    /// Remove the entries of the files that no longer exist, such as the tip objects of an
    /// earlier tip compile. Such a file cannot come back with the same content.
    fn remove_missing(&self) {
        self.files.lock().unwrap().retain(|path, _| path.exists());
    }
}

/// The canonical hash of every function that the wasm objects in `paths` define, by symbol
/// name. A path is an rlib archive or a single object file.
///
/// The hash covers the code of the function with every relocation site set to zero, and the
/// relocations of that code as (offset, type, target name, addend), so that two compiles of the
/// same source give the same hash when the object around the function changed.
///
/// The function hashes only the files that `cache` does not hold, and takes the other files
/// from `cache`.
pub fn function_hashes(paths: &[PathBuf], cache: &FileHashCache) -> anyhow::Result<ObjectIndex> {
    let started = std::time::Instant::now();
    let mut index = ObjectIndex::default();
    let mut hashed = 0;
    for (source, path) in paths.iter().enumerate() {
        let (file, miss) = cache.file_index(path)?;
        hashed += usize::from(miss);
        let offset = index.objects.len();
        index
            .objects
            .extend(file.objects.iter().map(|object| PatchObject {
                source,
                ..object.clone()
            }));
        index
            .functions
            .extend(file.functions.iter().map(|(name, function)| {
                (
                    name.clone(),
                    FunctionOrigin {
                        hash: function.hash,
                        object: function.object + offset,
                    },
                )
            }));
    }
    if hashed > 0 {
        cache.remove_missing();
    }
    tracing::debug!(
        "Hashed the functions of {hashed} of {} files, the cache held the others, in {:?}",
        paths.len(),
        started.elapsed()
    );
    Ok(index)
}

/// The index of the functions and the objects of one rlib or object file. Each
/// `PatchObject::source` is 0.
fn hash_file_functions(path: &Path) -> anyhow::Result<ObjectIndex> {
    let mut index = ObjectIndex::default();
    let bytes =
        std::fs::read(path).with_context(|| format!("Could not read {}", path.display()))?;
    match object::read::archive::ArchiveFile::parse(&*bytes) {
        Ok(archive) => {
            for member in archive.members() {
                let member = member?;
                let (offset, size) = member.file_range();
                let range = offset as usize..(offset + size) as usize;
                let data = member.data(&*bytes)?;
                if data.starts_with(b"\0asm") {
                    hash_object_functions(data, 0, Some(range), &mut index)
                        .with_context(|| format!("In a member of {}", path.display()))?;
                }
            }
        }
        Err(_) => hash_object_functions(&bytes, 0, None, &mut index)
            .with_context(|| format!("In {}", path.display()))?,
    }
    Ok(index)
}

impl ObjectIndex {
    /// The objects that a link of the functions in `wanted` needs: the object that defines
    /// each one, and the objects that define the hidden functions those objects take the
    /// address of, in turn. Returns the object indices, and the names of the wanted or
    /// address-taken functions that no object defines.
    pub fn objects_for<'a>(
        &self,
        wanted: impl IntoIterator<Item = &'a str>,
    ) -> (Vec<usize>, Vec<String>) {
        let mut selected = vec![false; self.objects.len()];
        let mut missing = Vec::new();
        let mut to_visit = Vec::new();
        for name in wanted {
            match self.functions.get(name) {
                Some(function) => to_visit.push(function.object),
                None => missing.push(name.to_string()),
            }
        }
        while let Some(object) = to_visit.pop() {
            if selected[object] {
                continue;
            }
            selected[object] = true;
            for name in &self.objects[object].hidden_address_refs {
                match self.functions.get(name) {
                    Some(function) => to_visit.push(function.object),
                    None => missing.push(name.clone()),
                }
            }
        }
        missing.sort_unstable();
        missing.dedup();
        let objects = (0..self.objects.len())
            .filter(|object| selected[*object])
            .collect();
        (objects, missing)
    }

    /// Write the objects with the indices in `objects` into `dir`, one file per archive member,
    /// and return their paths in link order. An object file keeps its own path.
    pub fn write_objects(
        &self,
        paths: &[PathBuf],
        objects: &[usize],
        dir: &Path,
    ) -> anyhow::Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        let mut archive: Option<(usize, Vec<u8>)> = None;
        for object in objects {
            let PatchObject { source, member, .. } = &self.objects[*object];
            let Some(range) = member else {
                out.push(paths[*source].clone());
                continue;
            };
            if archive.as_ref().is_none_or(|(loaded, _)| loaded != source) {
                archive = Some((*source, std::fs::read(&paths[*source])?));
            }
            let bytes = &archive.as_ref().unwrap().1;
            let stem = paths[*source]
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("member");
            let path = dir.join(format!("{stem}.{}.o", range.start));
            std::fs::write(&path, &bytes[range.clone()])?;
            out.push(path);
        }
        Ok(out)
    }
}

/// Hash every defined function of one wasm object into `index`, and record the object. See
/// `function_hashes`.
fn hash_object_functions(
    bytes: &[u8],
    source: usize,
    member: Option<Range<usize>>,
    index: &mut ObjectIndex,
) -> anyhow::Result<()> {
    use std::hash::{Hash, Hasher};
    use wasmparser::{KnownCustom, TypeRef};

    let mut import_funcs: Vec<String> = Vec::new();
    let mut import_globals: Vec<String> = Vec::new();
    let mut import_tables: Vec<String> = Vec::new();
    let mut import_tags: Vec<String> = Vec::new();
    let mut types: Vec<String> = Vec::new();
    let mut code_start = 0usize;
    let mut bodies: Vec<std::ops::Range<usize>> = Vec::new();
    let mut symbols: Vec<SymbolInfo> = Vec::new();
    let mut relocs: Vec<wasmparser::RelocationEntry> = Vec::new();
    let mut data_relocs: Vec<wasmparser::RelocationEntry> = Vec::new();

    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        match payload? {
            Payload::TypeSection(section) => {
                for group in section {
                    for sub in group?.into_types() {
                        types.push(format!("{:?}", sub.composite_type));
                    }
                }
            }
            Payload::ImportSection(section) => {
                for import in section {
                    let import = import?;
                    // The symbol of an undefined function has the name of the import, without
                    // the module, so it matches the symbol of the object that defines it.
                    let name = import.name.to_string();
                    match import.ty {
                        TypeRef::Func(_) => import_funcs.push(name),
                        TypeRef::Global(_) => import_globals.push(name),
                        TypeRef::Table(_) => import_tables.push(name),
                        TypeRef::Tag(_) => import_tags.push(name),
                        _ => {}
                    }
                }
            }
            Payload::CodeSectionStart { range, .. } => code_start = range.start,
            Payload::CodeSectionEntry(body) => bodies.push(body.range()),
            Payload::CustomSection(section) => match section.as_known() {
                KnownCustom::Linking(reader) => {
                    for subsection in reader.subsections() {
                        if let Linking::SymbolTable(map) = subsection? {
                            symbols = map.into_iter().collect::<Result<Vec<_>, _>>()?;
                        }
                    }
                }
                KnownCustom::Reloc(reader) if section.name() == "reloc.CODE" => {
                    relocs = reader
                        .entries()
                        .into_iter()
                        .collect::<Result<Vec<_>, _>>()?;
                }
                KnownCustom::Reloc(reader) if section.name() == "reloc.DATA" => {
                    data_relocs = reader
                        .entries()
                        .into_iter()
                        .collect::<Result<Vec<_>, _>>()?;
                }
                _ => {}
            },
            _ => {}
        }
    }

    // The name of the target of a relocation, by symbol index. An undefined function, global,
    // table or tag takes the name of its import.
    let import_name = |imports: &[String], index: u32, name: Option<&str>| -> String {
        match name {
            Some(name) => name.to_string(),
            None => imports
                .get(index as usize)
                .cloned()
                .unwrap_or_else(|| format!("#{index}")),
        }
    };
    let symbol_names: Vec<String> = symbols
        .iter()
        .map(|symbol| match symbol {
            SymbolInfo::Func { index, name, .. } => import_name(&import_funcs, *index, *name),
            SymbolInfo::Data { name, .. } => name.to_string(),
            SymbolInfo::Global { index, name, .. } => import_name(&import_globals, *index, *name),
            SymbolInfo::Table { index, name, .. } => import_name(&import_tables, *index, *name),
            SymbolInfo::Event { index, name, .. } => import_name(&import_tags, *index, *name),
            SymbolInfo::Section { section, .. } => format!("section:{section}"),
        })
        .collect();
    let mut defined_names: HashMap<u32, &str> = HashMap::new();
    for symbol in &symbols {
        if let SymbolInfo::Func {
            index,
            name: Some(name),
            ..
        } = symbol
        {
            if *index as usize >= import_funcs.len() {
                defined_names.insert(*index, name);
            }
        }
    }

    // The hidden functions that the object takes the address of but does not define.
    let mut hidden_address_refs: Vec<String> = relocs
        .iter()
        .chain(&data_relocs)
        .filter(|reloc| {
            use wasmparser::RelocationType::*;
            matches!(
                reloc.ty,
                TableIndexSleb
                    | TableIndexI32
                    | TableIndexRelSleb
                    | TableIndexSleb64
                    | TableIndexI64
                    | TableIndexRelSleb64
            )
        })
        .filter_map(|reloc| match symbols.get(reloc.index as usize) {
            Some(SymbolInfo::Func { flags, .. })
                if flags.contains(wasmparser::SymbolFlags::UNDEFINED)
                    && flags.contains(wasmparser::SymbolFlags::VISIBILITY_HIDDEN) =>
            {
                symbol_names.get(reloc.index as usize).cloned()
            }
            _ => None,
        })
        .collect();
    hidden_address_refs.sort_unstable();
    hidden_address_refs.dedup();
    let object = index.objects.len();
    index.objects.push(PatchObject {
        source,
        member,
        hidden_address_refs,
    });

    relocs.sort_by_key(|reloc| reloc.offset);
    let mut next_reloc = 0;
    for (position, body) in bodies.iter().enumerate() {
        let index_of_function = (import_funcs.len() + position) as u32;
        let start = body.start - code_start;
        let end = body.end - code_start;
        let mut code = bytes[body.clone()].to_vec();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        while next_reloc < relocs.len() && (relocs[next_reloc].offset as usize) < end {
            let reloc = relocs[next_reloc];
            next_reloc += 1;
            let site = reloc.relocation_range();
            if site.start < start {
                continue;
            }
            let local = (site.start - start)..(site.end - start).min(code.len());
            code[local].fill(0);
            (site.start - start).hash(&mut hasher);
            (reloc.ty as u8).hash(&mut hasher);
            if reloc.ty == wasmparser::RelocationType::TypeIndexLeb {
                types.get(reloc.index as usize).hash(&mut hasher);
            } else {
                symbol_names.get(reloc.index as usize).hash(&mut hasher);
            }
            reloc.addend.hash(&mut hasher);
        }
        code.hash(&mut hasher);
        if let Some(name) = defined_names.get(&index_of_function) {
            index.functions.insert(
                (*name).to_string(),
                FunctionOrigin {
                    hash: hasher.finish(),
                    object,
                },
            );
        }
    }
    Ok(())
}

fn collect_func_ifuncs(m: &Module) -> HashMap<&str, i32> {
    // Collect all the functions in the module that are ifuncs
    let mut func_to_offset = HashMap::new();
    for el in m.elements.iter() {
        let ElementKind::Active { offset, .. } = &el.kind else {
            continue;
        };

        let offset = match offset {
            // Handle explicit offsets
            ConstExpr::Value(value) => match value {
                walrus::ir::Value::I32(idx) => *idx,
                walrus::ir::Value::I64(idx) => *idx as i32,
                _ => continue,
            },

            // Globals are usually imports and thus don't add a specific offset
            // ie the ifunc table is offset by a global, so we don't need to push the offset out
            ConstExpr::Global(_) => 0,
            _ => continue,
        };

        match &el.items {
            ElementItems::Functions(ids) => {
                for (idx, id) in ids.iter().enumerate() {
                    if let Some(name) = m.funcs.get(*id).name.as_deref() {
                        func_to_offset.insert(name, offset + idx as i32);
                    }
                }
            }
            ElementItems::Expressions(_ref_type, _const_exprs) => {}
        }
    }

    func_to_offset
}

/// Map every ifunc-table index to the wasm signature of the function in that slot.
///
/// Used to decide which `map` entries are safe to overwrite in place at runtime: only pairs whose
/// old and new slots share the exact same `(params, results)` may be repointed. A name-matched pair
/// is *not* enough, because at high opt levels several differently-typed symbols can collapse onto a
/// single ifunc index.
fn collect_ifunc_signatures(m: &Module) -> HashMap<i32, (Vec<ValType>, Vec<ValType>)> {
    let mut idx_to_sig = HashMap::new();
    for el in m.elements.iter() {
        let ElementKind::Active { offset, .. } = &el.kind else {
            continue;
        };

        let offset = match offset {
            ConstExpr::Value(walrus::ir::Value::I32(idx)) => *idx,
            ConstExpr::Value(walrus::ir::Value::I64(idx)) => *idx as i32,
            ConstExpr::Global(_) => 0,
            _ => continue,
        };

        if let ElementItems::Functions(ids) = &el.items {
            for (i, id) in ids.iter().enumerate() {
                let ty = m.types.get(m.funcs.get(*id).ty());
                idx_to_sig.insert(
                    offset + i as i32,
                    (ty.params().to_vec(), ty.results().to_vec()),
                );
            }
        }
    }

    idx_to_sig
}

/// Resolve the undefined symbols in the incrementals against the original binary, returning an object
/// file that can be linked along the incrementals.
///
/// This makes it possible to dlopen the resulting object file and use the original binary's symbols
/// bypassing the dynamic linker.
///
/// This is very similar to malware :) but it's not!
///
/// Note - this function is not defined to run on WASM binaries. The `object` crate does not
///
/// todo... we need to wire up the cache
pub fn create_undefined_symbol_stub(
    cache: &HotpatchModuleCache,
    incrementals: &[PathBuf],
    triple: &Triple,
    aslr_reference: u64,
) -> Result<Vec<u8>> {
    let sorted: Vec<_> = incrementals.iter().sorted().collect();

    // Find all the undefined symbols in the incrementals
    let mut undefined_symbols = HashSet::new();
    let mut defined_symbols = HashSet::new();

    for path in sorted {
        collect_stub_symbols_from_path(path, &mut undefined_symbols, &mut defined_symbols)?;
    }
    let undefined_symbols: Vec<_> = undefined_symbols
        .difference(&defined_symbols)
        .cloned()
        .collect();

    tracing::trace!("Undefined symbols: {:#?}", undefined_symbols);

    // Create a new object file (architecture doesn't matter much for our purposes)
    let mut obj = object::write::Object::new(
        match triple.binary_format {
            target_lexicon::BinaryFormat::Elf => object::BinaryFormat::Elf,
            target_lexicon::BinaryFormat::Macho => object::BinaryFormat::MachO,
            target_lexicon::BinaryFormat::Coff => object::BinaryFormat::Coff,
            target_lexicon::BinaryFormat::Wasm => object::BinaryFormat::Wasm,
            target_lexicon::BinaryFormat::Xcoff => object::BinaryFormat::Xcoff,
            _ => return Err(PatchError::UnsupportedPlatform(triple.to_string())),
        },
        match triple.architecture {
            Architecture::Aarch64(_) => object::Architecture::Aarch64,
            Architecture::Wasm32 => object::Architecture::Wasm32,
            Architecture::X86_64 => object::Architecture::X86_64,
            _ => return Err(PatchError::UnsupportedPlatform(triple.to_string())),
        },
        match triple.endianness() {
            Ok(target_lexicon::Endianness::Little) => Endianness::Little,
            Ok(target_lexicon::Endianness::Big) => Endianness::Big,
            _ => Endianness::Little,
        },
    );

    // Write the headers so we load properly in ios/macos
    #[allow(clippy::identity_op)]
    match triple.operating_system {
        OperatingSystem::Darwin(_) => {
            obj.set_macho_build_version({
                let mut build_version = MachOBuildVersion::default();
                build_version.platform = macho::PLATFORM_MACOS;
                build_version.minos = (11 << 16) | (0 << 8) | 0; // 11.0.0
                build_version.sdk = (11 << 16) | (0 << 8) | 0; // SDK 11.0.0
                build_version
            });
        }
        OperatingSystem::IOS(_) => {
            obj.set_macho_build_version({
                let mut build_version = MachOBuildVersion::default();
                build_version.platform = match triple.environment {
                    target_lexicon::Environment::Sim => macho::PLATFORM_IOSSIMULATOR,
                    _ => macho::PLATFORM_IOS,
                };
                build_version.minos = (14 << 16) | (0 << 8) | 0; // 14.0.0
                build_version.sdk = (14 << 16) | (0 << 8) | 0; // SDK 14.0.0
                build_version
            });
        }

        _ => {}
    }

    // Get the offset from the main module and adjust the addresses by the slide;
    let aslr_ref_address = cache
        .symbol_table
        .get(main_sentinel(triple))
        .context("failed to find '_main' symbol in patch")?
        .address;

    if aslr_reference < aslr_ref_address {
        return Err(PatchError::InvalidModule(format!(
            "ASLR reference is less than the main module's address - is there a `main`?. {aslr_reference:x} < {aslr_ref_address:x}"
        )));
    }

    let aslr_offset = aslr_reference - aslr_ref_address;

    // we need to assemble a PLT/GOT so direct calls to the patch symbols work
    // for each symbol we either write the address directly (as a symbol) or create a PLT/GOT entry
    let text_section = obj.section_id(StandardSection::Text);
    for name in undefined_symbols {
        let Some(sym) = cache
            .symbol_table
            .get(name.as_str().trim_start_matches("__imp_"))
        else {
            tracing::debug!("Symbol not found: {}", name);
            continue;
        };

        // Undefined symbols tend to be import symbols (darwin gives them an address of 0 until defined).
        // If we fail to skip these, then we end up with stuff like alloc at 0x0 which is quite bad!
        if sym.is_undefined {
            continue;
        }

        // ld64 likes to prefix symbols in intermediate object files with an underscore, but our symbol
        // table doesn't, so we need to strip it off.
        let name_offset = match triple.operating_system {
            OperatingSystem::MacOSX(_) | OperatingSystem::Darwin(_) | OperatingSystem::IOS(_) => 1,
            _ => 0,
        };

        let abs_addr = sym.address + aslr_offset;

        match sym.kind {
            // Handle synthesized window linker cross-dll statics.
            //
            // The `__imp_` prefix is a rather poorly documented feature of link.exe that makes it possible
            // to reference statics in DLLs via text sections. The linker will synthesize a function
            // that returns the address of the static, so calling that function will return the address.
            // We want to satisfy it by creating a data symbol with the contents of the *actual* symbol
            // in the original binary.
            //
            // We ca't use the `__imp_` from the original binary because it was not properly compiled
            // with this in mind. Instead we have to create the new symbol.
            //
            // This is currently only implemented for 64bit architectures (haven't tested 32bit yet).
            //
            // https://stackoverflow.com/questions/5159353/how-can-i-get-rid-of-the-imp-prefix-in-the-linker-in-vc
            _ if name.starts_with("__imp_") => {
                let data_section = obj.section_id(StandardSection::Data);

                // Add a pointer to the resolved address
                let offset = obj.append_section_data(
                    data_section,
                    &abs_addr.to_le_bytes(),
                    8, // Use proper alignment
                );

                // Add the symbol as a data symbol in our data section
                obj.add_symbol(Symbol {
                    name: name.as_bytes().to_vec(),
                    value: offset, // Offset within the data section
                    size: 8,       // Size of pointer
                    scope: SymbolScope::Linkage,
                    kind: SymbolKind::Data, // Always Data for IAT entries
                    weak: false,
                    section: SymbolSection::Section(data_section),
                    flags: SymbolFlags::None,
                });
            }

            // Text symbols are normal code symbols. We need to assemble stubs that resolve the undefined
            // symbols and jump to the original address in the original binary.
            //
            // Unfortunately this isn't simply cross-platform, so we need to handle Unix and Windows
            // calling conventions separately. It also depends on the architecture, making it even more
            // complicated.
            SymbolKind::Text => {
                let jump_asm = match triple.operating_system {
                    // The windows ABI and calling convention is different than the SystemV ABI.
                    OperatingSystem::Windows => match triple.architecture {
                        Architecture::X86_64 => {
                            // Windows x64 has specific requirements for alignment and position-independent code
                            let mut code = vec![
                                0x48, 0xB8, // movabs RAX, imm64 (move 64-bit immediate to RAX)
                            ];
                            // Append the absolute 64-bit address
                            code.extend_from_slice(&abs_addr.to_le_bytes());
                            // jmp RAX (jump to the address in RAX)
                            code.extend_from_slice(&[0xFF, 0xE0]);
                            code
                        }
                        Architecture::X86_32(_) => {
                            // On Windows 32-bit, we can use direct jump but need proper alignment
                            let mut code = vec![
                                0xB8, // mov EAX, imm32 (move immediate value to EAX)
                            ];
                            // Append the absolute 32-bit address
                            code.extend_from_slice(&(abs_addr as u32).to_le_bytes());
                            // jmp EAX (jump to the address in EAX)
                            code.extend_from_slice(&[0xFF, 0xE0]);
                            code
                        }
                        Architecture::Aarch64(_) => {
                            // Use MOV/MOVK sequence to load 64-bit address into X16
                            // This is more reliable than ADRP+LDR for direct hotpatching
                            let mut code = Vec::new();

                            // MOVZ X16, #imm16_0 (bits 0-15 of address)
                            let imm16_0 = (abs_addr & 0xFFFF) as u16;
                            let movz = 0xD2800010u32 | ((imm16_0 as u32) << 5);
                            code.extend_from_slice(&movz.to_le_bytes());

                            // MOVK X16, #imm16_1, LSL #16 (bits 16-31 of address)
                            let imm16_1 = ((abs_addr >> 16) & 0xFFFF) as u16;
                            let movk1 = 0xF2A00010u32 | ((imm16_1 as u32) << 5);
                            code.extend_from_slice(&movk1.to_le_bytes());

                            // MOVK X16, #imm16_2, LSL #32 (bits 32-47 of address)
                            let imm16_2 = ((abs_addr >> 32) & 0xFFFF) as u16;
                            let movk2 = 0xF2C00010u32 | ((imm16_2 as u32) << 5);
                            code.extend_from_slice(&movk2.to_le_bytes());

                            // MOVK X16, #imm16_3, LSL #48 (bits 48-63 of address)
                            let imm16_3 = ((abs_addr >> 48) & 0xFFFF) as u16;
                            let movk3 = 0xF2E00010u32 | ((imm16_3 as u32) << 5);
                            code.extend_from_slice(&movk3.to_le_bytes());

                            // BR X16 (Branch to address in X16)
                            code.extend_from_slice(&[0x00, 0x02, 0x1F, 0xD6]);

                            code
                        }
                        Architecture::Arm(_) => {
                            // For Windows 32-bit ARM, we need a different approach
                            let mut code = Vec::new();
                            // LDR r12, [pc, #8] ; Load the address into r12
                            code.extend_from_slice(&[0x08, 0xC0, 0x9F, 0xE5]);
                            // BX r12 ; Branch to the address in r12
                            code.extend_from_slice(&[0x1C, 0xFF, 0x2F, 0xE1]);
                            // 4-byte alignment padding
                            code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                            // Store the 32-bit address - 4-byte aligned
                            code.extend_from_slice(&(abs_addr as u32).to_le_bytes());
                            code
                        }
                        _ => return Err(PatchError::UnsupportedPlatform(triple.to_string())),
                    },
                    _ => match triple.architecture {
                        Architecture::X86_64 => {
                            // Use JMP instruction to absolute address: FF 25 followed by 32-bit offset
                            // Then the 64-bit absolute address
                            let mut code = vec![0xFF, 0x25, 0x00, 0x00, 0x00, 0x00]; // jmp [rip+0]
                                                                                     // Append the 64-bit address
                            code.extend_from_slice(&abs_addr.to_le_bytes());
                            code
                        }
                        Architecture::X86_32(_) => {
                            // For 32-bit Intel, use JMP instruction with absolute address
                            let mut code = vec![0xE9]; // jmp rel32
                            let rel_addr = abs_addr as i32 - 5; // Relative address (offset from next instruction)
                            code.extend_from_slice(&rel_addr.to_le_bytes());
                            code
                        }
                        Architecture::Aarch64(_) => {
                            // For ARM64, we load the address into a register and branch
                            let mut code = Vec::new();
                            // LDR X16, [PC, #0]  ; Load from the next instruction
                            code.extend_from_slice(&[0x50, 0x00, 0x00, 0x58]);
                            // BR X16            ; Branch to the address in X16
                            code.extend_from_slice(&[0x00, 0x02, 0x1F, 0xD6]);
                            // Store the 64-bit address
                            code.extend_from_slice(&abs_addr.to_le_bytes());
                            code
                        }
                        Architecture::Arm(_) => {
                            // For 32-bit ARM, use LDR PC, [PC, #-4] to load the address and branch
                            let mut code = Vec::new();
                            // LDR PC, [PC, #-4] ; Load the address into PC (branching to it)
                            code.extend_from_slice(&[0x04, 0xF0, 0x1F, 0xE5]);
                            // Store the 32-bit address
                            code.extend_from_slice(&(abs_addr as u32).to_le_bytes());
                            code
                        }
                        _ => return Err(PatchError::UnsupportedPlatform(triple.to_string())),
                    },
                };
                let offset = obj.append_section_data(text_section, &jump_asm, 8);
                obj.add_symbol(Symbol {
                    name: name.as_bytes()[name_offset..].to_vec(),
                    value: offset,
                    size: jump_asm.len() as u64,
                    scope: SymbolScope::Linkage,
                    kind: SymbolKind::Text,
                    weak: false,
                    section: SymbolSection::Section(text_section),
                    flags: SymbolFlags::None, // ignore for these stubs
                });
            }

            // Rust code typically generates Tls accessors as functions (text), but they are referenced
            // indirectly as data symbols. We end up handling this by adding the TLS symbol as a data
            // symbol with the initializer as the address of the original tls initializer. That way
            // if new TLS are added at runtime, they get initialized properly, but otherwise, the
            // tls initialization check (cbz) properly skips re-initialization on patches.
            //
            // ```
            // __ZN17crossbeam_channel5waker17current_thread_id9THREAD_ID29_$u7b$$u7b$constant$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$17h33618d877d86bb77E:
            //    stp     x20, x19, [sp, #-0x20]!
            //    stp     x29, x30, [sp, #0x10]
            //    add     x29, sp, #0x10
            //    adrp    x19, 21603 ; 0x1054bd000
            //    add     x19, x19, #0x998
            //    ldr     x20, [x19]
            //    mov     x0, x19
            //    blr     x20
            //    ldr     x8, [x0]
            //    cbz     x8, 0x10005acc0
            //    mov     x0, x19
            //    blr     x20
            //    ldp     x29, x30, [sp, #0x10]
            //    ldp     x20, x19, [sp], #0x20
            //    ret
            //    mov     x0, x19
            //    blr     x20
            //    bl      __ZN3std3sys12thread_local6native4lazy20Storage$LT$T$C$D$GT$10initialize17h818476638edff4e6E
            //    b       0x10005acac
            // ```
            SymbolKind::Tls => {
                let tls_section = obj.section_id(StandardSection::Tls);

                let pointer_width = match triple.pointer_width().unwrap() {
                    PointerWidth::U16 => 2,
                    PointerWidth::U32 => 4,
                    PointerWidth::U64 => 8,
                };

                // Resolve the TLS init data offset and size.
                //
                // On ELF: sym.address IS the TLS offset and sym.size is the data size.
                // On Mach-O: sym.address points to __thread_vars (TLV descriptor), NOT
                // __thread_data. Mach-O nlist has no size field (always 0). We look up
                // the corresponding $tlv$init symbol (LLVM convention) to get the real
                // offset and size within __thread_data.
                //
                // Note: each patch gets its own TLS copy (not shared with the main exe).
                // TLS variables reset to their initial value on patch.
                // Use the full name (with Mach-O `_` prefix) since tls_init_sizes
                // keys come from the same symbol table and include the prefix.
                let init_key = format!("{}$tlv$init", name);
                let (tls_offset, size) =
                    if let Some(&(offset, size)) = cache.tls_init_sizes.get(&init_key) {
                        // macOS: found the $tlv$init symbol with correct offset and size
                        (offset, size)
                    } else if sym.size > 0 {
                        // ELF: sym.address is the TLS offset, sym.size is the data size
                        (sym.address, sym.size)
                    } else if !cache.tls_init_sizes.is_empty() {
                        // macOS fallback: $tlv$init not found but map isn't empty (binary
                        // might be partially stripped). Use entire tdata as upper bound.
                        (0, cache.tls_init_data.len() as u64)
                    } else {
                        // Last resort (ELF with size=0): use pointer width
                        (sym.address, pointer_width)
                    };

                let align = size.min(pointer_width).next_power_of_two();

                let start = tls_offset as usize;
                let end = start + size as usize;
                let init = if end <= cache.tls_init_data.len() {
                    cache.tls_init_data[start..end].to_vec()
                } else {
                    // Beyond .tdata bounds (.tbss) or Mach-O fallback: zero-init
                    vec![0u8; size as usize]
                };

                // Use add_symbol_data() so the object crate's Mach-O writer auto-creates
                // __thread_vars TLV descriptors (via macho_add_thread_var). Without this,
                // the symbol stays in __thread_data and the runtime misinterprets raw init
                // bytes as a TLV descriptor — first 8 bytes become the thunk pointer.
                let sym_id = obj.add_symbol(Symbol {
                    name: name.as_bytes()[name_offset..].to_vec(),
                    value: 0,
                    size: 0,
                    scope: SymbolScope::Linkage,
                    kind: SymbolKind::Tls,
                    weak: false,
                    section: SymbolSection::Undefined,
                    flags: SymbolFlags::None,
                });
                obj.add_symbol_data(sym_id, tls_section, &init, align);
            }

            // We just assume all non-text symbols are data (globals, statics, etc)
            _ => {
                // darwin statics show up as "unknown" symbols even though they are data symbols.
                let kind = match sym.kind {
                    SymbolKind::Unknown => SymbolKind::Data,
                    k => k,
                };

                // plain linux *wants* these flags, but android doesn't.
                // unsure what's going on here, but this is special cased for now.
                // I think the more advanced linkers don't want these flags, but the default linux linker (ld) does.
                let flags = match triple.environment {
                    target_lexicon::Environment::Android => SymbolFlags::None,
                    _ => sym.flags,
                };

                obj.add_symbol(Symbol {
                    name: name.as_bytes()[name_offset..].to_vec(),
                    value: abs_addr,
                    size: 0,
                    scope: SymbolScope::Linkage,
                    kind,
                    weak: sym.is_weak,
                    section: SymbolSection::Absolute,
                    flags,
                });
            }
        }
    }

    Ok(obj.write()?)
}

fn collect_stub_symbols_from_path(
    path: &Path,
    undefined_symbols: &mut HashSet<String>,
    defined_symbols: &mut HashSet<String>,
) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("failed to read {path:?}"))?;

    if path
        .extension()
        .is_some_and(|ext| matches!(ext.to_str(), Some("rlib" | "a")))
    {
        let mut archive = ar::Archive::new(std::io::Cursor::new(bytes));
        while let Some(entry) = archive.next_entry() {
            let mut entry = entry?;
            let name = std::str::from_utf8(entry.header().identifier()).unwrap_or_default();

            if name.ends_with(".rmeta") || !(name.ends_with(".o") || name.ends_with(".obj")) {
                continue;
            }

            let mut entry_bytes = Vec::with_capacity(entry.header().size() as usize);
            entry.read_to_end(&mut entry_bytes)?;
            collect_stub_symbols_from_bytes(&entry_bytes, undefined_symbols, defined_symbols)?;
        }

        return Ok(());
    }

    collect_stub_symbols_from_bytes(&bytes, undefined_symbols, defined_symbols)
}

fn collect_stub_symbols_from_bytes(
    bytes: &[u8],
    undefined_symbols: &mut HashSet<String>,
    defined_symbols: &mut HashSet<String>,
) -> Result<()> {
    let file = File::parse(bytes)?;
    for symbol in file.symbols() {
        if symbol.is_undefined() {
            undefined_symbols.insert(symbol.name()?.to_string());
        } else if symbol.is_global() {
            defined_symbols.insert(symbol.name()?.to_string());
        }
    }

    Ok(())
}

/// Drop wasm-ld linker-sidecar custom sections (`linking` and `reloc.*`) from a wasm binary.
///
/// These sections only exist because we link with `--emit-relocs` to support the hot-patch
/// flow. Once `HotpatchModuleCache` has consumed them server-side, nothing in the browser
/// reads them — the relocs that matter are baked into `__wasm_apply_data_relocs` /
/// `__wasm_apply_global_relocs` function bodies during link, and the JumpTable is sent to
/// the browser as JSON via the devtools websocket. On real-world hot-patch builds these
/// sections can be 30-50% of the served bytes (e.g. ~99 MB of a 214 MB binary).
///
/// This is a streaming strip: we walk the section header sequence, skip the matching
/// custom sections, and copy every other section's bytes verbatim without parsing their
/// payloads. Cost is ~one memcpy of the input. On a 200 MB wasm it runs in tens of ms,
/// well below the threshold where it would slow a fat or patch build noticeably.
pub fn strip_linker_sidecars(input: &[u8], keep_names: bool, strip_dwarf: bool) -> Vec<u8> {
    if input.len() < 8 || &input[..4] != b"\0asm" {
        return input.to_vec();
    }
    let mut out = Vec::with_capacity(input.len());
    out.extend_from_slice(&input[..8]); // magic + version
    let mut pos = 8;
    while pos < input.len() {
        let section_start = pos;
        let section_id = input[pos];
        pos += 1;
        let Some((section_size, leb_len)) = read_uleb128(&input[pos..]) else {
            // Malformed - bail out, return original bytes untouched.
            return input.to_vec();
        };
        pos += leb_len;
        let payload_end = pos + section_size as usize;
        if payload_end > input.len() {
            return input.to_vec();
        }

        let mut keep = true;
        if section_id == 0 {
            if let Some((name_len, name_leb_len)) = read_uleb128(&input[pos..]) {
                let name_start = pos + name_leb_len;
                let name_end = name_start + name_len as usize;
                if name_end <= payload_end {
                    if let Ok(name) = std::str::from_utf8(&input[name_start..name_end]) {
                        if should_strip_custom_section(name, keep_names, strip_dwarf) {
                            keep = false;
                        }
                    }
                }
            }
        }

        if keep {
            out.extend_from_slice(&input[section_start..payload_end]);
        }
        pos = payload_end;
    }
    out
}

/// Resolve a function's index (in the module's function index space, imports first) by symbol name.
///
/// Prefers the wasm-ld `linking` section symbol table — these patches are linked with `--emit-relocs`
/// so it's always present, and it's the same source walrus reads to populate `Function::name` (the
/// modules carry no `name` custom section). Falls back to the `name` section for completeness.
fn find_patch_func_index(bytes: &[u8], target: &str) -> Option<u32> {
    if let Ok(section) = parse_bytes_to_data_segment(bytes) {
        if let Some(&idx) = section.code_symbol_map.get(target) {
            return Some(idx as u32);
        }
    }
    find_wasm_func_index_by_name(bytes, target)
}

/// Find a function's index (in the module's function index space, imports first) by its `name`
/// custom-section entry. That index space is exactly what an export entry must reference.
fn find_wasm_func_index_by_name(bytes: &[u8], target: &str) -> Option<u32> {
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        let Ok(Payload::CustomSection(s)) = payload else {
            continue;
        };
        if s.name() != "name" {
            continue;
        }
        let reader = wasmparser::NameSectionReader::new(BinaryReader::new(s.data(), 0));
        for subsection in reader {
            let Ok(wasmparser::Name::Function(map)) = subsection else {
                continue;
            };
            for naming in map {
                let Ok(naming) = naming else { continue };
                if naming.name == target {
                    return Some(naming.index);
                }
            }
        }
    }
    None
}

/// Produce the bytes we serve for a fast-path patch directly from the raw linker output.
///
/// Three edits, all at the section/byte level so the code, data, and import sections — and the DWARF
/// that indexes them — pass through untouched:
///   1. strip the custom sections the browser never reads (see [`should_strip_custom_section`]),
///   2. drop the start section (patch code must never auto-run; the runtime drives ctors/relocs), and
///   3. export `__wasm_apply_global_relocs` so the runtime can call it (wasm-ld refuses to export it).
///
/// Cost is ~one memcpy of the input plus a re-encode of the (tiny) export section.
/// Rewrite the element section of a patch so that the active segment on table 0 lists every
/// defined function of the patch.
///
/// The base module calls a function of another crate through a slot of the shared table. The
/// jump table repoints that slot to the patch only when the patch holds a function with the same
/// name in its own element segment. The linker puts a function into the segment only when an
/// object of the patch takes its address. A function that only a skipped dependent calls has no
/// such reference, so this pass appends it. The existing items keep their slots, because the
/// relocation thunk of the patch refers to them by index.
///
/// A patch without an element section gets one, with the offset `global.get __table_base`.
fn extend_element_segment(input: &[u8]) -> Result<Vec<u8>> {
    const SECTION_ELEMENT: u8 = 9;
    const SECTION_CODE: u8 = 10;

    if input.len() < 8 || &input[..4] != b"\0asm" {
        return Ok(input.to_vec());
    }

    // First pass: the function index space and the `__table_base` global.
    let mut imported_funcs = 0u32;
    let mut imported_globals = 0u32;
    let mut table_base_global = None;
    let mut defined_funcs = 0u32;
    for payload in wasmparser::Parser::new(0).parse_all(input) {
        match payload? {
            Payload::ImportSection(reader) => {
                for import in reader {
                    let import = import?;
                    match import.ty {
                        wasmparser::TypeRef::Func(_) => imported_funcs += 1,
                        wasmparser::TypeRef::Global(_) => {
                            if import.name == "__table_base" {
                                table_base_global = Some(imported_globals);
                            }
                            imported_globals += 1;
                        }
                        _ => {}
                    }
                }
            }
            Payload::FunctionSection(reader) => defined_funcs = reader.count(),
            Payload::CodeSectionStart { .. } => break,
            _ => {}
        }
    }
    if defined_funcs == 0 {
        return Ok(input.to_vec());
    }

    // Second pass: find the element section by a header scan, so that the splice keeps every
    // other section byte for byte.
    let mut pos = 8;
    let mut element: Option<(usize, usize, usize)> = None; // (section start, payload start, end)
    let mut code_start = None;
    while pos < input.len() {
        let section_start = pos;
        let section_id = input[pos];
        pos += 1;
        let Some((section_size, leb_len)) = read_uleb128(&input[pos..]) else {
            return Ok(input.to_vec());
        };
        pos += leb_len;
        let payload_start = pos;
        let payload_end = pos + section_size as usize;
        if payload_end > input.len() {
            return Ok(input.to_vec());
        }
        pos = payload_end;
        match section_id {
            SECTION_ELEMENT => element = Some((section_start, payload_start, payload_end)),
            SECTION_CODE if code_start.is_none() => code_start = Some(section_start),
            _ => {}
        }
    }

    // The segments of the existing section. The target is the first active segment on table 0
    // with a `global.get` offset. Every other segment is copied as is.
    let mut segments: Vec<Vec<u8>> = Vec::new();
    let mut target: Option<(usize, Vec<u8>, Vec<u32>)> = None; // (position, offset expr, items)
    if let Some((_, payload_start, payload_end)) = element {
        let payload = &input[payload_start..payload_end];
        let reader =
            wasmparser::ElementSectionReader::new(BinaryReader::new(payload, payload_start))?;
        for segment in reader {
            let segment = segment?;
            let raw = input[segment.range.start..segment.range.end].to_vec();
            if target.is_none() {
                if let wasmparser::ElementKind::Active {
                    table_index,
                    offset_expr,
                } = &segment.kind
                {
                    let mut expr = offset_expr.get_binary_reader();
                    let expr_bytes = expr.read_bytes(expr.bytes_remaining())?.to_vec();
                    let is_global_get = matches!(
                        offset_expr.get_operators_reader().read()?,
                        wasmparser::Operator::GlobalGet { .. }
                    );
                    if table_index.unwrap_or(0) == 0 && is_global_get {
                        if let wasmparser::ElementItems::Functions(funcs) = segment.items {
                            let ids = funcs
                                .into_iter()
                                .collect::<std::result::Result<Vec<u32>, _>>()?;
                            target = Some((segments.len(), expr_bytes, ids));
                            segments.push(Vec::new());
                            continue;
                        }
                    }
                }
            }
            segments.push(raw);
        }
    }
    let (position, offset_expr, mut ids) = match target {
        Some(target) => target,
        None => {
            let Some(global) = table_base_global else {
                tracing::debug!("The patch has no `__table_base` global; the element segment stays");
                return Ok(input.to_vec());
            };
            let mut expr = vec![0x23]; // global.get
            write_uleb128(&mut expr, global);
            expr.push(0x0b); // end
            segments.push(Vec::new());
            (segments.len() - 1, expr, Vec::new())
        }
    };

    let present: HashSet<u32> = ids.iter().copied().collect();
    let added = (imported_funcs..imported_funcs + defined_funcs)
        .filter(|id| !present.contains(id))
        .collect::<Vec<_>>();
    tracing::debug!(
        "Element segment: {} functions from the linker, {} appended",
        ids.len(),
        added.len()
    );
    ids.extend(added);

    let mut segment = vec![0x00]; // flags: active, table 0, funcref indices
    segment.extend_from_slice(&offset_expr);
    write_uleb128(&mut segment, ids.len() as u32);
    for id in &ids {
        write_uleb128(&mut segment, *id);
    }
    segments[position] = segment;

    let mut payload = Vec::new();
    write_uleb128(&mut payload, segments.len() as u32);
    for segment in &segments {
        payload.extend_from_slice(segment);
    }
    let mut section = vec![SECTION_ELEMENT];
    write_uleb128(&mut section, payload.len() as u32);
    section.extend_from_slice(&payload);

    let (splice_start, splice_end) = match element {
        Some((section_start, _, payload_end)) => (section_start, payload_end),
        None => {
            let Some(code_start) = code_start else {
                return Ok(input.to_vec());
            };
            (code_start, code_start)
        }
    };
    let mut out = Vec::with_capacity(input.len() + section.len());
    out.extend_from_slice(&input[..splice_start]);
    out.extend_from_slice(&section);
    out.extend_from_slice(&input[splice_end..]);
    Ok(out)
}

fn finalize_patch_wasm(
    input: &[u8],
    reloc_export: Option<u32>,
    keep_names: bool,
    strip_dwarf: bool,
) -> Result<Vec<u8>> {
    const SECTION_EXPORT: u8 = 7;
    const SECTION_START: u8 = 8;

    if input.len() < 8 || &input[..4] != b"\0asm" {
        return Ok(input.to_vec());
    }

    let mut out = Vec::with_capacity(input.len());
    out.extend_from_slice(&input[..8]); // magic + version

    let mut pos = 8;
    let mut export_emitted = false;
    while pos < input.len() {
        let section_start = pos;
        let section_id = input[pos];
        pos += 1;
        let Some((section_size, leb_len)) = read_uleb128(&input[pos..]) else {
            return Ok(input.to_vec());
        };
        pos += leb_len;
        let payload_start = pos;
        let payload_end = pos + section_size as usize;
        if payload_end > input.len() {
            return Ok(input.to_vec());
        }
        pos = payload_end;

        // If the module had no export section (patches always do, but be safe), synthesize one right
        // before the first ordered section that must follow it.
        if !export_emitted && section_id != 0 && section_id > SECTION_EXPORT {
            if reloc_export.is_some() {
                out.extend_from_slice(&encode_export_section(&[], reloc_export));
            }
            export_emitted = true;
        }

        match section_id {
            0 => {
                let mut keep = true;
                if let Some((name_len, name_leb)) = read_uleb128(&input[payload_start..]) {
                    let name_start = payload_start + name_leb;
                    let name_end = name_start + name_len as usize;
                    if name_end <= payload_end {
                        if let Ok(name) = std::str::from_utf8(&input[name_start..name_end]) {
                            if should_strip_custom_section(name, keep_names, strip_dwarf) {
                                keep = false;
                            }
                        }
                    }
                }
                if keep {
                    out.extend_from_slice(&input[section_start..payload_end]);
                }
            }
            SECTION_START => { /* drop: never auto-run patch code */ }
            SECTION_EXPORT => {
                out.extend_from_slice(&encode_export_section(
                    &input[payload_start..payload_end],
                    reloc_export,
                ));
                export_emitted = true;
            }
            _ => out.extend_from_slice(&input[section_start..payload_end]),
        }
    }

    Ok(out)
}

/// Re-encode the export section: keep the exports that the runtime calls, and append the
/// `__wasm_apply_global_relocs` function export when `reloc_export` names its index.
///
/// The thin link exports every function of the base table, because those exports are the
/// roots of the link. The runtime does not call them, and the export names cost as much as
/// the `name` section, so this pass keeps `main` and the relocation and constructor thunks
/// only. `existing_payload` is the original export section payload, `count` followed by the
/// entries, or empty to synthesize a fresh section.
fn encode_export_section(existing_payload: &[u8], reloc_export: Option<u32>) -> Vec<u8> {
    const NAME: &str = "__wasm_apply_global_relocs";
    const KEEP: [&str; 3] = ["main", "__wasm_apply_data_relocs", "__wasm_call_ctors"];

    let (count, mut entries): (u32, &[u8]) = match read_uleb128(existing_payload) {
        Some((c, l)) => (c, &existing_payload[l..]),
        None => (0, &[]),
    };

    let mut kept = Vec::new();
    let mut kept_count = 0u32;
    for _ in 0..count {
        // An entry is: name length, name bytes, kind byte, index.
        let Some((name_len, name_leb)) = read_uleb128(entries) else {
            break;
        };
        let name_end = name_leb + name_len as usize;
        let Some((_, index_leb)) = entries
            .get(name_end + 1..)
            .and_then(read_uleb128)
        else {
            break;
        };
        let entry_end = name_end + 1 + index_leb;
        let name = std::str::from_utf8(&entries[name_leb..name_end]).unwrap_or("");
        if KEEP.contains(&name) || name == NAME {
            kept.extend_from_slice(&entries[..entry_end]);
            kept_count += 1;
        }
        entries = &entries[entry_end..];
    }

    if let Some(func_index) = reloc_export {
        write_uleb128(&mut kept, NAME.len() as u32);
        kept.extend_from_slice(NAME.as_bytes());
        kept.push(0x00); // export kind: function
        write_uleb128(&mut kept, func_index);
        kept_count += 1;
    }

    let mut payload = Vec::new();
    write_uleb128(&mut payload, kept_count);
    payload.extend_from_slice(&kept);

    let mut section = vec![0x07u8]; // export section id
    write_uleb128(&mut section, payload.len() as u32);
    section.extend_from_slice(&payload);
    section
}

fn write_uleb128(out: &mut Vec<u8>, mut value: u32) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Custom sections the served patch doesn't need, stripped to shrink the bytes the browser fetches
/// and the runtime instantiates.
///
/// The browser only needs the standard sections to *run* the patch. We deliberately keep the DWARF
/// sections that `addr2line` consumes for click-to-source symbolication — `.debug_info`,
/// `.debug_abbrev`, `.debug_line`(+`.debug_line_str`), `.debug_str`(+`.debug_str_offsets`),
/// `.debug_addr`, and `.debug_ranges`/`.debug_rnglists` — plus `dylink.0`. Everything else goes:
/// wasm-ld bookkeeping (`linking`/`reloc.*`), build metadata, the wasm `name` section (the
/// symbolicator reads raw `0x` code offsets from the stack and resolves names from DWARF, never from
/// this section — so it's pure weight, and the largest strippable one), and the DWARF
/// accelerator/variable sections `addr2line` never reads for line/frame lookup.
///
/// `keep_names` (from the `--keep-names` CLI flag) preserves the `name` section: tools like
/// `console_error_panic_hook` print human-readable backtraces from it without a browser extension,
/// which is worth the extra bytes when profiling/debugging.
///
/// `strip_dwarf` removes every `.debug_*` section. The build with `--dwarf-sidecar` writes the
/// DWARF to a sidecar file, so the served module does not need it.
fn should_strip_custom_section(name: &str, keep_names: bool, strip_dwarf: bool) -> bool {
    if name == "name" {
        return !keep_names;
    }
    if strip_dwarf && name.starts_with(".debug_") {
        return true;
    }
    name.starts_with("reloc.")
        || name.contains("manganis")
        || name.contains("__wasm_bindgen")
        || matches!(
            name,
            "linking"
                | "producers"
                | "target_features"
                | ".debug_aranges"
                | ".debug_pubnames"
                | ".debug_pubtypes"
                | ".debug_gnu_pubnames"
                | ".debug_gnu_pubtypes"
                | ".debug_loc"
                | ".debug_loclists"
                | ".debug_frame"
                | ".eh_frame"
                | ".debug_macro"
                | ".debug_macinfo"
                | ".debug_cu_index"
                | ".debug_tu_index"
        )
}

fn read_uleb128(bytes: &[u8]) -> Option<(u32, usize)> {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;
    for (i, &b) in bytes.iter().enumerate().take(5) {
        result |= ((b & 0x7f) as u32) << shift;
        if b & 0x80 == 0 {
            return Some((result, i + 1));
        }
        shift += 7;
    }
    None
}

/// Prepares the base module before running wasm-bindgen.
///
/// This tries to work around how wasm-bindgen works by intelligently promoting non-wasm-bindgen functions
/// to the export table.
///
/// It also moves all functions and memories to be callable indirectly.
///
/// With `keep_dwarf`, walrus converts the DWARF of the module and writes it back. Without it,
/// walrus drops the DWARF, which saves most of the time on a large module. A build with a DWARF
/// sidecar passes `false`: the sidecar is a copy of the input, and it holds the DWARF.
pub fn prepare_wasm_base_module(bytes: &[u8], keep_dwarf: bool) -> Result<Vec<u8>> {
    let ParsedModule {
        mut module,
        ids,
        symbols,
        ..
    } = parse_module_with_ids_config(bytes, keep_dwarf)?;

    // Due to monomorphizations, functions will get merged and multiple names will point to the same function.
    // Walrus loses this information, so we need to manually parse the names table to get the indices
    // and names of these functions.
    //
    // Unfortunately, the indices it gives us ARE NOT VALID.
    // We need to work around it by using the FunctionId from the module as a link between the merged function names.
    let ifunc_map = collect_func_ifuncs(&module);
    let ifuncs = module
        .funcs
        .par_iter()
        .filter_map(|f| ifunc_map.get(f.name.as_deref()?).map(|_| f.id()))
        .collect::<HashSet<_>>();

    let imported_funcs = module
        .imports
        .iter()
        .filter_map(|i| match i.kind {
            ImportKind::Function(id) => Some((id, i.id())),
            _ => None,
        })
        .collect::<HashMap<_, _>>();

    let mut exported = HashSet::new();

    // Wasm-bindgen will synthesize imports to satisfy its external calls. This facilitates things
    // like inline-js, snippets, and literally the `#[wasm_bindgen]` macro. All calls to JS are
    // just `extern "wbg"` blocks!
    //
    // However, wasm-bindgen will run a GC pass on the module, removing any unused imports.
    let mut make_indirect = vec![];
    for (imported_func, importid) in imported_funcs {
        // Pull out the import's metadata so the `&module.imports` borrow is released before
        // any `&mut module` calls below (`replace_imported_func` takes `&mut self`).
        let (import_module, import_name) = {
            let import = module.imports.get(importid);
            (import.module.to_string(), import.name.to_string())
        };
        let name_is_wbg =
            import_name.starts_with("__wbindgen") || import_name.starts_with("__wbg_");

        if name_is_wbg && !name_is_bindgen_symbol(&import_name) {
            let func = module.funcs.get(imported_func);

            let ty = module.types.get(func.ty());
            let params = ty.params().to_vec();
            let results = ty.results().to_vec();

            let mut builder = FunctionBuilder::new(&mut module.types, &params, &results);
            let mut body = builder
                .name(format!("__saved_wbg_{}", import_name))
                .func_body();

            let locals = params
                .iter()
                .map(|ty| module.locals.add(*ty))
                .collect::<Vec<_>>();

            for l in locals.iter() {
                body.local_get(*l);
            }

            body.call(imported_func);

            let new_func_id = module.funcs.add_local(builder.local_func(locals));

            let saved_name = format!("__saved_wbg_{}", import_name);
            if exported.insert(saved_name.clone()) {
                module.exports.add(&saved_name, new_func_id);
            }

            make_indirect.push(new_func_id);
        } else if import_module == "env" && !name_is_wbg {
            // We also stub out any stray non-wbg `env` imports here. The fat build links with
            // `--no-gc-sections` so every symbol survives for future hot patches, which means any
            // unresolved C dep (e.g. `isprint` pulled in by a `cc`-compiled tree-sitter) stays in the
            // module as `(import "env" <name>)`. wasm-bindgen, which runs after this, doesn't own the
            // `env` namespace — it forwards the import verbatim as `import * as importN from "env"` in
            // the JS loader, and the browser then rejects it with `TypeError: Module name, 'env' does
            // not resolve to a valid URL`. Cold (Base) builds dodge this because default wasm-ld
            // dead-strips the unreachable C call sites and never emits the import. We mirror that
            // effect at the module level: replace the import with a local function whose body is a
            // single `unreachable`, and register it in the ifunc table so a thin patch's later env
            // reference resolves through `name_to_ifunc_old` in `create_wasm_jump_table`.
            //
            // Walrus parses the wasm name section into `Function::name`, so the imported
            // function already carries its original name (e.g. "isprint"). Save it before
            // replacement so we can put it back on the new local function — `collect_func_ifuncs`
            // and the cache's `symbol_ifunc_map` both key off `Function::name`, and a patch's
            // `env` import will look the symbol up by that exact name.
            let original_name = module.funcs.get(imported_func).name.clone();
            let new_fid = module
                .replace_imported_func(imported_func, |(body, _args)| {
                    body.unreachable();
                })
                .map_err(|e| {
                    PatchError::InvalidModule(format!(
                        "Failed to stub env import {import_name}: {e}"
                    ))
                })?;
            module.funcs.get_mut(new_fid).name = original_name;
            make_indirect.push(new_fid);
        }
    }

    for (name, index) in symbols.code_symbol_map.iter() {
        if name_is_bindgen_symbol(name) {
            continue;
        }

        let func = module.funcs.get(ids[*index]);

        // We want to preserve the intrinsics from getting gc-ed out.
        //
        // These will create corresponding shim functions in the main module, that the patches will
        // then call. Wasm-bindgen doesn't actually check if anyone uses the `__wbindgen` exports and
        // forcefully deletes them literally by checking for symbols that start with `__wbindgen`. We
        // preserve these symbols by naming them `__saved_wbg_<name>` and then exporting them.
        //
        // When wasm-bindgen runs, it will wrap these intrinsics with an `externref shim`, but we
        // want to preserve the actual underlying function so side modules can call them directly.
        //
        // https://github.com/rustwasm/wasm-bindgen/blob/c35cc9369d5e0dc418986f7811a0dd702fb33ef9/crates/cli-support/src/wit/mod.rs#L1505
        if name.starts_with("__wbindgen") {
            let saved_name = format!("__saved_wbg_{}", name);
            if exported.insert(saved_name.clone()) {
                module.exports.add(&saved_name, func.id());
            }
        }

        // This is basically `--export-all` but designed to work around wasm-bindgen not properly gc-ing
        // imports like __wbindgen_placeholder__ and __wbindgen_externref__
        //
        // We only export local functions, and then make sure they can be accessible indirectly.
        // If we weren't dealing with PIC code, then we could just create local ifuncs in the patch that
        // call the original function directly. Unfortunately, this would require adding a new relocation
        // to corresponding GOT.func entry, which we don't want to deal with.
        //
        // Note that we don't export via the export table, but rather the ifunc table. This is to work
        // around issues on large projects where we hit the maximum number of exports.
        //
        // https://github.com/emscripten-core/emscripten/issues/22863
        if let FunctionKind::Local(_) = &func.kind {
            if !ifuncs.contains(&func.id()) {
                make_indirect.push(func.id());
            }
        }
    }

    // Now we need to make sure to add the new ifuncs to the ifunc segment initializer.
    // We just assume the last segment is the safest one we can add to which is common practice.
    let segment = module
        .elements
        .iter_mut()
        .last()
        .context("Missing ifunc table")?;
    let make_indirect_count = make_indirect.len() as u64;
    let ElementItems::Functions(segment_ids) = &mut segment.items else {
        return Err(PatchError::InvalidModule(
            "Expected ifunc table to be a function table".into(),
        ));
    };

    for func in make_indirect {
        segment_ids.push(func);
    }

    if let ElementKind::Active { table, .. } = segment.kind {
        let table = module.tables.get_mut(table);
        table.initial += make_indirect_count;
        if let Some(max) = table.maximum {
            table.maximum = Some(max + make_indirect_count);
        }
    }

    // Embed a per-build identity the runtime can read back to detect a stale base. On wasm a patch
    // is just a set of table indices; applying one built against a different base silently dispatches
    // into the wrong functions. Exporting it as a global lets `subsecond` compare it against the
    // patch's `JumpTable::base_id` and refuse the mismatch. The `HotpatchModuleCache` reads the same
    // value back from the post-bindgen module, so both sides agree; if wasm-bindgen drops the export
    // the cache reads `None` and the check goes inert (no false positives).
    let base_id = base_build_id();
    let gid = module.globals.add_local(
        walrus::ValType::I32,
        false,
        false,
        ConstExpr::Value(walrus::ir::Value::I32(base_id)),
    );
    module.exports.add(SUBSECOND_BASE_ID_EXPORT, gid);

    Ok(module.emit_wasm())
}

/// Name of the exported `i32` global that carries the base module's per-build identity.
/// `subsecond` reads this at patch-apply time to reject patches built against a different base.
const SUBSECOND_BASE_ID_EXPORT: &str = "__subsecond_base_id";

/// A per-build identity for the base module. Just needs to differ across base rebuilds (including
/// body-only changes that can still shuffle ifunc indices), so the low bits of the wall clock at
/// build time suffice.
fn base_build_id() -> i32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i32)
        .unwrap_or(0)
}

/// Read the value of an exported, locally-defined `i32` global by export name. Returns `None` if the
/// export is missing, isn't a global, or isn't an `i32` constant (e.g. wasm-bindgen rewrote it).
fn read_exported_i32_global(module: &Module, export_name: &str) -> Option<i32> {
    let export = module.exports.iter().find(|e| e.name == export_name)?;
    let walrus::ExportItem::Global(gid) = export.item else {
        return None;
    };
    match &module.globals.get(gid).kind {
        walrus::GlobalKind::Local(ConstExpr::Value(walrus::ir::Value::I32(v))) => Some(*v),
        _ => None,
    }
}

/// Check if the name is a wasm-bindgen symbol
///
/// todo(jon): I believe we can just look at all the functions the wasm_bindgen describe export references.
/// this is kinda hacky on slow.
///
/// Uses the heuristics from the wasm-bindgen source code itself:
///
/// <https://github.com/rustwasm/wasm-bindgen/blob/c35cc9369d5e0dc418986f7811a0dd702fb33ef9/crates/cli-support/src/wit/mod.rs#L1165>
///
/// Symbols arrive in both mangling schemes (legacy `_ZN..$LT$..$GT$..` and v0 `_R..`, the default
/// since rustc 1.97), so each describe pattern needs a matcher per scheme. The v0 patterns match
/// the trailing `<Trait><method>` identifiers rather than the `wasm_bindgen` crate path because v0
/// backrefs (`NtB5_` etc.) routinely compress the path away. If any of these slip through, the
/// describe functions get pinned into the ifunc table, wasm-bindgen's GC can't delete them, and
/// the final module ships an unsatisfiable `__wbindgen_placeholder__.__wbindgen_describe` import.
/// Check if the name is a `wasm_bindgen::__rt::wbg_cast` instantiation (excluding its inner
/// `breaks_if_inlined` helper). These functions need their bodies rewritten to call the base
/// module's JS-bound versions via the ifunc table — if one slips through, the patch keeps its
/// local copy, which calls `breaks_if_inlined` → `describe::inform` → the stubbed
/// `__wbindgen_describe` import and traps with "null function" the first time the patched code
/// casts a value (e.g. creating an event-listener closure).
///
/// In legacy mangling the path appears as `wasm_bindgen4__rt8wbg_cast`; in v0 (default since
/// rustc 1.97) identifiers starting with `_` get a `_` separator after their length, so `__rt`
/// encodes as `4___rt`.
fn name_is_wbg_cast_symbol(name: &str) -> bool {
    (name.contains("wasm_bindgen4__rt8wbg_cast") || name.contains("wasm_bindgen4___rt8wbg_cast"))
        && !name.contains("breaks_if_inline")
}

fn name_is_bindgen_symbol(name: &str) -> bool {
    name.contains("__wbindgen_describe")
        || name.contains("__wbindgen_externref")
        || name.contains("wasm_bindgen8describe6inform")
        || name.contains("wasm_bindgen..describe..WasmDescribe")
        || name.contains("12WasmDescribe8describe")
        || name.contains("18WasmDescribeVector15describe_vector")
        || (name.contains("wasm_bindgen..closure..WasmClosure") && name.contains("describe"))
        || (name.contains("11WasmClosure") && name.contains("describe"))
        || (name.contains("wasm_bindgen7closure16Closure") && name.contains("describe"))
        || (name.contains("7closure7Closure") && name.contains("describe"))
        || (name.contains("wasm_bindgen7convert8closures") && name.contains("describe_invoke"))
}

// Test for bindgen symbols. As we find more bad symbols, add them here
#[test]
fn bindgen_symbol_catch() {
    let symbol = "_ZN12wasm_bindgen7convert8closures1_142_$LT$impl$u20$wasm_bindgen..closure..WasmClosure$u20$for$u20$dyn$u20$core..ops..function..FnMut$LT$$LP$$RP$$GT$$u2b$Output$u20$$u3d$$u20$R$GT$15describe_invoke17h4373f8b6570333dcE";
    assert!(name_is_bindgen_symbol(symbol));

    // matches_legacy_wasm_bindgen_closure_describe_symbols
    let symbol = "_ZN12wasm_bindgen7closure16Closure$LT$T$GT$4wrap8describe17h1234567890abcdefE";
    assert!(name_is_bindgen_symbol(symbol));

    // v0 mangling (default since rustc 1.97): `<T as WasmDescribe>::describe` impl with the
    // full wasm_bindgen path spelled out
    let symbol = "_RNvXNvNtNtCs9jB4f2OZCsR_7web_sys8features36gen_TransformStreamDefaultController1__NtB4_32TransformStreamDefaultControllerNtNtCs9tRDgkfeYnK_12wasm_bindgen8describe12WasmDescribe8describe";
    assert!(name_is_bindgen_symbol(symbol));

    // v0 describe impl where a backref (NtB5_) compresses the wasm_bindgen::describe path away
    let symbol = "_RNvXNvNtNtCs9jB4f2OZCsR_7web_sys8features8gen_Node4NodeENtB5_12WasmDescribe8describeCs3X5Dvr2wWzv_21dioxus_interpreter_js";
    assert!(name_is_bindgen_symbol(symbol));

    // v0 `<T as WasmDescribeVector>::describe_vector` impl (legacy mangling matches these via
    // the `wasm_bindgen..describe..WasmDescribe` prefix substring, v0 needs its own pattern)
    let symbol = "_RNvXs4_NtNtCs9tRDgkfeYnK_12wasm_bindgen7convert6slicesNtNtCscHiZRFGp0KF_5alloc6string6StringNtNtB9_8describe18WasmDescribeVector15describe_vector";
    assert!(name_is_bindgen_symbol(symbol));

    // v0 describe_vector impl in a downstream crate with the trait path backref-compressed
    let symbol = "_RNvXsf_NtCs4ofacjxbDm2_10dioxus_web8documentNtB5_7JSOwnerNtNtCs9tRDgkfeYnK_12wasm_bindgen8describe18WasmDescribeVector15describe_vector";
    assert!(name_is_bindgen_symbol(symbol));

    // v0 closure describe_invoke (WasmClosure trait path is backref-compressed to `NtNtBc_`)
    let symbol = "_RINvXs1_NvNtNtCs9tRDgkfeYnK_12wasm_bindgen7convert8closuress8_1__DINtNtNtCs9WN6KVdqFxk_4core3ops8function5FnMutTNtBc_7JsValueB1M_mNtCsezy3jvZZ1sp_6js_sys5ArrayEEp6OutputB1M_EL_NtNtBc_7closure11WasmClosure15describe_invokeKb1_EB26_";
    assert!(name_is_bindgen_symbol(symbol));

    // v0 name of the __wbindgen_describe import shim
    let symbol = "_RNvCs9tRDgkfeYnK_12wasm_bindgen19___wbindgen_describe";
    assert!(name_is_bindgen_symbol(symbol));

    // does_not_match_saved_runtime_exports
    assert!(!name_is_bindgen_symbol("__wbindgen_malloc"));
    assert!(!name_is_bindgen_symbol("__wbindgen_realloc"));
    assert!(!name_is_bindgen_symbol("__wbindgen_free"));

    // does_not_match_ordinary_user_symbols_in_either_mangling
    assert!(!name_is_bindgen_symbol(
        "_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h1234567890abcdefE"
    ));
    assert!(!name_is_bindgen_symbol(
        "_RNvXs5_NtCs9tRDgkfeYnK_12wasm_bindgen5__rt5LazyINtB5_4LazyNtNtCsezy3jvZZ1sp_6js_sys6ObjectE5force"
    ));
}

#[test]
fn wbg_cast_symbol_catch() {
    // legacy mangling: wbg_cast instantiation matches, its breaks_if_inlined helper does not
    assert!(name_is_wbg_cast_symbol(
        "_ZN12wasm_bindgen4__rt8wbg_cast17h1234567890abcdefE"
    ));
    assert!(!name_is_wbg_cast_symbol(
        "_ZN12wasm_bindgen4__rt8wbg_cast17breaks_if_inlined17h1234567890abcdefE"
    ));

    // v0 mangling: `__rt` encodes as `4___rt` (length 4, `_` separator, then `__rt`)
    assert!(name_is_wbg_cast_symbol(
        "_RINvNtCsa7akE1TfegA_12wasm_bindgen4___rt8wbg_castINtB4_7closure12OwnedClosureDINtNtNtCs9WN6KVdqFxk_4core3ops8function5FnMutTNtNtNtCs5qlPUvWaqlJ_7web_sys8features14gen_MouseEvent10MouseEventEEp6OutputuEL_Kb1_ENtBO_9JsClosureECsjFep1nV9Dzo_32dioxus_playwright_web_patch_test"
    ));
    assert!(!name_is_wbg_cast_symbol(
        "_RINvNvNtCsa7akE1TfegA_12wasm_bindgen4___rt8wbg_cast17breaks_if_inlinedINtNtB6_7closure12OwnedClosureDINtNtNtCs9WN6KVdqFxk_4core3ops8function5FnMutTNtNtNtCs5qlPUvWaqlJ_7web_sys8features14gen_MouseEvent10MouseEventEEp6OutputuEL_Kb1_ENtB19_9JsClosureECsjFep1nV9Dzo_32dioxus_playwright_web_patch_test"
    ));
}

/// Run via:
///   PREPARE_WASM_INPUT=path/to/cargo-built.wasm cargo test \
///     -p dioxus-cli --target aarch64-apple-darwin -- \
///     prepare_wasm_preserves_dwarf --nocapture --ignored
#[test]
#[ignore]
fn prepare_wasm_preserves_dwarf() {
    let path = match std::env::var("PREPARE_WASM_INPUT") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("set PREPARE_WASM_INPUT=<path-to-wasm> to run this test");
            return;
        }
    };
    let bytes = std::fs::read(&path).expect("read input wasm");
    eprintln!("input bytes={}", bytes.len());

    let out = prepare_wasm_base_module(&bytes, true).expect("prepare_wasm_base_module");
    eprintln!("output bytes={}", out.len());

    let mut debug_total = 0usize;
    for payload in wasmparser::Parser::new(0).parse_all(&out) {
        if let Ok(Payload::CustomSection(s)) = payload {
            if s.name().starts_with(".debug_") {
                eprintln!("  output has {} (size={})", s.name(), s.data().len());
                debug_total += s.data().len();
            }
        }
    }
    assert!(
        debug_total > 0,
        ".debug_* sections were stripped by walrus — DWARF lost"
    );

    if let Ok(out_path) = std::env::var("PREPARE_WASM_OUTPUT") {
        std::fs::write(&out_path, &out).expect("write output wasm");
        eprintln!("wrote prepared wasm to {}", out_path);
    }
}

/// Run via:
///   STRIP_WASM_INPUT=path/to/wasm cargo test \
///     -p dioxus-cli --target aarch64-apple-darwin --bin dx -- \
///     strip_linker_sidecars_bench --nocapture --ignored
#[test]
#[ignore]
fn strip_linker_sidecars_bench() {
    let path = match std::env::var("STRIP_WASM_INPUT") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("set STRIP_WASM_INPUT=<path-to-wasm> to run this benchmark");
            return;
        }
    };
    let bytes = std::fs::read(&path).expect("read input wasm");
    eprintln!(
        "input size: {} bytes ({:.2} MB)",
        bytes.len(),
        bytes.len() as f64 / 1024.0 / 1024.0
    );

    let runs = 5;
    let mut times_ms = Vec::with_capacity(runs);
    let mut last = Vec::new();
    for _ in 0..runs {
        let start = std::time::Instant::now();
        last = strip_linker_sidecars(&bytes, false, false);
        let elapsed = start.elapsed();
        times_ms.push(elapsed.as_secs_f64() * 1000.0);
    }
    eprintln!(
        "output size: {} bytes ({:.2} MB) -- {:.1}% of input",
        last.len(),
        last.len() as f64 / 1024.0 / 1024.0,
        last.len() as f64 * 100.0 / bytes.len() as f64
    );
    eprintln!("strip times (ms): {:?}", times_ms);
    let avg = times_ms.iter().sum::<f64>() / runs as f64;
    let min = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!("avg = {:.1} ms, best = {:.1} ms", avg, min);

    // Verify the output is still a valid wasm (parses cleanly) and that the
    // sections we wanted gone are gone, while the ones we wanted to keep are present.
    let mut linking_present = false;
    let mut reloc_present = false;
    let mut name_present = false;
    let mut debug_present = false;
    for payload in wasmparser::Parser::new(0).parse_all(&last) {
        match payload.expect("stripped wasm fails to parse") {
            Payload::CustomSection(s) => match s.name() {
                "linking" => linking_present = true,
                n if n.starts_with("reloc.") => reloc_present = true,
                "name" => name_present = true,
                n if n.starts_with(".debug_") => debug_present = true,
                _ => {}
            },
            _ => {}
        }
    }
    assert!(!linking_present, "linking section survived strip");
    assert!(!reloc_present, "reloc.* section survived strip");
    eprintln!(
        "post-strip: name={} debug_present={}",
        name_present, debug_present
    );

    if let Ok(out_path) = std::env::var("STRIP_WASM_OUTPUT") {
        std::fs::write(&out_path, &last).expect("write output wasm");
        eprintln!("wrote stripped wasm to {}", out_path);
    }
}

#[test]
#[ignore]
fn find_reloc_index_probe() {
    let path = std::env::var("PROBE_WASM").expect("set PROBE_WASM=<path>");
    let bytes = std::fs::read(&path).expect("read wasm");
    eprintln!("custom sections present:");
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let Ok(Payload::CustomSection(s)) = payload {
            eprintln!("  {} ({} bytes)", s.name(), s.data().len());
        }
    }
    let by_name = find_wasm_func_index_by_name(&bytes, "__wasm_apply_global_relocs");
    eprintln!("find_wasm_func_index_by_name -> {by_name:?}");
    if let Ok(sec) = parse_bytes_to_data_segment(&bytes) {
        let by_link = sec
            .code_symbol_map
            .get("__wasm_apply_global_relocs")
            .copied();
        eprintln!("linking code_symbol_map -> {by_link:?}");
        eprintln!(
            "linking code_symbol_map total funcs: {}",
            sec.code_symbol_map.len()
        );
    } else {
        eprintln!("no linking section");
    }
}

#[test]
fn extend_element_segment_lists_every_defined_function() {
    fn section(id: u8, payload: &[u8]) -> Vec<u8> {
        let mut s = vec![id];
        write_uleb128(&mut s, payload.len() as u32);
        s.extend_from_slice(payload);
        s
    }
    fn segments_of(module: &[u8]) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        for payload in wasmparser::Parser::new(0).parse_all(module) {
            if let Payload::ElementSection(reader) = payload.expect("the module must parse") {
                for segment in reader {
                    let segment = segment.unwrap();
                    let wasmparser::ElementItems::Functions(funcs) = segment.items else {
                        panic!("expected function indices");
                    };
                    out.push(funcs.into_iter().map(|f| f.unwrap()).collect());
                }
            }
        }
        out
    }

    // type: () -> ()
    let types = section(1, &[0x01, 0x60, 0x00, 0x00]);
    // imports: one function `env.f` of type 0, one global `env.__table_base` (i32, const)
    let mut imports = Vec::new();
    write_uleb128(&mut imports, 2);
    for (name, desc) in [("f", vec![0x00u8, 0x00]), ("__table_base", vec![0x03, 0x7f, 0x00])] {
        write_uleb128(&mut imports, 3);
        imports.extend_from_slice(b"env");
        write_uleb128(&mut imports, name.len() as u32);
        imports.extend_from_slice(name.as_bytes());
        imports.extend_from_slice(&desc);
    }
    let imports = section(2, &imports);
    // three defined functions of type 0: indices 1, 2 and 3
    let funcs = section(3, &[0x03, 0x00, 0x00, 0x00]);
    // code: three empty bodies
    let mut code = Vec::new();
    write_uleb128(&mut code, 3);
    for _ in 0..3 {
        code.extend_from_slice(&[0x02, 0x00, 0x0b]);
    }
    let code = section(10, &code);

    let mut head = Vec::new();
    head.extend_from_slice(b"\0asm");
    head.extend_from_slice(&1u32.to_le_bytes());
    head.extend_from_slice(&types);
    head.extend_from_slice(&imports);
    head.extend_from_slice(&funcs);

    // A module whose linker segment holds function 2 only: the pass appends 1 and 3 after it.
    let mut with_segment = head.clone();
    with_segment.extend_from_slice(&section(9, &[0x01, 0x00, 0x23, 0x00, 0x0b, 0x01, 0x02]));
    with_segment.extend_from_slice(&code);
    let out = extend_element_segment(&with_segment).unwrap();
    assert_eq!(segments_of(&out), vec![vec![2, 1, 3]]);
    assert!(out.ends_with(&code), "the code section stays as it was");

    // A module without an element section gets one before the code section.
    let mut without = head.clone();
    without.extend_from_slice(&code);
    let out = extend_element_segment(&without).unwrap();
    assert_eq!(segments_of(&out), vec![vec![1, 2, 3]]);
    assert!(out.ends_with(&code), "the code section stays as it was");
}

#[test]
fn finalize_patch_wasm_edits() {
    use std::collections::HashSet;

    fn section(id: u8, payload: &[u8]) -> Vec<u8> {
        let mut s = vec![id];
        write_uleb128(&mut s, payload.len() as u32);
        s.extend_from_slice(payload);
        s
    }
    fn custom(name: &str, data: &[u8]) -> Vec<u8> {
        let mut p = Vec::new();
        write_uleb128(&mut p, name.len() as u32);
        p.extend_from_slice(name.as_bytes());
        p.extend_from_slice(data);
        section(0, &p)
    }

    let mut m = Vec::new();
    m.extend_from_slice(b"\0asm");
    m.extend_from_slice(&1u32.to_le_bytes());
    // type: () -> ()
    m.extend_from_slice(&section(1, &[0x01, 0x60, 0x00, 0x00]));
    // func: one function of type 0
    m.extend_from_slice(&section(3, &[0x01, 0x00]));
    // export: "main" -> func 0, and a linker root "extra" -> func 0 that must go
    let mut exp = Vec::new();
    write_uleb128(&mut exp, 2);
    for name in ["main", "extra"] {
        write_uleb128(&mut exp, name.len() as u32);
        exp.extend_from_slice(name.as_bytes());
        exp.push(0x00);
        write_uleb128(&mut exp, 0);
    }
    m.extend_from_slice(&section(7, &exp));
    // start: func 0 (must be dropped)
    m.extend_from_slice(&section(8, &[0x00]));
    // code: one empty body
    let body = [0x00u8, 0x0b];
    let mut code = Vec::new();
    write_uleb128(&mut code, 1);
    write_uleb128(&mut code, body.len() as u32);
    code.extend_from_slice(&body);
    m.extend_from_slice(&section(10, &code));
    // customs: name + manganis stripped, .debug_info + dylink.0 kept
    m.extend_from_slice(&custom("name", &[0x00]));
    m.extend_from_slice(&custom(".debug_info", &[0x01, 0x02]));
    m.extend_from_slice(&custom("manganis", &[0x03]));
    m.extend_from_slice(&custom("dylink.0", &[0x04]));

    let out = finalize_patch_wasm(&m, Some(0), false, false).unwrap();

    let mut has_start = false;
    let mut exports = HashSet::new();
    let mut customs = HashSet::new();
    for payload in wasmparser::Parser::new(0).parse_all(&out) {
        match payload.expect("finalized wasm must parse") {
            Payload::StartSection { .. } => has_start = true,
            Payload::ExportSection(r) => {
                for e in r {
                    exports.insert(e.unwrap().name.to_string());
                }
            }
            Payload::CustomSection(s) => {
                customs.insert(s.name().to_string());
            }
            _ => {}
        }
    }

    assert!(!has_start, "start section should be dropped");
    assert!(exports.contains("main"), "existing exports preserved");
    assert!(!exports.contains("extra"), "a linker root export is dropped");
    assert!(
        exports.contains("__wasm_apply_global_relocs"),
        "relocs export added"
    );
    assert!(!customs.contains("name"), "name section stripped");
    assert!(!customs.contains("manganis"), "manganis stripped");
    assert!(customs.contains(".debug_info"), "DWARF preserved");
    assert!(customs.contains("dylink.0"), "dylink preserved");

    // With keep_names, the `name` section survives while everything else is still stripped.
    let kept = finalize_patch_wasm(&m, Some(0), true, false).unwrap();
    let mut kept_customs = HashSet::new();
    for payload in wasmparser::Parser::new(0).parse_all(&kept) {
        if let Payload::CustomSection(s) = payload.expect("finalized wasm must parse") {
            kept_customs.insert(s.name().to_string());
        }
    }
    assert!(kept_customs.contains("name"), "name section kept with --keep-names");
    assert!(!kept_customs.contains("manganis"), "manganis still stripped");

    // With a DWARF sidecar, the `.debug_*` sections go too, and the `name` section stays.
    let split = finalize_patch_wasm(&m, Some(0), true, true).unwrap();
    let mut split_customs = HashSet::new();
    for payload in wasmparser::Parser::new(0).parse_all(&split) {
        if let Payload::CustomSection(s) = payload.expect("finalized wasm must parse") {
            split_customs.insert(s.name().to_string());
        }
    }
    assert!(!split_customs.contains(".debug_info"), "DWARF stripped with a sidecar");
    assert!(split_customs.contains("name"), "name section kept with a sidecar");
    assert!(split_customs.contains("dylink.0"), "dylink preserved with a sidecar");
}

/// Manually parse the data section from a wasm module
///
/// We need to do this for data symbols because walrus doesn't provide the right range and offset
/// information for data segments. Fortunately, it provides it for code sections, so we only need to
/// do a small amount extra of parsing here.
fn parse_bytes_to_data_segment(bytes: &[u8]) -> Result<RawDataSection<'_>> {
    let parser = wasmparser::Parser::new(0);
    let mut parser = parser.parse_all(bytes);
    let mut segments = vec![];
    let mut data_range = 0..0;
    let mut symbols = vec![];

    // Process the payloads in the raw wasm file so we can extract the specific sections we need
    while let Some(Ok(payload)) = parser.next() {
        match payload {
            Payload::DataSection(section) => {
                data_range = section.range();
                segments = section
                    .into_iter()
                    .collect::<Result<Vec<_>, BinaryReaderError>>()?
            }
            Payload::CustomSection(section) if section.name() == "linking" => {
                let reader = BinaryReader::new(section.data(), 0);
                let reader = LinkingSectionReader::new(reader)?;
                for subsection in reader.subsections() {
                    if let Linking::SymbolTable(map) = subsection? {
                        symbols = map.into_iter().collect::<Result<Vec<_>, _>>()?;
                    }
                }
            }
            Payload::CustomSection(section) => {
                tracing::trace!("Skipping Custom section: {:?}", section.name());
            }
            _ => {}
        }
    }

    // Accumulate the data symbols into a btreemap for later use
    let mut data_symbols = BTreeMap::new();
    let mut data_symbol_map = HashMap::new();
    let mut code_symbol_map = BTreeMap::new();
    for (index, symbol) in symbols.iter().enumerate() {
        if let SymbolInfo::Func { name, index, .. } = symbol {
            if let Some(name) = name {
                code_symbol_map.insert(*name, *index as usize);
            }
            continue;
        }

        let SymbolInfo::Data {
            symbol: Some(symbol),
            name,
            ..
        } = symbol
        else {
            continue;
        };

        data_symbol_map.insert(*name, index);

        let data_segment = segments
            .get(symbol.index as usize)
            .context("Failed to find data segment")?;
        let offset: usize =
            data_segment.range.end - data_segment.data.len() + (symbol.offset as usize);
        let range = offset..(offset + symbol.size as usize);

        data_symbols.insert(
            index,
            DataSymbol {
                _index: index,
                _range: range,
                segment_offset: symbol.offset as usize,
                _symbol_size: symbol.size as usize,
                which_data_segment: symbol.index as usize,
            },
        );
    }

    Ok(RawDataSection {
        _data_range: data_range,
        symbols,
        data_symbols,
        data_symbol_map,
        code_symbol_map,
    })
}

struct RawDataSection<'a> {
    _data_range: Range<usize>,
    symbols: Vec<SymbolInfo<'a>>,
    code_symbol_map: BTreeMap<&'a str, usize>,
    data_symbols: BTreeMap<usize, DataSymbol>,
    data_symbol_map: HashMap<&'a str, usize>,
}

#[derive(Debug)]
struct DataSymbol {
    _index: usize,
    _range: Range<usize>,
    segment_offset: usize,
    _symbol_size: usize,
    which_data_segment: usize,
}

struct ParsedModule<'a> {
    module: Module,
    ids: Vec<FunctionId>,
    symbols: RawDataSection<'a>,
}

/// Parse a module and return the mapping of index to FunctionID.
/// We'll use this mapping to remap ModuleIDs
fn parse_module_with_ids(bindgened: &[u8]) -> Result<ParsedModule<'_>> {
    parse_module_with_ids_config(bindgened, false)
}

/// Same as `parse_module_with_ids` but lets the caller request DWARF preservation.
///
/// Walrus drops every `.debug_*` custom section at emit time unless `generate_dwarf` is set.
/// Callers that re-emit the wasm to disk (e.g. `prepare_wasm_base_module`) need
/// `generate_dwarf=true` so source-level stack traces survive the transform; callers that
/// only inspect the parsed `Module` should leave it `false` to skip the extra DWARF parse.
fn parse_module_with_ids_config(
    bindgened: &[u8],
    generate_dwarf: bool,
) -> Result<ParsedModule<'_>> {
    let ids = Arc::new(RwLock::new(Vec::new()));
    let ids_ = ids.clone();
    let mut config = ModuleConfig::new();
    if generate_dwarf {
        config.generate_dwarf(true);
    }
    config.on_parse(move |_m, our_ids| {
        let mut ids = ids_.write().expect("No shared writers");
        let mut idx = 0;
        while let Ok(entry) = our_ids.get_func(idx) {
            ids.push(entry);
            idx += 1;
        }

        Ok(())
    });
    let module = Module::from_buffer_with_config(bindgened, &config)?;
    let mut ids_ = ids.write().expect("No shared writers");
    let mut ids = vec![];
    std::mem::swap(&mut ids, &mut *ids_);

    let symbols = parse_bytes_to_data_segment(bindgened).context("Failed to parse data segment")?;

    Ok(ParsedModule {
        module,
        ids,
        symbols,
    })
}

/// Get the main sentinel symbol for the given target triple
///
/// We need to special case darwin since `main` is the entrypoint but `_main` is the actual symbol.
/// The entrypoint ends up outside the text section, seemingly, and breaks our aslr detection.
fn main_sentinel(triple: &Triple) -> &'static str {
    match triple.operating_system {
        // The symbol in the symtab is called "_main" but in the dysymtab it is called "main"
        OperatingSystem::MacOSX(_) | OperatingSystem::Darwin(_) | OperatingSystem::IOS(_) => {
            "_main"
        }

        _ => "main",
    }
}
