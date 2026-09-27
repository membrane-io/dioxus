use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    path::PathBuf,
};

/// See `JumpTable::earlier_patches`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct PreviousPatch {
    /// The `lib` of an earlier jump table. A tab that did not apply that patch, because the
    /// page loaded after it, skips the repoint.
    pub lib: PathBuf,
    /// `(slot in the region of the earlier patch, new ifunc index)`.
    pub repoint: Vec<(u64, u64)>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct JumpTable {
    /// The dylib containing the patch. This should be a valid path so you can just pass it to LibLoading
    ///
    /// On wasm you will need to fetch() this file and then pass it to the WebAssembly.instantiate() function
    pub lib: PathBuf,

    /// old -> new
    /// does not take into account the base address of the patch when loaded into memory - need dlopen for that
    ///
    /// These are intended to be `*const ()` pointers but need to be `u64` for the hashmap. On 32-bit platforms
    /// you will need to cast to `usize` before using them.
    pub map: AddressMap,

    /// the address of the base address of the old original binary
    ///
    /// machos: this is the address of the `_mh_execute_header` symbol usually at 0x100000000 and loaded near 0x100000000
    /// linux: this is the address of the `__executable_start` symbol usually at 0x0 but loaded around 0x555555550000
    /// windows: this is the address of the `ImageBase` field of the PE header
    /// wasm: not useful since there's no ASLR
    ///
    /// While we can generally guess that these values are, it's possible they are different and thus reading
    /// them dynamically is worthwhile.
    pub aslr_reference: u64,

    /// the address of the base address of the new binary
    ///
    /// machos: this is the address of the `_mh_execute_header` symbol usually at 0x100000000 and loaded near 0x100000000
    /// linux: this is the address of the `__executable_start` symbol usually at 0x0 but loaded around 0x555555550000
    /// windows: this is the address of the `ImageBase` field of the PE header
    /// wasm: not useful since there's no ASLR
    ///
    /// While we can generally guess that these values are, it's possible they are different and thus reading
    /// them dynamically is worthwhile.
    pub new_base_address: u64,

    /// The amount of ifuncs this will register. This is used by WASM to know how much space to allocate
    /// for the ifuncs in the ifunc table
    pub ifunc_count: u64,

    /// (wasm only) The DWARF sidecar of `lib`, when dx moved the DWARF of the patch out of the
    /// served bytes. The runtime fetches it after the patch runs and attaches it to the patch
    /// module through `wasmStackTrace`, so that stack traces and the DWARF inspector read it.
    #[serde(default)]
    pub dwarf_sidecar: Option<PathBuf>,

    /// (wasm only) Pairs of `(old_ifunc_index, new_ifunc_index)` whose old/new functions have
    /// identical wasm signatures and are therefore safe to overwrite in place in the indirect
    /// function table after the patch is instantiated.
    ///
    /// Overwriting old table slots with the new function makes already-existing, type-erased values
    /// (trait-object vtables, `drop_in_place`/`type_id` glue captured before the patch) dispatch into
    /// the patched code instead of the stale original, which avoids crashes when such values outlive a
    /// patch. The signature filter is essential: at high opt levels several differently-typed symbols
    /// can collapse onto one ifunc index, so a name-matched `map` entry may pair an old slot with a
    /// new function of a different signature — overwriting those would corrupt unrelated `call_indirect`
    /// sites. New indices are pre-rebase (the runtime adds `__table_base`).
    #[serde(default)]
    pub ifunc_repoint: Vec<(u64, u64)>,

    /// (wasm only) The earlier patches of the session that define a function of this patch,
    /// with the pairs `(slot in the region of that patch, new ifunc index)` whose functions
    /// have the same signature. The vtables and the function pointers in the data of an
    /// earlier patch point into its region. The runtime repoints those slots, and every other
    /// slot of an earlier region that holds a replaced function. New indices are pre-rebase,
    /// like `ifunc_repoint`.
    #[serde(default)]
    pub earlier_patches: Vec<PreviousPatch>,

    /// (wasm only) The `lib` of the patch that this patch builds on. This patch defines only
    /// the functions that changed since that patch, and it calls the other functions of the
    /// earlier patches through the base slots. A tab that did not apply that patch as its last
    /// patch must not apply this one. `None` means that the patch defines every function that
    /// differs from the base, so any tab can apply it.
    #[serde(default)]
    pub builds_on: Option<PathBuf>,

    /// (wasm only) The `lib` of each earlier patch whose functions this patch replaced
    /// completely. The runtime tells the stack trace code that those modules are old.
    #[serde(default)]
    pub retired: Vec<PathBuf>,

    /// (wasm only) Values the runtime must supply for the patch's dynamic-linking imports at
    /// instantiate time. When present, the CLI served the patch *without* rewriting it (no walrus
    /// round-trip), leaving every `GOT.func` / `GOT.mem` / `env` / `__wbindgen_placeholder__` import
    /// in place so the code section — and the DWARF that indexes it — stays byte-identical to the
    /// linker's output. The runtime resolves these imports against the host's shared function table
    /// and exports instead. `None` means the patch was self-satisfied the old way (walrus rewrite).
    #[serde(default)]
    pub wasm: Option<WasmFixups>,

    /// (wasm only) Identity of the base module this patch was built against, read from the base's
    /// exported `__subsecond_base_id` global. The runtime compares it against the running base's
    /// value and refuses to apply the patch on a mismatch — a stale base (e.g. the page wasn't
    /// reloaded after a full rebuild) has a different ifunc-table layout, so applying a patch built
    /// against another base silently dispatches into the wrong functions. `None` disables the check
    /// (older CLI, non-wasm, or wasm-bindgen dropped the export), preserving prior behavior.
    #[serde(default)]
    pub base_id: Option<i32>,
}

/// Host-supplied values for a side-module patch's dynamic-linking imports.
///
/// wasm-ld emits PIC side modules whose `GOT.*` / `env` / `__wbindgen_placeholder__` imports are
/// meant to be filled in by a loader at instantiate time. Rather than mutating the module to bake
/// these in (which forces a full walrus re-encode and invalidates DWARF), the CLI ships the raw
/// values here and the runtime builds the import object. Anything not listed is resolved by the
/// runtime itself: `env` functions from the host exports, `__wbindgen_placeholder__` from the base's
/// `__saved_wbg_*` exports, and anything still missing gets a trapping stub so instantiation can't
/// fail with a `LinkError`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct WasmFixups {
    /// `GOT.func.<name>` → old indirect-function-table index. Supplied as an imported `i32` global.
    pub got_func: Vec<(String, i32)>,

    /// `GOT.mem.<name>` → absolute offset of the data symbol in the base module's linear memory.
    /// Supplied as an imported `i32` global.
    pub got_mem: Vec<(String, i32)>,

    /// `env.<name>` function imports the base module doesn't export but that exist in the shared
    /// indirect function table, paired with that table index. The runtime supplies the funcref via
    /// `table.get(index)`. Only signature-matched entries are listed; mismatches are left to the
    /// trapping-stub fallback (same observable behavior as the old `call_indirect` path: traps only
    /// if actually called).
    pub env_ifunc: Vec<(String, i32)>,

    /// Whether the `GOT.*` imported globals are declared mutable (wasm-ld emits them mutable). The
    /// supplied `WebAssembly.Global` must match or instantiation fails with a `LinkError`.
    pub got_mutable: bool,
}

/// An address to address hashmap that does not hash addresses since addresses are by definition unique.
pub type AddressMap = HashMap<u64, u64, BuildAddressHasher>;
pub type BuildAddressHasher = BuildHasherDefault<AddressHasher>;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AddressHasher(u64);
impl Hasher for AddressHasher {
    fn write(&mut self, _: &[u8]) {
        panic!("Invalid use of NoHashHasher")
    }
    fn write_u8(&mut self, n: u8) {
        self.0 = u64::from(n)
    }
    fn write_u16(&mut self, n: u16) {
        self.0 = u64::from(n)
    }
    fn write_u32(&mut self, n: u32) {
        self.0 = u64::from(n)
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n
    }
    fn write_usize(&mut self, n: usize) {
        self.0 = n as u64
    }
    fn write_i8(&mut self, n: i8) {
        self.0 = n as u64
    }
    fn write_i16(&mut self, n: i16) {
        self.0 = n as u64
    }
    fn write_i32(&mut self, n: i32) {
        self.0 = n as u64
    }
    fn write_i64(&mut self, n: i64) {
        self.0 = n as u64
    }
    fn write_isize(&mut self, n: isize) {
        self.0 = n as u64
    }
    fn finish(&self) -> u64 {
        self.0
    }
}
