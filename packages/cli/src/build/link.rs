//! Hotpatching: Fat and Thin Linking
//!
//! This module implements the dance we need to perform around manually linking projects using dx itself.
//! This is done by being the `RUSTC_WORKSPACE_WRAPPER` as well as `LINKER`. By intercepting both of these,
//! we can perform various optimizations like persisting rustc arguments for hotpatching.
//!
//! The flow looks like this
//! - bust fingerprint for tip of crate to ensure we always get final linker invocation
//! - run build, intercepting with wrapper, dumping rustc args to disk
//! - read dumped rustc args into a local cache for the given build under artifacts
//! - track changed crates
//! - for each changed crate, re-run the direct rustc invocation all the way to the tip
//! - relink the rlib set with the `.o` from the tip -> final binary
//!
//! source of truth is the read-out of the link args after the initial build

use super::HotpatchModuleCache;
use crate::build::patch::{ObjectIndex, PatchIfuncs};
use crate::{BuildArtifacts, BuildMode, TipObjects, WorkspaceRustcArgs};
use crate::{BuildContext, Error, LinkerFlavor, Result, RustcArgs, Workspace};
use crate::{BuildRequest, DX_RUSTC_WRAPPER_ENV_VAR};
use anyhow::{Context, bail, ensure};
use cargo_metadata::diagnostic::Diagnostic;
use depinfo::RustcDepInfo;
use itertools::Itertools;
use serde::Serialize;
use sha1::Digest;
use sha2::Sha256;
use std::process::Stdio;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    ffi::OsString,
};
use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use subsecond_types::JumpTable;
use target_lexicon::{Architecture, OperatingSystem};
use tokio::{io::AsyncBufReadExt, process::Command};
use uuid::Uuid;

/// Marks an error of the replay step of a thin build, so that the fallback of the interface
/// gate does not retry a compile error in the changed crate itself.
#[derive(Debug)]
struct ReplayFailed;

impl std::fmt::Display for ReplayFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the replay of a workspace crate failed")
    }
}

/// Marks a thin build whose edit changed no function since the fat build. A cascade compiles
/// the same functions again, so the fallback of the interface gate must not retry it.
#[derive(Debug)]
struct NothingToPatch;

impl std::fmt::Display for NothingToPatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("No function changed since the fat build, so there is nothing to patch")
    }
}

/// The tip compile of one thin build: the exe, the link args that name the tip objects, and
/// the dep-info of the tip.
struct TipBuild {
    exe: PathBuf,
    link_args: Vec<String>,
    depinfo: RustcDepInfo,
    time_start: SystemTime,
}

/// Changes to the captured rustc args of one replay.
#[derive(Default)]
struct ReplayOverrides {
    /// Write the outputs into this directory instead of the captured `--out-dir`.
    out_dir: Option<PathBuf>,
    /// Link these crates from these rlibs instead of the captured `--extern` paths. The key is
    /// the crate name, as in the `lib<name>-<hash>.rlib` file name.
    extern_rlibs: HashMap<String, PathBuf>,
    /// Add these directories to the library search path, before the captured ones.
    search_dirs: Vec<PathBuf>,
    /// Drop the `-C incremental` arg.
    no_incremental: bool,
}

impl ReplayOverrides {
    /// Rewrite one `--extern name=path` value when `extern_rlibs` holds the crate of `path`.
    fn rewrite_extern(&self, value: &str) -> String {
        let Some((name, path)) = value.split_once('=') else {
            return value.to_string();
        };
        let Some(crate_name) = rlib_crate_name(Path::new(path)) else {
            return value.to_string();
        };
        match self.extern_rlibs.get(crate_name) {
            Some(rlib) => format!("{name}={}", rlib.display()),
            None => value.to_string(),
        }
    }
}

/// The crate name in an rlib file name, `lib<name>-<hash>.rlib`.
fn rlib_crate_name(path: &Path) -> Option<&str> {
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_prefix("lib")?.strip_suffix(".rlib")?;
    Some(stem.rsplit_once('-').map(|(name, _)| name).unwrap_or(stem))
}

/// The object members of an rlib, by member name.
fn rlib_objects(path: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read the rlib '{}'", path.display()))?;
    let archive = object::read::archive::ArchiveFile::parse(data.as_slice())
        .with_context(|| format!("Failed to parse the rlib '{}'", path.display()))?;
    let mut objects = BTreeMap::new();
    for member in archive.members() {
        let member = member?;
        let name = String::from_utf8_lossy(member.name()).to_string();
        if !name.ends_with(".o") {
            continue;
        }
        objects.insert(name, member.data(data.as_slice())?.to_vec());
    }
    Ok(objects)
}

/// The names of the object members that differ between two rlibs.
fn rlib_object_differences(old: &Path, new: &Path) -> Result<Vec<String>> {
    let old_objects = rlib_objects(old)?;
    let new_objects = rlib_objects(new)?;
    let mut differences = vec![];
    for (name, data) in &old_objects {
        match new_objects.get(name) {
            Some(new_data) if new_data == data => {}
            Some(_) => differences.push(format!("{name} (content)")),
            None => differences.push(format!("{name} (only in the old rlib)")),
        }
    }
    for name in new_objects.keys() {
        if !old_objects.contains_key(name) {
            differences.push(format!("{name} (only in the new rlib)"));
        }
    }
    Ok(differences)
}

impl BuildRequest {
    /// We're going to create a DAG of modified crates, replay their rustc commands directly, and then
    /// manually link at the end.
    ///
    /// # Compilation
    ///
    /// We compile dirty crates by computing a dag across the workspace and then replaying the original
    /// rustc commands that generated their artifacts. For most crates, this results in an rlib being
    /// written to disk. In the case of hotpatching, the rlib is overwritten in-place since we're
    /// replaying the original rustc command. The nice thing here is that the rlibs remain stable
    /// in the linking command we've captured. The `.o` files in the linking command almost always
    /// come from the main tip crate.
    ///
    /// # Linking
    ///
    /// Run our custom linker setup to generate a patch file in the right location
    ///
    /// This should be the only case where the cargo output is a "dummy" file and requires us to
    /// manually do any linking.
    ///
    /// We also run some post processing steps here, like extracting out any new assets.
    ///
    /// Workspace support replays captured rustc invocations into the modified crate chain first,
    /// updating their on-disk outputs in place. The final patch link then combines the tip crate's
    /// fresh `.rcgu.o` files with the updated workspace rlibs from that replay.
    ///
    /// # Stub creation
    ///
    /// During this phase, we call out to `create_undefined_symbol_stub`. This function reads the
    /// rlibs and .o files that are about to be linked, identifies missing symbols, and then generates
    /// new assembly on the fly that satisfies these missing symbols. The assembly we generate outputs
    /// new functions with the corresponding symbol name that jump into known addresses of the originally
    /// loaded binary that's running and receiving patch updates.
    ///
    /// On wasm, we don't call this since WASM is much more complex and actually requires a full rewrite
    /// of the final binary. The `--allow-undefined` flag of wasm-ld lets us generate unrunnable binaries
    /// that we then fixup for load.
    ///
    /// # Linking command format
    ///
    /// When rustc links your project, it passes the args as how a linker would expect, but with
    /// a somewhat reliable ordering. These are all internal details to cargo/rustc, so we can't
    /// rely on them *too* much, but the *are* fundamental to how rust compiles your projects, and
    /// linker interfaces probably won't change drastically for another 40 years.
    ///
    /// We need to tear apart this command and only pass the args that are relevant to our thin link.
    /// Mainly, we don't want any dependency (non-workspace) rlibs to be linked. Occasionally some
    /// libraries like objc_exception export a folder with their artifacts - unsure if we actually
    /// need to include them. Generally you can err on the side that most *libraries* don't need to
    /// be linked here since dlopen satisfies those symbols anyways when the binary is loaded. In the
    /// future, if there are weird issues with a non-rust crate being linked incorrectly during hotpatch,
    /// the logic here would be a good place to check first.
    ///
    /// The format of this command roughly follows:
    /// ```
    /// clang
    ///     /dioxus/target/debug/subsecond-cli
    ///     /var/folders/zs/gvrfkj8x33d39cvw2p06yc700000gn/T/rustcAqQ4p2/symbols.o
    ///     /dioxus/target/subsecond-dev/deps/subsecond_harness-acfb69cb29ffb8fa.05stnb4bovskp7a00wyyf7l9s.rcgu.o
    ///     /dioxus/target/subsecond-dev/deps/subsecond_harness-acfb69cb29ffb8fa.08rgcutgrtj2mxoogjg3ufs0g.rcgu.o
    ///     /dioxus/target/subsecond-dev/deps/subsecond_harness-acfb69cb29ffb8fa.0941bd8fa2bydcv9hfmgzzne9.rcgu.o
    ///     /dioxus/target/subsecond-dev/deps/libbincode-c215feeb7886f81b.rlib
    ///     /dioxus/target/subsecond-dev/deps/libanyhow-e69ac15c094daba6.rlib
    ///     /dioxus/target/subsecond-dev/deps/libratatui-c3364579b86a1dfc.rlib
    ///     /.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/lib/libstd-019f0f6ae6e6562b.rlib
    ///     /.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/lib/libpanic_unwind-7387d38173a2eb37.rlib
    ///     /.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/lib/libobject-2b03cf6ece171d21.rlib
    ///     -framework AppKit
    ///     -lc
    ///     -framework Foundation
    ///     -framework Carbon
    ///     -lSystem
    ///     -framework CoreFoundation
    ///     -lobjc
    ///     -liconv
    ///     -lm
    ///     -arch arm64
    ///     -mmacosx-version-min=11.0.0
    ///     -L /dioxus/target/subsecond-dev/build/objc_exception-dc226cad0480ea65/out
    ///     -o /dioxus/target/subsecond-dev/deps/subsecond_harness-acfb69cb29ffb8fa
    ///     -nodefaultlibs
    ///     -Wl,-all_load
    /// ```
    ///
    /// Many args are passed twice, too, which can be confusing, but generally don't have any real
    /// effect. Note that on macos/ios, there's a special macho header that needs to be set, otherwise
    /// dyld will complain.
    ///
    /// Also, some flags in darwin land might become deprecated, need to be super conservative:
    /// - <https://developer.apple.com/forums/thread/773907>
    ///
    /// We need to be careful about which linker we're interpreting too. Some are old, some are new,
    /// some are experimental, and each has their own syntax ie `-C, /C, --C, C=` which need to be handlded.
    pub async fn compile_workspace_hotpatch(&self, ctx: &BuildContext) -> Result<BuildArtifacts> {
        let BuildMode::Thin {
            modified_crates,
            replay_in_place,
            replay_out_of_place,
            out_of_place_crates,
            additive_crates,
            previous_tip,
            ..
        } = &ctx.mode
        else {
            bail!("Not thin mode!")
        };

        let mut link_crates = modified_crates.clone();
        link_crates.extend(out_of_place_crates.iter().cloned());

        // The patch runs the code of the tip and calls into the base module for every crate
        // that the link leaves out. So a call from the tip to a changed function must pass
        // through patch code only: the patch links the rlib of every workspace crate between
        // the tip and a changed crate. Those rlibs come from the fat build, which compiled
        // them as PIC, or from an earlier in-place replay. They do not compile again.
        link_crates.extend(self.workspace_dependents_cascade(out_of_place_crates));

        let attempt = self
            .compile_workspace_hotpatch_attempt(
                ctx,
                replay_in_place,
                replay_out_of_place,
                &link_crates,
                out_of_place_crates,
                previous_tip.as_ref(),
            )
            .await;

        match attempt {
            Ok(mut artifacts) => {
                artifacts.replayed_in_place = replay_in_place.clone();
                Ok(artifacts)
            }
            // The interface gate is a heuristic. When a gated patch fails after its replay, the
            // gate can have missed an interface change: replay the changed crates in place with
            // their full dependent cascade, as an ungated patch does. A failed replay is a
            // compile error in the changed crate itself, and a cascade cannot fix it, unless an
            // additive crate holds a new item that the failed crate uses: the replay compiled
            // against the old metadata. The tip compiles against the old metadata too, so a
            // tip error with an additive crate gets the same retry. An edit that changed no
            // function gives the cascade nothing to link either.
            Err(err)
                if (!replay_out_of_place.is_empty() || !additive_crates.is_empty())
                    && (err.downcast_ref::<ReplayFailed>().is_none()
                        || !additive_crates.is_empty())
                    && err.downcast_ref::<NothingToPatch>().is_none() =>
            {
                let mut roots = replay_out_of_place.clone();
                roots.extend(additive_crates.iter().cloned());
                tracing::warn!(
                    "The gated patch for {roots:?} failed: {err:#}. \
                     Retrying with the full dependent cascade."
                );
                let cascade = self.workspace_dependents_cascade(&roots);
                let mut in_place = replay_in_place.clone();
                in_place.extend(cascade.iter().cloned());
                let mut out_of_place = out_of_place_crates.clone();
                for crate_name in &cascade {
                    out_of_place.remove(crate_name);
                }
                let mut link_crates = modified_crates.clone();
                link_crates.extend(cascade.iter().cloned());
                link_crates.extend(out_of_place.iter().cloned());
                // The cascade replays in place, so the tip must compile again.
                let mut artifacts = self
                    .compile_workspace_hotpatch_attempt(
                        ctx,
                        &in_place,
                        &HashSet::new(),
                        &link_crates,
                        &out_of_place,
                        None,
                    )
                    .await?;
                artifacts.replayed_in_place = in_place;
                Ok(artifacts)
            }
            Err(err) => Err(err),
        }
    }

    /// Check the interface gate for the crates in `out_of_place`: compile each workspace
    /// dependent of those crates twice, once against the old rlibs in the cargo target
    /// directory and once against the new out-of-place rlibs, and compare the object members of
    /// the two rlibs. Equal objects show that the dependents did not need a replay. A difference
    /// means that the gate missed an exported body; dx then logs a warning.
    ///
    /// Both compiles run without the incremental cache, so that they are deterministic. The
    /// pass runs in the background after a patch and removes its directory at the end.
    pub(crate) async fn verify_skipped_dependents(
        self,
        workspace_rustc_args: WorkspaceRustcArgs,
        out_of_place: HashSet<String>,
    ) -> Result<()> {
        let started = SystemTime::now();
        let mut dependents = self.workspace_dependents_cascade(&out_of_place);
        for crate_name in &out_of_place {
            dependents.remove(crate_name);
        }
        if dependents.is_empty() {
            return Ok(());
        }
        let levels = self.workspace_hotpatch_replay_levels(&dependents, &workspace_rustc_args)?;
        let dependents: Vec<String> = levels.iter().flatten().cloned().collect();
        tracing::debug!(
            "Verifying the skip of {} dependents of {out_of_place:?}: {levels:?}",
            dependents.len()
        );

        let pass = format!(
            "{}",
            started
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or_default()
        );
        let pass_dir = self.hotpatch_verify_dir(&pass);
        let old_dir = pass_dir.join("old");
        let new_dir = pass_dir.join("new");

        // The dummy context drops every status update: the pass must not move the TUI progress.
        let (tx, _rx) = futures_channel::mpsc::unbounded();
        let ctx = BuildContext::new(tx, BuildMode::Base, crate::BuildId::PRIMARY);

        // The new rlibs: the out-of-place replays, and then the dependents as they compile.
        let mut new_rlibs: HashMap<String, PathBuf> = HashMap::new();
        let mut search_dirs: Vec<PathBuf> = vec![];
        for crate_name in &out_of_place {
            let Some(args) = self.workspace_hotpatch_replay_args(&workspace_rustc_args, crate_name)
            else {
                continue;
            };
            let dir = self.hotpatch_replay_dir(crate_name);
            new_rlibs.insert(
                crate_name.clone(),
                self.find_rlib_for_crate(crate_name, args, Some(dir.clone()))?,
            );
            search_dirs.push(dir);
        }

        let result = async {
            for level in &levels {
                let mut old_overrides: HashMap<&String, ReplayOverrides> = HashMap::new();
                let mut new_overrides: HashMap<&String, ReplayOverrides> = HashMap::new();
                for crate_name in level {
                    old_overrides.insert(
                        crate_name,
                        ReplayOverrides {
                            out_dir: Some(old_dir.join(crate_name)),
                            no_incremental: true,
                            ..ReplayOverrides::default()
                        },
                    );
                    new_overrides.insert(
                        crate_name,
                        ReplayOverrides {
                            out_dir: Some(new_dir.join(crate_name)),
                            extern_rlibs: new_rlibs.clone(),
                            search_dirs: search_dirs.clone(),
                            no_incremental: true,
                        },
                    );
                }
                let one_level = std::slice::from_ref(level);
                self.replay_levels(&ctx, one_level, &workspace_rustc_args, &old_overrides)
                    .await?;
                self.replay_levels(&ctx, one_level, &workspace_rustc_args, &new_overrides)
                    .await?;
                for crate_name in level {
                    let args = self
                        .workspace_hotpatch_replay_args(&workspace_rustc_args, crate_name)
                        .with_context(|| format!("Missing rustc args for '{crate_name}'"))?;
                    let dir = new_dir.join(crate_name);
                    new_rlibs.insert(
                        crate_name.clone(),
                        self.find_rlib_for_crate(crate_name, args, Some(dir.clone()))?,
                    );
                    search_dirs.push(dir);
                }
            }

            let mut differing: Vec<(String, Vec<String>)> = vec![];
            for crate_name in &dependents {
                let args = self
                    .workspace_hotpatch_replay_args(&workspace_rustc_args, crate_name)
                    .with_context(|| format!("Missing rustc args for '{crate_name}'"))?;
                let old = self.find_rlib_for_crate(crate_name, args, Some(old_dir.join(crate_name)))?;
                let new = self.find_rlib_for_crate(crate_name, args, Some(new_dir.join(crate_name)))?;
                let differences = rlib_object_differences(&old, &new)?;
                if !differences.is_empty() {
                    differing.push((crate_name.clone(), differences));
                }
            }
            Ok::<_, Error>(differing)
        }
        .await;

        _ = std::fs::remove_dir_all(&pass_dir);

        let differing = result?;
        let elapsed = started.elapsed().map(|d| d.as_secs_f32()).unwrap_or_default();
        if differing.is_empty() {
            tracing::info!(
                "Verified the skip of {} dependents of {out_of_place:?} in {elapsed:.1}s: \
                 the objects are identical",
                dependents.len()
            );
        } else {
            for (crate_name, differences) in &differing {
                tracing::warn!(
                    "The skipped dependent '{crate_name}' compiles to different objects against \
                     the new rlibs of {out_of_place:?}. The interface gate missed an exported \
                     body. Differences: {differences:?}"
                );
            }
        }
        Ok(())
    }

    /// The crates in `roots` and every workspace crate that transitively depends on one of them.
    /// The tip crate is not part of the result.
    fn workspace_dependents_cascade(&self, roots: &HashSet<String>) -> HashSet<String> {
        let tip = self.tip_package_name();
        let mut result = HashSet::new();
        let mut to_visit: Vec<String> = roots.iter().cloned().collect();
        while let Some(crate_name) = to_visit.pop() {
            if crate_name == tip || !result.insert(crate_name.clone()) {
                continue;
            }
            to_visit.extend(self.workspace_dependents_of(&crate_name));
        }
        result
    }

    /// One attempt at a thin build: replay `replay_in_place` into the cargo target directory and
    /// `replay_out_of_place` into the dx replay directory, compile the tip, and link the patch
    /// from the tip objects and the rlibs of `link_crates`. A crate in `out_of_place_crates`
    /// links its out-of-place rlib. With `previous_tip` and no in-place replay, the attempt
    /// links the tip objects of the previous patch instead of a new tip compile.
    async fn compile_workspace_hotpatch_attempt(
        &self,
        ctx: &BuildContext,
        replay_in_place: &HashSet<String>,
        replay_out_of_place: &HashSet<String>,
        link_crates: &HashSet<String>,
        out_of_place_crates: &HashSet<String>,
        previous_tip: Option<&TipObjects>,
    ) -> Result<BuildArtifacts> {
        let BuildMode::Thin {
            aslr_reference,
            workspace_rustc_args,
            cache,
            modified_crates,
            ..
        } = &ctx.mode
        else {
            bail!("Not thin mode!")
        };

        tracing::debug!(
            "Replay in place {replay_in_place:?}, out of place {replay_out_of_place:?}, \
             link {link_crates:?}, out-of-place rlibs {out_of_place_crates:?}"
        );

        // Replay the rustcs for the workspace crates of this patch. This is not the final tip binary.
        // Note that the final tip might include itself as a lib (lib.rs + main.rs) which gets covered here.
        ctx.profile_phase("Workspace hotpatch replay");
        let mut replay_set = replay_in_place.clone();
        replay_set.extend(replay_out_of_place.iter().cloned());
        let replay_levels =
            self.workspace_hotpatch_replay_levels(&replay_set, workspace_rustc_args)?;
        let replayed_crates: Vec<String> = replay_levels.iter().flatten().cloned().collect();
        tracing::debug!(
            "replaying {} crates in {} dependency levels: {replay_levels:?}",
            replayed_crates.len(),
            replay_levels.len()
        );

        let replay_overrides: HashMap<&String, ReplayOverrides> = replay_out_of_place
            .iter()
            .map(|crate_name| {
                (
                    crate_name,
                    ReplayOverrides {
                        out_dir: Some(self.hotpatch_replay_dir(crate_name)),
                        ..ReplayOverrides::default()
                    },
                )
            })
            .collect();
        let replay = async {
            self.replay_levels(ctx, &replay_levels, workspace_rustc_args, &replay_overrides)
                .await
                .map_err(|err| err.context(ReplayFailed))
        };

        // The tip compiles against the metadata in the cargo target directory. An in-place
        // replay changes that metadata, so the tip waits for the replay. An out-of-place replay
        // does not touch it, so the tip compiles at the same time as the replay.
        let tip = self.compile_tip(ctx, previous_tip.filter(|_| replay_in_place.is_empty()));
        let tip = if replay_in_place.is_empty() {
            futures_util::future::try_join(replay, tip).await?.1
        } else {
            replay.await?;
            tip.await?
        };

        ctx.status_writing_patch();
        ctx.profile_phase("Patch: Cache Tip Objects");

        // The link args of the tip compile name the tip objects and the linker inputs.
        let link_args = &tip.link_args;
        let tip_bin_key = format!("{}.bin", self.tip_crate_name());
        let args = workspace_rustc_args
            .rustc_args
            .get(&tip_bin_key)
            .cloned()
            .with_context(|| {
                format!(
                    "Missing rustc args for tip bin target '{tip_bin_key}' \
                     (available keys: {:?})",
                    workspace_rustc_args.rustc_args.keys().collect::<Vec<_>>()
                )
            })?;

        let mut dylibs = vec![];

        // The tip objects. After the link they move to `hotpatch_tip_dir`, for the next patch.
        let temp_objects: Vec<PathBuf> = link_args
            .iter()
            .filter(|arg| arg.ends_with(".rcgu.o"))
            .sorted()
            .map(PathBuf::from)
            .collect();

        let workspace_rlibs = self.workspace_hotpatch_link_rlibs(
            workspace_rustc_args,
            link_crates,
            out_of_place_crates,
        )?;

        // Merge both sets for the linker. Merge order
        let mut object_files: Vec<PathBuf> = temp_objects.clone();
        object_files.extend(workspace_rlibs.iter().cloned());

        // A wasm patch links only the objects that define the functions it must hold, see
        // `HotpatchModuleCache::patch_functions`. The linker keeps a function only when a
        // root reaches it, and it loads a member of an rlib only when a loaded object
        // references a symbol of that member. A whole rlib in the link, or `main` as a root,
        // pulls the closure of every referenced member: fifty megabytes of code for a change
        // in one low crate. So the link input is the tip objects and the rlib members that
        // define the needed functions, and nothing else. Every other reference becomes an
        // import that the jump table resolves to a slot of the base table or to an export of
        // the base. The roots are the needed functions that have a slot in the base table,
        // through `--export-if-defined`, in a response file because a name can hold a space
        // or an arrow. `finalize_patch_wasm` removes the exports again.
        let mut roots: Vec<String> = Vec::new();
        let patch_objects_dir = self.patch_exe(tip.time_start).with_extension("objects");
        if self.linker_flavor() == LinkerFlavor::WasmLld {
            let t_select = std::time::Instant::now();
            let index = crate::build::patch::function_hashes(&object_files)?;
            let seeds = self.changed_functions_since_base(
                workspace_rustc_args,
                modified_crates,
                out_of_place_crates,
                &object_files,
                temp_objects.len(),
                &index,
            )?;
            let functions = cache.patch_functions(seeds.iter().map(String::as_str));
            // A changed function that the base does not hold, a new function, is not in
            // `needed`. The patch holds it too, for a caller that the patch holds.
            // An earlier patch since the fat build repointed the slots of the functions it
            // defined. The patch defines them again, so those slots repoint to the newest
            // code, also when this edit took the code back to the base. An empty set here
            // means the edit changed no code, and the app keeps the last patch.
            let earlier: Vec<String> = cache
                .patched_functions
                .read()
                .unwrap()
                .iter()
                .cloned()
                .collect();
            let earlier_functions = cache.patch_functions(earlier.iter().map(String::as_str));
            let mut wanted: HashSet<&str> = functions.needed.clone();
            wanted.extend(seeds.iter().map(String::as_str));
            wanted.extend(earlier_functions.needed.iter().copied());
            wanted.extend(earlier.iter().map(String::as_str));
            if wanted.is_empty() {
                return Err(anyhow::anyhow!(
                    "the link input holds no function that differs from the fat build"
                )
                .context(NothingToPatch));
            }
            cache
                .patched_functions
                .write()
                .unwrap()
                .extend(wanted.iter().map(|name| name.to_string()));
            let (objects, missing) = index.objects_for(wanted.iter().copied());
            if !missing.is_empty() {
                tracing::debug!(
                    "{} needed functions are in no object of the link, they become imports: {:?}",
                    missing.len(),
                    &missing[..missing.len().min(20)]
                );
            }
            _ = std::fs::remove_dir_all(&patch_objects_dir);
            std::fs::create_dir_all(&patch_objects_dir)
                .context("Could not create the patch object dir")?;
            object_files = index.write_objects(&object_files, &objects, &patch_objects_dir)?;
            roots = functions
                .roots
                .iter()
                .chain(&earlier_functions.roots)
                .map(|name| name.to_string())
                .collect();
            roots.sort_unstable();
            roots.dedup();
            tracing::debug!(
                "Patch link input: {} changed functions, {} needed, {} roots, {} of {} objects, in {:?}",
                seeds.len(),
                wanted.len(),
                roots.len(),
                object_files.len(),
                index.objects.len(),
                t_select.elapsed()
            );
        }

        // On non-wasm platforms, we generate a special shim object file which converts symbols from
        // fat binary into direct addresses from the running process.
        //
        // Our wasm approach is quite specific to wasm. We don't need to resolve any missing symbols
        // there since wasm is relocatable, but there is considerable pre and post processing work to
        // satisfy undefined symbols that we do by munging the binary directly.
        //
        // todo: can we adjust our wasm approach to also use a similar system?
        // todo: don't require the aslr reference and just patch the got when loading.
        //
        // Requiring the ASLR offset here is necessary but unfortunately might be flakey in practice.
        // Android apps can take a long time to open, and a hot patch might've been issued in the interim,
        // making this hotpatch a failure.
        if !self.is_wasm_or_wasi() {
            let stub_bytes = crate::build::create_undefined_symbol_stub(
                cache,
                &object_files,
                &self.triple,
                *aslr_reference,
            )
            .expect("failed to resolve patch symbols");

            // Currently we're dropping stub.o in the exe dir, but should probably just move to a tempfile?
            let patch_file = self.main_exe().with_file_name("stub.o");
            std::fs::write(&patch_file, stub_bytes)?;
            object_files.push(patch_file);

            // Add the dylibs/sos to the linker args
            // Make sure to use the one in the bundle, not the ones in the target dir or system.
            for arg in link_args {
                if arg.ends_with(".dylib") || arg.ends_with(".so") {
                    let path = PathBuf::from(arg);
                    dylibs.push(self.frameworks_folder().join(path.file_name().unwrap()));
                }
            }
        }

        // And now we can run the linker with our new args
        let linker = self.select_linker()?;
        let out_exe = self.patch_exe(tip.time_start);
        let out_arg = match self.triple.operating_system {
            OperatingSystem::Windows => vec![format!("/OUT:{}", out_exe.display())],
            _ => vec!["-o".to_string(), out_exe.display().to_string()],
        };

        tracing::trace!("Linking with {:?} using args: {:#?}", linker, object_files);
        tracing::trace!("Workspace hotpatch rlibs: {:#?}", workspace_rlibs);

        let mut out_args: Vec<OsString> = vec![];
        out_args.extend(object_files.iter().map(Into::into));
        out_args.extend(dylibs.iter().map(Into::into));
        out_args.extend(self.thin_link_args(link_args)?.iter().map(Into::into));

        let roots_file = out_exe.with_extension("roots.rsp");
        if self.linker_flavor() == LinkerFlavor::WasmLld {
            let mut contents = String::new();
            for name in roots.iter().sorted() {
                contents.push_str("\"--export-if-defined=");
                contents.push_str(&name.replace('\\', "\\\\").replace('"', "\\\""));
                contents.push_str("\"\n");
            }
            std::fs::write(&roots_file, contents).context("Could not write the linker roots")?;
            out_args.push(format!("@{}", roots_file.display()).into());
        }
        out_args.extend(out_arg.iter().map(Into::into));

        if cfg!(windows) {
            let cmd_contents: String = out_args
                .iter()
                .map(|s| format!("\"{}\"", s.to_string_lossy()))
                .join(" ");
            std::fs::write(self.windows_command_file(), cmd_contents)
                .context("Failed to write linker command file")?;
            out_args = vec![format!("@{}", self.windows_command_file().display()).into()];
        }

        // Add more search paths for the linker
        let mut command_envs = args.envs.clone();

        // On linux, we need to set a more complete PATH for the linker to find its libraries
        if cfg!(target_os = "linux") {
            command_envs.push(("PATH".to_string(), std::env::var("PATH").unwrap()));
        }

        // Run the linker directly!
        //
        // We dump its output directly into the patch exe location which is different than how rustc
        // does it since it uses llvm-objcopy into the `target/debug/` folder.
        ctx.profile_phase("Patch: Link");
        let res = Command::new(linker)
            .args(out_args)
            .env_clear()
            .envs(command_envs)
            .output()
            .await?;
        _ = std::fs::remove_file(&roots_file);
        _ = std::fs::remove_dir_all(&patch_objects_dir);

        if !res.stderr.is_empty() {
            let errs = String::from_utf8_lossy(&res.stderr);
            if !self.patch_exe(tip.time_start).exists() || !res.status.success() {
                tracing::error!(
                    telemetry = %serde_json::json!({ "event": "hotpatch_linker_failed" }),
                    "Failed to generate patch: {}",
                    errs.trim()
                );
            } else {
                tracing::trace!("Linker output during thin linking: {}", errs.trim());
            }
        }

        // For some really weird reason that I think is because of dlopen caching, future loads of the
        // jump library will fail if we don't remove the original fat file. I think this could be
        // because of library versioning and namespaces, but really unsure.
        //
        // The errors if you forget to do this are *extremely* cryptic - missing symbols that never existed.
        //
        // Fortunately, this binary exists in two places - the deps dir and the target out dir. We
        // can just remove the one in the deps dir and the problem goes away.
        if let Some(idx) = link_args.iter().position(|arg| *arg == "-o") {
            _ = std::fs::remove_file(PathBuf::from(link_args[idx + 1].as_str()));
        }

        // Keep the tip objects for the next patch. A patch that does not change the tip links
        // them again instead of a new tip compile.
        let link_args = self.keep_tip_objects(&temp_objects, link_args)?;

        // Now extract linker metadata from the fat binary (assets, plugin data)
        let assets = self
            .collect_assets_and_metadata(&self.patch_exe(tip.time_start), ctx)
            .await?;

        // If this is a web build, reset the index.html file in case it was modified by SSG
        self.write_index_html(&assets)
            .context("Failed to write index.html")?;

        Ok(BuildArtifacts {
            root_dir: self.root_dir(),
            exe: tip.exe,
            workspace_rustc: WorkspaceRustcArgs {
                link_args,
                rustc_args: workspace_rustc_args.rustc_args.clone(),
            },
            time_start: tip.time_start,
            time_end: SystemTime::now(),
            assets,
            mode: ctx.mode.clone(),
            patch_cache: None,
            depinfo: tip.depinfo,
            build_id: ctx.build_id,
            replayed_in_place: HashSet::new(),
        })
    }

    /// Compile the tip crate, or take the tip objects of the previous patch.
    async fn compile_tip(
        &self,
        ctx: &BuildContext,
        previous_tip: Option<&TipObjects>,
    ) -> Result<TipBuild> {
        if let Some(previous) = previous_tip {
            tracing::debug!("Reusing the tip objects of the previous patch");
            return Ok(TipBuild {
                exe: previous.exe.clone(),
                link_args: previous.link_args.clone(),
                depinfo: previous.depinfo.clone(),
                time_start: SystemTime::now(),
            });
        }
        let artifacts = self.cargo_build(ctx).await?;
        Ok(TipBuild {
            exe: artifacts.exe,
            link_args: artifacts.workspace_rustc.link_args,
            depinfo: artifacts.depinfo,
            time_start: artifacts.time_start,
        })
    }

    /// Replay the crates of `levels` with rustc, one level after the other. The crates of one
    /// level have no dependency edge between them, so they compile at the same time. A crate in
    /// `overrides` replays with those overrides.
    async fn replay_levels(
        &self,
        ctx: &BuildContext,
        levels: &[Vec<String>],
        workspace_rustc_args: &WorkspaceRustcArgs,
        overrides: &HashMap<&String, ReplayOverrides>,
    ) -> Result<()> {
        let no_overrides = ReplayOverrides::default();
        for level in levels {
            let mut level_jobs = Vec::with_capacity(level.len());
            for crate_name in level {
                let rustc_args = self
                    .workspace_hotpatch_replay_args(workspace_rustc_args, crate_name)
                    .with_context(|| format!("Missing rustc args for replay: '{crate_name}'"))?;
                let crate_overrides = overrides.get(crate_name).unwrap_or(&no_overrides);
                level_jobs.push((crate_name, rustc_args, crate_overrides));
            }
            let level_futures =
                level_jobs
                    .into_iter()
                    .map(|(crate_name, rustc_args, crate_overrides)| async move {
                        self.compile_dep_crate(ctx, crate_name, rustc_args, crate_overrides)
                            .await
                            .with_context(|| {
                                format!("Failed to replay workspace crate '{crate_name}'")
                            })
                    });
            futures_util::future::try_join_all(level_futures).await?;
        }
        Ok(())
    }

    /// Move the tip objects of this patch into `hotpatch_tip_dir` and remove the objects of the
    /// previous patch from that directory. Returns `link_args` with the new object paths.
    fn keep_tip_objects(&self, objects: &[PathBuf], link_args: &[String]) -> Result<Vec<String>> {
        let dir = self.hotpatch_tip_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create the tip object dir '{}'", dir.display()))?;

        let mut moved: HashMap<String, String> = HashMap::new();
        for object in objects {
            let Some(file_name) = object.file_name() else {
                continue;
            };
            let target = dir.join(file_name);
            if *object != target {
                std::fs::rename(object, &target).with_context(|| {
                    format!(
                        "Failed to move the tip object '{}' to '{}'",
                        object.display(),
                        target.display()
                    )
                })?;
            }
            moved.insert(
                object.display().to_string(),
                target.display().to_string(),
            );
        }

        let kept: HashSet<PathBuf> = moved.values().map(PathBuf::from).collect();
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            if !kept.contains(&entry.path()) {
                _ = std::fs::remove_file(entry.path());
            }
        }

        Ok(link_args
            .iter()
            .map(|arg| moved.get(arg).cloned().unwrap_or_else(|| arg.clone()))
            .collect())
    }

    /// Take the original args passed to the "fat" build and then create the "thin" variant.
    ///
    /// This is basically just stripping away the rlibs and other libraries that will be satisfied
    /// by our stub step.
    fn thin_link_args(&self, original_args: &[String]) -> Result<Vec<String>> {
        let mut out_args = vec![];

        match self.linker_flavor() {
            // wasm32-unknown-unknown -> use wasm-ld (gnu-lld)
            //
            // We need to import a few things - namely the memory and ifunc table.
            //
            // We can safely export everything, I believe, though that led to issues with the "fat"
            // binaries that also might lead to issues here too. wasm-bindgen chokes on some symbols
            // and the resulting JS has issues.
            //
            // We turn on both --pie and --experimental-pic but I think we only need --pie.
            //
            // We don't use *any* of the original linker args since they do lots of custom exports
            // and other things that we don't need.
            //
            // The trickiest one here is -Crelocation-model=pic, which forces data symbols
            // into a GOT, making it possible to import them from the main module.
            //
            // I think we can make relocation-model=pic work for non-wasm platforms, enabling
            // fully relocatable modules with no host coordination in lieu of sending out
            // the aslr slide at runtime.
            LinkerFlavor::WasmLld => {
                out_args.extend([
                    "--fatal-warnings".to_string(),
                    "--verbose".to_string(),
                    "--import-memory".to_string(),
                    "--import-table".to_string(),
                    "--growable-table".to_string(),
                    "--allow-undefined".to_string(),
                    "--no-demangle".to_string(),
                    "--no-entry".to_string(),
                    "--pie".to_string(),
                    "--experimental-pic".to_string(),
                ]);

                // The captured args hold one `--export <symbol>` pair per exported symbol of
                // the whole program, thousands for a large app. Each pair is a root for the
                // linker, so the patch would keep every function that an export reaches,
                // five times the functions that `main` reaches. The runtime calls three
                // exports of a patch only, the relocation and constructor thunks, and the
                // jump table reads the element table and the `name` section. So the thin
                // link drops those pairs, and `main` too. The roots of the link are the needed
                // functions of the base table instead, see the response file in
                // `compile_workspace_hotpatch_attempt`.
            }

            // This uses "cc" and these args need to be ld compatible
            //
            // Most importantly, we want to pass `-dylib` to both CC and the linker to indicate that
            // we want to generate the shared library instead of an executable.
            LinkerFlavor::Darwin => {
                out_args.extend(["-Wl,-dylib".to_string()]);

                // Preserve the original args. We only preserve:
                // -framework
                // -arch
                // -lxyz
                // There might be more, but some flags might break our setup.
                for (idx, arg) in original_args.iter().enumerate() {
                    if *arg == "-framework"
                        || *arg == "-arch"
                        || *arg == "-L"
                        || *arg == "-target"
                        || (*arg == "-isysroot"
                            && matches!(
                                self.triple.operating_system,
                                target_lexicon::OperatingSystem::IOS(_)
                            ))
                    {
                        out_args.push(arg.to_string());
                        out_args.push(original_args[idx + 1].to_string());
                    }

                    if arg.starts_with("-l")
                        || arg.starts_with("-m")
                        || arg.starts_with("-nodefaultlibs")
                    {
                        out_args.push(arg.to_string());
                    }
                }
            }

            // android/linux need to be compatible with lld
            //
            // android currently drags along its own libraries and other zany flags
            LinkerFlavor::Gnu => {
                out_args.extend([
                    "-shared".to_string(),
                    "-Wl,--eh-frame-hdr".to_string(),
                    "-Wl,-z,noexecstack".to_string(),
                    "-Wl,-z,relro,-z,now".to_string(),
                    "-nodefaultlibs".to_string(),
                    "-Wl,-Bdynamic".to_string(),
                ]);

                // Preserve the original args. We only preserve:
                // -L <path>
                // -lxyz
                // -m (arch/emulation)
                // -B<path>  (gcc program search path — Rust 1.86+ injects -B/gcc-ld + -fuse-ld=lld
                //            so that cc picks up the bundled lld; we must forward it for the patch
                //            linker invocation too, otherwise cc falls back to the system `ld`)
                // -fuse-ld  (linker selection)
                // There might be more, but some flags might break our setup.
                for (idx, arg) in original_args.iter().enumerate() {
                    if *arg == "-L" {
                        out_args.push(arg.to_string());
                        out_args.push(original_args[idx + 1].to_string());
                    }

                    if arg.starts_with("-l")
                        || arg.starts_with("-m")
                        || arg.starts_with("-Wl,--target=")
                        || arg.starts_with("-Wl,-fuse-ld")
                        || arg.starts_with("-fuse-ld")
                        || arg.starts_with("-B")
                        || arg.contains("-ld-path")
                    {
                        out_args.push(arg.to_string());
                    }
                }
            }

            LinkerFlavor::Msvc => {
                out_args.extend([
                    "shlwapi.lib".to_string(),
                    "kernel32.lib".to_string(),
                    "advapi32.lib".to_string(),
                    "ntdll.lib".to_string(),
                    "userenv.lib".to_string(),
                    "ws2_32.lib".to_string(),
                    "dbghelp.lib".to_string(),
                    "/defaultlib:msvcrt".to_string(),
                    "/DLL".to_string(),
                    "/DEBUG".to_string(),
                    "/PDBALTPATH:%_PDB%".to_string(),
                    "/EXPORT:main".to_string(),
                    "/HIGHENTROPYVA:NO".to_string(),
                ]);
            }

            LinkerFlavor::Unsupported => {
                bail!("Unsupported platform for thin linking")
            }
        }

        let extract_value = |arg: &str| -> Option<String> {
            original_args
                .iter()
                .position(|a| *a == arg)
                .map(|i| original_args[i + 1].to_string())
        };

        if let Some(vale) = extract_value("-target") {
            out_args.push("-target".to_string());
            out_args.push(vale);
        }

        if let Some(vale) = extract_value("-isysroot") {
            if matches!(
                self.triple.operating_system,
                target_lexicon::OperatingSystem::IOS(_)
            ) {
                out_args.push("-isysroot".to_string());
                out_args.push(vale);
            }
        }

        Ok(out_args)
    }

    /// Compile a workspace crate directly with `rustc` using its captured args.
    ///
    /// This produces updated outputs at the same paths cargo originally wrote to.
    /// Used during thin builds to replay the modified workspace chain before the tip crate.
    /// Copy the rlib of `crate_name` into `hotpatch_base_dir` when that directory holds no rlib
    /// yet. See `changed_functions_since_base`.
    fn keep_base_rlib(&self, crate_name: &str, rustc_args: &RustcArgs) -> Result<()> {
        let Ok(rlib) = self.find_rlib_for_crate(crate_name, rustc_args, None) else {
            return Ok(());
        };
        let dir = self.hotpatch_base_dir(crate_name);
        let target = dir.join(rlib.file_name().context("The rlib has no file name")?);
        if target.exists() || !rlib.exists() {
            return Ok(());
        }
        std::fs::create_dir_all(&dir)?;
        std::fs::copy(&rlib, &target)
            .with_context(|| format!("Could not keep the base rlib '{}'", rlib.display()))?;
        Ok(())
    }

    async fn compile_dep_crate(
        &self,
        ctx: &BuildContext,
        crate_name: &str,
        rustc_args: &RustcArgs,
        overrides: &ReplayOverrides,
    ) -> Result<()> {
        let mut cmd = Command::new("rustc");
        cmd.current_dir(rustc_args.cwd.clone());
        cmd.env_clear();

        let out_dir = overrides.out_dir.as_deref();
        match out_dir {
            Some(out_dir) => {
                std::fs::create_dir_all(out_dir).with_context(|| {
                    format!(
                        "Failed to create the replay directory '{}'",
                        out_dir.display()
                    )
                })?;
            }
            // An in-place replay overwrites the rlib of the fat build. The next patch compares
            // its functions against that rlib, so the first replay after a fat build keeps a
            // copy of it in `hotpatch_base_dir`.
            None => self.keep_base_rlib(crate_name, rustc_args)?,
        }

        // Skip args[0] which is the rustc binary path captured by the wrapper.
        // We must also strip the dx linker override so replayed crates produce real outputs
        // instead of re-entering our no-link interception path.
        let mut replay_args = Vec::with_capacity(rustc_args.args.len().saturating_sub(1));
        let mut idx = 1;
        while idx < rustc_args.args.len() {
            let arg = &rustc_args.args[idx];

            if arg.starts_with("-Clinker=") {
                idx += 1;
                continue;
            }

            if arg == "-C"
                && rustc_args
                    .args
                    .get(idx + 1)
                    .is_some_and(|next| next.starts_with("linker="))
            {
                idx += 2;
                continue;
            }

            // An out-of-place replay writes its rlib into the dx replay directory, so that the
            // cargo target directory keeps the rlib that the dependents compiled against.
            if let Some(out_dir) = out_dir {
                if arg == "--out-dir" {
                    replay_args.push(arg.clone());
                    replay_args.push(out_dir.display().to_string());
                    idx += 2;
                    continue;
                }
                if arg.starts_with("--out-dir=") {
                    replay_args.push(format!("--out-dir={}", out_dir.display()));
                    idx += 1;
                    continue;
                }
            }

            // A verify compile links a dependency from a different rlib than the captured one.
            if !overrides.extern_rlibs.is_empty() {
                if arg == "--extern" {
                    if let Some(next) = rustc_args.args.get(idx + 1) {
                        replay_args.push(arg.clone());
                        replay_args.push(overrides.rewrite_extern(next));
                        idx += 2;
                        continue;
                    }
                }
                if let Some(value) = arg.strip_prefix("--extern=") {
                    replay_args.push(format!("--extern={}", overrides.rewrite_extern(value)));
                    idx += 1;
                    continue;
                }
            }

            // A verify compile runs without the incremental cache, so that two compiles of one
            // crate produce the same objects.
            if overrides.no_incremental {
                if arg == "-C"
                    && rustc_args
                        .args
                        .get(idx + 1)
                        .is_some_and(|next| next.starts_with("incremental="))
                {
                    idx += 2;
                    continue;
                }
                if arg.starts_with("-Cincremental=") {
                    idx += 1;
                    continue;
                }
            }

            replay_args.push(arg.clone());
            idx += 1;
        }

        for dir in &overrides.search_dirs {
            replay_args.push("-L".to_string());
            replay_args.push(format!("dependency={}", dir.display()));
        }

        cmd.args(&replay_args);

        // Restore the captured environment, filtering out wrapper env vars and
        // stale cargo jobserver vars to prevent recursive invocation and warnings.
        let filtered_env_keys = [
            "RUSTC_WORKSPACE_WRAPPER",
            "RUSTC_WRAPPER",
            DX_RUSTC_WRAPPER_ENV_VAR,
            "CARGO_MAKEFLAGS",
            "MAKEFLAGS",
        ];
        cmd.envs(
            rustc_args
                .envs
                .iter()
                .filter(|(k, _)| {
                    !filtered_env_keys.contains(&k.as_str()) && !k.starts_with("DX_LINK")
                })
                .cloned(),
        );

        // Wasm hotpatches are linked as relocatable PIC modules, so replayed workspace crate
        // compilations need to emit PIC-compatible objects too.
        if self.is_wasm_or_wasi() {
            cmd.arg("-Crelocation-model=pic");
        }

        // Lower each replay compiler's OS scheduling priority so the dx process keeps enough CPU to
        // render the TUI and, crucially, drain this child's stdout/stderr. A wide replay level runs
        // several rustc concurrently — each already multi-threaded via `-Zthreads`/`codegen-units` —
        // which can saturate every core and starve dx's pipe-draining task. rustc emits a large
        // (JSON) diagnostic stream; once the pipe buffer fills it blocks on write, stalling the
        // build. Deprioritizing the compilers lets dx win enough scheduling to keep the pipes
        // flowing. Best-effort and Unix-only; a failed `setpriority` is harmless.
        #[cfg(unix)]
        unsafe {
            cmd.pre_exec(|| {
                libc::setpriority(libc::PRIO_PROCESS as _, 0, 10);
                Ok(())
            });
        }

        // Stream stdout/stderr and collect diagnostics/text lines. The captured rustc args
        // include `--error-format=json-*`, so each diagnostic arrives as a single JSON line that
        // parses into a `Diagnostic` with a pre-rendered string attached.
        //
        // We buffer rather than forward-as-we-go so that on failure the rendered diagnostics can
        // be bundled into the `bail!` message — otherwise the "Build failed: ..." log lands
        // *after* the streamed diagnostics in the TUI, making the error look detached from the
        // output that explains it. On success we still forward everything, just after the child
        // exits.
        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn rustc replay")?;

        let mut stdout = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();
        let mut stderr = tokio::io::BufReader::new(child.stderr.take().unwrap()).lines();

        let mut diagnostics: Vec<Diagnostic> = vec![];

        loop {
            let line = tokio::select! {
                Ok(Some(line)) = stdout.next_line() => line,
                Ok(Some(line)) = stderr.next_line() => line,
                else => break,
            };

            // Only Diagnostic-shaped JSON is interesting. Other rustc output (artifact/emit
            // JSON lines, linker chatter, etc.) gets dropped — we don't want it polluting the
            // bail message, and on success the replayed build shouldn't be surfacing it either.
            if let Ok(diag) = serde_json::from_str::<Diagnostic>(&line) {
                diagnostics.push(diag);
            }
        }

        let status = child
            .wait()
            .await
            .context("Failed to wait for rustc replay")?;

        if !status.success() {
            let mut rendered = String::new();
            for diag in &diagnostics {
                if let Some(r) = diag.rendered.as_deref() {
                    rendered.push('\n');
                    rendered.push_str(r.trim_end());
                }
            }
            bail!("Failed to compile workspace dep crate '{crate_name}':{rendered}");
        }

        // On success, forward any diagnostics (warnings, notes) through the normal channel.
        for diag in diagnostics {
            ctx.status_build_diagnostic(diag);
        }

        // Surface the source-file list rustc just emitted into the `.d` file so the runner can
        // fold this crate's inputs (incl. `include_str!` targets, generated files in `OUT_DIR`,
        // etc.) into its filemap. Without this, edits to those files in workspace dep crates
        // wouldn't trigger a rebuild because they only show up via the dep crate's depinfo —
        // never via the tip exe's depinfo, which is the only one we historically loaded.
        //
        // Rustc emits dep-info paths relative to the cwd we just invoked it under, so canonicalize
        // against that same path.
        if let Some(dep_info_path) = dep_info_path_for_rustc_args(&replay_args) {
            match RustcDepInfo::from_file(&dep_info_path) {
                Ok(info) => {
                    ctx.status_dep_info_discovered(info.canonicalize(rustc_args.cwd.clone()).files);
                }
                Err(err) => tracing::debug!(
                    "Failed to read dep-info for replayed crate '{crate_name}' at {}: {err}",
                    dep_info_path.display()
                ),
            }
        }

        Ok(())
    }

    fn workspace_hotpatch_replay_args<'a>(
        &self,
        workspace_rustc_args: &'a WorkspaceRustcArgs,
        crate_name: &str,
    ) -> Option<&'a RustcArgs> {
        let lib_key = format!("{crate_name}.lib");
        // if crate_name == self.tip_crate_name() {
        //     return workspace_rustc_args
        //         .rustc_args
        //         .get(&format!("{crate_name}.bin"));
        // }

        workspace_rustc_args.rustc_args.get(&lib_key).or_else(|| {
            workspace_rustc_args
                .rustc_args
                .get(&format!("{crate_name}.bin"))
        })
    }

    /// Topological sort of modified workspace crates for rustc replay, grouped into dependency
    /// *levels*.
    ///
    /// The caller (builder) already guarantees that every crate in `modified_crates` transitively
    /// reaches the tip. This excludes the tip crate itself — it gets compiled separately via
    /// `cargo_build` after the replay.
    ///
    /// Every crate in a level depends only on crates in earlier levels, so crates within a single
    /// level have no edge between them and can be compiled concurrently. Levels are emitted in
    /// dependency order (each level's rlibs must exist before the next level compiles), and crates
    /// within a level are sorted lexicographically for determinism.
    fn workspace_hotpatch_replay_levels(
        &self,
        modified_crates: &HashSet<String>,
        workspace_rustc_args: &WorkspaceRustcArgs,
    ) -> Result<Vec<Vec<String>>> {
        // Exclude the tip crate — it's compiled separately via cargo_build after replay.
        // Modified crates are tracked by *package* name, which can differ from the bin
        // target name (`tip_crate_name()`), e.g. package `browser` with `[[bin]] name = "blitz"`.
        let tip = self.tip_package_name();

        // `modified_crates` is derived from the raw Cargo workspace graph (`workspace_dependents_of`),
        // which is feature- and target-agnostic. A crate can appear here yet never have been compiled
        // into the fat baseline — most commonly an OPTIONAL dependency behind a disabled feature: its
        // outgoing edges make it a "dependent" of things you edit, but the build never emitted it, so
        // no rustc args were captured. Replaying such a crate is impossible (nothing to replay) and
        // wrong (it isn't in the binary), so drop it. We key off captured args as the source of truth
        // for "what the fat build actually compiled".
        let skipped: Vec<&String> = modified_crates
            .iter()
            .filter(|name| {
                **name != tip
                    && self
                        .workspace_hotpatch_replay_args(workspace_rustc_args, name)
                        .is_none()
            })
            .collect();
        if !skipped.is_empty() {
            tracing::debug!(
                "Skipping {} modified crate(s) with no captured rustc args (not in the fat build, \
                 e.g. optional deps behind disabled features): {skipped:?}",
                skipped.len(),
            );
        }

        let crates: HashSet<&String> = modified_crates
            .iter()
            .filter(|name| {
                **name != tip
                    && self
                        .workspace_hotpatch_replay_args(workspace_rustc_args, name)
                        .is_some()
            })
            .collect();

        // Build the subgraph: edge A→B means "A must compile before B".
        let mut indegree: HashMap<&String, usize> = crates.iter().map(|name| (*name, 0)).collect();
        let mut edges: HashMap<&String, Vec<&String>> = HashMap::new();

        for crate_name in &crates {
            for dependent in self.workspace_dependents_of(crate_name) {
                if let Some(dep) = crates.get(&dependent) {
                    *indegree.entry(dep).or_default() += 1;
                    edges.entry(crate_name).or_default().push(dep);
                }
            }
        }

        // Layered Kahn's algorithm. Each pass drains every currently-ready (indegree 0) crate as a
        // single level, then releases their dependents into the next level. BTreeSet keeps each
        // level lexicographically ordered for determinism.
        let mut ready: BTreeSet<&String> = indegree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(name, _)| *name)
            .collect();
        let mut levels: Vec<Vec<String>> = Vec::new();
        let mut emitted = 0usize;
        while !ready.is_empty() {
            let level: Vec<&String> = std::mem::take(&mut ready).into_iter().collect();
            emitted += level.len();
            for &name in &level {
                for dep in edges.get(name).into_iter().flatten() {
                    let deg = indegree.get_mut(dep).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        ready.insert(dep);
                    }
                }
            }
            levels.push(level.into_iter().cloned().collect());
        }

        ensure!(
            emitted == crates.len(),
            "Cycle in workspace dependency graph — cannot determine replay order"
        );

        Ok(levels)
    }

    /// Collect the rlib paths for every replayed workspace crate, ordered for the linker.
    ///
    /// Each crate in `link_crates` is resolved to its on-disk `.rlib` using the captured
    /// rustc args from the fat build (specifically `--out-dir` and `-C extra-filename`).
    /// Every crate must resolve — a missing rlib would produce a corrupted patch binary.
    ///
    /// The returned paths preserve the original link order from the fat build's captured
    /// linker arguments. Any rlibs not found in that order are appended at the end.
    fn workspace_hotpatch_link_rlibs(
        &self,
        args: &WorkspaceRustcArgs,
        link_crates: &HashSet<String>,
        out_of_place_crates: &HashSet<String>,
    ) -> Result<Vec<PathBuf>> {
        // Resolve every linked crate to its rlib path. A crate without captured `.lib` args was
        // not in the fat build (an optional dependency behind a disabled feature) and has no
        // rlib. Every other crate must resolve — a missing rlib means we'd link a corrupted binary.
        let tip = self.tip_package_name();
        let mut wanted = HashSet::new();
        for crate_name in link_crates.iter().sorted() {
            if *crate_name == tip {
                continue;
            }
            let Some(rustc_args) = args.rustc_args.get(&format!("{crate_name}.lib")) else {
                continue;
            };

            let out_dir = out_of_place_crates
                .contains(crate_name)
                .then(|| self.hotpatch_replay_dir(crate_name));
            let rlib = self
                .find_rlib_for_crate(crate_name, rustc_args, out_dir)
                .with_context(|| {
                    format!("Could not find rlib for workspace crate '{crate_name}'")
                })?;

            wanted.insert(rlib);
        }

        // Preserve the link order from the original fat build for any rlibs that appear
        // in the captured link args.
        let mut ordered = Vec::new();
        let mut seen = HashSet::new();

        for arg in &args.link_args {
            if !arg.ends_with(".rlib") {
                continue;
            }

            let path = PathBuf::from(arg);
            if wanted.contains(&path) && seen.insert(path.clone()) {
                ordered.push(path);
            }
        }

        // Any rlibs not in the captured link order get appended at the end.
        let mut remaining: Vec<_> = wanted.into_iter().filter(|p| !seen.contains(p)).collect();
        remaining.sort();
        ordered.extend(remaining);

        Ok(ordered)
    }

    /// The functions of the changed crates and of the tip whose code differs from the base
    /// module. The base copy of every other function stays valid.
    ///
    /// `index` holds the hash of every function of `object_files`, the link inputs, whose
    /// first `tip_count` entries are the tip objects. The comparison runs against the objects
    /// of the fat build: the rlib in the cargo target directory for a crate that replayed out
    /// of place, the copy in `hotpatch_base_dir` for a crate that replayed in place, and the
    /// tip objects in `hotpatch_base_tip_dir`. Without a base copy, every function of the
    /// crate counts as changed.
    fn changed_functions_since_base(
        &self,
        args: &WorkspaceRustcArgs,
        modified_crates: &HashSet<String>,
        out_of_place_crates: &HashSet<String>,
        object_files: &[PathBuf],
        tip_count: usize,
        index: &ObjectIndex,
    ) -> Result<Vec<String>> {
        fn changed(old: &ObjectIndex, new: &ObjectIndex, sources: &HashSet<usize>) -> Vec<String> {
            new.functions
                .iter()
                .filter(|(_, function)| sources.contains(&new.objects[function.object].source))
                .filter(|(name, function)| {
                    old.functions
                        .get(*name)
                        .is_none_or(|o| o.hash != function.hash)
                })
                .map(|(name, _)| name.clone())
                .sorted()
                .collect()
        }

        let tip = self.tip_package_name();
        let mut changed_crates: HashSet<String> = modified_crates.clone();
        changed_crates.extend(out_of_place_crates.iter().cloned());
        let mut seeds = Vec::new();
        for crate_name in changed_crates.iter().sorted() {
            if *crate_name == tip {
                continue;
            }
            let Some(rustc_args) = args.rustc_args.get(&format!("{crate_name}.lib")) else {
                continue;
            };
            let out_of_place = out_of_place_crates.contains(crate_name);
            let new = self.find_rlib_for_crate(
                crate_name,
                rustc_args,
                out_of_place.then(|| self.hotpatch_replay_dir(crate_name)),
            )?;
            let Some(source) = object_files.iter().position(|path| *path == new) else {
                tracing::warn!(
                    "The rlib of the changed crate {crate_name} is not a link input: {}",
                    new.display()
                );
                continue;
            };
            let old = if out_of_place {
                Some(self.find_rlib_for_crate(crate_name, rustc_args, None)?)
            } else {
                self.find_rlib_for_crate(
                    crate_name,
                    rustc_args,
                    Some(self.hotpatch_base_dir(crate_name)),
                )
                .ok()
            };
            let old: Vec<PathBuf> = old.into_iter().filter(|path| path.exists()).collect();
            let old = crate::build::patch::function_hashes(&old)?;
            let changed = changed(&old, index, &HashSet::from([source]));
            tracing::debug!(
                "{} functions of {crate_name} differ from the base",
                changed.len()
            );
            seeds.extend(changed);
        }

        let base_tip: Vec<PathBuf> = std::fs::read_dir(self.hotpatch_base_tip_dir())
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "o"))
            .collect();
        let old = crate::build::patch::function_hashes(&base_tip)?;
        let changed = changed(&old, index, &(0..tip_count).collect());
        tracing::debug!(
            "{} functions of the tip differ from the base",
            changed.len()
        );
        seeds.extend(changed);
        Ok(seeds)
    }

    /// Patches are stored in the same directory as the main executable, but with a name based on the
    /// time the patch started compiling.
    ///
    /// - lib{name}-patch-{time}.(so/dll/dylib) (next to the main exe)
    ///
    /// Note that weirdly enough, the name of dylibs can actually matter. In some environments, libs
    /// can override each other with symbol interposition.
    ///
    /// Also, on Android - and some Linux, we *need* to start the lib name with `lib` for the dynamic
    /// loader to consider it a shared library.
    ///
    /// todo: the time format might actually be problematic if two platforms share the same build folder.
    pub(crate) fn patch_exe(&self, time_start: SystemTime) -> PathBuf {
        let path = self.main_exe().with_file_name(format!(
            "lib{}-patch-{}",
            self.executable_name(),
            time_start
                .duration_since(UNIX_EPOCH)
                .map(|f| f.as_millis())
                .unwrap_or(0),
        ));

        let extension = match self.linker_flavor() {
            LinkerFlavor::Darwin => "dylib",
            LinkerFlavor::Gnu => "so",
            LinkerFlavor::WasmLld => "wasm",
            LinkerFlavor::Msvc => "dll",
            LinkerFlavor::Unsupported => "",
        };

        path.with_extension(extension)
    }

    /// Sweep leftover `-Csave-temps=true` byproducts out of the thin-build `deps/` directory.
    ///
    /// Thin (hot-patch) builds pass `-Csave-temps=true` (see [`BuildRequest::cargo_build_arguments`])
    /// so that the per-codegen-unit `.rcgu.o` object files survive on disk for us to link the patch
    /// manually. As a side effect, rustc *also* leaves the LLVM IR/bitcode siblings it produced for
    /// each codegen unit — `.no-opt.bc`, `.opt.bc`, `.rcgu.bc`, `.ll`, `.s` — and nothing in the
    /// patch flow ever reads those again. Because codegen-unit filenames are content-hashed, every
    /// patch that touches code emits a *fresh* set of these files without overwriting the previous
    /// one, so a long `dx serve` session accumulates them without bound (observed: tens of GB).
    ///
    /// The only loose objects we ever link are the tip crate's current `.rcgu.o` set, and those are
    /// deleted immediately after the link in [`BuildRequest::compile_workspace_hotpatch`] (workspace
    /// dep crates are linked via their `.rlib`, not loose objects). So any `.rcgu.o` still sitting in
    /// `deps/` at sweep time is also a stale byproduct and safe to remove — rustc will regenerate
    /// whatever the next build needs. We deliberately leave `.rmeta`/`.rlib` and the `incremental/`
    /// cache alone; those are cargo's to manage.
    ///
    /// `link_args` are the captured rustc link args from the just-completed build; we derive the
    /// `deps/` directory from them (rather than reconstructing `target/<triple>/<profile>/deps`) so
    /// the path is correct regardless of whether cargo emitted a triple subdir.
    ///
    /// Returns `(files_removed, bytes_freed)`.
    pub(crate) fn clean_thin_build_byproducts(&self, link_args: &[String]) -> (usize, u64) {
        // Find the deps dir from the first link arg that lives under our target dir and points at a
        // codegen-unit object. If we can't (e.g. an all-rlib link), there's nothing to sweep.
        // The tip objects live in `hotpatch_tip_dir` after the link, so an rlib in `deps/`
        // names the directory too.
        let platform_dir = self.platform_dir();
        let Some(deps_dir) = link_args
            .iter()
            .map(PathBuf::from)
            .find(|p| {
                p.starts_with(&self.target_dir)
                    && !p.starts_with(&platform_dir)
                    && (p.to_string_lossy().ends_with(".rcgu.o")
                        || p.to_string_lossy().ends_with(".rlib"))
            })
            .and_then(|p| p.parent().map(Path::to_path_buf))
        else {
            return (0, 0);
        };

        sweep_thin_build_byproducts(&deps_dir)
    }

    /// When we link together the fat binary, we need to make sure every `.o` file in *every* rlib
    /// is taken into account. This is the same work that the rust compiler does when assembling
    /// staticlibs.
    ///
    /// <https://github.com/rust-lang/rust/blob/191df20fcad9331d3a948aa8e8556775ec3fe69d/compiler/rustc_codegen_ssa/src/back/link.rs#L448>
    ///
    /// Since we're going to be passing these to the linker, we need to make sure and not provide any
    /// weird files (like the rmeta) file that rustc generates.
    ///
    /// We discovered the need for this after running into issues with wasm-ld not being able to
    /// handle the rmeta file.
    ///
    /// <https://github.com/llvm/llvm-project/issues/55786>
    ///
    /// Also, crates might not drag in all their dependent code. The monorphizer won't lift trait-based generics:
    ///
    /// <https://github.com/rust-lang/rust/blob/191df20fcad9331d3a948aa8e8556775ec3fe69d/compiler/rustc_monomorphize/src/collector.rs>
    ///
    /// When Rust normally handles this, it uses the +whole-archive directive which adjusts how the rlib
    /// is written to disk.
    ///
    /// Since creating this object file can be a lot of work, we cache it in the target dir by hashing
    /// the names of the rlibs in the command and storing it in the target dir. That way, when we run
    /// this command again, we can just used the cached object file.
    ///
    /// In theory, we only need to do this for every crate accessible by the current crate, but that's
    /// hard acquire without knowing the exported symbols from each crate.
    ///
    /// todo: I think we can traverse our immediate dependencies and inspect their symbols, unless they `pub use` a crate
    /// todo: we should try and make this faster with memmapping
    pub(crate) async fn run_fat_link(
        &self,
        ctx: &BuildContext,
        exe: &Path,
        set: &WorkspaceRustcArgs,
    ) -> Result<()> {
        // Get the tip crate rustc argsa
        let rustc_args = set
            .rustc_args
            .get(&format!("{}.bin", self.tip_crate_name()))
            .context("Missing rustc capture")?;

        ensure!(
            !set.link_args.is_empty(),
            "Missing linker args for fat link of '{}'. The tip crate likely did not run through linker interception for this build.",
            self.tip_crate_name()
        );

        let link_start = SystemTime::now();
        ctx.status_starting_fat_link();

        // Filter out the rlib files from the arguments
        let rlibs = set
            .link_args
            .iter()
            .filter(|arg| arg.ends_with(".rlib"))
            .map(PathBuf::from)
            .collect::<Vec<_>>();

        // Acquire a hash from the rlib names, sizes, modified times, and dx's git commit hash
        // This ensures that any changes in dx or the rlibs will cause a new hash to be generated
        // The hash relies on both dx and rustc hashes, so it should be thoroughly unique. Keep it
        // short to avoid long file names.
        let hash_id = Uuid::new_v5(
            &Uuid::NAMESPACE_OID,
            rlibs
                .iter()
                .map(|p| {
                    format!(
                        "{}-{}-{}-{}",
                        p.file_name().unwrap().to_string_lossy(),
                        p.metadata().map(|m| m.len()).unwrap_or_default(),
                        p.metadata()
                            .ok()
                            .and_then(|m| m.modified().ok())
                            .and_then(|f| f.duration_since(UNIX_EPOCH).map(|f| f.as_secs()).ok())
                            .unwrap_or_default(),
                        crate::dx_build_info::GIT_COMMIT_HASH.unwrap_or_default()
                    )
                })
                .collect::<String>()
                .as_bytes(),
        )
        .to_string()
        .chars()
        .take(8)
        .collect::<String>();

        // Check if we already have a cached object file
        let out_ar_path = exe.with_file_name(format!("libdeps-{hash_id}.a",));
        let out_rlibs_list = exe.with_file_name(format!("rlibs-{hash_id}.txt"));
        let mut archive_has_contents = out_ar_path.exists();

        // Use the rlibs list if it exists
        let mut compiler_rlibs = std::fs::read_to_string(&out_rlibs_list)
            .ok()
            .map(|s| s.lines().map(PathBuf::from).collect::<Vec<_>>())
            .unwrap_or_default();

        // Create it by dumping all the rlibs into it
        // This will include the std rlibs too, which can severely bloat the size of the archive
        //
        // The nature of this process involves making extremely fat archives, so we should try and
        // speed up the future linking process by caching the archive.
        //
        // Since we're using the git hash for the CLI entropy, debug builds should always regenerate
        // the archive since their hash might not change, but the logic might.
        if !archive_has_contents || cfg!(debug_assertions) {
            compiler_rlibs.clear();

            let mut bytes = vec![];
            let mut out_ar = ar::Builder::new(&mut bytes);
            for rlib in &rlibs {
                // Skip compiler rlibs since they're missing bitcode
                //
                // https://github.com/rust-lang/rust/issues/94232#issuecomment-1048342201
                //
                // if the rlib is not in the target directory, we skip it.
                if !rlib.starts_with(self.workspace_dir()) {
                    compiler_rlibs.push(rlib.clone());
                    tracing::trace!("Skipping rlib: {:?}", rlib);
                    continue;
                }

                tracing::trace!("Adding rlib to staticlib: {:?}", rlib);

                let rlib_contents = std::fs::read(rlib)?;
                let mut reader = ar::Archive::new(std::io::Cursor::new(rlib_contents));
                let mut keep_linker_rlib = false;
                while let Some(Ok(object_file)) = reader.next_entry() {
                    let name = std::str::from_utf8(object_file.header().identifier()).unwrap();
                    if name.ends_with(".rmeta") {
                        continue;
                    }

                    if object_file.header().size() == 0 {
                        continue;
                    }

                    // rlibs might contain dlls/sos/lib files which we don't want to include
                    //
                    // This catches .dylib, .so, .dll, .lib, .o, etc files that are not compatible with
                    // our "fat archive" linking process.
                    //
                    // We only trust `.rcgu.o` files to make it into the --all_load archive.
                    // This is a temporary stopgap to prevent issues with libraries that generate
                    // object files that are not compatible with --all_load.
                    // see https://github.com/DioxusLabs/dioxus/issues/4237
                    if !(name.ends_with(".rcgu.o") || name.ends_with(".obj")) {
                        keep_linker_rlib = true;
                        continue;
                    }

                    archive_has_contents = true;
                    out_ar
                        .append(&object_file.header().clone(), object_file)
                        .context("Failed to add object file to archive")?;
                }

                // Some rlibs contain weird artifacts that we don't want to include in the fat archive.
                // However, we still want them around in the linker in case the regular linker can handle them.
                if keep_linker_rlib {
                    compiler_rlibs.push(rlib.clone());
                }
            }

            let bytes = out_ar.into_inner().context("Failed to finalize archive")?;
            std::fs::write(&out_ar_path, bytes).context("Failed to write archive")?;
            tracing::debug!("Wrote fat archive to {:?}", out_ar_path);

            // Run the ranlib command to index the archive. This slows down this process a bit,
            // but is necessary for some linkers to work properly.
            // We ignore its error in case it doesn't recognize the architecture
            if self.linker_flavor() == LinkerFlavor::Darwin {
                if let Some(ranlib) = Workspace::select_ranlib() {
                    _ = Command::new(ranlib).arg(&out_ar_path).output().await;
                }
            }
        }

        compiler_rlibs.dedup();

        // We're going to replace the first rlib in the args with our fat archive
        // And then remove the rest of the rlibs
        //
        // We also need to insert the -force_load flag to force the linker to load the archive
        let mut args: Vec<_> = set.link_args.clone();
        if let Some(last_object) = args.iter().rposition(|arg| arg.ends_with(".o")) {
            if archive_has_contents {
                match self.linker_flavor() {
                    LinkerFlavor::WasmLld => {
                        args.insert(last_object, "--whole-archive".to_string());
                        args.insert(last_object + 1, out_ar_path.display().to_string());
                        args.insert(last_object + 2, "--no-whole-archive".to_string());
                        args.retain(|arg| !arg.ends_with(".rlib"));
                        for rlib in compiler_rlibs.iter().rev() {
                            args.insert(last_object + 3, rlib.display().to_string());
                        }
                    }
                    LinkerFlavor::Gnu => {
                        args.insert(last_object, "-Wl,--whole-archive".to_string());
                        args.insert(last_object + 1, out_ar_path.display().to_string());
                        args.insert(last_object + 2, "-Wl,--no-whole-archive".to_string());
                        args.retain(|arg| !arg.ends_with(".rlib"));
                        for rlib in compiler_rlibs.iter().rev() {
                            args.insert(last_object + 3, rlib.display().to_string());
                        }
                    }
                    LinkerFlavor::Darwin => {
                        args.insert(last_object, "-Wl,-force_load".to_string());
                        args.insert(last_object + 1, out_ar_path.display().to_string());
                        args.retain(|arg| !arg.ends_with(".rlib"));
                        for rlib in compiler_rlibs.iter().rev() {
                            args.insert(last_object + 2, rlib.display().to_string());
                        }
                    }
                    LinkerFlavor::Msvc => {
                        args.insert(
                            last_object,
                            format!("/WHOLEARCHIVE:{}", out_ar_path.display()),
                        );
                        args.retain(|arg| !arg.ends_with(".rlib"));
                        for rlib in compiler_rlibs.iter().rev() {
                            args.insert(last_object + 1, rlib.display().to_string());
                        }
                    }
                    LinkerFlavor::Unsupported => {
                        tracing::error!("Unsupported platform for fat linking: {}", self.triple);
                    }
                };
            }
        }

        // Add custom args to the linkers
        match self.linker_flavor() {
            LinkerFlavor::Gnu => {
                // Export `main` so subsecond can use it for a reference point
                args.push("-Wl,--export-dynamic-symbol,main".to_string());
            }
            LinkerFlavor::Darwin => {
                args.push("-Wl,-exported_symbol,_main".to_string());
            }
            LinkerFlavor::Msvc => {
                // Prevent alsr from overflowing 32 bits
                args.push("/HIGHENTROPYVA:NO".to_string());

                // Export `main` so subsecond can use it for a reference point
                args.push("/EXPORT:main".to_string());
            }
            LinkerFlavor::WasmLld | LinkerFlavor::Unsupported => {}
        }

        // We also need to remove the `-o` flag since we want the linker output to end up in the
        // rust exe location, not in the deps dir as it normally would.
        if let Some(idx) = args
            .iter()
            .position(|arg| *arg == "-o" || *arg == "--output")
        {
            args.remove(idx + 1);
            args.remove(idx);
        }

        // same but windows support
        if let Some(idx) = args.iter().position(|arg| arg.starts_with("/OUT")) {
            args.remove(idx);
        }

        // We want to go through wasm-ld directly, so we need to remove the -flavor flag
        if let Some(flavor_idx) = args.iter().position(|arg| *arg == "-flavor") {
            args.remove(flavor_idx + 1);
            args.remove(flavor_idx);
        }

        // Note: Swift sources are now compiled as dynamic frameworks during the main build flow.
        // Dynamic frameworks are loaded at runtime, not linked statically, so we don't add
        // them to the linker args here. The framework will be installed to the Frameworks
        // folder by compile_swift_sources() in the main bundle creation phase.
        if matches!(
            self.triple.operating_system,
            OperatingSystem::IOS(_) | OperatingSystem::MacOSX { .. } | OperatingSystem::Darwin(_)
        ) {
            let workspace_dir = self.workspace_dir();
            let swift_sources =
                super::apple::extract_swift_metadata_from_link_args(&set.link_args, &workspace_dir);

            if !swift_sources.is_empty() {
                tracing::debug!(
                    "Found {} Swift plugin source(s) - will be compiled as dynamic framework during bundle creation",
                    swift_sources.len()
                );
            }
        }

        // Set the output file
        match self.triple.operating_system {
            OperatingSystem::Windows => args.push(format!("/OUT:{}", exe.display())),
            _ => args.extend(["-o".to_string(), exe.display().to_string()]),
        }

        // And now we can run the linker with our new args
        let linker = self.select_linker()?;

        tracing::trace!("Fat linking with args: {:?} {:#?}", linker, args);
        tracing::trace!("Fat linking with env:");
        for e in rustc_args.envs.iter() {
            tracing::trace!("  {}={}", e.0, e.1);
        }

        // Handle windows command files
        let mut out_args = args.clone();
        if cfg!(windows) {
            let cmd_contents: String = out_args.iter().map(|f| format!("\"{f}\"")).join(" ");
            std::fs::write(self.windows_command_file(), cmd_contents)
                .context("Failed to write linker command file")?;
            out_args = vec![format!("@{}", self.windows_command_file().display())];
        }

        // Add more search paths for the linker
        let mut command_envs = rustc_args.envs.clone();

        // On linux, we need to set a more complete PATH for the linker to find its libraries
        if cfg!(target_os = "linux") {
            command_envs.push(("PATH".to_string(), std::env::var("PATH").unwrap()));
        }

        // Run the linker directly!
        let res = Command::new(linker)
            .args(out_args)
            .env_clear()
            .envs(command_envs)
            .output()
            .await?;

        if !res.status.success() {
            let stderr = String::from_utf8_lossy(&res.stderr);
            let stdout = String::from_utf8_lossy(&res.stdout);
            let combined = match (stdout.trim().is_empty(), stderr.trim().is_empty()) {
                (false, false) => format!("{}\n{}", stdout.trim(), stderr.trim()),
                (false, true) => stdout.trim().to_string(),
                (true, false) => stderr.trim().to_string(),
                (true, true) => format!("linker exited with status {}", res.status),
            };

            tracing::error!(
                telemetry = %serde_json::json!({ "event": "hotpatch_fat_binary_generation_failed" }),
                "Failed to generate fat binary: {}",
                combined
            );
            bail!("Failed to generate fat binary: {combined}");
        }

        if !res.stderr.is_empty() {
            let errs = String::from_utf8_lossy(&res.stderr);
            tracing::trace!("Warnings during fat linking: {}", errs.trim());
        }

        if !res.stdout.is_empty() {
            let out = String::from_utf8_lossy(&res.stdout);
            tracing::trace!("Output from fat linking: {}", out.trim());
        }

        // Every patch compares its objects against the objects of this fat build, see
        // `changed_functions_since_base`. The tip objects move into `hotpatch_base_tip_dir`
        // on wasm, and the base rlibs of the previous fat build go away. On the other
        // platforms the temps go away.
        let objects: Vec<PathBuf> = args
            .iter()
            .filter(|arg| arg.ends_with(".rcgu.o"))
            .map(PathBuf::from)
            .collect();
        if self.is_wasm_or_wasi() {
            self.keep_base_tip_objects(&objects)?;
        } else {
            for object in &objects {
                _ = std::fs::remove_file(object);
            }
        }

        // Cache the rlibs list
        _ = std::fs::write(
            &out_rlibs_list,
            compiler_rlibs
                .into_iter()
                .map(|s| s.display().to_string())
                .join("\n"),
        );

        tracing::debug!(
            "Fat linking completed in {}us",
            SystemTime::now()
                .duration_since(link_start)
                .unwrap()
                .as_micros()
        );

        Ok(())
    }

    /// Build the jump table of a patch. `previous` holds the table slots of the last patch of
    /// the session on wasm, and the result carries the slots of this patch for the next call.
    pub(crate) fn create_jump_table(
        &self,
        patch: &Path,
        cache: &HotpatchModuleCache,
        previous: Option<&PatchIfuncs>,
    ) -> Result<(JumpTable, Option<PatchIfuncs>)> {
        use crate::build::patch::{
            create_native_jump_table, create_wasm_jump_table, create_windows_jump_table,
        };

        let root_dir = self.root_dir();
        let base_path = self.base_path();
        let triple = &self.triple;

        // Symbols are stored differently based on the platform, so we need to handle them differently.
        // - Wasm requires the walrus crate and actually modifies the patch file
        // - windows requires the pdb crate and pdb files
        // - nix requires the object crate
        let mut ifuncs = None;
        let mut jump_table = match triple.operating_system {
            OperatingSystem::Windows => create_windows_jump_table(patch, cache)?,
            _ if triple.architecture == Architecture::Wasm32 => {
                let (table, patch_ifuncs) = create_wasm_jump_table(
                    patch,
                    cache,
                    self.keep_wasm_names(),
                    self.dwarf_sidecar,
                    previous,
                )?;
                ifuncs = patch_ifuncs;
                table
            }
            _ => create_native_jump_table(patch, triple, cache)?,
        };

        // root_dir: &Path,
        //     base_path: Option<&str>,
        // Rebase the wasm binary to be relocatable once the jump table is generated
        if triple.architecture == target_lexicon::Architecture::Wasm32 {
            // Make sure we use the dir relative to the public dir, so the web can load it as a proper URL
            //
            // ie we would've shipped `/Users/foo/Projects/dioxus/target/dx/project/debug/web/public/wasm/lib.wasm`
            //    but we want to ship `/wasm/lib.wasm`
            let url_root = PathBuf::from(
                "/".to_string() + base_path.unwrap_or_default().trim_start_matches('/'),
            );
            jump_table.lib = url_root.join(jump_table.lib.strip_prefix(&root_dir).unwrap());
            if let Some(sidecar) = jump_table.dwarf_sidecar.as_mut() {
                *sidecar = url_root.join(sidecar.strip_prefix(&root_dir).unwrap());
            }
        }

        Ok((jump_table, ifuncs))
    }

    /// Automatically detect the linker flavor based on the target triple and any custom linkers.
    ///
    /// This tries to replicate what rustc does when selecting the linker flavor based on the linker
    /// and triple.
    fn linker_flavor(&self) -> LinkerFlavor {
        if let Some(custom) = self.custom_linker.as_ref() {
            let name = custom.file_name().unwrap().to_ascii_lowercase();
            match name.to_str() {
                Some("lld-link") => return LinkerFlavor::Msvc,
                Some("lld-link.exe") => return LinkerFlavor::Msvc,
                Some("wasm-ld") => return LinkerFlavor::WasmLld,
                Some("ld64.lld") => return LinkerFlavor::Darwin,
                Some("ld.lld") => return LinkerFlavor::Gnu,
                Some("ld.gold") => return LinkerFlavor::Gnu,
                Some("mold") => return LinkerFlavor::Gnu,
                Some("sold") => return LinkerFlavor::Gnu,
                Some("wild") => return LinkerFlavor::Gnu,
                _ => {}
            }
        }

        match self.triple.environment {
            target_lexicon::Environment::Gnu
            | target_lexicon::Environment::Gnuabi64
            | target_lexicon::Environment::Gnueabi
            | target_lexicon::Environment::Gnueabihf
            | target_lexicon::Environment::GnuLlvm => LinkerFlavor::Gnu,
            target_lexicon::Environment::Musl => LinkerFlavor::Gnu,
            target_lexicon::Environment::Android => LinkerFlavor::Gnu,
            target_lexicon::Environment::Msvc => LinkerFlavor::Msvc,
            target_lexicon::Environment::Macabi => LinkerFlavor::Darwin,
            _ => match self.triple.operating_system {
                OperatingSystem::Darwin(_) => LinkerFlavor::Darwin,
                OperatingSystem::IOS(_) => LinkerFlavor::Darwin,
                OperatingSystem::MacOSX(_) => LinkerFlavor::Darwin,
                OperatingSystem::Linux => LinkerFlavor::Gnu,
                OperatingSystem::Windows => LinkerFlavor::Msvc,
                _ => match self.triple.architecture {
                    target_lexicon::Architecture::Wasm32 => LinkerFlavor::WasmLld,
                    target_lexicon::Architecture::Wasm64 => LinkerFlavor::WasmLld,
                    _ => LinkerFlavor::Unsupported,
                },
            },
        }
    }

    /// Select the linker to use for this platform.
    ///
    /// We prefer to use the rust-lld linker when we can since it's usually there.
    /// On macos, we use the system linker since macho files can be a bit finicky.
    ///
    /// This means we basically ignore the linker flavor that the user configured, which could
    /// cause issues with a custom linker setup. In theory, rust translates most flags to the right
    /// linker format.
    fn select_linker(&self) -> Result<PathBuf, Error> {
        if let Some(linker) = self.custom_linker.clone() {
            return Ok(linker);
        }

        let cc = match self.linker_flavor() {
            LinkerFlavor::WasmLld => self.workspace.wasm_ld(),

            // On macOS, we use the system linker since it's usually there.
            // We could also use `lld` here, but it might not be installed by default.
            //
            // Note that this is *clang*, not `lld`.
            LinkerFlavor::Darwin => self.workspace.cc(),

            // On Linux, we use the system linker since it's usually there.
            LinkerFlavor::Gnu => self.workspace.cc(),

            // On windows, instead of trying to find the system linker, we just go with the lld.link
            // that rustup provides. It's faster and more stable then reyling on link.exe in path.
            LinkerFlavor::Msvc => self.workspace.lld_link(),

            // The rest of the platforms use `cc` as the linker which should be available in your path,
            // provided you have build-tools setup. On mac/linux this is the default, but on Windows
            // it requires msvc or gnu downloaded, which is a requirement to use rust anyways.
            //
            // The default linker might actually be slow though, so we could consider using lld or rust-lld
            // since those are shipping by default on linux as of 1.86. Window's linker is the really slow one.
            //
            // https://blog.rust-lang.org/2024/05/17/enabling-rust-lld-on-linux.html
            //
            // Note that "cc" is *not* a linker. It's a compiler! The arguments we pass need to be in
            // the form of `-Wl,<args>` for them to make it to the linker. This matches how rust does it
            // which is confusing.
            LinkerFlavor::Unsupported => self.workspace.cc(),
        };

        Ok(cc)
    }

    /// Find the rlib path for a workspace crate from its captured rustc args.
    ///
    /// Extracts `--out-dir` and `-C extra-filename` from the args to construct the exact
    /// rlib filename. This is important because multiple rlibs for the same crate can coexist
    /// in the deps directory (e.g., from different dx builds that produce different `-C metadata`),
    /// and globbing would return an arbitrary one.
    fn find_rlib_for_crate(
        &self,
        crate_name: &str,
        rustc_args: &RustcArgs,
        out_dir_override: Option<PathBuf>,
    ) -> Result<PathBuf> {
        // Extract --out-dir from the captured args, unless the crate replayed out of place.
        let out_dir = match out_dir_override {
            Some(dir) => dir,
            None => rustc_args
                .args
                .iter()
                .zip(rustc_args.args.iter().skip(1))
                .find(|(flag, _)| *flag == "--out-dir")
                .map(|(_, dir)| PathBuf::from(dir))
                .with_context(|| {
                    format!("No --out-dir in captured rustc args for '{crate_name}'")
                })?,
        };

        // Extract -C extra-filename from captured args.
        // Cargo passes this to rustc to disambiguate output filenames via metadata hash.
        // Handle all forms: `-Cextra-filename=X`, `-C extra-filename=X`, and `-C` `extra-filename=X`.
        let extra_filename = rustc_args.args.iter().enumerate().find_map(|(i, arg)| {
            arg.strip_prefix("-Cextra-filename=")
                .map(|s| s.to_string())
                .or_else(|| {
                    if arg == "-C" {
                        rustc_args.args.get(i + 1).and_then(|next| {
                            next.strip_prefix("extra-filename=").map(|s| s.to_string())
                        })
                    } else {
                        None
                    }
                })
        });

        // If we have an exact extra-filename, construct the precise rlib path.
        if let Some(extra) = &extra_filename {
            let exact = out_dir.join(format!("lib{crate_name}{extra}.rlib"));
            if exact.exists() {
                return Ok(exact);
            }
        }

        // Fallback: glob for lib<crate_name>-<hash>.rlib in the output directory.
        // Prefer the most recently modified rlib to avoid picking up stale artifacts.
        let prefix = format!("lib{crate_name}-");
        let mut best: Option<(PathBuf, std::time::SystemTime)> = None;
        for entry in std::fs::read_dir(&out_dir)
            .with_context(|| format!("Could not read --out-dir '{}'", out_dir.display()))?
            .flatten()
        {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with(&prefix) && name.ends_with(".rlib") {
                    if let Ok(meta) = entry.metadata() {
                        if let Ok(mtime) = meta.modified() {
                            if best.as_ref().is_none_or(|(_, t)| mtime > *t) {
                                best = Some((entry.path(), mtime));
                            }
                        }
                    }
                }
            }
        }

        best.map(|(path, _)| path).with_context(|| {
            format!(
                "No rlib found for '{crate_name}' in '{}' \
                 (looked for lib{crate_name}*.rlib, extra-filename={:?})",
                out_dir.display(),
                extra_filename
            )
        })
    }

    fn rustc_wrapper_capture_mode(&self, build_mode: &BuildMode) -> &'static str {
        match build_mode {
            BuildMode::Fat => "fat",
            BuildMode::Base => "base",
            BuildMode::Thin { .. } => "thin",
        }
    }

    pub fn rustc_wrapper_scope_dir_name(&self, build_mode: &BuildMode) -> Result<String> {
        #[derive(Debug, Serialize)]
        struct RustcWrapperScope {
            version: u8,
            capture_mode: &'static str,
            bundle: String,
            triple: String,
            profile: String,
            package: String,
            main_target: String,
            executable_type: String,
            rustc_version: String,
            features: Vec<String>,
            all_features: bool,
            rustflags: Vec<String>,
            extra_cargo_args: Vec<String>,
            extra_rustc_args: Vec<String>,
        }

        let scope = RustcWrapperScope {
            version: 1,
            capture_mode: self.rustc_wrapper_capture_mode(build_mode),
            bundle: self.bundle.to_string(),
            triple: self.triple.to_string(),
            profile: self.profile.clone(),
            package: self.package.clone(),
            main_target: self.main_target.clone(),
            executable_type: format!("{:?}", self.executable_type()),
            rustc_version: self.workspace.rustc_version.clone(),
            features: self.features.clone(),
            all_features: self.all_features,
            rustflags: self.build_rustflags(build_mode).flags,
            extra_cargo_args: self.extra_cargo_args.clone(),
            extra_rustc_args: self.extra_rustc_args.clone(),
        };

        let encoded =
            serde_json::to_vec(&scope).context("Failed to serialize rustc wrapper scope")?;
        let mut hasher = Sha256::new();
        hasher.update(encoded);
        let scope_hash = format!("{:x}", hasher.finalize());
        Ok(format!(
            "{}-{}-{}-{}",
            self.tip_crate_name(),
            self.triple,
            self.profile,
            &scope_hash[..16]
        ))
    }
}

/// Reconstruct the dep-info `.d` path that rustc will write for an invocation, by parsing the
/// `--out-dir`, `--crate-name`, and `-C extra-filename=` from the captured args. This mirrors
/// rustc's own naming convention: `<out_dir>/<crate_name><extra_filename>.d`.
fn dep_info_path_for_rustc_args(args: &[String]) -> Option<PathBuf> {
    let mut out_dir: Option<&str> = None;
    let mut crate_name: Option<&str> = None;
    let mut extra: String = String::new();

    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "--out-dir" => {
                out_dir = args.get(i + 1).map(String::as_str);
                i += 2;
                continue;
            }
            "--crate-name" => {
                crate_name = args.get(i + 1).map(String::as_str);
                i += 2;
                continue;
            }
            "-C" => {
                if let Some(next) = args.get(i + 1) {
                    if let Some(val) = next.strip_prefix("extra-filename=") {
                        extra = val.to_string();
                    }
                }
                i += 2;
                continue;
            }
            _ => {}
        }

        if let Some(rest) = arg.strip_prefix("--out-dir=") {
            out_dir = Some(rest);
        } else if let Some(rest) = arg.strip_prefix("--crate-name=") {
            crate_name = Some(rest);
        } else if let Some(rest) = arg.strip_prefix("-Cextra-filename=") {
            extra = rest.to_string();
        }

        i += 1;
    }

    let out_dir = out_dir?;
    let crate_name = crate_name?;
    Some(PathBuf::from(out_dir).join(format!("{crate_name}{extra}.d")))
}

/// Filename suffixes that are pure `-Csave-temps=true` byproducts: rustc emits one per codegen unit
/// and nothing in the patch/link flow ever reads them back. `.rcgu.o` is included because the only
/// loose objects we link are the tip crate's *current* set, which is deleted right after the link in
/// [`BuildRequest::compile_workspace_hotpatch`] — any `.rcgu.o` lingering at sweep time is stale.
const THIN_BUILD_BYPRODUCT_SUFFIXES: &[&str] =
    &[".no-opt.bc", ".opt.bc", ".rcgu.bc", ".rcgu.o", ".ll", ".s"];

fn is_thin_build_byproduct(file_name: &str) -> bool {
    THIN_BUILD_BYPRODUCT_SUFFIXES
        .iter()
        .any(|suffix| file_name.ends_with(suffix))
}

/// Delete every [`is_thin_build_byproduct`] file directly inside `deps_dir`. Non-recursive on
/// purpose — rustc writes these alongside the `.rlib`/`.rmeta` outputs in the single `deps/` dir.
/// Returns `(files_removed, bytes_freed)`.
fn sweep_thin_build_byproducts(deps_dir: &Path) -> (usize, u64) {
    let Ok(entries) = std::fs::read_dir(deps_dir) else {
        return (0, 0);
    };

    let mut files_removed = 0;
    let mut bytes_freed = 0;
    for entry in entries.flatten() {
        if !is_thin_build_byproduct(&entry.file_name().to_string_lossy()) {
            continue;
        }

        let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
        if std::fs::remove_file(entry.path()).is_ok() {
            files_removed += 1;
            bytes_freed += len;
        }
    }

    (files_removed, bytes_freed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byproduct_predicate_matches_only_save_temps_files() {
        // Byproducts that should be swept.
        for name in [
            "app-1a2b.05stnb4bovskp7a00wyyf7l9s.rcgu.o",
            "app-1a2b.05stnb4bovskp7a00wyyf7l9s.rcgu.bc",
            "app-1a2b.05stnb4bovskp7a00wyyf7l9s.no-opt.bc",
            "app-1a2b.05stnb4bovskp7a00wyyf7l9s.opt.bc",
            "app-1a2b.ll",
            "app-1a2b.s",
        ] {
            assert!(is_thin_build_byproduct(name), "{name} should be swept");
        }

        // Artifacts cargo/dx still need must be preserved.
        for name in [
            "libapp-1a2b.rlib",
            "libapp-1a2b.rmeta",
            "app-1a2b.d",
            "libapp.so",
            "app.wasm",
            "stub.o",
        ] {
            assert!(!is_thin_build_byproduct(name), "{name} should be kept");
        }
    }

    #[test]
    fn sweep_removes_byproducts_and_keeps_real_outputs() {
        let dir = tempfile::tempdir().unwrap();
        let deps = dir.path();

        let byproducts = [
            "app-1a2b.cgu0.rcgu.o",
            "app-1a2b.cgu0.no-opt.bc",
            "app-1a2b.cgu1.opt.bc",
            "dep-9z8y.cgu0.rcgu.o",
        ];
        let keep = ["libapp-1a2b.rlib", "libapp-1a2b.rmeta", "app-1a2b.d"];

        for name in byproducts.iter().chain(keep.iter()) {
            std::fs::write(deps.join(name), b"x").unwrap();
        }

        let (removed, bytes) = sweep_thin_build_byproducts(deps);
        assert_eq!(removed, byproducts.len());
        assert_eq!(bytes, byproducts.len() as u64); // 1 byte each

        for name in byproducts {
            assert!(!deps.join(name).exists(), "{name} should be gone");
        }
        for name in keep {
            assert!(deps.join(name).exists(), "{name} should remain");
        }
    }

    #[test]
    fn sweep_missing_dir_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");
        assert_eq!(sweep_thin_build_byproducts(&missing), (0, 0));
    }
}
