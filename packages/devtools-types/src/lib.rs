use dioxus_core::internal::HotReloadTemplateWithLocation;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use subsecond_types::JumpTable;

/// A message the hot reloading server sends to the client
#[non_exhaustive]
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum DevserverMsg {
    /// Attempt a hotreload
    /// This includes all the templates/literals/assets/binary patches that have changed in one shot
    HotReload(HotReloadMsg),

    /// Starting a hotpatch
    HotPatchStart,

    /// The devserver is starting a full rebuild.
    FullReloadStart,

    /// The full reload failed.
    FullReloadFailed { errors: Vec<BuildError> },

    /// The app should reload completely if it can
    FullReloadCommand,

    /// The program is shutting down completely - maybe toss up a splash screen or something?
    Shutdown,
}

/// A message the client sends from the frontend to the devserver
///
/// This is used to communicate with the devserver
#[non_exhaustive]
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum ClientMsg {
    Log {
        level: String,
        messages: Vec<String>,
    },
    /// Ask the devserver to do a full rebuild — equivalent to pressing `r` in the TUI.
    /// Useful when a hot-patch leaves the running app in an unrecoverable state.
    FullRebuild,
    /// The client received a patch that builds on a patch that it did not apply. The
    /// devserver sends the patches of the session again, to this client only.
    MissedPatch,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct HotReloadMsg {
    pub templates: Vec<HotReloadTemplateWithLocation>,
    pub assets: Vec<PathBuf>,
    pub ms_elapsed: u64,
    pub jump_table: Option<JumpTable>,
    pub for_build_id: Option<u64>,
    pub for_pid: Option<u32>,
}

/// A build error with rendered compiler output and source locations.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BuildError {
    pub message: String,
    pub rendered: String,
    pub locations: Vec<SourceLocation>,
}

/// A source location associated with a build error.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SourceLocation {
    pub path: PathBuf,
    pub line: usize,
    pub column: usize,
}

impl HotReloadMsg {
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty() && self.assets.is_empty() && self.jump_table.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_reload_failed_serializes_errors_and_locations() {
        let message = DevserverMsg::FullReloadFailed {
            errors: vec![BuildError {
                message: "cannot find value `missing`".into(),
                rendered: "error[E0425]: cannot find value `missing`".into(),
                locations: vec![SourceLocation {
                    path: "/workspace/src/main.rs".into(),
                    line: 12,
                    column: 7,
                }],
            }],
        };

        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: DevserverMsg = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, message);
        assert!(serialized.contains(r#""line":12"#));
        assert!(serialized.contains(r#""column":7"#));
    }
}
