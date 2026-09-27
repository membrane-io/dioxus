use crate::{BuildId, BuilderUpdate, BundleFormat, Error, TraceMsg};
use axum::extract::ws::Message as WsMessage;
use std::path::PathBuf;

/// One fat enum to rule them all....
///
/// Thanks to libraries like winit for the inspiration
#[allow(clippy::large_enum_variant)]
pub(crate) enum ServeUpdate {
    NewConnection {
        id: BuildId,
        aslr_reference: Option<u64>,
        pid: Option<u32>,
    },
    WsMessage {
        bundle: BundleFormat,
        msg: WsMessage,
        /// The index of the socket that sent the message, for a reply with
        /// `WebServer::send_hotreload_to`. The index is valid until the next call of
        /// `WebServer::wait`.
        socket: usize,
    },

    /// An update regarding the state of the build and running app from an AppBuilder
    BuilderUpdate {
        id: BuildId,
        update: BuilderUpdate,
    },

    FilesChanged {
        files: Vec<PathBuf>,
    },

    OpenApp,

    RequestRebuild,

    CycleHotreloadMode,

    ToggleSkipDependents,

    OpenDebugger {
        id: BuildId,
    },

    Redraw,

    TracingLog {
        log: TraceMsg,
    },

    Exit {
        error: Option<Error>,
    },
}
