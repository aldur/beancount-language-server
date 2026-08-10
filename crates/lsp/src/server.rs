use crate::beancount_data::BeancountData;
use crate::checkers::BeancountChecker;
use crate::checkers::CheckerRegistry;
use crate::config::Config;
use crate::dispatcher::NotificationDispatcher;
use crate::dispatcher::RequestRouter;
use crate::document::Document;
use crate::document_store::DocumentStore;
use crate::document_store::DocumentStoreMaps;
use crate::forest;
use crate::progress::Progress;
use crate::providers::completion;
use crate::providers::definition;
use crate::providers::document_symbol;
use crate::providers::folding_range;
use crate::providers::formatting;
use crate::providers::hover;
use crate::providers::inlay_hints;
use crate::providers::references;
use crate::providers::semantic_tokens;
use crate::providers::text_document;
use crate::providers::workspace_symbol;
use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
use lsp_types::Notification;
use ropey::Rope;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tree_sitter_beancount::tree_sitter;

pub(crate) type RequestHandler = fn(&mut LspServerState, lsp_server::Response);
pub(crate) type ForestData = Box<
    Option<(
        PathBuf,
        Arc<tree_sitter::Tree>,
        Arc<BeancountData>,
        Arc<Rope>,
    )>,
>;

#[derive(Debug)]
pub(crate) enum ProgressMsg {
    BeanCheck {
        total: usize,
        done: usize,
        checker_name: String,
        // Unique id for a single bean-check run to avoid token collisions
        run_id: u64,
    },
    ForestInit {
        total: usize,
        done: usize,
        data: ForestData,
    },
}

#[derive(Debug)]
pub(crate) enum Task {
    Response(lsp_server::Response),
    Progress(ProgressMsg),
    /// A document parsed on the thread pool.
    Parsed {
        path: PathBuf,
        version: i32,
        tree: Option<Arc<tree_sitter::Tree>>,
    },
    /// A completed diagnostics run, to be diffed against what is displayed.
    Diagnostics(HashMap<PathBuf, Vec<lsp_types::Diagnostic>>),
    /// Semantic data rebuilt on the thread pool after an edit.
    SemanticData {
        path: PathBuf,
        version: i32,
        data: Arc<BeancountData>,
    },
}

#[derive(Debug)]
pub(crate) enum Event {
    Lsp(lsp_server::Message),
    Task(Task),
}

/*
struct LspServer {
    client: tower_lsp::Client,
    session: Session,
}
*/

pub(crate) struct LspServerState {
    /// Owns open_docs, parsers (private), forest, and beancount_data with coordinated updates.
    pub doc_store: DocumentStore,

    // the lsp server config options
    pub config: Config,

    // The request queue keeps track of all incoming and outgoing requests.
    pub req_queue: lsp_server::ReqQueue<(String, Instant), RequestHandler>,

    // Channel to send language server messages to the client
    pub sender: Sender<lsp_server::Message>,

    // True if the client requested that we shut down
    pub shutdown_requested: bool,

    // Channel to send tasks to from background operations
    pub task_sender: Sender<Task>,
    /// Monotone id for diagnostics runs, assigned on the main loop so it
    /// follows event order; see `published_diag_run`.
    pub next_diag_run: u64,
    /// Highest diagnostics run published so far. Runs execute on the thread
    /// pool and can finish out of order; a slower, earlier run must not
    /// overwrite a newer run's results with stale ones.
    pub published_diag_run: Arc<std::sync::atomic::AtomicU64>,

    // Channel to receive tasks on from background operations
    pub task_receiver: Receiver<Task>,

    // Thread pool for async execution
    pub thread_pool: threadpool::ThreadPool,

    /// Files with a semantic-data rebuild in flight. Rebuilding a large
    /// ledger takes a while, so fast typing must not queue one extraction per
    /// keystroke: at most one runs per file, and it re-checks the document
    /// version when it lands.
    pub extracting: std::collections::HashSet<PathBuf>,

    /// Files with a parse in flight, for the same reason as `extracting`.
    pub parsing: std::collections::HashSet<PathBuf>,

    /// Files that currently show diagnostics in the client, so a run can
    /// clear exactly those that no longer have any.
    pub published_diagnostics: std::collections::HashSet<PathBuf>,

    /// Requests the client has cancelled. Shared with the thread pool so a
    /// queued request can be dropped instead of computed.
    pub cancelled: Arc<std::sync::Mutex<std::collections::HashSet<lsp_server::RequestId>>>,

    // Cached checker instance (created once and reused)
    pub checker_registry: CheckerRegistry,

    // Request router with registered handlers
    pub request_router: Arc<RequestRouter>,
}

/// A snapshot of the state of the language server.
///
/// The three map fields are `Arc<HashMap<…>>` — cheap to clone and share across
/// thread-pool tasks.  Access them like plain `HashMap`s via `Deref`.
pub(crate) struct LspServerStateSnapshot {
    pub beancount_data: Arc<HashMap<PathBuf, Arc<BeancountData>>>,
    pub config: Config,
    pub forest: Arc<HashMap<PathBuf, Arc<tree_sitter::Tree>>>,
    /// Rope content for non-open forest files. Use `open_docs` first for open files.
    pub forest_content: Arc<HashMap<PathBuf, Arc<Rope>>>,
    pub open_docs: Arc<HashMap<PathBuf, Document>>,
    pub checker: Option<Arc<dyn BeancountChecker>>,
}

impl LspServerStateSnapshot {
    pub fn tree_and_document_for_uri(
        &self,
        uri: &lsp_types::Uri,
    ) -> Result<(&Arc<tree_sitter::Tree>, &Document)> {
        let path = uri
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Failed to convert URI to file path: {}", uri.as_str()))?;

        let tree = self
            .forest
            .get(&path)
            .with_context(|| format!("No parsed tree found for file: {}", path.display()))?;
        let doc = self
            .open_docs
            .get(&path)
            .with_context(|| format!("Document not found for file: {}", path.display()))?;
        Ok((tree, doc))
    }
}

/*
impl LspServer {
    /// Create a new [`Server`] instance.
    fn new(client: Client) -> Self {
        let session = Session::new(client.clone());
        Self { client, session }
    }
}
*/
impl LspServerState {
    pub fn new(sender: Sender<lsp_server::Message>, config: Config) -> Self {
        let (task_sender, task_receiver) = crossbeam_channel::unbounded();
        //let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let request_router = Arc::new(Self::build_request_router());
        Self {
            doc_store: DocumentStore::new(),
            config,
            req_queue: lsp_server::ReqQueue::default(),
            sender,
            shutdown_requested: false,
            task_sender,
            next_diag_run: 0,
            published_diag_run: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            task_receiver,
            thread_pool: threadpool::ThreadPool::default(),
            extracting: std::collections::HashSet::new(),
            parsing: std::collections::HashSet::new(),
            published_diagnostics: std::collections::HashSet::new(),
            cancelled: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
            checker_registry: CheckerRegistry::new(),
            request_router,
        }
    }

    pub fn run(&mut self, receiver: Receiver<lsp_server::Message>) -> Result<()> {
        tracing::info!("LSP server starting main event loop");

        // Register file watchers for .beancount files
        self.register_file_watchers();

        // Initialize checker once (can be slow); report progress to users.
        self.ensure_checker();

        // init forest
        if let Some(file) = self.config.journal_root.as_ref() {
            let journal_root = if file.is_relative() {
                self.config.root_dir.join(file)
            } else {
                file.clone()
            };

            // Check if exists
            if !journal_root.exists() {
                let error_msg = format!("Journal root does not exist: {}", journal_root.display());
                tracing::error!("{}", error_msg);

                // Send error message to client
                self.send_notification::<lsp_types::ShowMessageNotification>(
                    lsp_types::ShowMessageParams {
                        kind: lsp_types::MessageType::Error,
                        message: error_msg.clone(),
                    },
                );

                // Log warning and continue without forest initialization instead of returning error
                // This allows the language server to continue functioning for open documents
                tracing::warn!(
                    "Continuing without forest initialization due to invalid journal root"
                );
            } else {
                tracing::info!(
                    "Initializing forest for journal root: {}",
                    journal_root.display()
                );
                let snapshot = self.snapshot();
                let sender = self.task_sender.clone();
                self.thread_pool.execute(move || {
                    match forest::parse_initial_forest(snapshot, journal_root, sender) {
                        Ok(_) => tracing::info!("Forest initialization completed successfully"),
                        Err(e) => tracing::error!("Forest initialization failed: {}", e),
                    }
                });
            }
        } else {
            tracing::warn!("No journal_root configured, skipping forest initialization");
        }

        tracing::debug!("Entering main event loop");
        while let Some(event) = self.next_event(&receiver) {
            if let Event::Lsp(lsp_server::Message::Notification(notification)) = &event
                && notification.method == lsp_types::ExitNotification::METHOD.as_str()
            {
                tracing::info!("Received exit notification, shutting down");
                return Ok(());
            }
            self.handle_event(event)?;
        }
        tracing::info!("Main event loop completed");
        Ok(())
    }

    // Blocks until new event is received
    pub fn next_event(&self, receiver: &Receiver<lsp_server::Message>) -> Option<Event> {
        crossbeam_channel::select! {
            recv(receiver) -> msg => msg.ok().map(Event::Lsp),
            recv(self.task_receiver) -> task => task.ok().map(Event::Task),
        }
    }

    // handles an event
    fn handle_event(&mut self, event: Event) -> Result<()> {
        let start_time = Instant::now();

        match event {
            Event::Task(task) => {
                tracing::debug!("Handling task: {:?}", task);
                self.handle_task(task)?;
            }
            Event::Lsp(msg) => match msg {
                lsp_server::Message::Request(req) => {
                    tracing::debug!("Handling LSP request: method={}, id={}", req.method, req.id);
                    self.on_request(req, start_time)?;
                }
                lsp_server::Message::Response(resp) => {
                    tracing::debug!("Handling LSP response: id={}", resp.id);
                    self.complete_request(resp);
                }
                lsp_server::Message::Notification(notif) => {
                    tracing::debug!("Handling LSP notification: method={}", notif.method);
                    self.on_notification(notif)?;
                }
            },
        };

        let duration = start_time.elapsed();
        if duration.as_millis() > 100 {
            tracing::warn!("Event handling took longer than expected: {:?}", duration);
        }

        Ok(())
    }

    // Handles a task sent by another async task
    fn handle_task(&mut self, task: Task) -> anyhow::Result<()> {
        match task {
            Task::Response(response) => {
                tracing::debug!("Sending response for request: {}", response.id);
                self.respond(response);
            }
            Task::Progress(progress_task) => {
                tracing::debug!("Handling progress task: {:?}", progress_task);
                self.handle_progress_task(progress_task)?;
            }
            Task::Diagnostics(diags) => {
                text_document::publish_diagnostics(self, diags);
            }
            Task::Parsed {
                path,
                version,
                tree,
            } => {
                self.parsing.remove(&path);
                match tree {
                    Some(tree) => {
                        if self.doc_store.install_tree(&path, tree, version) {
                            // A tree without semantic data is only half useful.
                            self.schedule_extraction(&path);
                        }
                    }
                    None => tracing::warn!("Failed to parse {:?}", path),
                }
                // The document moved on while it was being parsed.
                if self
                    .doc_store
                    .parse_inputs(&path)
                    .is_some_and(|(_, current)| current != version)
                {
                    self.schedule_parse(&path);
                }
            }
            Task::SemanticData {
                path,
                version,
                data,
            } => {
                self.extracting.remove(&path);
                self.doc_store.install_beancount_data(&path, data, version);
                // The document moved on while this was being built: coalesce
                // all those edits into one fresh rebuild.
                if self
                    .doc_store
                    .extraction_inputs(&path)
                    .is_some_and(|(_, _, current)| current != version)
                {
                    self.schedule_extraction(&path);
                }
            }
        }
        Ok(())
    }

    fn handle_progress_task(&mut self, task: ProgressMsg) -> Result<()> {
        match task {
            ProgressMsg::BeanCheck {
                total,
                done,
                checker_name,
                run_id,
            } => {
                let progress_state = if done == 0 {
                    Progress::Begin
                } else if done < total {
                    Progress::Report
                } else {
                    Progress::End
                };
                // Use a per-run unique token suffix to prevent collisions when
                // multiple diagnostics tasks overlap.
                self.report_progress(
                    &format!("bean check ({})", checker_name),
                    progress_state,
                    Some(format!("{done}/{total}")),
                    Some(Progress::fraction(done, total)),
                    Some(&format!("/{}", run_id)),
                )
            }
            ProgressMsg::ForestInit { total, done, data } => {
                if let Some((path, tree, beancount_data, rope)) = *data {
                    // The forest task parsed this file from disk, possibly long
                    // ago. If the user opened (and maybe edited) it meanwhile,
                    // the open buffer is the truth: overwriting its forest tree
                    // with the disk parse pairs a stale tree with the live rope,
                    // and every byte range in it is a latent panic.
                    if self.doc_store.has_open_doc(&path) {
                        tracing::debug!(
                            "Forest init: {:?} is open in the editor, keeping the buffer parse",
                            path
                        );
                    } else {
                        self.doc_store
                            .insert_tree_and_data(path, tree, beancount_data, rope);
                    }
                }
                let progress_state = if done == 0 {
                    Progress::Begin
                } else if done < total {
                    Progress::Report
                } else {
                    Progress::End
                };
                self.report_progress(
                    "generating forest",
                    progress_state,
                    Some(format!("{done}/{total}")),
                    Some(Progress::fraction(done, total)),
                    None,
                )
            }
        }
        Ok(())
    }

    // Registers a request with the server. We register all these request to make
    // sure they all get handled and so we can measure the time it takes for them
    // to complete from the point of view of the client.
    fn register_request(&mut self, request: &lsp_server::Request, start_time: Instant) {
        self.req_queue
            .incoming
            .register(request.id.clone(), (request.method.clone(), start_time))
    }

    // Handles a language server protocol request
    fn on_request(&mut self, req: lsp_server::Request, start_time: Instant) -> Result<()> {
        self.register_request(&req, start_time);
        if self.shutdown_requested {
            tracing::warn!("Request {} received after shutdown was requested", req.id);
            self.respond(lsp_server::Response::new_err(
                req.id,
                lsp_server::ErrorCode::InvalidRequest as i32,
                "shutdown was requested".to_string(),
            ));
            return Ok(());
        }

        tracing::debug!("Processing request: method={}, id={}", req.method, req.id);

        self.request_router.clone().dispatch(self, req);

        Ok(())
    }

    // Handles a response to a request we made. The response gets forwarded to where we made the request from.
    fn complete_request(&mut self, resp: lsp_server::Response) {
        // A duplicate or spurious response id must not panic the main loop.
        match self.req_queue.outgoing.complete(resp.id.clone()) {
            Some(handler) => handler(self, resp),
            None => tracing::error!("received response for unknown request: {:?}", resp.id),
        }
    }

    // Handles a notification from the language server client
    fn on_notification(&mut self, notif: lsp_server::Notification) -> Result<()> {
        NotificationDispatcher::new(self, notif)
            .on::<lsp_types::DidOpenTextDocumentNotification>(text_document::did_open)?
            .on::<lsp_types::DidCloseTextDocumentNotification>(text_document::did_close)?
            .on::<lsp_types::DidSaveTextDocumentNotification>(text_document::did_save)?
            .on::<lsp_types::DidChangeTextDocumentNotification>(text_document::did_change)?
            .on::<lsp_types::DidChangeWatchedFilesNotification>(
                text_document::did_change_watched_files,
            )?
            .on::<lsp_types::CancelNotification>(Self::cancel_request)?
            .finish();
        Ok(())
    }

    /// Note a cancellation so a queued request can be skipped.
    ///
    /// The request still gets a response — the protocol requires one — but it
    /// is an error rather than the result of work nobody is waiting for.
    fn cancel_request(&mut self, params: lsp_types::CancelParams) -> Result<()> {
        let id: lsp_server::RequestId = match params.id {
            lsp_types::Id::Int(n) => n.into(),
            lsp_types::Id::String(s) => s.into(),
        };
        tracing::debug!("cancelling request {id}");
        if let Ok(mut cancelled) = self.cancelled.lock() {
            cancelled.insert(id);
        }
        Ok(())
    }

    // Sends a response to the client. This method logs the time it took us to reply to a request from the client.
    pub(crate) fn respond(&mut self, response: lsp_server::Response) {
        if let Ok(mut cancelled) = self.cancelled.lock() {
            cancelled.remove(&response.id);
        }
        if let Some((method, start)) = self.req_queue.incoming.complete(&response.id) {
            let duration = start.elapsed();
            let is_error = response.error.is_some();

            if is_error {
                tracing::warn!(
                    "Request {} ({}) completed with error in {:?}: {:?}",
                    response.id,
                    method,
                    duration,
                    response.error
                );
            } else {
                tracing::trace!(
                    "Request {} ({}) completed successfully in {:?}",
                    response.id,
                    method,
                    duration
                );
            }

            if duration.as_millis() > 1000 {
                tracing::warn!("Slow request detected: {} took {:?}", method, duration);
            }

            self.send(response.into());
        } else {
            tracing::warn!("Received response for unknown request: {}", response.id);
        }
    }

    /// Sends a message to the client
    pub(crate) fn send(&mut self, message: lsp_server::Message) {
        match &message {
            lsp_server::Message::Request(req) => {
                tracing::debug!(
                    "Sending request to client: method={}, id={}",
                    req.method,
                    req.id
                );
            }
            lsp_server::Message::Response(resp) => {
                tracing::debug!(
                    "Sending response to client: id={}, has_error={}",
                    resp.id,
                    resp.error.is_some()
                );
            }
            lsp_server::Message::Notification(notif) => {
                tracing::debug!("Sending notification to client: method={}", notif.method);
            }
        }

        if let Err(e) = self.sender.send(message) {
            tracing::error!("Failed to send LSP message to client: {}", e);
        }
    }

    // Sends a request to the client and registers the request so that we can handle the response.
    pub(crate) fn send_request<R: lsp_types::Request>(
        &mut self,
        params: R::Params,
        handler: RequestHandler,
    ) {
        let request = self
            .req_queue
            .outgoing
            .register(R::METHOD.to_string(), params, handler);
        self.send(request.into());
    }

    // Sends a notification to the client
    pub(crate) fn send_notification<N: lsp_types::Notification>(&mut self, params: N::Params) {
        let not = lsp_server::Notification::new(N::METHOD.to_string(), params);
        self.send(not.into());
    }

    /// Rebuild a file's semantic data on the thread pool, at most one at a
    /// time per file.
    /// Parse a document on the thread pool, at most one parse per file.
    ///
    /// A full parse of a multi-megabyte ledger takes seconds; doing it in the
    /// notification handler blocks every buffer and every request.
    pub(crate) fn schedule_parse(&mut self, uri: &PathBuf) {
        if self.parsing.contains(uri) {
            return;
        }
        let Some((text, version)) = self.doc_store.parse_inputs(uri) else {
            return;
        };
        self.parsing.insert(uri.clone());
        let sender = self.task_sender.clone();
        let path = uri.clone();
        self.thread_pool.execute(move || {
            // The depth check belongs here, next to the parse: a tree deeper
            // than the walkers will follow is worse than no tree, because
            // every query over it is unbounded work.
            let tree = crate::treesitter_utils::parse_beancount(&text)
                .filter(|tree| {
                    if crate::treesitter_utils::tree_depth_exceeds(
                        tree,
                        crate::treesitter_utils::MAX_TREE_DEPTH,
                    ) {
                        tracing::warn!("Parse tree for {path:?} is pathologically deep");
                        false
                    } else {
                        true
                    }
                })
                .map(Arc::new);
            if let Err(e) = sender.send(Task::Parsed {
                path,
                version,
                tree,
            }) {
                tracing::debug!("Failed to deliver parse result: {e}");
            }
        });
    }

    /// Schedule rebuilds for every forest file that has no semantic data yet.
    pub(crate) fn schedule_missing_extractions(&mut self) {
        for path in self.doc_store.files_missing_data() {
            self.schedule_extraction(&path);
        }
    }

    pub(crate) fn schedule_extraction(&mut self, uri: &PathBuf) {
        if self.extracting.contains(uri) {
            return;
        }
        let Some((tree, rope, version)) = self.doc_store.extraction_inputs(uri) else {
            return;
        };
        self.extracting.insert(uri.clone());
        let sender = self.task_sender.clone();
        let path = uri.clone();
        self.thread_pool.execute(move || {
            let data = Arc::new(BeancountData::new(&tree, &rope));
            if let Err(e) = sender.send(Task::SemanticData {
                path,
                version,
                data,
            }) {
                tracing::debug!("Failed to deliver semantic data: {e}");
            }
        });
    }

    pub(crate) fn snapshot(&mut self) -> LspServerStateSnapshot {
        // Deliberately no extraction here: this runs on the main loop for
        // every request, and rebuilding semantic data is unbounded work.
        // Whoever changes a document schedules the rebuild instead.
        let DocumentStoreMaps {
            open_docs,
            forest,
            beancount_data,
            forest_content,
        } = self.doc_store.snapshot_maps();
        LspServerStateSnapshot {
            beancount_data,
            config: self.config.clone(),
            forest,
            forest_content,
            open_docs,
            checker: self.checker_registry.get(),
        }
    }

    fn build_request_router() -> RequestRouter {
        let mut router = RequestRouter::new();
        router
            .on_sync::<lsp_types::ShutdownRequest>(|state, _request| {
                tracing::info!("Received shutdown request");
                state.shutdown_requested = true;
                Ok(())
            })
            .expect("Failed to register Shutdown handler")
            .on::<lsp_types::HoverRequest>(hover::hover)
            .expect("Failed to register Hover handler")
            .on::<lsp_types::CompletionRequest>(completion::completion)
            .expect("Failed to register Completion handler")
            .on::<lsp_types::DocumentFormattingRequest>(formatting::formatting)
            .expect("Failed to register Formatting handler")
            .on::<lsp_types::RenameRequest>(references::rename)
            .expect("Failed to register Rename handler")
            .on::<lsp_types::ReferencesRequest>(references::references)
            .expect("Failed to register References handler")
            .on::<lsp_types::DefinitionRequest>(definition::definition)
            .expect("Failed to register GotoDefinition handler")
            .on::<lsp_types::SemanticTokensRequest>(semantic_tokens::semantic_tokens_full)
            .expect("Failed to register SemanticTokens handler")
            .on::<lsp_types::InlayHintRequest>(inlay_hints::inlay_hints)
            .expect("Failed to register InlayHint handler")
            .on::<lsp_types::FoldingRangeRequest>(folding_range::folding_ranges)
            .expect("Failed to register FoldingRange handler")
            .on::<lsp_types::DocumentSymbolRequest>(document_symbol::document_symbols)
            .expect("Failed to register DocumentSymbol handler")
            .on::<lsp_types::WorkspaceSymbolRequest>(workspace_symbol::workspace_symbols)
            .expect("Failed to register WorkspaceSymbol handler");

        router
    }

    /// Register file watchers with the client to detect external file changes.
    /// This enables real-time detection of files modified outside the editor.
    fn register_file_watchers(&mut self) {
        use lsp_types::{
            DidChangeWatchedFilesRegistrationOptions, FileSystemWatcher, GlobPattern, Registration,
            WatchKind,
        };

        let watch_kind = WatchKind::Create | WatchKind::Change | WatchKind::Delete;

        // `include` directives are extension-agnostic, so the watcher must
        // cover more than the two canonical extensions; `.include` is a
        // common convention for ledger fragments. Files pulled in under yet
        // other extensions still end up in the forest but go stale on
        // external changes.
        let watchers = vec![FileSystemWatcher {
            glob_pattern: GlobPattern::Pattern("**/*.{bean,beancount,include}".to_string()),
            kind: Some(watch_kind),
        }];

        let registration_options = DidChangeWatchedFilesRegistrationOptions { watchers };

        let registration = Registration {
            id: "beancount-file-watcher".to_string(),
            method: "workspace/didChangeWatchedFiles".to_string(),
            register_options: Some(
                serde_json::to_value(registration_options)
                    .expect("Failed to serialize file watcher options"),
            ),
        };

        let params = lsp_types::RegistrationParams {
            registrations: vec![registration],
        };

        // Send registration request to client (fire-and-forget, we don't need the response)
        self.send_request::<lsp_types::RegistrationRequest>(params, |_state, response| {
            if let Some(error) = response.error {
                tracing::warn!(
                    "Failed to register file watchers: {} (code: {})",
                    error.message,
                    error.code
                );
            } else {
                tracing::info!("File watchers registered successfully for *.beancount files");
            }
        });
    }

    fn ensure_checker(&mut self) -> Option<Arc<dyn BeancountChecker>> {
        if let Some(checker) = self.checker_registry.get() {
            return Some(checker);
        }

        self.report_progress(
            "checker auto",
            Progress::Begin,
            Some("discovering available checkers".to_string()),
            None,
            None,
        );

        let checker = self
            .checker_registry
            .get_or_init(&self.config.bean_check, &self.config.root_dir);

        if let Some(ref c) = checker {
            self.report_progress(
                "checker auto",
                Progress::End,
                Some(format!("using {}", c.name())),
                None,
                None,
            );
        } else {
            self.report_progress(
                "checker auto",
                Progress::End,
                Some("no checker available".to_string()),
                None,
                None,
            );
        }

        checker
    }
}
