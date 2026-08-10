use crate::beancount_data::BeancountData;
use crate::document::Document;
use crate::treesitter_utils::{
    MAX_TREE_DEPTH, lsp_textdocchange_to_ts_inputedit, parse_with_budget, tree_depth_exceeds,
};
use anyhow::Result;
use ropey::Rope;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tree_sitter_beancount::tree_sitter;

/// Version stand-in for files that are in the forest but not open in the
/// editor, and therefore have no document version.
pub(crate) const NO_VERSION: i32 = i32::MIN;

/// Arc-wrapped views of the public maps, used to construct `LspServerStateSnapshot`.
///
/// Each field is an `Arc<HashMap<…>>` so that taking a snapshot is a cheap pointer
/// clone; the underlying HashMap is only copied (via [`Arc::make_mut`]) when the
/// `DocumentStore` actually modifies it.
pub(crate) struct DocumentStoreMaps {
    pub open_docs: Arc<HashMap<PathBuf, Document>>,
    pub forest: Arc<HashMap<PathBuf, Arc<tree_sitter::Tree>>>,
    pub beancount_data: Arc<HashMap<PathBuf, Arc<BeancountData>>>,
    /// Rope content for non-open forest files. Open files use `open_docs` as the
    /// source of truth; `forest_content` holds their rope only after they are closed.
    pub forest_content: Arc<HashMap<PathBuf, Arc<Rope>>>,
}

/// Owns all document-related maps and enforces their consistency invariants:
/// - Every open document has a parser (private, hidden from callers).
/// - When a tree is invalidated, its `beancount_data` is removed atomically.
/// - `beancount_data` is extracted lazily via `ensure_beancount_data`.
/// - `forest_content` holds rope content for non-open forest files.
///   Open files use `open_docs` as the source of truth; `forest_content` is
///   populated for a file when it is inserted via `insert_parsed` /
///   `insert_tree_and_data`, and the open_docs entry takes precedence while
///   the file is open.
///
/// All public maps are stored as `Arc<HashMap<…>>` so that
/// [`snapshot_maps`][DocumentStore::snapshot_maps] is an O(1) pointer clone.
/// [`Arc::make_mut`] is used before every mutation to ensure copy-on-write
/// semantics: if a snapshot is currently live the HashMap is cloned once, then
/// mutated; otherwise the existing allocation is reused.
pub(crate) struct DocumentStore {
    open_docs: Arc<HashMap<PathBuf, Document>>,
    /// Stateful parsers for incremental re-parsing. Private: callers never need a parser directly.
    parsers: HashMap<PathBuf, tree_sitter::Parser>,
    forest: Arc<HashMap<PathBuf, Arc<tree_sitter::Tree>>>,
    beancount_data: Arc<HashMap<PathBuf, Arc<BeancountData>>>,
    /// Rope content for non-open forest files (open files use `open_docs`).
    forest_content: Arc<HashMap<PathBuf, Arc<Rope>>>,
}

impl DocumentStore {
    pub(crate) fn new() -> Self {
        Self {
            open_docs: Arc::new(HashMap::new()),
            parsers: HashMap::new(),
            forest: Arc::new(HashMap::new()),
            beancount_data: Arc::new(HashMap::new()),
            forest_content: Arc::new(HashMap::new()),
        }
    }

    /// Open a document: insert the doc buffer, initialise (or reuse) a parser, do a
    /// fresh parse, and eagerly extract `BeancountData`.
    ///
    /// Always parses fresh — the file may have been modified externally between
    /// close and reopen, so cached trees cannot be trusted.
    pub(crate) fn open(&mut self, uri: PathBuf, text: &str, version: i32) {
        let content = ropey::Rope::from_str(text);
        Arc::make_mut(&mut self.open_docs).insert(uri.clone(), Document { content, version });

        self.parsers.entry(uri.clone()).or_insert_with(|| {
            let mut parser = tree_sitter::Parser::new();
            parser
                .set_language(&tree_sitter_beancount::language())
                .expect("Failed to set language for tree-sitter parser");
            parser
        });

        // Neither the parse nor the extraction happens here: a full parse of a
        // multi-megabyte ledger takes seconds, and this runs in the didOpen
        // handler on the main loop. The caller schedules both, and anything
        // that needs a tree declines until it lands.
        Arc::make_mut(&mut self.forest).remove(&uri);
        Arc::make_mut(&mut self.beancount_data).remove(&uri);
        Arc::make_mut(&mut self.forest_content).remove(&uri);
    }

    /// Install a tree parsed off-thread, unless the document moved on.
    ///
    /// Returns true when the tree was installed.
    pub(crate) fn install_tree(
        &mut self,
        uri: &PathBuf,
        tree: Arc<tree_sitter::Tree>,
        version: i32,
    ) -> bool {
        let current = match self.open_docs.get(uri) {
            Some(doc) => doc.version,
            None => return false,
        };
        if current != version {
            tracing::trace!(
                "Dropping tree for {:?}: parsed at v{version}, document at v{current}",
                uri
            );
            return false;
        }
        Arc::make_mut(&mut self.forest).insert(uri.clone(), tree);
        true
    }

    /// The text and version to parse off-thread.
    pub(crate) fn parse_inputs(&self, uri: &PathBuf) -> Option<(String, i32)> {
        let doc = self.open_docs.get(uri)?;
        Some((doc.text_string(), doc.version))
    }

    /// Apply incremental content changes to an open document.
    ///
    /// Updates the rope, does an incremental tree-sitter re-parse, and lazily
    /// invalidates `beancount_data` (removed so it is re-extracted on next demand).
    pub(crate) fn apply_change(
        &mut self,
        uri: &PathBuf,
        changes: &[lsp_types::TextDocumentContentChangeEvent],
        new_version: i32,
    ) -> Result<()> {
        // Steps 1+2 — convert each change to a tree-sitter edit and apply it to
        // the rope, one change at a time. Each change's positions refer to the
        // document as left by the *previous* change (LSP sync semantics), so
        // converting the whole batch up front against the pre-edit rope reads
        // lines that do not exist yet: the edits desync the tree, and a change
        // that lands on a line created earlier in the batch is out of bounds
        // for the old rope and panics.
        let mut ts_edits = Vec::with_capacity(changes.len());
        {
            let doc = match Arc::make_mut(&mut self.open_docs).get_mut(uri) {
                Some(d) => d,
                None => {
                    tracing::warn!("Document not found in open_docs: {:?}", uri);
                    return Ok(());
                }
            };

            let current_version = doc.version;
            if new_version <= current_version {
                tracing::warn!(
                    "Received out-of-order or duplicate change: current version={}, received version={}",
                    current_version,
                    new_version
                );
            }
            tracing::trace!("Document version: {} -> {}", current_version, new_version);

            for change in changes {
                ts_edits.push(lsp_textdocchange_to_ts_inputedit(&doc.content, change)?);
                let (text, range) = match change {
                    lsp_types::TextDocumentContentChangeEvent::TextDocumentContentChangePartial(c) => {
                        (c.text.as_str(), c.range)
                    }
                    lsp_types::TextDocumentContentChangeEvent::TextDocumentContentChangeWholeDocument(c) => {
                        let end_line = (doc.content.len_lines().saturating_sub(1)) as u32;
                        let end_line_len = if doc.content.len_lines() > 0 {
                            let last_line = doc.content.line(end_line as usize);
                            last_line.len_chars().saturating_sub(1) as u32
                        } else {
                            0
                        };
                        let r = lsp_types::Range {
                            start: lsp_types::Position::new(0, 0),
                            end: lsp_types::Position::new(end_line, end_line_len),
                        };
                        (c.text.as_str(), r)
                    }
                };

                // Clamp line indices like the UTF-16 columns below: a position
                // beyond the document must degrade to the document end, not
                // panic the main loop (`line_to_char` accepts at most len_lines).
                let start_row_char_idx = doc
                    .content
                    .line_to_char((range.start.line as usize).min(doc.content.len_lines()));
                let end_row_char_idx = doc
                    .content
                    .line_to_char((range.end.line as usize).min(doc.content.len_lines()));

                let start_line_utf16_cu = doc.content.char_to_utf16_cu(start_row_char_idx);
                let start_utf16_idx = (start_line_utf16_cu + range.start.character as usize)
                    .min(doc.content.len_utf16_cu());
                let start_col_char_idx =
                    doc.content.utf16_cu_to_char(start_utf16_idx) - start_row_char_idx;

                let end_line_utf16_cu = doc.content.char_to_utf16_cu(end_row_char_idx);
                let end_utf16_idx = (end_line_utf16_cu + range.end.character as usize)
                    .min(doc.content.len_utf16_cu());
                let end_col_char_idx =
                    doc.content.utf16_cu_to_char(end_utf16_idx) - end_row_char_idx;

                let start_char_idx = start_row_char_idx + start_col_char_idx;
                // A protocol-invalid inverted range must not panic `remove`.
                let end_char_idx = (end_row_char_idx + end_col_char_idx).max(start_char_idx);

                tracing::trace!(
                    "Applying change: range={}:{}-{}:{}, char_idx={}-{}, text_len={}",
                    range.start.line,
                    range.start.character,
                    range.end.line,
                    range.end.character,
                    start_char_idx,
                    end_char_idx,
                    text.len()
                );

                doc.content.remove(start_char_idx..end_char_idx);
                if !text.is_empty() {
                    doc.content.insert(start_char_idx, text);
                }
            }

            doc.version = new_version;
            // doc borrow released
        }

        // Step 3 — clone the old tree (applying ts_edits) and snapshot the text.
        // Both borrows are released before step 4 mutates `parsers`.
        //
        // No tree yet (the initial parse is still in flight, or an earlier one
        // failed): the rope is updated, and `did_change` schedules a parse.
        // Never parse from scratch here — that is main-loop work.
        if !self.forest.contains_key(uri) {
            return Ok(());
        }
        let (edited_old_tree, text_str) = {
            let old_tree_arc = match self.forest.get(uri) {
                Some(t) => t,
                None => {
                    tracing::warn!("Tree not found in forest: {:?}", uri);
                    return Ok(());
                }
            };
            let mut old_tree = (**old_tree_arc).clone();
            for edit in &ts_edits {
                old_tree.edit(edit);
            }
            let text_str = self
                .open_docs
                .get(uri)
                .expect("doc should exist")
                .text_string();
            (old_tree, text_str)
            // forest and open_docs borrows released
        };

        // Step 4 — incremental parse (only mutates `parsers`).
        let new_tree = {
            let parser = match self.parsers.get_mut(uri) {
                Some(p) => p,
                None => {
                    tracing::warn!("Parser not found for document: {:?}", uri);
                    return Ok(());
                }
            };
            parse_with_budget(parser, &text_str, Some(&edited_old_tree))
        };

        // Step 5 — commit new tree, lazily invalidate beancount_data.
        if let Some(tree) = new_tree.filter(|t| {
            if tree_depth_exceeds(t, MAX_TREE_DEPTH) {
                tracing::warn!("Edited tree for {:?} is pathologically deep, dropping it", uri);
                false
            } else {
                true
            }
        }) {
            *Arc::make_mut(&mut self.forest)
                .get_mut(uri)
                .expect("tree should exist in forest") = Arc::new(tree);
            // The previous semantic data is kept deliberately: re-extracting
            // it here (or lazily on the next request) costs a full pass over
            // the document on the main loop, which for a large ledger is
            // hundreds of milliseconds per keystroke. `did_change` schedules
            // the rebuild on the thread pool instead, and completions use
            // data that is at most one keystroke stale.
        } else {
            // Keep rope and tree from disagreeing: without a usable tree the
            // file leaves the forest until a later edit parses cleanly.
            Arc::make_mut(&mut self.forest).remove(uri);
            Arc::make_mut(&mut self.beancount_data).remove(uri);
        }

        Ok(())
    }

    /// Close a document: transition it from open to a non-open forest file.
    ///
    /// Removes the buffer from `open_docs` but keeps the tree, beancount_data,
    /// and parser so the file remains part of the forest for cross-file operations
    /// (references, diagnostics). The rope is transferred to `forest_content` so
    /// providers can still access the file's text without falling back to disk.
    pub(crate) fn close(&mut self, uri: &PathBuf) {
        // Deliberately no extraction here: building semantic data for a 5.8MB
        // buffer costs a quarter of a second on the main loop, and closing a
        // file is not the moment to spend it. Whatever data exists stays (it is
        // at most one keystroke stale) and the caller schedules a rebuild from
        // the rope below.

        // Transfer rope to forest_content so it stays available after close.
        if let Some(doc) = self.open_docs.get(uri) {
            Arc::make_mut(&mut self.forest_content)
                .insert(uri.clone(), Arc::new(doc.content.clone()));
        }

        Arc::make_mut(&mut self.open_docs).remove(uri);
        // forest, beancount_data, and parsers intentionally kept for non-open file tracking
    }

    /// Insert a freshly parsed external file (includes, watched-file reloads).
    ///
    /// Wraps the tree in `Arc`, creates `BeancountData`, stores both, and stores
    /// the rope in `forest_content` for text-level access by providers.
    /// Does not touch `open_docs` or `parsers`.
    pub(crate) fn insert_parsed(&mut self, uri: PathBuf, tree: tree_sitter::Tree, content: &str) {
        let tree_arc = Arc::new(tree);
        let rope = Rope::from_str(content);
        Arc::make_mut(&mut self.forest).insert(uri.clone(), tree_arc);
        // Extraction is the caller's job, off the main loop (see `open`).
        Arc::make_mut(&mut self.beancount_data).remove(&uri);
        Arc::make_mut(&mut self.forest_content).insert(uri, Arc::new(rope));
    }

    /// Insert pre-computed `Arc`-wrapped tree, data, and rope (used by the ForestInit background task).
    pub(crate) fn insert_tree_and_data(
        &mut self,
        uri: PathBuf,
        tree: Arc<tree_sitter::Tree>,
        data: Arc<BeancountData>,
        rope: Arc<Rope>,
    ) {
        Arc::make_mut(&mut self.forest).insert(uri.clone(), tree);
        Arc::make_mut(&mut self.beancount_data).insert(uri.clone(), data);
        Arc::make_mut(&mut self.forest_content).insert(uri, rope);
    }

    /// Remove all caches for an externally deleted file.
    pub(crate) fn remove_external(&mut self, uri: &PathBuf) {
        Arc::make_mut(&mut self.forest).remove(uri);
        Arc::make_mut(&mut self.beancount_data).remove(uri);
        Arc::make_mut(&mut self.forest_content).remove(uri);
        self.parsers.remove(uri);
    }

    /// Clear stale caches for an externally changed file before re-parsing.
    pub(crate) fn invalidate_external(&mut self, uri: &PathBuf) {
        Arc::make_mut(&mut self.forest).remove(uri);
        Arc::make_mut(&mut self.beancount_data).remove(uri);
        Arc::make_mut(&mut self.forest_content).remove(uri);
        // open_docs and parsers untouched — file is not open in the editor
    }

    /// Drop forest entries that are neither open nor reachable from `root`
    /// via include directives. Returns the pruned paths.
    ///
    /// The include graph changes with every edit of a journal; without
    /// pruning the forest is add-only, and a file whose include line was
    /// removed keeps feeding completions, references and diagnostics
    /// forever.
    pub(crate) fn retain_reachable(&mut self, root: &std::path::Path) -> Vec<PathBuf> {
        let mut keep: std::collections::HashSet<PathBuf> =
            self.open_docs.keys().cloned().collect();
        keep.insert(root.to_path_buf());
        let mut queue: Vec<PathBuf> = keep.iter().cloned().collect();
        while let Some(path) = queue.pop() {
            if !self.forest.contains_key(&path) {
                continue;
            }
            let Some(text) = self.get_content(&path) else {
                continue;
            };
            for include in crate::forest::extract_include_paths(&text, &path) {
                if keep.insert(include.clone()) {
                    queue.push(include);
                }
            }
        }

        let pruned: Vec<PathBuf> = self
            .forest
            .keys()
            .filter(|path| !keep.contains(*path))
            .cloned()
            .collect();
        for path in &pruned {
            Arc::make_mut(&mut self.forest).remove(path);
            Arc::make_mut(&mut self.beancount_data).remove(path);
            Arc::make_mut(&mut self.forest_content).remove(path);
            self.parsers.remove(path);
        }
        pruned
    }

    /// The tree, rope and version needed to rebuild semantic data off-thread.
    pub(crate) fn extraction_inputs(
        &self,
        uri: &PathBuf,
    ) -> Option<(Arc<tree_sitter::Tree>, Arc<Rope>, i32)> {
        let tree = self.forest.get(uri)?.clone();
        match self.open_docs.get(uri) {
            Some(doc) => Some((tree, Arc::new(doc.content.clone()), doc.version)),
            // Not open: the cached rope is the source of truth and carries no
            // version, so `install_beancount_data` accepts it unconditionally.
            None => Some((tree, self.forest_content.get(uri)?.clone(), NO_VERSION)),
        }
    }

    /// Files that are in the forest but have no semantic data yet.
    pub(crate) fn files_missing_data(&self) -> Vec<PathBuf> {
        self.forest
            .keys()
            .filter(|path| !self.beancount_data.contains_key(*path))
            .cloned()
            .collect()
    }

    /// Install semantic data computed off-thread, unless the document moved on.
    pub(crate) fn install_beancount_data(
        &mut self,
        uri: &PathBuf,
        data: Arc<BeancountData>,
        version: i32,
    ) {
        // Not-open files have no version to compare; accept while they are
        // still part of the forest.
        if version == NO_VERSION {
            if self.forest.contains_key(uri) {
                Arc::make_mut(&mut self.beancount_data).insert(uri.clone(), data);
            }
            return;
        }
        match self.open_docs.get(uri) {
            Some(doc) if doc.version == version => {
                Arc::make_mut(&mut self.beancount_data).insert(uri.clone(), data);
            }
            Some(doc) => tracing::trace!(
                "Dropping semantic data for {:?}: built at v{}, document at v{}",
                uri,
                version,
                doc.version
            ),
            None => {}
        }
    }

    /// Lazily extract `BeancountData` for `uri` if it is absent.
    ///
    /// Called before requests that need semantic data (completion, hover, …).
    /// `beancount_data` is absent after every `apply_change` to avoid blocking
    /// per-keystroke parsing; it is (re-)created here on the first read after
    /// each edit.
    pub(crate) fn ensure_beancount_data(&mut self, uri: &PathBuf) {
        if self.beancount_data.contains_key(uri) {
            return;
        }
        if let (Some(tree), Some(doc)) = (self.forest.get(uri), self.open_docs.get(uri)) {
            let beancount_data = BeancountData::new(tree, &doc.content);
            Arc::make_mut(&mut self.beancount_data).insert(uri.clone(), Arc::new(beancount_data));
            tracing::debug!("Lazy extraction: BeancountData extracted for {:?}", uri);
        }
    }

    // ── Readers ──────────────────────────────────────────────────────────────

    pub(crate) fn get_tree(&self, uri: &PathBuf) -> Option<&Arc<tree_sitter::Tree>> {
        self.forest.get(uri)
    }

    /// The text matching this file's forest tree: the open buffer if there is
    /// one, else the cached rope from the last (re-)parse.
    pub(crate) fn get_content(&self, uri: &PathBuf) -> Option<String> {
        self.open_docs
            .get(uri)
            .map(|d| d.text_string())
            .or_else(|| self.forest_content.get(uri).map(|r| r.to_string()))
    }

    pub(crate) fn has_open_doc(&self, uri: &PathBuf) -> bool {
        self.open_docs.contains_key(uri)
    }

    pub(crate) fn open_doc_keys(&self) -> impl Iterator<Item = &PathBuf> {
        self.open_docs.keys()
    }


    // ── Snapshot ─────────────────────────────────────────────────────────────

    /// Clone the three public map Arcs for constructing `LspServerStateSnapshot`.
    ///
    /// This is an O(1) operation: only the Arc reference counts are incremented.
    /// The underlying HashMaps are not copied unless the store subsequently
    /// mutates them (copy-on-write via [`Arc::make_mut`]).
    pub(crate) fn snapshot_maps(&self) -> DocumentStoreMaps {
        DocumentStoreMaps {
            open_docs: Arc::clone(&self.open_docs),
            forest: Arc::clone(&self.forest),
            beancount_data: Arc::clone(&self.beancount_data),
            forest_content: Arc::clone(&self.forest_content),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tree_sitter_beancount::tree_sitter::Parser;

    fn make_parser() -> Parser {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_beancount::language())
            .expect("Failed to set language");
        parser
    }

    fn parse(content: &str) -> tree_sitter::Tree {
        make_parser().parse(content, None).expect("Failed to parse")
    }

    const CONTENT: &str = "2024-01-01 open Assets:Checking USD\n";

    /// Open a document and land its parse and semantic data, the way the
    /// scheduler does asynchronously in the running server.
    fn open_parsed(store: &mut DocumentStore, uri: &PathBuf, text: &str, version: i32) {
        store.open(uri.clone(), text, version);
        assert!(store.install_tree(uri, Arc::new(parse(text)), version));
        store.ensure_beancount_data(uri);
    }

    #[test]
    fn test_open_populates_all_maps() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");

        store.open(uri.clone(), CONTENT, 1);

        // Parsing and extraction are scheduled by the caller, off the main
        // loop, so open() alone yields a buffer and nothing else.
        assert!(store.open_docs.contains_key(&uri));
        assert_eq!(store.open_docs.get(&uri).unwrap().version, 1);
        assert!(store.get_tree(&uri).is_none());
        assert!(!store.beancount_data.contains_key(&uri));

        // Once the scheduled work lands, the maps are populated.
        assert!(store.install_tree(&uri, Arc::new(parse(CONTENT)), 1));
        store.ensure_beancount_data(&uri);
        assert!(store.get_tree(&uri).is_some());
        assert!(store.beancount_data.contains_key(&uri));
    }

    #[test]
    fn test_apply_change_keeps_previous_semantic_data() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, CONTENT, 1);

        #[allow(deprecated)]
        let change = lsp_types::TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            lsp_types::TextDocumentContentChangePartial {
                range: lsp_types::Range {
                    start: lsp_types::Position::new(0, 0),
                    end: lsp_types::Position::new(0, 0),
                },
                range_length: None,
                text: "".to_string(),
            },
        );
        store.apply_change(&uri, &[change], 2).unwrap();

        // Semantic data is deliberately kept (stale) rather than dropped:
        // re-extracting per keystroke blocked the main loop. did_change
        // schedules the rebuild on the thread pool.
        assert!(store.beancount_data.contains_key(&uri));
        assert!(store.get_tree(&uri).is_some());
        assert!(store.open_docs.get(&uri).is_some());
    }

    #[test]
    fn test_apply_change_updates_content_and_version() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, "hello", 1);

        #[allow(deprecated)]
        let change = lsp_types::TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            lsp_types::TextDocumentContentChangePartial {
                range: lsp_types::Range {
                    start: lsp_types::Position::new(0, 0),
                    end: lsp_types::Position::new(0, 5),
                },
                range_length: None,
                text: "world".to_string(),
            },
        );
        store.apply_change(&uri, &[change], 2).unwrap();

        let doc = store.open_docs.get(&uri).unwrap();
        assert_eq!(doc.text_string(), "world");
        assert_eq!(doc.version, 2);
    }

    #[allow(deprecated)]
    fn partial(
        start: (u32, u32),
        end: (u32, u32),
        text: &str,
    ) -> lsp_types::TextDocumentContentChangeEvent {
        lsp_types::TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            lsp_types::TextDocumentContentChangePartial {
                range: lsp_types::Range {
                    start: lsp_types::Position::new(start.0, start.1),
                    end: lsp_types::Position::new(end.0, end.1),
                },
                range_length: None,
                text: text.to_string(),
            },
        )
    }

    #[test]
    fn test_apply_change_batch_applies_sequentially() {
        // A didChange batch applies sequentially: a later change may reference
        // lines created by an earlier one in the same batch. Converting the
        // whole batch against the pre-edit rope panicked (line out of bounds)
        // and desynced the tree-sitter edits.
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, "2024-01-01 open Assets:Cash\n", 1);

        let changes = vec![
            partial((1, 0), (1, 0), "line1\nline2\nline3\n"),
            // line 3 does not exist in the pre-batch document
            partial((3, 0), (3, 5), "edited"),
        ];
        store.apply_change(&uri, &changes, 2).unwrap();

        let doc = store.open_docs.get(&uri).unwrap();
        assert_eq!(
            doc.text_string(),
            "2024-01-01 open Assets:Cash\nline1\nline2\nedited\n"
        );
    }

    #[test]
    fn test_apply_change_incremental_parse_matches_fresh_parse() {
        // The InputEdits handed to tree-sitter must describe each change
        // exactly (see lsp_textdocchange_to_ts_inputedit): wrong Points
        // corrupt subtree reuse, yielding trees whose structure — and even
        // byte ranges — diverge from a fresh parse of the same text.
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(
            &mut store,
            &uri,
            "2024-01-02 * \"Café ☕\" \"Espresso\"\n  Expenses:Caffè  2.50 EUR\n  Assets:Bank:Checking\n",
            1,
        );

        // A mix of edits that all previously produced corrupt Points:
        // multi-line insert at EOF, replacement shrinking a span mid-doc,
        // single-char append at the very end.
        let edits = vec![
            partial((3, 0), (3, 0), "2024-02-01 * \"New\"\n  Expenses:Food  1.00 EUR\n"),
            partial((1, 2), (2, 2), "X"),
            partial((3, 24), (3, 24), " ;🎉"),
        ];
        for (i, change) in edits.into_iter().enumerate() {
            store.apply_change(&uri, &[change], i as i32 + 2).unwrap();
        }

        let text = store.open_docs.get(&uri).unwrap().text_string();
        let incremental = store.get_tree(&uri).unwrap();
        let fresh = parse(&text);
        assert!(
            incremental.root_node().end_byte() <= text.len(),
            "incremental tree overruns the document: {} > {}",
            incremental.root_node().end_byte(),
            text.len()
        );
        assert_eq!(
            incremental.root_node().to_sexp(),
            fresh.root_node().to_sexp(),
            "incremental parse diverged from fresh parse of: {text:?}"
        );
    }

    #[test]
    fn test_apply_change_inverted_range_does_not_panic() {
        // Protocol-invalid, but a desynced client can send it; it must not
        // panic the main loop.
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, "2024-01-01 open Assets:Cash\n", 1);

        let changes = vec![partial((0, 9), (0, 2), "X")];
        store.apply_change(&uri, &changes, 2).unwrap();
    }

    #[test]
    fn test_apply_change_out_of_bounds_line_is_clamped() {
        // A position beyond the last line degrades to the document end
        // instead of panicking the main loop.
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, "2024-01-01 open Assets:Cash\n", 1);

        let changes = vec![partial((99, 7), (99, 9), "; tail")];
        store.apply_change(&uri, &changes, 2).unwrap();

        let doc = store.open_docs.get(&uri).unwrap();
        assert_eq!(doc.text_string(), "2024-01-01 open Assets:Cash\n; tail");
    }

    #[test]
    fn test_close_transitions_to_non_open_forest_file() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, CONTENT, 1);

        store.close(&uri);

        // Removed from open_docs
        assert!(store.open_docs.get(&uri).is_none());
        // Kept in forest and beancount_data for cross-file operations
        assert!(store.get_tree(&uri).is_some());
        assert!(store.beancount_data.contains_key(&uri));
        // Rope transferred to forest_content
        assert!(store.forest_content.contains_key(&uri));
        // Parser kept for reuse
        assert!(store.parsers.contains_key(&uri));
    }

    #[test]
    fn test_close_keeps_state_without_extracting() {
        // Closing must not rebuild semantic data: that is a full pass over the
        // document on the main loop. Existing data stays and the tree and rope
        // remain available for cross-file work.
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, CONTENT, 1);
        let before = store.beancount_data.get(&uri).unwrap().clone();

        store.close(&uri);

        assert!(store.open_docs.get(&uri).is_none());
        assert!(store.get_tree(&uri).is_some());
        assert!(store.forest_content.contains_key(&uri));
        assert!(Arc::ptr_eq(store.beancount_data.get(&uri).unwrap(), &before));
    }

    #[test]
    fn test_close_without_data_leaves_it_absent_for_the_scheduler() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        store.open(uri.clone(), CONTENT, 1);
        assert!(store.install_tree(&uri, Arc::new(parse(CONTENT)), 1));

        store.close(&uri);

        // No synchronous extraction happened; the rope is cached so the
        // scheduled rebuild can still run.
        assert!(!store.beancount_data.contains_key(&uri));
        assert!(store.extraction_inputs(&uri).is_some());
    }

    #[test]
    fn test_reopen_after_close_gets_fresh_state() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, CONTENT, 1);
        store.close(&uri);

        let new_content = "2024-06-01 open Liabilities:CreditCard USD\n";
        open_parsed(&mut store, &uri, new_content, 2);

        let doc = store.open_docs.get(&uri).unwrap();
        assert_eq!(doc.version, 2);
        assert!(doc.text_string().contains("Liabilities"));
        // forest_content cleared when re-opened (open_docs takes over)
        assert!(!store.forest_content.contains_key(&uri));
    }

    #[test]
    fn test_ensure_beancount_data_lazy_extraction() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, CONTENT, 1);

        // Data absent but tree present: the fallback must rebuild it.
        Arc::make_mut(&mut store.beancount_data).remove(&uri);
        assert!(!store.beancount_data.contains_key(&uri));

        store.ensure_beancount_data(&uri);
        assert!(store.beancount_data.contains_key(&uri));
    }

    #[test]
    fn test_install_beancount_data_respects_document_version() {
        // Data built off-thread must not overwrite a newer document's data.
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, CONTENT, 1);
        let (tree, rope, version) = store.extraction_inputs(&uri).unwrap();
        assert_eq!(version, 1);
        let fresh = Arc::new(BeancountData::new(&tree, &rope));

        // Same version: installed.
        store.install_beancount_data(&uri, fresh.clone(), 1);
        assert!(Arc::ptr_eq(store.beancount_data.get(&uri).unwrap(), &fresh));

        // Older version than the document: dropped.
        store.open(uri.clone(), CONTENT, 5);
        assert!(store.install_tree(&uri, Arc::new(parse(CONTENT)), 5));
        store.install_beancount_data(&uri, fresh.clone(), 1);
        assert!(!store.beancount_data.contains_key(&uri));

        // A forest file that is not open carries no version and is accepted.
        let other = PathBuf::from("/test/included.beancount");
        store.insert_parsed(other.clone(), parse(CONTENT), CONTENT);
        let (tree, rope, version) = store.extraction_inputs(&other).unwrap();
        assert_eq!(version, NO_VERSION);
        store.install_beancount_data(&other, Arc::new(BeancountData::new(&tree, &rope)), version);
        assert!(store.beancount_data.contains_key(&other));
    }

    #[test]
    fn test_ensure_beancount_data_skips_if_present() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, CONTENT, 1);

        let first_ptr = Arc::as_ptr(store.beancount_data.get(&uri).unwrap());
        store.ensure_beancount_data(&uri);
        let second_ptr = Arc::as_ptr(store.beancount_data.get(&uri).unwrap());

        assert_eq!(
            first_ptr, second_ptr,
            "should not re-extract if data present"
        );
    }

    #[test]
    fn test_ensure_beancount_data_does_nothing_without_tree() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        // doc exists but no tree
        Arc::make_mut(&mut store.open_docs).insert(
            uri.clone(),
            Document {
                content: ropey::Rope::from_str(CONTENT),
                version: 1,
            },
        );

        store.ensure_beancount_data(&uri); // must not panic
        assert!(!store.beancount_data.contains_key(&uri));
    }

    #[test]
    fn test_insert_parsed_stores_tree_data_and_rope() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/included.beancount");
        let tree = parse(CONTENT);

        store.insert_parsed(uri.clone(), tree, CONTENT);

        assert!(store.get_tree(&uri).is_some());
        assert!(store.forest_content.contains_key(&uri));
        // not an open doc
        assert!(store.open_docs.get(&uri).is_none());
        // Semantic data is scheduled by the caller, not built here.
        assert!(!store.beancount_data.contains_key(&uri));
    }

    #[test]
    fn test_insert_tree_and_data() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/bg.beancount");
        let tree = Arc::new(parse(CONTENT));
        let rope = ropey::Rope::from_str(CONTENT);
        let data = Arc::new(BeancountData::new(&tree, &rope));

        store.insert_tree_and_data(uri.clone(), tree, data, Arc::new(rope));

        assert!(store.get_tree(&uri).is_some());
        assert!(store.beancount_data.contains_key(&uri));
        assert!(store.forest_content.contains_key(&uri));
    }

    #[test]
    fn test_remove_external_clears_all_caches() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/ext.beancount");
        let tree = parse(CONTENT);
        store.insert_parsed(uri.clone(), tree, CONTENT);

        store.remove_external(&uri);

        assert!(store.get_tree(&uri).is_none());
        assert!(!store.beancount_data.contains_key(&uri));
    }

    #[test]
    fn test_invalidate_external_clears_tree_and_data() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/ext.beancount");
        let tree = parse(CONTENT);
        store.insert_parsed(uri.clone(), tree, CONTENT);

        store.invalidate_external(&uri);

        assert!(store.get_tree(&uri).is_none());
        assert!(!store.beancount_data.contains_key(&uri));
    }

    #[test]
    fn test_retain_reachable_prunes_unincluded_files() {
        // Files dropped from the include graph must leave the forest, or
        // they keep feeding completions and diagnostics forever.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("main.beancount");
        let kept = dir.path().join("kept.beancount");
        let ghost = dir.path().join("ghost.beancount");
        std::fs::write(&root, "include \"kept.beancount\"\n").unwrap();
        std::fs::write(&kept, "2020-01-01 open Assets:Kept\n").unwrap();
        std::fs::write(&ghost, "2020-01-01 open Assets:Ghost\n").unwrap();

        let mut store = DocumentStore::new();
        for path in [&root, &kept, &ghost] {
            let content = std::fs::read_to_string(path).unwrap();
            store.insert_parsed(path.clone(), parse(&content), &content);
        }

        let pruned = store.retain_reachable(&root);

        assert_eq!(pruned, vec![ghost.clone()]);
        assert!(store.get_tree(&ghost).is_none());
        assert!(store.get_tree(&kept).is_some());
        assert!(store.get_tree(&root).is_some());
    }

    #[test]
    fn test_retain_reachable_keeps_open_docs() {
        // An open buffer stays even when unreachable from the journal.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("main.beancount");
        let open_file = dir.path().join("scratch.beancount");
        std::fs::write(&root, "2020-01-01 open Assets:Root\n").unwrap();

        let mut store = DocumentStore::new();
        let content = std::fs::read_to_string(&root).unwrap();
        store.insert_parsed(root.clone(), parse(&content), &content);
        open_parsed(&mut store, &open_file, "2020-01-01 open Assets:Open\n", 1);

        let pruned = store.retain_reachable(&root);

        assert!(pruned.is_empty());
        assert!(store.get_tree(&open_file).is_some());
    }

    #[test]
    fn test_snapshot_maps_clones_all_maps() {
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, CONTENT, 1);

        let maps = store.snapshot_maps();

        assert!(maps.open_docs.contains_key(&uri));
        assert!(maps.forest.contains_key(&uri));
        assert!(maps.beancount_data.contains_key(&uri));
        // open file: forest_content not populated (open_docs is the source of truth)
        assert!(!maps.forest_content.contains_key(&uri));
        // parsers NOT in snapshot
    }

    #[test]
    fn test_snapshot_maps_shares_arc_identity() {
        // Snapshot should share the same Arc allocation (pointer equality),
        // not clone the underlying HashMaps.
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, CONTENT, 1);

        let maps1 = store.snapshot_maps();
        let maps2 = store.snapshot_maps();

        assert!(
            Arc::ptr_eq(&maps1.forest, &maps2.forest),
            "consecutive snapshots should share forest Arc"
        );
        assert!(
            Arc::ptr_eq(&maps1.beancount_data, &maps2.beancount_data),
            "consecutive snapshots should share beancount_data Arc"
        );
        assert!(
            Arc::ptr_eq(&maps1.open_docs, &maps2.open_docs),
            "consecutive snapshots should share open_docs Arc"
        );
        assert!(
            Arc::ptr_eq(&maps1.forest_content, &maps2.forest_content),
            "consecutive snapshots should share forest_content Arc"
        );
    }

    #[test]
    fn test_mutation_after_snapshot_does_not_alias() {
        // After a mutation, the live snapshot must not reflect the change
        // (copy-on-write: make_mut allocates a new HashMap).
        let mut store = DocumentStore::new();
        let uri = PathBuf::from("/test/file.beancount");
        open_parsed(&mut store, &uri, CONTENT, 1);

        let snapshot_before = store.snapshot_maps();

        // Mutate by inserting another key
        let uri2 = PathBuf::from("/test/file2.beancount");
        open_parsed(&mut store, &uri2, CONTENT, 1);

        // snapshot_before should still point to the old allocation
        assert!(
            !Arc::ptr_eq(&snapshot_before.forest, &store.snapshot_maps().forest),
            "snapshot taken before mutation should not alias the new forest"
        );
        // The old snapshot should not contain the new key
        assert!(
            !snapshot_before.forest.contains_key(&uri2),
            "old snapshot must not see keys added after snapshot"
        );
    }
}
