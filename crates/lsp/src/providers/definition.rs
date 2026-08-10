use crate::beancount_data::get_unified_query;
use crate::document::Document;
use crate::query_cache;
use crate::server::LspServerStateSnapshot;
use crate::treesitter_utils::{
    lsp_position_to_tree_sitter_point_range, text_for_tree_sitter_node,
    tree_sitter_node_to_lsp_range,
};
use anyhow::Context;
use anyhow::Result;
use lsp_types::DefinitionResponse;
use lsp_types::Location;
use lsp_types::LocationLink;
use ropey::Rope;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tree_sitter::StreamingIterator;
use tree_sitter_beancount::NodeKind;
use tree_sitter_beancount::tree_sitter;

/// Provider function for `textDocument/definition`.
pub(crate) fn definition(
    snapshot: LspServerStateSnapshot,
    params: lsp_types::DefinitionParams,
) -> Result<Option<DefinitionResponse>> {
    let doc_uri = &params.text_document_position_params.text_document.uri;
    let position = params.text_document_position_params.position;

    let (tree, doc) = snapshot
        .tree_and_document_for_uri(doc_uri)
        .context("Failed to get tree/document for definition")?;
    let content = doc.content.clone();

    let (start, end) = lsp_position_to_tree_sitter_point_range(&content, position)?;

    let Some(node) = tree
        .root_node()
        .named_descendant_for_point_range(start, end)
    else {
        return Ok(None);
    };

    let (query, capture_name) = match node.kind().into() {
        NodeKind::Account => (get_unified_query(), "account"),
        NodeKind::Currency => (query_cache::commodity_definition_query(), "currency"),
        _ => return Ok(None),
    };

    let origin_selection_range = tree_sitter_node_to_lsp_range(&content, &node);

    let node_text = text_for_tree_sitter_node(&content, &node);
    let locs = find_definitions(
        &snapshot.forest,
        &snapshot.open_docs,
        &snapshot.forest_content,
        query,
        capture_name,
        node_text,
    );
    if locs.is_empty() {
        return Ok(None);
    }

    let links: Vec<LocationLink> = locs
        .into_iter()
        .map(|loc| LocationLink {
            origin_selection_range: Some(origin_selection_range),
            target_uri: loc.uri,
            target_range: loc.range,
            target_selection_range: loc.range,
        })
        .collect();

    Ok(Some(DefinitionResponse::DefinitionLinkList(links)))
}

fn find_definitions(
    forest: &HashMap<PathBuf, Arc<tree_sitter::Tree>>,
    open_docs: &HashMap<PathBuf, Document>,
    forest_content: &HashMap<PathBuf, Arc<Rope>>,
    query: &tree_sitter::Query,
    capture_name: &str,
    node_text: String,
) -> Vec<Location> {
    forest
        .iter()
        .flat_map(|(url, tree)| {
            let capture_index = match query.capture_index_for_name(capture_name) {
                Some(index) => index,
                None => {
                    tracing::warn!("Query missing capture '{capture_name}'");
                    return vec![];
                }
            };

            // The text must match the forest tree, so it comes from the
            // snapshot (open buffer or cached rope) — never from disk, which
            // may have changed since the tree was parsed and whose bytes the
            // tree's ranges would then overrun.
            let (text, rope) = if let Some(doc) = open_docs.get(url) {
                (doc.text().to_string(), doc.content.clone())
            } else if let Some(stored) = forest_content.get(url) {
                (stored.to_string(), (**stored).clone())
            } else {
                tracing::debug!("No cached content for forest file: {:?}", url);
                return vec![];
            };

            let Ok(uri) = lsp_types::Uri::from_file_path(url) else {
                tracing::debug!("Failed to convert file path to URI: {}", url.display());
                return vec![];
            };

            let source = text.as_bytes();
            let mut query_cursor = tree_sitter::QueryCursor::new();
            let mut matches = query_cursor.matches(query, tree.root_node(), source);
            let mut results = Vec::new();
            while let Some(m) = matches.next() {
                if let Some(node) = m.nodes_for_capture_index(capture_index).next() {
                    let m_text = match node.utf8_text(source) {
                        Ok(text) => text,
                        Err(err) => {
                            tracing::debug!("Failed to read node text: {err}");
                            continue;
                        }
                    };
                    if m_text == node_text {
                        results.push(Location::new(
                            uri.clone(),
                            tree_sitter_node_to_lsp_range(&rope, &node),
                        ));
                    }
                }
            }
            results
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beancount_data::BeancountData;
    use crate::config::Config;
    use ropey::Rope;
    use tree_sitter::Parser;

    struct TestState {
        snapshot: LspServerStateSnapshot,
        path: PathBuf,
    }

    impl TestState {
        fn new(content: &str) -> anyhow::Result<Self> {
            let path = std::env::current_dir()?.join("test.beancount");
            let rope_content = Rope::from_str(content);

            let mut parser = tree_sitter::Parser::new();
            parser.set_language(&tree_sitter_beancount::language())?;
            let tree = parser.parse(content, None).unwrap();

            let mut forest = HashMap::new();
            forest.insert(path.clone(), Arc::new(tree.clone()));

            let mut open_docs = HashMap::new();
            open_docs.insert(
                path.clone(),
                Document {
                    content: rope_content.clone(),
                    version: 0,
                },
            );

            let mut beancount_data = HashMap::new();
            beancount_data.insert(
                path.clone(),
                Arc::new(BeancountData::new(&tree, &rope_content)),
            );

            let config = Config::new(path.clone());

            Ok(Self {
                snapshot: LspServerStateSnapshot {
                    forest: Arc::new(forest),
                    forest_content: Arc::new(HashMap::new()),
                    open_docs: Arc::new(open_docs),
                    beancount_data: Arc::new(beancount_data),
                    config,
                    checker: None,
                },
                path,
            })
        }

        fn definition_params(&self, line: u32, character: u32) -> lsp_types::DefinitionParams {
            lsp_types::DefinitionParams {
                text_document_position_params: lsp_types::TextDocumentPositionParams {
                    text_document: lsp_types::TextDocumentIdentifier {
                        uri: lsp_types::Uri::from_file_path(&self.path).unwrap(),
                    },
                    position: lsp_types::Position { line, character },
                },
                work_done_progress_params: Default::default(),
                partial_result_params: Default::default(),
            }
        }
    }

    fn make_tree(text: &str) -> tree_sitter::Tree {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_beancount::language())
            .unwrap();
        parser.parse(text, None).unwrap()
    }

    fn make_doc(text: &str) -> Document {
        Document {
            content: Rope::from_str(text),
            version: 1,
        }
    }

    fn find_account_open_definitions(
        forest: &HashMap<PathBuf, Arc<tree_sitter::Tree>>,
        open_docs: &HashMap<PathBuf, Document>,
        node_text: String,
    ) -> Vec<Location> {
        find_definitions(
            forest,
            open_docs,
            &HashMap::new(),
            get_unified_query(),
            "account",
            node_text,
        )
    }

    fn find_commodity_definitions(
        forest: &HashMap<PathBuf, Arc<tree_sitter::Tree>>,
        open_docs: &HashMap<PathBuf, Document>,
        node_text: String,
    ) -> Vec<Location> {
        find_definitions(
            forest,
            open_docs,
            &HashMap::new(),
            query_cache::commodity_definition_query(),
            "currency",
            node_text,
        )
    }

    #[test]
    fn test_find_account_open_definitions_single_match() {
        let text = "2024-01-01 open Assets:Cash\n";
        let path = std::env::temp_dir().join("definition_test.bean");
        let tree = Arc::new(make_tree(text));

        let mut forest = HashMap::new();
        forest.insert(path.clone(), tree);

        let mut open_docs = HashMap::new();
        open_docs.insert(path.clone(), make_doc(text));

        let locs = find_account_open_definitions(&forest, &open_docs, "Assets:Cash".to_string());

        assert_eq!(locs.len(), 1);
        let loc = &locs[0];
        assert_eq!(loc.range.start.line, 0);
        assert_eq!(loc.range.start.character, 16);
        assert_eq!(loc.range.end.line, 0);
        assert_eq!(loc.range.end.character, 27);

        let expected_uri = lsp_types::Uri::from_file_path(&path).unwrap();
        assert_eq!(loc.uri, expected_uri);
    }

    #[test]
    fn test_find_account_open_definitions_multiple_files() {
        let text_a = "2024-01-01 open Assets:Cash\n";
        let text_b = "2024-01-02 open Assets:Cash\n";
        let path_a = std::env::temp_dir().join("definition_test_a.bean");
        let path_b = std::env::temp_dir().join("definition_test_b.bean");

        let mut forest = HashMap::new();
        forest.insert(path_a.clone(), Arc::new(make_tree(text_a)));
        forest.insert(path_b.clone(), Arc::new(make_tree(text_b)));

        let mut open_docs = HashMap::new();
        open_docs.insert(path_a, make_doc(text_a));
        open_docs.insert(path_b, make_doc(text_b));

        let locs = find_account_open_definitions(&forest, &open_docs, "Assets:Cash".to_string());

        assert_eq!(locs.len(), 2);
    }

    #[test]
    fn test_find_account_open_definitions_no_match() {
        let text = "2024-01-01 open Assets:Cash\n";
        let path = std::env::temp_dir().join("definition_test_none.bean");
        let tree = Arc::new(make_tree(text));

        let mut forest = HashMap::new();
        forest.insert(path.clone(), tree);

        let mut open_docs = HashMap::new();
        open_docs.insert(path, make_doc(text));

        let locs =
            find_account_open_definitions(&forest, &open_docs, "Liabilities:Card".to_string());

        assert!(locs.is_empty());
    }

    #[test]
    fn test_find_commodity_definitions_single_match() {
        let text = "2024-01-01 commodity USD\n\n2024-01-02 * \"Payee\" \"Narration\"\n  Assets:Cash  100.00 USD\n  Expenses:Misc\n";
        let path = std::env::temp_dir().join("definition_test_commodity.bean");
        let tree = Arc::new(make_tree(text));

        let mut forest = HashMap::new();
        forest.insert(path.clone(), tree);

        let mut open_docs = HashMap::new();
        open_docs.insert(path.clone(), make_doc(text));

        let locs = find_commodity_definitions(&forest, &open_docs, "USD".to_string());

        // Only the commodity directive matches, not the posting usage
        assert_eq!(locs.len(), 1);
        let loc = &locs[0];
        assert_eq!(loc.range.start.line, 0);
        assert_eq!(loc.range.start.character, 21);
        assert_eq!(loc.range.end.line, 0);
        assert_eq!(loc.range.end.character, 24);

        let expected_uri = lsp_types::Uri::from_file_path(&path).unwrap();
        assert_eq!(loc.uri, expected_uri);
    }

    #[test]
    fn test_find_commodity_definitions_no_match() {
        let text = "2024-01-01 commodity USD\n";
        let path = std::env::temp_dir().join("definition_test_commodity_none.bean");
        let tree = Arc::new(make_tree(text));

        let mut forest = HashMap::new();
        forest.insert(path.clone(), tree);

        let mut open_docs = HashMap::new();
        open_docs.insert(path, make_doc(text));

        let locs = find_commodity_definitions(&forest, &open_docs, "EUR".to_string());

        assert!(locs.is_empty());
    }

    #[test]
    fn test_definition_handler_account() {
        let content = r#"
2024-01-01 open Assets:Checking
2024-01-02 * "Test"
  Assets:Checking  100.00 USD
  Expenses:Food   -100.00 USD
"#;
        let state = TestState::new(content).unwrap();
        let uri = lsp_types::Uri::from_file_path(&state.path).unwrap();

        // Cursor on "Assets:Checking" in the posting
        let params = state.definition_params(3, 5);
        let result = definition(state.snapshot, params).unwrap();

        let Some(DefinitionResponse::DefinitionLinkList(links)) = result else {
            panic!("Expected DefinitionLinkList");
        };
        assert_eq!(links.len(), 1);
        let link = &links[0];
        assert_eq!(link.target_uri, uri);
        // Target is the account in the open directive
        assert_eq!(link.target_range.start.line, 1);
        assert_eq!(link.target_range.start.character, 16);
        assert_eq!(link.target_range.end.character, 31);
        // Origin covers the account under the cursor
        let origin = link.origin_selection_range.expect("origin range");
        assert_eq!(origin.start.line, 3);
        assert_eq!(origin.start.character, 2);
        assert_eq!(origin.end.character, 17);
    }

    #[test]
    fn test_definition_handler_commodity() {
        let content = r#"
2024-01-01 commodity USD
2024-01-02 * "Test"
  Assets:Checking  100.00 USD
  Expenses:Food   -100.00 USD
"#;
        let state = TestState::new(content).unwrap();
        let uri = lsp_types::Uri::from_file_path(&state.path).unwrap();

        // Cursor on "USD" in the posting
        let params = state.definition_params(3, 27);
        let result = definition(state.snapshot, params).unwrap();

        let Some(DefinitionResponse::DefinitionLinkList(links)) = result else {
            panic!("Expected DefinitionLinkList");
        };
        assert_eq!(links.len(), 1);
        let link = &links[0];
        assert_eq!(link.target_uri, uri);
        // Target is the currency in the commodity directive
        assert_eq!(link.target_range.start.line, 1);
        assert_eq!(link.target_range.start.character, 21);
        assert_eq!(link.target_range.end.character, 24);
        // Origin covers the currency under the cursor
        let origin = link.origin_selection_range.expect("origin range");
        assert_eq!(origin.start.line, 3);
        assert_eq!(origin.start.character, 26);
        assert_eq!(origin.end.character, 29);
    }

    #[test]
    fn test_definition_handler_commodity_undeclared() {
        let content = r#"
2024-01-01 commodity USD
2024-01-02 * "Test"
  Assets:Checking  100.00 EUR
  Expenses:Food   -100.00 EUR
"#;
        let state = TestState::new(content).unwrap();

        // Cursor on "EUR", which has no commodity directive
        let params = state.definition_params(3, 27);
        let result = definition(state.snapshot, params).unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn test_definition_handler_other_node() {
        let content = r#"
2024-01-01 commodity USD
2024-01-02 * "Test"
  Assets:Checking  100.00 USD
"#;
        let state = TestState::new(content).unwrap();

        // Cursor on the narration string: not an account or currency
        let params = state.definition_params(2, 14);
        let result = definition(state.snapshot, params).unwrap();

        assert!(result.is_none());
    }
}
