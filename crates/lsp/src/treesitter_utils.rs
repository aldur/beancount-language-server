use std::ops::ControlFlow;
use std::time::{Duration, Instant};
use tree_sitter_beancount::tree_sitter;

/// Wall-clock budget for a single parse.
///
/// tree-sitter's error recovery is superlinear on some pathological input
/// (tens of thousands of nested `(`): a single paste of such a file parsed
/// for minutes, and since documents are parsed on the main loop that froze
/// the entire server, not just one request.
pub(crate) const PARSE_BUDGET: Duration = Duration::from_secs(5);

/// Parse `text`, giving up after [`PARSE_BUDGET`].
///
/// Returns `None` when the budget is exhausted; callers must treat that like
/// any other parse failure rather than assuming a tree exists.
pub(crate) fn parse_with_budget(
    parser: &mut tree_sitter::Parser,
    text: &str,
    old_tree: Option<&tree_sitter::Tree>,
) -> Option<tree_sitter::Tree> {
    let deadline = Instant::now() + PARSE_BUDGET;
    let mut cancelled = false;
    let tree = {
        let mut progress = |_: &tree_sitter::ParseState| {
            if Instant::now() >= deadline {
                cancelled = true;
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let options = tree_sitter::ParseOptions::new().progress_callback(&mut progress);
        let bytes = text.as_bytes();
        let mut read = |offset: usize, _: tree_sitter::Point| bytes.get(offset..).unwrap_or(&[]);
        parser.parse_with_options(&mut read, old_tree, Some(options))
    };
    if cancelled {
        // A cancelled parse leaves the parser mid-parse; the next call must
        // start clean.
        parser.reset();
        tracing::warn!("parse exceeded {PARSE_BUDGET:?} and was cancelled");
        return None;
    }
    tree
}

/// Byte-offset → LSP position lookups over a `&str`, without a rope.
///
/// The providers that walk whole documents (symbols, semantic tokens) used to
/// convert every node through the rope: `byte_to_lsp_position` alone is five
/// O(log n) tree walks, and a symbol costs two of those plus a text slice per
/// field. On a 240k-line ledger that is millions of walks and seconds per
/// request.
///
/// Building this index is one linear scan; each lookup is then an integer
/// binary search, plus — only for lines that actually contain non-ASCII — a
/// short scan of the line prefix.
pub(crate) struct LineIndex<'a> {
    text: &'a str,
    /// Byte offset where each line starts; always begins with 0.
    line_starts: Vec<u32>,
    /// Whether each line is pure ASCII, in which case the UTF-16 column is
    /// simply the byte offset within the line.
    line_ascii: Vec<bool>,
}

impl<'a> LineIndex<'a> {
    pub(crate) fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        let mut line_starts = Vec::with_capacity(bytes.len() / 32 + 1);
        let mut line_ascii = Vec::with_capacity(bytes.len() / 32 + 1);
        line_starts.push(0);
        let mut ascii = true;
        for (i, &b) in bytes.iter().enumerate() {
            if !b.is_ascii() {
                ascii = false;
            }
            if b == b'\n' {
                line_ascii.push(ascii);
                ascii = true;
                line_starts.push((i + 1) as u32);
            }
        }
        line_ascii.push(ascii);
        Self {
            text,
            line_starts,
            line_ascii,
        }
    }

    /// The whole document.
    pub(crate) fn text(&self) -> &'a str {
        self.text
    }

    /// The line containing `byte`.
    fn line_of(&self, byte: usize) -> usize {
        // `line_starts` is sorted and starts at 0, so this never underflows.
        self.line_starts.partition_point(|&start| start as usize <= byte) - 1
    }

    /// Largest char boundary at or below `byte`, clamped to the document.
    fn floor_boundary(&self, byte: usize) -> usize {
        let mut byte = byte.min(self.text.len());
        while byte > 0 && !self.text.is_char_boundary(byte) {
            byte -= 1;
        }
        byte
    }

    /// LSP position (line, UTF-16 column) of a byte offset.
    pub(crate) fn position(&self, byte: usize) -> lsp_types::Position {
        let byte = self.floor_boundary(byte);
        let line = self.line_of(byte);
        let start = self.line_starts[line] as usize;
        let column = if self.line_ascii[line] {
            (byte - start) as u32
        } else {
            self.text[start..byte].encode_utf16().count() as u32
        };
        lsp_types::Position::new(line as u32, column)
    }

    /// LSP range covering a node.
    pub(crate) fn range(&self, node: &tree_sitter::Node) -> lsp_types::Range {
        lsp_types::Range {
            start: self.position(node.start_byte()),
            end: self.position(node.end_byte()),
        }
    }

    /// Byte offset where a line begins (clamped to the last line).
    pub(crate) fn line_start_byte(&self, line: usize) -> usize {
        let line = line.min(self.line_starts.len() - 1);
        self.line_starts[line] as usize
    }

    /// Byte offset just past a line, i.e. the start of the next one. Includes
    /// the line's own terminator, matching a rope's `line_to_char(n + 1)`.
    pub(crate) fn line_end_byte(&self, line: usize) -> usize {
        match self.line_starts.get(line + 1) {
            Some(&next) => next as usize,
            None => self.text.len(),
        }
    }

    /// Number of lines, counted the way a rope does (a trailing newline
    /// yields one final empty line).
    pub(crate) fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// A line's text, terminator included.
    pub(crate) fn line_text(&self, line: usize) -> &'a str {
        &self.text[self.line_start_byte(line)..self.line_end_byte(line)]
    }

    /// A node's source text, borrowed rather than copied.
    pub(crate) fn slice(&self, node: &tree_sitter::Node) -> &'a str {
        let start = self.floor_boundary(node.start_byte());
        let end = self.floor_boundary(node.end_byte()).max(start);
        &self.text[start..end]
    }
}

/// Maximum tree depth the recursive walkers will descend.
///
/// Real ledgers nest a handful of levels; pathological input does not.
/// `1+1+1+…` a hundred thousand times parses into an equally deep
/// expression tree, and a recursive walk over it overflows the stack —
/// which aborts the whole process, on any thread, uncatchably.
pub(crate) const MAX_TREE_DEPTH: usize = 256;

/// True when `tree` is deeper than `limit`.
///
/// Depth is the one property that makes an otherwise ordinary document
/// pathological for everything downstream: tree-sitter's own queries go
/// superlinear on very deep trees (a document with tens of thousands of
/// nested `(` froze the main loop inside a query, not in the parse), and the
/// recursive walkers have to truncate. Rejecting such a tree once, here,
/// keeps every consumer safe without a budget at each call site.
pub(crate) fn tree_depth_exceeds(tree: &tree_sitter::Tree, limit: usize) -> bool {
    let mut cursor = tree.walk();
    let mut depth = 0usize;
    loop {
        while cursor.goto_first_child() {
            depth += 1;
            if depth > limit {
                return true;
            }
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return false;
            }
            depth -= 1;
        }
    }
}

/// Parse a beancount document from scratch.
/// Returns `None` only if tree-sitter itself fails (extremely rare).
pub(crate) fn parse_beancount(text: &str) -> Option<tree_sitter::Tree> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_beancount::language())
        .ok()?;
    parse_with_budget(&mut parser, text, None)
}

/// Convert an LSP UTF-16 position into a tree-sitter `Point` (byte-based column).
pub fn lsp_position_to_tree_sitter_point(
    source: &ropey::Rope,
    position: lsp_types::Position,
) -> anyhow::Result<tree_sitter::Point> {
    Ok(lsp_position_to_core(source, position)?.point)
}

pub fn lsp_position_to_tree_sitter_point_range(
    source: &ropey::Rope,
    position: lsp_types::Position,
) -> anyhow::Result<(tree_sitter::Point, tree_sitter::Point)> {
    // Use a 1-character-wide point range (pos-1..pos) so token-boundary cursors
    // still resolve to the intended node.
    let start_pos = lsp_types::Position {
        line: position.line,
        character: position.character.saturating_sub(1),
    };

    let start = lsp_position_to_tree_sitter_point(source, start_pos)?;
    let end = lsp_position_to_tree_sitter_point(source, position)?;
    Ok((start, end))
}

pub fn tree_sitter_node_to_lsp_range(
    source: &ropey::Rope,
    node: &tree_sitter::Node,
) -> lsp_types::Range {
    let start = byte_to_lsp_position(source, node.start_byte());
    let end = byte_to_lsp_position(source, node.end_byte());
    lsp_types::Range { start, end }
}

pub fn lsp_textdocchange_to_ts_inputedit(
    source: &ropey::Rope,
    change: &lsp_types::TextDocumentContentChangeEvent,
) -> anyhow::Result<tree_sitter::InputEdit> {
    let (text, range) = match change {
        lsp_types::TextDocumentContentChangeEvent::TextDocumentContentChangePartial(partial) => {
            (partial.text.as_str(), partial.range)
        }
        lsp_types::TextDocumentContentChangeEvent::TextDocumentContentChangeWholeDocument(
            partial,
        ) => {
            // Full document replacement: range covers the entire OLD document
            let start = byte_to_lsp_position(source, 0);
            let end = byte_to_lsp_position(source, source.len_bytes());
            let range = lsp_types::Range { start, end };
            (partial.text.as_str(), range)
        }
    };

    let text_bytes = text.as_bytes();
    let text_end_byte_idx = text_bytes.len();

    let start = lsp_position_to_core(source, range.start)?;
    let old_end = lsp_position_to_core(source, range.end)?;

    let new_end_byte = start.byte as usize + text_end_byte_idx;

    // The new end is the start plus the extent of the inserted text itself.
    // It must not be derived from `source`: the rope still holds the OLD
    // document, so mapping the new byte offset through it (or worse, adding
    // line counts to the old total) yields garbage Points — and tree-sitter
    // trusts InputEdits, so garbage here silently corrupts every subsequent
    // incremental parse.
    let new_end_position = match text.rfind('\n') {
        Some(last_nl) => tree_sitter::Point::new(
            start.point.row + text.bytes().filter(|&b| b == b'\n').count(),
            text_end_byte_idx - last_nl - 1,
        ),
        None => tree_sitter::Point::new(start.point.row, start.point.column + text_end_byte_idx),
    };

    Ok(tree_sitter::InputEdit {
        start_byte: start.byte as usize,
        old_end_byte: old_end.byte as usize,
        new_end_byte: u32::try_from(new_end_byte)? as usize,
        start_position: start.point,
        old_end_position: old_end.point,
        new_end_position,
    })
}

pub(crate) fn byte_to_lsp_position(text: &ropey::Rope, byte_idx: usize) -> lsp_types::Position {
    let line_idx = text.byte_to_line(byte_idx);

    let line_utf16_cu_idx = {
        let char_idx = text.line_to_char(line_idx);
        text.char_to_utf16_cu(char_idx)
    };

    let character_utf16_cu_idx = {
        let char_idx = text.byte_to_char(byte_idx);
        text.char_to_utf16_cu(char_idx)
    };

    let line = line_idx;
    let character = character_utf16_cu_idx - line_utf16_cu_idx;

    lsp_types::Position::new(line as u32, character as u32)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TextPosition {
    pub char: u32,
    pub byte: u32,
    pub code: u32,
    pub point: tree_sitter::Point,
}

fn lsp_position_to_core(
    source: &ropey::Rope,
    position: lsp_types::Position,
) -> anyhow::Result<TextPosition> {
    // Clamp the line like the UTF-16 column below: `line_to_char`/`line_to_byte`
    // accept at most len_lines, and an out-of-bounds position from the client
    // must degrade to the document end rather than panic.
    let row_idx = (position.line as usize).min(source.len_lines());

    // LSP `character` is a *line-relative* UTF-16 code-unit offset.
    let col_utf16_cu_idx = position.character as usize;

    // Convert the *line-relative* UTF-16 column into an absolute UTF-16 index.
    let row_char_idx = source.line_to_char(row_idx);
    let row_utf16_cu_idx = source.char_to_utf16_cu(row_char_idx);
    let abs_utf16_cu_idx = row_utf16_cu_idx + col_utf16_cu_idx;
    // Clamp to document bounds to prevent panic if client sends invalid positions
    let abs_utf16_cu_idx = abs_utf16_cu_idx.min(source.len_utf16_cu());

    // Convert absolute UTF-16 index -> absolute char index -> absolute byte index.
    let abs_char_idx = source.utf16_cu_to_char(abs_utf16_cu_idx);
    let abs_byte_idx = source.char_to_byte(abs_char_idx);

    // tree-sitter Point columns are byte offsets from the *start of the row*.
    let row_byte_idx = source.line_to_byte(row_idx);
    let col_byte_offset = abs_byte_idx.saturating_sub(row_byte_idx);
    let point = tree_sitter::Point::new(row_idx, col_byte_offset);

    Ok(TextPosition {
        char: u32::try_from(abs_char_idx)?,
        byte: u32::try_from(abs_byte_idx)?,
        code: u32::try_from(abs_utf16_cu_idx)?,
        point,
    })
}

#[cfg(test)]
fn byte_to_tree_sitter_point(
    source: &ropey::Rope,
    byte_idx: usize,
) -> anyhow::Result<tree_sitter::Point> {
    let line_idx = source.byte_to_line(byte_idx);
    let line_byte_idx = source.line_to_byte(line_idx);
    let row = u32::try_from(line_idx)? as usize;
    let column = u32::try_from(byte_idx - line_byte_idx)? as usize;
    Ok(tree_sitter::Point::new(row, column))
}

pub fn text_for_tree_sitter_node(
    source: &ropey::Rope,
    node: &tree_sitter::Node,
) -> std::string::String {
    let start = source.byte_to_char(node.start_byte());
    let end = source.byte_to_char(node.end_byte());
    let slice = source.slice(start..end);
    slice.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lsp_types::{
        Position, Range, TextDocumentContentChangeEvent, TextDocumentContentChangePartial,
        TextDocumentContentChangeWholeDocument,
    };
    use ropey::Rope;
    use tree_sitter::Point;

    #[test]
    fn test_lsp_textdocchange_simple_insertion() {
        let source = Rope::from("Hello World");
        let change = TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            TextDocumentContentChangePartial {
                range: Range {
                    start: Position::new(0, 5),
                    end: Position::new(0, 5),
                },
                text: " Beautiful".to_string(),
                ..Default::default()
            },
        );

        let edit = lsp_textdocchange_to_ts_inputedit(&source, &change).unwrap();
        assert_eq!(edit.start_byte, 5);
        assert_eq!(edit.old_end_byte, 5);
        assert_eq!(edit.new_end_byte, 15); // Added 10 bytes
        assert_eq!(edit.start_position, Point::new(0, 5));
        assert_eq!(edit.old_end_position, Point::new(0, 5));
    }

    #[test]
    fn test_lsp_textdocchange_simple_deletion() {
        let source = Rope::from("Hello World");
        let change = TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            TextDocumentContentChangePartial {
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(0, 6),
                },
                text: String::new(),
                ..Default::default()
            },
        );

        let edit = lsp_textdocchange_to_ts_inputedit(&source, &change).unwrap();
        assert_eq!(edit.start_byte, 0);
        assert_eq!(edit.old_end_byte, 6);
        assert_eq!(edit.new_end_byte, 0);
        assert_eq!(edit.start_position, Point::new(0, 0));
        assert_eq!(edit.old_end_position, Point::new(0, 6));
        assert_eq!(edit.new_end_position, Point::new(0, 0));
    }

    #[test]
    fn test_lsp_textdocchange_replacement() {
        let source = Rope::from("Hello World");
        let change = TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            TextDocumentContentChangePartial {
                range: Range {
                    start: Position::new(0, 6),
                    end: Position::new(0, 11),
                },
                text: "Rust".to_string(),
                ..Default::default()
            },
        );

        let edit = lsp_textdocchange_to_ts_inputedit(&source, &change).unwrap();
        assert_eq!(edit.start_byte, 6);
        assert_eq!(edit.old_end_byte, 11);
        assert_eq!(edit.new_end_byte, 10); // "Rust" is 4 bytes
        assert_eq!(edit.start_position, Point::new(0, 6));
    }

    #[test]
    fn test_lsp_textdocchange_full_document_replacement() {
        let source = Rope::from("Old content");
        let change = TextDocumentContentChangeEvent::TextDocumentContentChangeWholeDocument(
            TextDocumentContentChangeWholeDocument {
                text: "New content".to_string(),
            },
        );

        let edit = lsp_textdocchange_to_ts_inputedit(&source, &change).unwrap();
        assert_eq!(edit.start_byte, 0);
        assert_eq!(edit.old_end_byte, 11);
        assert_eq!(edit.new_end_byte, 11);
        assert_eq!(edit.start_position, Point::new(0, 0));
    }

    #[test]
    fn test_lsp_textdocchange_multiline_insertion() {
        let source = Rope::from("Line 1\nLine 2");
        let change = TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            TextDocumentContentChangePartial {
                range: Range {
                    start: Position::new(1, 0),
                    end: Position::new(1, 0),
                },
                text: "New line\n".to_string(),
                ..Default::default()
            },
        );

        let edit = lsp_textdocchange_to_ts_inputedit(&source, &change).unwrap();
        assert_eq!(edit.start_byte, 7); // After "Line 1\n"
        assert_eq!(edit.old_end_byte, 7);
        assert_eq!(edit.new_end_byte, 16); // Added "New line\n" (9 bytes)
    }

    #[test]
    fn test_lsp_textdocchange_with_multibyte_utf8() {
        let source = Rope::from("Hello 世界");
        let change = TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            TextDocumentContentChangePartial {
                range: Range {
                    start: Position::new(0, 6),
                    end: Position::new(0, 8),
                },
                text: "🌍".to_string(),
                ..Default::default()
            },
        );

        let edit = lsp_textdocchange_to_ts_inputedit(&source, &change).unwrap();
        assert_eq!(edit.start_byte, 6);
        // 世界 is 6 bytes (3 bytes each), but we're replacing it with 🌍 (4 bytes)
        assert_eq!(edit.old_end_byte, 12);
        assert_eq!(edit.new_end_byte, 10); // 6 + 4
    }

    #[test]
    fn test_lsp_textdocchange_empty_document() {
        let source = Rope::from("");
        let change = TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            TextDocumentContentChangePartial {
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(0, 0),
                },
                text: "New content".to_string(),
                ..Default::default()
            },
        );

        let edit = lsp_textdocchange_to_ts_inputedit(&source, &change).unwrap();
        assert_eq!(edit.start_byte, 0);
        assert_eq!(edit.old_end_byte, 0);
        assert_eq!(edit.new_end_byte, 11);
        assert_eq!(edit.start_position, Point::new(0, 0));
        assert_eq!(edit.old_end_position, Point::new(0, 0));
    }

    #[test]
    fn test_lsp_textdocchange_to_empty() {
        let source = Rope::from("Content to delete");
        let change = TextDocumentContentChangeEvent::TextDocumentContentChangeWholeDocument(
            TextDocumentContentChangeWholeDocument {
                text: String::new(),
            },
        );

        let edit = lsp_textdocchange_to_ts_inputedit(&source, &change).unwrap();
        assert_eq!(edit.start_byte, 0);
        assert_eq!(edit.old_end_byte, 17);
        assert_eq!(edit.new_end_byte, 0);
    }

    #[test]
    fn test_new_end_position_measures_inserted_text_not_old_rope() {
        // Replacing "a\nb" (2 lines) with "xyz" (1 line): the new end is
        // (0, 3). The old code mapped the new byte offset through the old
        // rope and answered (1, 1).
        let source = Rope::from("a\nb cd");
        let change = TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            TextDocumentContentChangePartial {
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(1, 1),
                },
                text: "xyz".to_string(),
                ..Default::default()
            },
        );
        let edit = lsp_textdocchange_to_ts_inputedit(&source, &change).unwrap();
        assert_eq!(edit.new_end_position, Point::new(0, 3));

        // Appending "x" at the end of a 1-line doc: new end is (0, len+1).
        // The old code answered row = len_lines + lines(text) = 2.
        let source = Rope::from("hello");
        let change = TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            TextDocumentContentChangePartial {
                range: Range {
                    start: Position::new(0, 5),
                    end: Position::new(0, 5),
                },
                text: "x".to_string(),
                ..Default::default()
            },
        );
        let edit = lsp_textdocchange_to_ts_inputedit(&source, &change).unwrap();
        assert_eq!(edit.new_end_position, Point::new(0, 6));

        // Multi-line insert: rows advance by the newline count, column
        // restarts after the last newline.
        let source = Rope::from("hello\n");
        let change = TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            TextDocumentContentChangePartial {
                range: Range {
                    start: Position::new(1, 0),
                    end: Position::new(1, 0),
                },
                text: "aa\nbbb\ncc".to_string(),
                ..Default::default()
            },
        );
        let edit = lsp_textdocchange_to_ts_inputedit(&source, &change).unwrap();
        assert_eq!(edit.new_end_position, Point::new(3, 2));
    }

    #[test]
    fn test_line_index_matches_rope_at_every_offset() {
        // The index replaces the rope in the hot providers, so it has to agree
        // with it everywhere — including around multibyte and astral
        // characters, CRLF, and the very end of the document.
        let documents = [
            "",
            "\n",
            "2020-01-01 open Assets:Cash EUR\n",
            "2020-01-01 open Assets:Caffè EUR\n  ; é comment\n",
            "2024-01-02 * \"Café ☕\" \"日本語 🎉\"\r\n  Expenses:Caffè  2.50 EUR\r\n",
            "no trailing newline 🎉",
            "a\r\nb\rc\nd",
            "🎉🎉🎉\n🎉\n",
        ];
        for text in documents {
            let rope = Rope::from_str(text);
            let index = LineIndex::new(text);
            for byte in 0..=text.len() {
                if !text.is_char_boundary(byte) {
                    continue;
                }
                assert_eq!(
                    index.position(byte),
                    byte_to_lsp_position(&rope, byte),
                    "offset {byte} of {text:?}"
                );
            }
        }
    }

    #[test]
    fn test_line_index_clamps_and_slices() {
        let text = "2020-01-01 open Assets:Caffè EUR\n";
        let index = LineIndex::new(text);
        // Past the end clamps to the document end.
        let last = index.position(text.len());
        assert_eq!(index.position(text.len() + 999), last);
        // Interior of a multibyte character floors to its start.
        let e_start = text.find('è').unwrap();
        assert_eq!(index.position(e_start + 1), index.position(e_start));
    }

    #[test]
    fn test_byte_to_lsp_position_simple() {
        let text = Rope::from("Hello\nWorld");
        let pos = byte_to_lsp_position(&text, 0);
        assert_eq!(pos.line, 0);
        assert_eq!(pos.character, 0);

        let pos = byte_to_lsp_position(&text, 6);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.character, 0);
    }

    #[test]
    fn test_lsp_position_to_core_simple() {
        let source = Rope::from("Hello\nWorld");
        let pos = Position::new(0, 5);
        let core_pos = lsp_position_to_core(&source, pos).unwrap();
        assert_eq!(core_pos.byte, 5);
        assert_eq!(core_pos.point, Point::new(0, 5));
    }

    #[test]
    fn test_lsp_position_to_core_second_line() {
        let source = Rope::from("Hello\nWorld");
        let pos = Position::new(1, 3);
        let core_pos = lsp_position_to_core(&source, pos).unwrap();
        assert_eq!(core_pos.byte, 9); // "Hello\n" is 6 bytes + 3
        assert_eq!(core_pos.point, Point::new(1, 3)); // Point column is byte offset from line start
    }

    #[test]
    fn test_lsp_position_to_core_second_line_with_utf8_in_first_line() {
        // First line contains multibyte UTF-8, which must not affect addressing on the next line.
        let source = Rope::from("财财\nABC");

        // On the second line, after "A" (column 1 in UTF-16), byte offset should be 1.
        let pos = Position::new(1, 1);
        let core_pos = lsp_position_to_core(&source, pos).unwrap();
        assert_eq!(core_pos.point, Point::new(1, 1));

        // The absolute byte index should be: bytes("财财\n") + 1.
        let prefix_len = "财财\n".len();
        assert_eq!(core_pos.byte as usize, prefix_len + 1);
    }

    #[test]
    fn test_byte_to_tree_sitter_point_simple() {
        let source = Rope::from("Hello\nWorld\nTest");
        let point = byte_to_tree_sitter_point(&source, 6).unwrap();
        assert_eq!(point, Point::new(1, 0));

        let point = byte_to_tree_sitter_point(&source, 12).unwrap();
        assert_eq!(point, Point::new(2, 0));
    }

    #[test]
    fn test_text_for_tree_sitter_node() {
        let source = Rope::from("2024-01-01 open Assets:Checking");

        // We need to parse the source to get a tree-sitter node
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_beancount::language())
            .unwrap();
        let tree = parser.parse(source.to_string(), None).unwrap();
        let root = tree.root_node();

        // Get the text for the entire tree
        let text = text_for_tree_sitter_node(&source, &root);
        assert_eq!(text, "2024-01-01 open Assets:Checking");
    }

    #[test]
    fn test_text_for_tree_sitter_node_with_utf8() {
        let source = Rope::from("2024-01-01 * \"Coffee ☕\"");

        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_beancount::language())
            .unwrap();
        let tree = parser.parse(source.to_string(), None).unwrap();
        let root = tree.root_node();

        let text = text_for_tree_sitter_node(&source, &root);
        assert_eq!(text, "2024-01-01 * \"Coffee ☕\"");
    }

    #[test]
    fn test_lsp_position_out_of_bounds() {
        // Test that out-of-bounds UTF-16 positions are clamped instead of panicking
        // This reproduces issue #820
        let source = Rope::from("2024-01-01 * \"Test\"\n");
        let total_utf16_len = source.len_utf16_cu();

        // Position beyond document bounds - should be clamped
        let pos = Position::new(0, (total_utf16_len + 100) as u32);
        let result = lsp_position_to_core(&source, pos);

        // Should not panic, should clamp to document end
        assert!(
            result.is_ok(),
            "Should handle out-of-bounds position gracefully"
        );
        let core_pos = result.unwrap();
        assert_eq!(
            core_pos.char as usize,
            source.len_chars(),
            "Should clamp to document end"
        );
    }

    #[test]
    fn test_lsp_textdocchange_out_of_bounds_range() {
        // Test that text changes with out-of-bounds ranges don't panic
        let source = Rope::from("Short text");
        let total_utf16_len = source.len_utf16_cu();

        // Change with end position beyond document - should be clamped
        let change = TextDocumentContentChangeEvent::TextDocumentContentChangePartial(
            TextDocumentContentChangePartial {
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(0, (total_utf16_len + 50) as u32),
                },
                text: "Replacement".to_string(),
                ..Default::default()
            },
        );

        let result = lsp_textdocchange_to_ts_inputedit(&source, &change);
        assert!(
            result.is_ok(),
            "Should handle out-of-bounds range gracefully"
        );
    }
}
