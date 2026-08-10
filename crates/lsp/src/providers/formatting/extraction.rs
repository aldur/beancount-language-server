use crate::query_cache;
use anyhow::Result;
use tree_sitter::StreamingIterator;
use tree_sitter_beancount::tree_sitter;

/// Represents a formateable line extracted from a Beancount file
/// Contains the components that bean-format uses for alignment
#[derive(Debug, Clone)]
pub(super) struct FormatableLine {
    /// Line number in the document
    pub(super) line_num: usize,
    /// Prefix text (account name or directive start)
    pub(super) prefix: String,
    /// Number text (amount value)
    pub(super) number: String,
    /// Rest of the line after the number (currency, comments, etc.)
    pub(super) rest: String,
}

/// Configuration for formatting calculations
#[derive(Debug)]
pub(super) struct FormatConfig {
    /// Final prefix width to use (may be overridden by config)
    pub(super) final_prefix_width: usize,
    /// Final number width to use (may be overridden by config)
    pub(super) final_num_width: usize,
}

/// Extracts formateable lines from the document using tree-sitter
/// This mimics bean-format's regex-based line extraction
pub(super) fn extract_formateable_lines(
    index: &crate::treesitter_utils::LineIndex,
    tree: &tree_sitter::Tree,
) -> Result<Vec<FormatableLine>> {
    let query = query_cache::format_query();

    let mut query_cursor = tree_sitter::QueryCursor::new();
    let mut matches = query_cursor.matches(query, tree.root_node(), index.text().as_bytes());

    let mut formateable_lines = Vec::new();

    while let Some(matched) = matches.next() {
        let mut prefix_node: Option<tree_sitter::Node> = None;
        let mut number_node: Option<tree_sitter::Node> = None;

        // Extract prefix and number nodes from captures
        for capture in matched.captures {
            let capture_name = query.capture_names()[capture.index as usize];
            match capture_name {
                "prefix" => prefix_node = Some(capture.node),
                "number" => number_node = Some(capture.node),
                _ => {}
            }
        }

        if let (Some(prefix), Some(number)) = (prefix_node, number_node)
            && let Some(line) = extract_line_components(index, prefix, number)
        {
            formateable_lines.push(line);
        }
    }

    Ok(formateable_lines)
}

/// Extracts the components (prefix, number, rest) from a single line
fn extract_line_components(
    index: &crate::treesitter_utils::LineIndex,
    prefix_node: tree_sitter::Node,
    number_node: tree_sitter::Node,
) -> Option<FormatableLine> {
    let line_num = prefix_node.start_position().row;
    let text = index.text();

    // Byte offsets straight from the tree; the index only supplies the line
    // bounds. Everything here is a borrow, not a rope walk.
    let line_start = index.line_start_byte(line_num);
    let line_end = index.line_end_byte(line_num);
    let prefix_end = prefix_node.end_byte().min(text.len());
    let number_start = number_node.start_byte().min(text.len());
    let number_end = number_node.end_byte().min(text.len());

    // The rebuilt line is prefix + padding + number + rest, so anything
    // sitting between the prefix and the number would be silently deleted by
    // the rebuild. The grammar's error recovery produces exactly such
    // (prefix, number) pairs on lines broken mid-edit.
    if number_node.start_position().row != line_num
        || number_start < prefix_end
        || !text.is_char_boundary(prefix_end)
        || !text.is_char_boundary(number_start)
        || !text.is_char_boundary(number_end)
        || text[prefix_end..number_start]
            .chars()
            .any(|c| !c.is_whitespace())
    {
        return None;
    }

    Some(FormatableLine {
        line_num,
        prefix: text[line_start..prefix_end].to_string(),
        number: text[number_start..number_end].to_string(),
        rest: if number_end < line_end {
            text[number_end..line_end].to_string()
        } else {
            String::new()
        },
    })
}

/// Calculates formatting configuration including maximum widths and overrides
pub(super) fn calculate_format_config(
    formateable_lines: &[FormatableLine],
    user_config: &crate::config::FormattingConfig,
) -> FormatConfig {
    // Calculate maximum widths across all lines (bean-format behavior)
    let max_prefix_width = formateable_lines
        .iter()
        .map(|line| line.prefix.trim_end().len())
        .max()
        .unwrap_or(0);

    let max_number_width = formateable_lines
        .iter()
        .map(|line| line.number.len())
        .max()
        .unwrap_or(0);

    // Use configuration overrides if provided (like bean-format's -w and -W options)
    let final_prefix_width = user_config.prefix_width.unwrap_or(max_prefix_width);
    let final_num_width = user_config.num_width.unwrap_or(max_number_width);

    FormatConfig {
        final_prefix_width,
        final_num_width,
    }
}
