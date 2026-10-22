import { useState } from "react";

/**
 * CitationCard Component
 *
 * Displays a single retrieved code chunk as a collapsible card with:
 * - File path badge (clickable header)
 * - Line range badge (L{start}–{end})
 * - Language tag
 * - Syntax-highlighted code block (via CSS pre/code)
 *
 * The first two citations are auto-expanded for immediate visibility.
 *
 * @param {Object} props
 * @param {Object} props.citation - Code citation with file_path, start_line, end_line, content, language
 * @param {number} props.index - 1-based display index shown on the card
 */
export default function CitationCard({ citation, index }) {
  // Auto-expand the first two citations so users see code immediately
  const [expanded, setExpanded] = useState(index <= 2);

  return (
    <div className="citation-card">
      <button
        className="citation-header"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
        type="button"
      >
        <span className="citation-index">{index}</span>

        <span className="citation-filepath" title={citation.file_path}>
          {citation.file_path}
        </span>

        <span className="citation-lines">
          L{citation.start_line}–{citation.end_line}
        </span>

        {citation.language && (
          <span className="citation-lang">{citation.language}</span>
        )}

        <span className="citation-toggle" aria-hidden="true">
          {expanded ? "▲" : "▼"}
        </span>
      </button>

      {expanded && (
        <div className="citation-body">
          <div className="citation-meta">
            <span className="citation-meta-path">{citation.file_path}</span>
            <span className="citation-meta-range">
              Lines {citation.start_line}–{citation.end_line}
            </span>
          </div>
          <pre className="citation-code">
            <code>{citation.content}</code>
          </pre>
        </div>
      )}
    </div>
  );
}
