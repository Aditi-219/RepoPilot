import { useState } from "react";
import type { CodeCitation } from "../types";

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={`citation-toggle${open ? " citation-toggle--open" : ""}`}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

function CopyIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

interface CitationCardProps {
  citation: CodeCitation;
  index: number;
}

export default function CitationCard({ citation, index }: CitationCardProps) {
  const [expanded, setExpanded] = useState(index <= 2);
  const [copied, setCopied] = useState(false);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(citation.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {});
  };

  return (
    <div className="citation-card">
      <button
        className="citation-header"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
        type="button"
        aria-controls={`citation-body-${index}`}
      >
        <span className="citation-index" aria-label={`Citation ${index}`}>
          {index}
        </span>

        <span className="citation-filepath" title={citation.file_path}>
          {citation.file_path}
        </span>

        <span className="citation-lines" aria-label={`Lines ${citation.start_line} to ${citation.end_line}`}>
          L{citation.start_line}–{citation.end_line}
        </span>

        {citation.language && (
          <span className="citation-lang">{citation.language}</span>
        )}

        <ChevronIcon open={expanded} />
      </button>

      {expanded && (
        <div className="citation-body" id={`citation-body-${index}`}>
          <div className="citation-meta" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <span className="citation-meta-path">{citation.file_path}</span>
              <span className="citation-meta-range">
                Lines {citation.start_line}–{citation.end_line}
              </span>
            </div>
            <button
              type="button"
              className="rp-btn rp-btn-ghost"
              onClick={handleCopy}
              title="Copy snippet to clipboard"
              aria-label="Copy snippet"
              style={{ height: "24px", fontSize: "11px", padding: "0 6px", gap: "4px" }}
            >
              {copied ? <CheckIcon /> : <CopyIcon />}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <pre className="citation-code">
            <code>{citation.content}</code>
          </pre>
        </div>
      )}
    </div>
  );
}
