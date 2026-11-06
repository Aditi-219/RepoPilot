/**
 * AssistantBubble Component
 *
 * Renders an AI response with full Markdown support (GitHub-Flavored Markdown),
 * syntax-styled code blocks with copy-to-clipboard, streaming cursor indicator,
 * and collapsible code citation panels.
 */

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { CodeCitation } from "../../types";
import CitationCard from "../CitationCard";

function DocumentIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}

function ChevronDownIcon({ open }: { open: boolean }) {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`chat-citations-chevron${open ? " chat-citations-chevron--open" : ""}`}
      aria-hidden="true"
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

function CopyIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function CodeBlock({ children, className }: { children: React.ReactNode; className?: string }) {
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || "");
  const language = match ? match[1] : "";
  const codeContent = String(children).replace(/\n$/, "");

  const handleCopy = () => {
    navigator.clipboard.writeText(codeContent).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {});
  };

  return (
    <div className="chat-code-block" style={{ margin: "10px 0", borderRadius: "8px", overflow: "hidden", border: "1px solid var(--color-border)" }}>
      <div
        className="chat-code-header"
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "5px 12px",
          background: "var(--color-surface-dim)",
          borderBottom: "1px solid var(--color-border)",
          fontSize: "11px",
          fontFamily: "var(--font-mono)",
          color: "var(--color-text-secondary)",
        }}
      >
        <span>{language || "code"}</span>
        <button
          type="button"
          onClick={handleCopy}
          className="rp-btn rp-btn-ghost"
          style={{ height: "22px", padding: "0 6px", fontSize: "11px", gap: "4px" }}
          aria-label="Copy code block"
        >
          {copied ? <CheckIcon /> : <CopyIcon />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre style={{ margin: 0, padding: "12px", background: "var(--color-surface)", overflowX: "auto" }}>
        <code className={className} style={{ fontFamily: "var(--font-mono)", fontSize: "13px", lineHeight: "1.5" }}>
          {codeContent}
        </code>
      </pre>
    </div>
  );
}

interface AssistantBubbleProps {
  id?: string;
  content: string;
  citations?: CodeCitation[];
  model?: string;
  isStreaming?: boolean;
}

export default function AssistantBubble({
  id = "bubble",
  content,
  citations,
  model,
  isStreaming = false,
}: AssistantBubbleProps) {
  const [citationsOpen, setCitationsOpen] = useState(true);

  if (!content && !isStreaming) {
    return null;
  }

  const citationControlsId = `chat-citations-list-${id}`;

  return (
    <div className={`chat-bubble chat-bubble--assistant${isStreaming ? " chat-bubble--streaming" : ""}`}>
      {/* Markdown-rendered answer text */}
      <div className="chat-bubble-text chat-markdown-body">
        {content ? (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              // Custom code block renderer with copy button
              code({ inline, className, children, ...props }: any) {
                if (!inline) {
                  return <CodeBlock className={className}>{children}</CodeBlock>;
                }
                return (
                  <code
                    className="chat-inline-code"
                    style={{
                      padding: "2px 5px",
                      borderRadius: "4px",
                      background: "var(--color-surface-dim)",
                      border: "1px solid var(--color-border)",
                      fontFamily: "var(--font-mono)",
                      fontSize: "12px",
                    }}
                    {...props}
                  >
                    {children}
                  </code>
                );
              },
            }}
          >
            {content}
          </ReactMarkdown>
        ) : (
          <span style={{ color: "var(--color-text-tertiary)", fontStyle: "italic" }}>
            Thinking
          </span>
        )}
        {isStreaming && <span className="chat-streaming-cursor" aria-hidden="true" />}
      </div>

      {/* Citations panel if available */}
      {citations && citations.length > 0 && (
        <div className="chat-citations">
          <button
            className="chat-citations-toggle"
            onClick={() => setCitationsOpen((o) => !o)}
            type="button"
            aria-expanded={citationsOpen}
            aria-controls={citationControlsId}
          >
            <span className="chat-citations-label">
              <span aria-hidden="true"><DocumentIcon /></span>
              <span>{citations.length} code {citations.length === 1 ? "snippet" : "snippets"} referenced</span>
              <span className="chat-citations-count">{citations.length}</span>
            </span>
            {model && model !== "none" && model !== "error" && (
              <span className="chat-model-badge">{model}</span>
            )}
            <ChevronDownIcon open={citationsOpen} />
          </button>

          {citationsOpen && (
            <div className="chat-citations-list" id={citationControlsId}>
              {citations.map((citation, i) => (
                <CitationCard
                  key={citation.chunk_id || `${id}-${i}`}
                  citation={citation}
                  index={i + 1}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
