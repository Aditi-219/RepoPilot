/**
 * ChatView Component — RepoPilot Phase 3 RAG Chatbot
 *
 * A conversational interface for asking questions about indexed codebases.
 * Uses the backend's hybrid retrieval + cross-encoder reranking + Gemini LLM
 * pipeline to return grounded answers with clickable file-level citations.
 *
 * Features:
 * - Repository selector (only shows indexed/completed repos)
 * - Real-time message thread (user + assistant bubbles)
 * - Per-message citation cards with expandable code blocks
 * - Typing indicator while generating
 * - Keyboard shortcut: Enter to send, Shift+Enter for newline
 * - Example question pills for quick start
 */

import { useEffect, useRef, useState } from "react";
import CitationCard from "./CitationCard";

// API base URL — mirrors the setting in App.jsx
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api";

/**
 * Shared API helper — throws on non-2xx with a user-readable message.
 *
 * @param {string} url
 * @param {RequestInit} [options]
 * @returns {Promise<any>}
 */
async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(
      typeof data.detail === "string" ? data.detail : "Request failed."
    );
  }
  return data;
}

/** Example questions shown in the sidebar to help users get started. */
const EXAMPLE_QUESTIONS = [
  "Where is authentication handled?",
  "Why might this API return a 404?",
  "How is the database session managed?",
  "What happens when indexing fails?",
  "How are code chunks embedded?",
  "Where are environment variables configured?",
];

/**
 * Renders a single AI-generated answer bubble, including its citations.
 *
 * @param {Object} props
 * @param {string} props.content - The answer text
 * @param {Array}  props.citations - Array of code citation objects
 * @param {string} props.model - Model name used (e.g. 'gemini-2.0-flash')
 */
function AssistantBubble({ content, citations, model }) {
  const [citationsOpen, setCitationsOpen] = useState(true);

  return (
    <div className="chat-bubble chat-bubble--assistant">
      {/* Answer text — preserve newlines */}
      <div className="chat-bubble-text">
        {content.split("\n").map((line, i, arr) => (
          <span key={i}>
            {line}
            {i < arr.length - 1 && <br />}
          </span>
        ))}
      </div>

      {/* Citations panel */}
      {citations && citations.length > 0 && (
        <div className="chat-citations">
          <button
            className="chat-citations-toggle"
            onClick={() => setCitationsOpen((o) => !o)}
            type="button"
          >
            <span>
              📄 {citations.length} code{" "}
              {citations.length === 1 ? "snippet" : "snippets"} referenced
            </span>
            {model && model !== "none" && model !== "error" && (
              <span className="chat-model-badge">{model}</span>
            )}
            <span className="chat-citations-chevron">
              {citationsOpen ? "▲" : "▼"}
            </span>
          </button>

          {citationsOpen && (
            <div className="chat-citations-list">
              {citations.map((citation, i) => (
                <CitationCard key={citation.chunk_id} citation={citation} index={i + 1} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Animated typing indicator shown while the backend is processing a request.
 */
function ThinkingBubble() {
  return (
    <div className="chat-bubble chat-bubble--assistant chat-bubble--thinking">
      <div className="chat-thinking">
        <span className="chat-thinking-dot" />
        <span className="chat-thinking-dot" />
        <span className="chat-thinking-dot" />
        <span className="chat-thinking-label">
          Searching codebase and generating answer…
        </span>
      </div>
    </div>
  );
}

/**
 * Main ChatView component.
 *
 * @param {Object}   props
 * @param {Array}    props.repositories - Full list of repos from the parent App
 * @param {number|string} [props.initialRepoId] - Optionally pre-selected repository ID
 */
export default function ChatView({ repositories, initialRepoId }) {
  const [selectedRepoId, setSelectedRepoId] = useState(
    initialRepoId ? String(initialRepoId) : ""
  );
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  // Sync selectedRepoId if an initialRepoId prop is passed or updated
  useEffect(() => {
    if (initialRepoId) {
      // oxlint-disable-next-line react/set-state-in-effect
      setSelectedRepoId(String(initialRepoId));
    }
  }, [initialRepoId]);

  // Auto-scroll to the latest message whenever messages or asking state changes
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  // Only completed (indexed) repositories can be queried
  const indexedRepos = repositories.filter((r) => r.status === "completed");
  const selectedRepo = repositories.find(
    (r) => r.id === Number(selectedRepoId)
  );

  /** Handle repository selection — clears the conversation on switch. */
  function handleRepoChange(event) {
    setSelectedRepoId(event.target.value);
    setMessages([]);
    setError("");
  }

  /** Inject an example question into the textarea. */
  function applyExampleQuestion(q) {
    setQuestion(q);
    textareaRef.current?.focus();
  }

  /** Submit the question to the backend /ask endpoint. */
  async function handleAsk(event) {
    event.preventDefault();

    const trimmed = question.trim();
    if (!trimmed || !selectedRepoId || asking) return;

    setQuestion("");
    setError("");
    setAsking(true);

    // Optimistically append the user message
    setMessages((prev) => [
      ...prev,
      { role: "user", content: trimmed },
    ]);

    try {
      const data = await apiFetch(
        `${API_URL}/repositories/${selectedRepoId}/ask`,
        {
          method: "POST",
          body: JSON.stringify({ question: trimmed, top_k: 8 }),
        }
      );

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          citations: data.citations,
          model: data.model,
        },
      ]);
    } catch (err) {
      setError(err.message || "Failed to get an answer. Is the backend running?");
      // Roll back the optimistically added user message
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setAsking(false);
    }
  }

  /** Allow Enter to submit (Shift+Enter inserts a newline). */
  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleAsk(event);
    }
  }

  return (
    <div className="chat-layout">
      {/* ------------------------------------------------------------------ */}
      {/* Sidebar                                                             */}
      {/* ------------------------------------------------------------------ */}
      <aside className="chat-sidebar">
        <h2 className="chat-sidebar-title">Repository</h2>

        {indexedRepos.length === 0 ? (
          <p className="text-secondary small">
            No indexed repositories yet.{" "}
            <strong>Index a repository from the Dashboard first.</strong>
          </p>
        ) : (
          <select
            className="form-select form-select-sm"
            value={selectedRepoId}
            onChange={handleRepoChange}
            aria-label="Select repository to query"
          >
            <option value="">Select a repository…</option>
            {indexedRepos.map((repo) => (
              <option key={repo.id} value={repo.id}>
                {repo.full_name}
              </option>
            ))}
          </select>
        )}

        {selectedRepo && (
          <div className="chat-repo-meta">
            {selectedRepo.language && (
              <span className="chat-repo-lang">{selectedRepo.language}</span>
            )}
            <span className="chat-repo-stars">
              ★ {selectedRepo.stars.toLocaleString()}
            </span>
          </div>
        )}

        {messages.length > 0 && (
          <button
            className="btn btn-outline-secondary btn-sm mt-3 w-100"
            type="button"
            onClick={() => {
              setMessages([]);
              setError("");
            }}
          >
            Clear conversation
          </button>
        )}

        {/* Example question pills */}
        <div className="chat-examples">
          <p className="chat-examples-title">Try asking</p>
          <ul className="chat-example-list">
            {EXAMPLE_QUESTIONS.map((q) => (
              <li key={q}>
                <button
                  type="button"
                  className="chat-example-pill"
                  onClick={() => applyExampleQuestion(q)}
                  disabled={!selectedRepoId}
                >
                  {q}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </aside>

      {/* ------------------------------------------------------------------ */}
      {/* Main Chat Area                                                       */}
      {/* ------------------------------------------------------------------ */}
      <div className="chat-main">
        {/* Message thread */}
        <div className="chat-thread" role="log" aria-live="polite">
          {/* Empty state — no repo selected */}
          {!selectedRepoId && (
            <div className="chat-empty">
              <div className="chat-empty-icon">💬</div>
              <h3 className="chat-empty-heading">Select a repository to start</h3>
              <p className="text-secondary">
                Choose an indexed repository from the sidebar, then ask any
                question about its codebase.
              </p>
            </div>
          )}

          {/* Empty state — repo selected but no messages yet */}
          {selectedRepoId && messages.length === 0 && !asking && (
            <div className="chat-empty">
              <div className="chat-empty-icon">🔍</div>
              <h3 className="chat-empty-heading">
                Ask about{" "}
                <strong>{selectedRepo?.full_name}</strong>
              </h3>
              <p className="text-secondary">
                I'll search the indexed code, retrieve the most relevant
                snippets, and give you a grounded answer with file citations.
              </p>
            </div>
          )}

          {/* Message list */}
          {messages.map((msg, idx) =>
            msg.role === "user" ? (
              /* User bubble */
              <div key={idx} className="chat-message chat-message--user">
                <div className="chat-bubble chat-bubble--user">
                  <div className="chat-bubble-text">{msg.content}</div>
                </div>
              </div>
            ) : (
              /* Assistant bubble */
              <div key={idx} className="chat-message chat-message--assistant">
                <AssistantBubble
                  content={msg.content}
                  citations={msg.citations}
                  model={msg.model}
                />
              </div>
            )
          )}

          {/* Animated typing indicator */}
          {asking && (
            <div className="chat-message chat-message--assistant">
              <ThinkingBubble />
            </div>
          )}

          {/* Error alert */}
          {error && (
            <div className="alert alert-danger mx-3 mt-2 mb-0" role="alert">
              {error}
            </div>
          )}

          {/* Scroll anchor */}
          <div ref={bottomRef} />
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* Input area                                                        */}
        {/* ---------------------------------------------------------------- */}
        <div className="chat-input-area">
          <form onSubmit={handleAsk} className="chat-input-form">
            <textarea
              ref={textareaRef}
              className="chat-textarea"
              placeholder={
                selectedRepoId
                  ? `Ask about ${selectedRepo?.full_name}…`
                  : "Select a repository first…"
              }
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!selectedRepoId || asking}
              rows={1}
              aria-label="Question input"
            />

            <button
              type="submit"
              className="chat-send-btn"
              disabled={!selectedRepoId || !question.trim() || asking}
              aria-label="Send question"
            >
              {asking ? (
                <span
                  className="spinner-border spinner-border-sm"
                  role="status"
                  aria-hidden="true"
                />
              ) : (
                /* Send icon */
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              )}
            </button>
          </form>

          <p className="chat-hint">
            Enter to send · Shift+Enter for new line · Powered by{" "}
            <strong>pgvector</strong> + <strong>Gemini 2.0 Flash</strong>
          </p>
        </div>
      </div>
    </div>
  );
}
