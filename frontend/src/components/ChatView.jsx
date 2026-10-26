/**
 * ChatView Component — RepoPilot LangGraph Conversational Agent
 *
 * A full-featured conversational interface with:
 * - Durable session threads persisted in PostgreSQL / Checkpointer
 * - Session switcher & "+ New Chat" session creation in sidebar
 * - Real-time SSE token streaming
 * - Active tool invocation status badges (file tree inspection, file reading)
 * - Markdown & code citation rendering
 * - Keyboard shortcuts (Enter to send, Shift+Enter for newline)
 * - Example quick-start questions
 */

import { useEffect, useRef, useState } from "react";
import CitationCard from "./CitationCard";
import {
  createChatSession,
  deleteChatSession,
  getChatHistory,
  listChatSessions,
  streamChatMessage,
} from "../services/api";

/** Example questions shown in the sidebar to help users get started. */
const EXAMPLE_QUESTIONS = [
  "Where is authentication handled in this repository?",
  "How is the database session and connection pool managed?",
  "What tools and dependencies are used across this project?",
  "Inspect the project file structure and give me an architectural overview.",
  "What happens when repository indexing fails or errors out?",
  "How are environment variables and configurations validated?",
];

/** Friendly human-readable label for backend tools. */
function formatToolName(toolName) {
  switch (toolName) {
    case "inspect_file_tree":
      return "Inspecting repository file tree…";
    case "read_source_file":
      return "Reading source file…";
    case "fetch_repo_metadata":
      return "Fetching repository metadata…";
    default:
      return `Executing tool: ${toolName}…`;
  }
}

/**
 * Renders a single AI-generated answer bubble, including optional citations.
 */
function AssistantBubble({ content, citations, model, isStreaming = false }) {
  const [citationsOpen, setCitationsOpen] = useState(true);

  return (
    <div className={`chat-bubble chat-bubble--assistant ${isStreaming ? "chat-bubble--streaming" : ""}`}>
      {/* Answer text — preserve newlines */}
      <div className="chat-bubble-text">
        {content ? (
          content.split("\n").map((line, i, arr) => (
            <span key={i}>
              {line}
              {i < arr.length - 1 && <br />}
            </span>
          ))
        ) : (
          <span className="text-muted">Thinking…</span>
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
                <CitationCard key={citation.chunk_id || i} citation={citation} index={i + 1} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Animated typing indicator shown while the backend is beginning to process a request.
 */
function ThinkingBubble() {
  return (
    <div className="chat-bubble chat-bubble--assistant chat-bubble--thinking">
      <div className="chat-thinking">
        <span className="chat-thinking-dot" />
        <span className="chat-thinking-dot" />
        <span className="chat-thinking-dot" />
        <span className="chat-thinking-label">
          Analyzing question and preparing tools…
        </span>
      </div>
    </div>
  );
}

/**
 * Tool status pill displayed during execution.
 */
function ToolExecutionBadge({ tool, status }) {
  const isRunning = status === "running";
  return (
    <div className={`chat-tool-pill ${isRunning ? "chat-tool-pill--running" : "chat-tool-pill--completed"}`}>
      <span className="chat-tool-icon">{isRunning ? "⚡" : "✓"}</span>
      <span className="chat-tool-text">{formatToolName(tool)}</span>
      {isRunning && <span className="chat-tool-spinner" />}
    </div>
  );
}

/**
 * Main ChatView component.
 */
export default function ChatView({ repositories, initialRepoId }) {
  const [selectedRepoId, setSelectedRepoId] = useState(
    initialRepoId ? String(initialRepoId) : ""
  );

  // Chat sessions state
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState("");

  // Conversation & streaming state
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState("");
  const [activeTools, setActiveTools] = useState([]);
  const [error, setError] = useState("");
  const loadingSessions = Boolean(selectedRepoId && sessions.length === 0 && !error);

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const accumulatedStreamRef = useRef("");

  // Handle external repo switch
  const [prevInitialRepoId, setPrevInitialRepoId] = useState(initialRepoId);
  if (initialRepoId !== prevInitialRepoId) {
    setPrevInitialRepoId(initialRepoId);
    if (initialRepoId) {
      setSelectedRepoId(String(initialRepoId));
      setSessions([]);
      setCurrentSessionId("");
      setMessages([]);
      setError("");
    }
  }

  // Only completed (indexed) repositories can be queried
  const indexedRepos = repositories.filter((r) => r.status === "completed");
  const selectedRepo = repositories.find(
    (r) => r.id === Number(selectedRepoId)
  );

  // Auto-scroll to the latest message whenever messages, asking, or streaming deltas change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking, streamingMessage, activeTools]);

  // Load chat sessions when selected repository changes
  useEffect(() => {
    if (!selectedRepoId) {
      return;
    }

    let isMounted = true;

    listChatSessions(selectedRepoId)
      .then(async (sessList) => {
        if (!isMounted) return;
        if (sessList && sessList.length > 0) {
          setSessions(sessList);
          const firstSessionId = sessList[0].id;
          setCurrentSessionId(firstSessionId);
          // Load history for the primary session
          try {
            const history = await getChatHistory(firstSessionId);
            if (isMounted) {
              setMessages(history.messages || []);
            }
          } catch {
            if (isMounted) setMessages([]);
          }
        } else {
          // Auto-create initial session
          try {
            const newSession = await createChatSession(selectedRepoId, "New Conversation");
            if (isMounted) {
              setSessions([newSession]);
              setCurrentSessionId(newSession.id);
              setMessages([]);
            }
          } catch {
            if (isMounted) setSessions([]);
          }
        }
      })
      .catch((err) => {
        if (isMounted) setError(err.message || "Failed to load chat sessions.");
      });

    return () => {
      isMounted = false;
    };
  }, [selectedRepoId]);

  /** Switch active session thread. */
  async function handleSelectSession(sessionId) {
    if (sessionId === currentSessionId || asking) return;
    setCurrentSessionId(sessionId);
    setError("");
    setMessages([]);

    try {
      const history = await getChatHistory(sessionId);
      setMessages(history.messages || []);
    } catch (err) {
      setError(err.message || "Failed to load session history.");
    }
  }

  /** Create a new conversation session. */
  async function handleCreateSession() {
    if (!selectedRepoId || asking) return;
    setError("");

    try {
      const newSession = await createChatSession(selectedRepoId, "New Conversation");
      setSessions((prev) => [newSession, ...prev]);
      setCurrentSessionId(newSession.id);
      setMessages([]);
    } catch (err) {
      setError(err.message || "Failed to create new conversation.");
    }
  }

  /** Delete a conversation session. */
  async function handleDeleteSession(event, sessionId) {
    event.stopPropagation();
    if (asking) return;

    try {
      await deleteChatSession(sessionId);
      const remaining = sessions.filter((s) => s.id !== sessionId);
      setSessions(remaining);

      if (sessionId === currentSessionId) {
        if (remaining.length > 0) {
          handleSelectSession(remaining[0].id);
        } else {
          // No sessions left, create a fresh one
          handleCreateSession();
        }
      }
    } catch (err) {
      setError(err.message || "Failed to delete chat session.");
    }
  }

  /** Handle repository selection change. */
  function handleRepoChange(event) {
    const nextRepoId = event.target.value;
    setSelectedRepoId(nextRepoId);
    setSessions([]);
    setCurrentSessionId("");
    setMessages([]);
    setError("");
  }

  /** Inject an example question into the textarea. */
  function applyExampleQuestion(q) {
    setQuestion(q);
    textareaRef.current?.focus();
  }

  /** Submit message and stream response via SSE. */
  async function handleAsk(event) {
    event.preventDefault();

    const trimmed = question.trim();
    if (!trimmed || !selectedRepoId || asking) return;

    // Ensure session exists
    let activeSessionId = currentSessionId;
    if (!activeSessionId) {
      try {
        const newSession = await createChatSession(selectedRepoId, "New Conversation");
        setSessions((prev) => [newSession, ...prev]);
        setCurrentSessionId(newSession.id);
        activeSessionId = newSession.id;
      } catch (err) {
        setError(err.message || "Failed to create conversation session.");
        return;
      }
    }

    setQuestion("");
    setError("");
    setAsking(true);
    setStreamingMessage("");
    setActiveTools([]);
    accumulatedStreamRef.current = "";

    // Append user message to thread
    setMessages((prev) => [
      ...prev,
      { role: "user", content: trimmed },
    ]);

    await streamChatMessage(selectedRepoId, activeSessionId, trimmed, {
      onToken: (delta) => {
        accumulatedStreamRef.current += delta;
        setStreamingMessage(accumulatedStreamRef.current);
      },
      onToolStart: (toolName) => {
        setActiveTools((prev) => [
          ...prev.filter((t) => t.tool !== toolName),
          { tool: toolName, status: "running" },
        ]);
      },
      onToolEnd: (toolName) => {
        setActiveTools((prev) =>
          prev.map((t) => (t.tool === toolName ? { ...t, status: "completed" } : t))
        );
      },
      onError: (err) => {
        setError(err.message || "Generation error encountered.");
        setAsking(false);
      },
      onDone: () => {
        const finalContent = accumulatedStreamRef.current;
        if (finalContent) {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: finalContent,
            },
          ]);
        }
        setStreamingMessage("");
        setActiveTools([]);
        setAsking(false);

        // Refresh session list to update auto-generated session titles
        listChatSessions(selectedRepoId)
          .then(setSessions)
          .catch(() => {});
      },
    });
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

        {/* Chat Sessions List */}
        {selectedRepoId && (
          <div className="chat-sessions-section">
            <div className="chat-sessions-header">
              <span className="chat-sessions-title">Conversations</span>
              <button
                type="button"
                className="btn btn-outline-primary btn-sm chat-new-btn"
                onClick={handleCreateSession}
                disabled={asking}
                title="Create a new conversation session"
              >
                + New
              </button>
            </div>

            {loadingSessions && sessions.length === 0 ? (
              <div className="chat-sessions-loading text-secondary small">
                Loading sessions…
              </div>
            ) : (
              <ul className="chat-sessions-list">
                {sessions.map((sess) => {
                  const isActive = sess.id === currentSessionId;
                  return (
                    <li key={sess.id}>
                      <button
                        type="button"
                        className={`chat-session-item ${isActive ? "chat-session-item--active" : ""}`}
                        onClick={() => handleSelectSession(sess.id)}
                        disabled={asking}
                      >
                        <span className="chat-session-name" title={sess.title}>
                          💬 {sess.title}
                        </span>
                        <span
                          className="chat-session-delete"
                          role="button"
                          tabIndex={0}
                          title="Delete session"
                          onClick={(e) => handleDeleteSession(e, sess.id)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              handleDeleteSession(e, sess.id);
                            }
                          }}
                        >
                          ✕
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
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
                  disabled={!selectedRepoId || asking}
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
                Ask about <strong>{selectedRepo?.full_name}</strong>
              </h3>
              <p className="text-secondary">
                LangGraph agent with tool access to repository metadata, file trees,
                and source files will assist you with streaming responses.
              </p>
            </div>
          )}

          {/* Message list */}
          {messages.map((msg, idx) => {
            if (msg.role === "user") {
              return (
                <div key={msg.id || idx} className="chat-message chat-message--user">
                  <div className="chat-bubble chat-bubble--user">
                    <div className="chat-bubble-text">{msg.content}</div>
                  </div>
                </div>
              );
            }

            if (msg.role === "tool") {
              return (
                <div key={msg.id || idx} className="chat-message chat-message--tool">
                  <div className="chat-tool-result">
                    <span className="chat-tool-icon">🛠</span>
                    <span className="chat-tool-output-preview">
                      Tool execution output received ({msg.content?.length || 0} chars)
                    </span>
                  </div>
                </div>
              );
            }

            // Assistant bubble
            return (
              <div key={msg.id || idx} className="chat-message chat-message--assistant">
                <AssistantBubble
                  content={msg.content}
                  citations={msg.citations}
                  model={msg.model}
                />
              </div>
            );
          })}

          {/* Active Tool Executions */}
          {activeTools.length > 0 && (
            <div className="chat-tools-container">
              {activeTools.map((t, idx) => (
                <ToolExecutionBadge key={idx} tool={t.tool} status={t.status} />
              ))}
            </div>
          )}

          {/* Streaming Assistant Bubble */}
          {asking && streamingMessage && (
            <div className="chat-message chat-message--assistant">
              <AssistantBubble content={streamingMessage} isStreaming={true} />
            </div>
          )}

          {/* Animated typing indicator when waiting before first token */}
          {asking && !streamingMessage && activeTools.length === 0 && (
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
            <strong>LangGraph</strong> + <strong>pgvector</strong>
          </p>
        </div>
      </div>
    </div>
  );
}
