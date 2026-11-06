import type { ChatSession, Repository } from "../../types";
import { EXAMPLE_QUESTIONS } from "./chatUtils";

function PlusIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function StarIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
  );
}

function ChatSessionIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

interface ChatSidebarProps {
  repositories: Repository[];
  selectedRepoId: string;
  onRepoChange: (repoId: string) => void;
  sessions: ChatSession[];
  currentSessionId: string;
  loadingSessions: boolean;
  asking: boolean;
  onSelectSession: (sessionId: string) => void;
  onCreateSession: () => void;
  onDeleteSession: (e: React.MouseEvent, sessionId: string) => void;
  onSelectExample: (question: string) => void;
}

export default function ChatSidebar({
  repositories,
  selectedRepoId,
  onRepoChange,
  sessions,
  currentSessionId,
  loadingSessions,
  asking,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  onSelectExample,
}: ChatSidebarProps) {
  const indexedRepos = repositories.filter((r) => r.status === "completed");
  const selectedRepo = repositories.find((r) => r.id === Number(selectedRepoId));

  return (
    <aside className="chat-sidebar" aria-label="Chat sidebar">
      <div className="chat-sidebar-inner">
        {/* Repository selector */}
        <p className="sidebar-section-label">Repository</p>

        {indexedRepos.length === 0 ? (
          <p style={{ fontSize: "13px", color: "var(--color-text-secondary)", lineHeight: "1.55" }}>
            No indexed repositories yet.{" "}
            <strong>Index a repository from the Dashboard first.</strong>
          </p>
        ) : (
          <div className="chat-repo-select-wrapper">
            <select
              className="rp-select"
              value={selectedRepoId}
              onChange={(e) => onRepoChange(e.target.value)}
              aria-label="Select repository to query"
            >
              <option value="">Select a repository</option>
              {indexedRepos.map((repo) => (
                <option key={repo.id} value={repo.id}>
                  {repo.full_name}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Selected repo meta */}
        {selectedRepo && (
          <div className="chat-repo-meta">
            {selectedRepo.language && (
              <span className="chat-repo-lang-badge">{selectedRepo.language}</span>
            )}
            <span className="chat-repo-stars-badge">
              <span className="chat-stars-icon" aria-hidden="true"><StarIcon /></span>
              {(selectedRepo.stars ?? 0).toLocaleString()}
            </span>
          </div>
        )}

        {/* Chat Sessions */}
        {selectedRepoId && (
          <div className="chat-sessions-section">
            <div className="chat-sessions-header">
              <span className="sidebar-section-label" style={{ marginBottom: 0 }}>
                Conversations
              </span>
              <button
                type="button"
                className="rp-btn rp-btn-ghost"
                onClick={onCreateSession}
                disabled={asking}
                title="Create a new conversation session"
                style={{ height: "26px", fontSize: "11px", fontWeight: 600, gap: "3px", padding: "0 8px" }}
              >
                <PlusIcon />
                New
              </button>
            </div>

            {loadingSessions ? (
              <div className="chat-sessions-loading">Loading sessions…</div>
            ) : (
              <ul className="chat-sessions-list" role="listbox" aria-label="Conversation sessions">
                {sessions.map((sess) => {
                  const isActive = sess.id === currentSessionId;
                  return (
                    <li key={sess.id} role="option" aria-selected={isActive} style={{ position: "relative" }}>
                      <div
                        className={`chat-session-item-row${isActive ? " chat-session-item-row--active" : ""}`}
                        style={{ display: "flex", alignItems: "center", width: "100%" }}
                      >
                        <button
                          type="button"
                          className={`chat-session-item${isActive ? " chat-session-item--active" : ""}`}
                          onClick={() => onSelectSession(sess.id)}
                          disabled={asking}
                          style={{ flex: 1, paddingRight: "30px", textAlign: "left" }}
                        >
                          <span className="chat-session-icon" aria-hidden="true">
                            <ChatSessionIcon />
                          </span>
                          <span className="chat-session-name" title={sess.title}>
                            {sess.title}
                          </span>
                        </button>
                        <button
                          type="button"
                          className="chat-session-delete-btn"
                          title={`Delete session: ${sess.title}`}
                          aria-label={`Delete session: ${sess.title}`}
                          onClick={(e) => onDeleteSession(e, sess.id)}
                          disabled={asking}
                          style={{
                            position: "absolute",
                            right: "6px",
                            background: "transparent",
                            border: "none",
                            cursor: "pointer",
                            padding: "4px",
                            display: "flex",
                            alignItems: "center",
                            color: "var(--color-text-tertiary)",
                          }}
                        >
                          <XIcon />
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        )}

        {/* Example question pills */}
        <div className="chat-examples-section">
          <p className="sidebar-section-label">Try asking</p>
          <ul className="chat-example-list" aria-label="Example questions">
            {EXAMPLE_QUESTIONS.map((q) => (
              <li key={q}>
                <button
                  type="button"
                  className="chat-example-pill"
                  onClick={() => onSelectExample(q)}
                  disabled={!selectedRepoId || asking}
                  title={q}
                >
                  {q}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </aside>
  );
}
