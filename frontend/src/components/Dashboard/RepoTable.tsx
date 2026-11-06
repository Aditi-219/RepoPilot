/**
 * RepoTable Component
 *
 * Renders the table of registered repositories in the user's workspace.
 *
 * Capabilities:
 * - Direct external links to GitHub
 * - Live lifecycle status badges with animated indicator
 * - "Chat" button for completed repositories
 * - "Index" button for pending or failed repositories
 * - Dev controls toggle to test state transitions safely
 * - Delete repository action with accessible icon button
 * - Manual refresh button
 * - Skeleton loading and empty states
 */

import { useState } from "react";
import type { Repository, RepositoryStatus } from "../../types";
import StatusBadge from "../common/StatusBadge";

/** Format an ISO date string into a localized readable date safely. */
function formatDate(value?: string | null): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (isNaN(parsed.getTime())) return "—";
  return parsed.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function ExternalLinkIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="23 4 23 10 17 10" />
      <polyline points="1 20 1 14 7 14" />
      <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
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

function ChatBubbleIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function IndexSpinner() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ animation: "rp-spin 0.8s linear infinite" }}>
      <line x1="12" y1="2" x2="12" y2="6" />
      <line x1="12" y1="18" x2="12" y2="22" />
      <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
      <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
      <line x1="2" y1="12" x2="6" y2="12" />
      <line x1="18" y1="12" x2="22" y2="12" />
      <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
      <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
    </svg>
  );
}

function WrenchIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
    </svg>
  );
}

function SkeletonRows() {
  return (
    <div className="skeleton-loading">
      {[1, 2, 3].map((n) => (
        <div key={n} className="skeleton-row">
          <div className="skeleton-block" style={{ width: "40%", height: "14px" }} />
          <div className="skeleton-block" style={{ width: "12%", height: "14px" }} />
          <div className="skeleton-block" style={{ width: "8%", height: "14px" }} />
          <div className="skeleton-block" style={{ width: "14%", height: "22px", borderRadius: "999px" }} />
          <div className="skeleton-block" style={{ width: "15%", height: "14px" }} />
          <div className="skeleton-block" style={{ width: "10%", height: "28px" }} />
        </div>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-state-icon">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M3 3h18v18H3z" />
          <path d="M9 9h6M9 12h6M9 15h4" />
        </svg>
      </div>
      <p className="empty-state-title">No repositories yet</p>
      <p className="empty-state-subtitle">
        Add your first public GitHub repository above to register it for indexing and AI Q&amp;A.
      </p>
    </div>
  );
}

interface RepoTableProps {
  repositories: Repository[];
  loading: boolean;
  indexingIds: Record<number, boolean>;
  onRefresh: () => void | Promise<void>;
  onTriggerIndexing: (id: number, fullName: string) => void | Promise<void>;
  onChangeStatus: (id: number, newStatus: RepositoryStatus) => void | Promise<void>;
  onDeleteRepo: (id: number, fullName: string) => void | Promise<void>;
  onOpenChat: (id: number) => void;
}

export default function RepoTable({
  repositories,
  loading,
  indexingIds,
  onRefresh,
  onTriggerIndexing,
  onChangeStatus,
  onDeleteRepo,
  onOpenChat,
}: RepoTableProps) {
  const [showDevControls, setShowDevControls] = useState(false);

  const isIndexingActive = (repo: Repository) =>
    repo.status === "indexing" || Boolean(indexingIds[repo.id]);

  return (
    <section className="rp-card repo-table-card" aria-label="Your repositories">
      {/* Header */}
      <div className="repo-table-header">
        <div>
          <h2 className="repo-table-title">Your repositories</h2>
          <p className="repo-table-subtitle">
            Registered GitHub projects ready for indexing and Q&amp;A
          </p>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <button
            type="button"
            className={`rp-btn ${showDevControls ? "rp-btn-primary" : "rp-btn-ghost"}`}
            onClick={() => setShowDevControls((prev) => !prev)}
            aria-label="Toggle developer controls"
            title="Toggle dev state override controls"
            style={{ height: "34px", fontSize: "12px", paddingInline: "10px", gap: "5px" }}
          >
            <WrenchIcon />
            Dev Mode
          </button>
          <button
            type="button"
            className="rp-btn rp-btn-secondary"
            onClick={onRefresh}
            aria-label="Refresh repository list"
            style={{ height: "34px", fontSize: "13px", paddingInline: "14px" }}
          >
            <RefreshIcon />
            Refresh
          </button>
        </div>
      </div>

      {/* Content area: loading | empty | table */}
      {loading ? (
        <SkeletonRows />
      ) : repositories.length === 0 ? (
        <EmptyState />
      ) : (
        <>
          <div className="rp-table-wrapper">
            <table className="rp-table" aria-label="Repositories list">
              <thead>
                <tr>
                  <th scope="col">Repository</th>
                  <th scope="col">Language</th>
                  <th scope="col">Stars</th>
                  <th scope="col">Status</th>
                  <th scope="col">Added</th>
                  <th scope="col">Actions</th>
                  {showDevControls && <th scope="col">Dev Override</th>}
                  <th scope="col">
                    <span className="visually-hidden">Delete</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {repositories.map((repo) => (
                  <tr key={repo.id}>
                    {/* Repository name + description */}
                    <td>
                      <a
                        href={repo.url}
                        target="_blank"
                        rel="noreferrer"
                        className="repo-name-link"
                        title={`Open ${repo.full_name} on GitHub`}
                      >
                        {repo.full_name}
                        {" "}
                        <ExternalLinkIcon />
                      </a>
                      {repo.description && (
                        <div className="repo-description-text" title={repo.description}>
                          {repo.description}
                        </div>
                      )}
                    </td>

                    {/* Language */}
                    <td>
                      {repo.language ? (
                        <span className="repo-language-badge">
                          <span className="repo-language-dot" aria-hidden="true" />
                          {repo.language}
                        </span>
                      ) : (
                        <span style={{ color: "var(--color-text-tertiary)" }}>—</span>
                      )}
                    </td>

                    {/* Stars */}
                    <td>
                      <span className="repo-stars">
                        <span className="repo-stars-icon" aria-hidden="true">
                          <StarIcon />
                        </span>
                        <span className="repo-stars-count">{(repo.stars ?? 0).toLocaleString()}</span>
                      </span>
                    </td>

                    {/* Status badge */}
                    <td>
                      <StatusBadge status={repo.status} />
                    </td>

                    {/* Date added */}
                    <td style={{ color: "var(--color-text-secondary)", fontSize: "12px" }}>
                      {formatDate(repo.created_at)}
                    </td>

                    {/* Primary action */}
                    <td>
                      {repo.status === "completed" ? (
                        <button
                          type="button"
                          className="rp-btn rp-btn-chat"
                          onClick={() => onOpenChat(repo.id)}
                          title={`Open AI Chat for ${repo.full_name}`}
                        >
                          <ChatBubbleIcon />
                          Chat
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="rp-btn rp-btn-index"
                          onClick={() => onTriggerIndexing(repo.id, repo.full_name)}
                          disabled={isIndexingActive(repo)}
                          title={isIndexingActive(repo) ? "Indexing in progress" : `Start indexing ${repo.full_name}`}
                        >
                          {isIndexingActive(repo) ? (
                            <>
                              <IndexSpinner />
                              Indexing
                            </>
                          ) : (
                            "Index"
                          )}
                        </button>
                      )}
                    </td>

                    {/* Dev Controls: manual status override (hidden unless Dev Mode toggled) */}
                    {showDevControls && (
                      <td>
                        <select
                          id={`status-override-${repo.id}`}
                          className="rp-select"
                          value={repo.status}
                          onChange={(event) =>
                            onChangeStatus(repo.id, event.target.value as RepositoryStatus)
                          }
                          aria-label={`Status override for ${repo.full_name}`}
                          style={{ width: "115px", height: "30px", fontSize: "11px" }}
                        >
                          <option value="pending">Pending</option>
                          <option value="indexing">Indexing</option>
                          <option value="completed">Completed</option>
                          <option value="failed">Failed</option>
                        </select>
                      </td>
                    )}

                    {/* Delete */}
                    <td>
                      <button
                        type="button"
                        className="rp-btn rp-btn-danger"
                        onClick={() => onDeleteRepo(repo.id, repo.full_name)}
                        title={`Delete ${repo.full_name}`}
                        aria-label={`Delete repository ${repo.full_name}`}
                      >
                        <TrashIcon />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="repo-table-footer">
            <span>{repositories.length} {repositories.length === 1 ? "repository" : "repositories"} registered</span>
            <span>Auto-refreshes during active indexing</span>
          </div>
        </>
      )}
    </section>
  );
}
