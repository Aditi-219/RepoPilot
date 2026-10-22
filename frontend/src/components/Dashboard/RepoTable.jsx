/**
 * RepoTable Component
 *
 * Renders the table of registered repositories in the user's workspace.
 *
 * Capabilities:
 * - Direct external links to GitHub
 * - Live lifecycle status badges
 * - "Chat ✨" button for completed repositories
 * - "Index" button for pending or failed repositories
 * - Dev controls dropdown to test state transitions
 * - Delete repository action
 * - Manual refresh button
 */

import StatusBadge from "../common/StatusBadge";

/**
 * Format an ISO date string into a localized readable date/time.
 */
function formatDate(value) {
  if (!value) return "—";
  return new Date(value).toLocaleString();
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
}) {
  return (
    <section className="card border-0 shadow-sm">
      <div className="card-body p-4">
        {/* Table header & refresh action */}
        <div className="d-flex flex-wrap justify-content-between align-items-center gap-3 mb-4">
          <div>
            <h2 className="h5 fw-bold mb-1">Your repositories</h2>
            <p className="text-secondary small mb-0">
              Registered GitHub projects ready for indexing and Q&amp;A
            </p>
          </div>
          <button
            type="button"
            className="btn btn-outline-secondary btn-sm"
            onClick={onRefresh}
          >
            Refresh
          </button>
        </div>

        {/* Loading and Empty States */}
        {loading ? (
          <p className="text-secondary">Loading repositories…</p>
        ) : repositories.length === 0 ? (
          <div className="empty-state text-center py-5">
            <h3 className="h6 fw-semibold">No repositories yet</h3>
            <p className="text-secondary mb-0">
              Add your first public GitHub repository above to get started.
            </p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="table align-middle">
              <thead>
                <tr>
                  <th>Repository</th>
                  <th>Language</th>
                  <th>Stars</th>
                  <th>Status</th>
                  <th>Added</th>
                  <th>Actions</th>
                  <th>Dev Controls</th>
                  <th>Delete</th>
                </tr>
              </thead>
              <tbody>
                {repositories.map((repo) => (
                  <tr key={repo.id}>
                    {/* Repository title and description */}
                    <td>
                      <a
                        href={repo.url}
                        target="_blank"
                        rel="noreferrer"
                        className="repo-link fw-semibold"
                      >
                        {repo.full_name}
                      </a>
                      <div className="text-secondary small repo-description">
                        {repo.description || "No description available"}
                      </div>
                    </td>

                    {/* Language & Stars */}
                    <td>{repo.language || "—"}</td>
                    <td>{repo.stars.toLocaleString()}</td>

                    {/* Status Badge */}
                    <td>
                      <StatusBadge status={repo.status} />
                    </td>

                    {/* Creation timestamp */}
                    <td className="small text-secondary">
                      {formatDate(repo.created_at)}
                    </td>

                    {/* Primary Action Button (Chat or Index) */}
                    <td>
                      {repo.status === "completed" ? (
                        <button
                          type="button"
                          className="btn btn-sm btn-chat me-2"
                          onClick={() => onOpenChat(repo.id)}
                          title="Open Chat to ask questions about this codebase"
                        >
                          Chat ✨
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="btn btn-sm btn-index"
                          onClick={() => onTriggerIndexing(repo.id, repo.full_name)}
                          disabled={
                            repo.status === "indexing" || indexingIds[repo.id]
                          }
                          title="Start indexing this repository"
                        >
                          {repo.status === "indexing" || indexingIds[repo.id]
                            ? "Indexing…"
                            : "Index"}
                        </button>
                      )}
                    </td>

                    {/* Dev Controls: Manual status override */}
                    <td>
                      <select
                        className="form-select form-select-sm"
                        value={repo.status}
                        aria-label={`Status override for ${repo.full_name}`}
                        onChange={(event) =>
                          onChangeStatus(repo.id, event.target.value)
                        }
                      >
                        <option value="pending">Pending</option>
                        <option value="indexing">Indexing</option>
                        <option value="completed">Completed</option>
                        <option value="failed">Failed</option>
                      </select>
                    </td>

                    {/* Delete Action */}
                    <td>
                      <button
                        type="button"
                        className="btn btn-outline-danger btn-sm"
                        onClick={() => onDeleteRepo(repo.id, repo.full_name)}
                        title="Delete repository"
                      >
                        🗑️
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
