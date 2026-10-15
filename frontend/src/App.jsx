/**
 * RepoPilot — Frontend Application Dashboard
 *
 * This component acts as the main user interface for Phase 1 of RepoPilot.
 * Key responsibilities:
 * 1. Registering new public GitHub repositories via URL input.
 * 2. Fetching and displaying the list of monitored repositories.
 * 3. Displaying dashboard metrics (total repositories, active indexing, completed).
 * 4. Updating repository processing statuses manually for testing workflows.
 */

import { useCallback, useEffect, useState } from "react";

// API Base URL - reads from environment variable if present, falls back to local FastAPI dev server
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api";

/**
 * StatusBadge Component
 * Displays a color-coded Bootstrap badge depending on the repository's lifecycle state.
 *
 * @param {Object} props
 * @param {'pending' | 'indexing' | 'completed' | 'failed'} props.status
 */
function StatusBadge({ status }) {
  // Mapping each lifecycle state to its corresponding Bootstrap color class
  const badgeClasses = {
    pending: "bg-secondary",
    indexing: "bg-primary",
    completed: "bg-success",
    failed: "bg-danger",
  };

  return (
    <span className={`badge ${badgeClasses[status] ?? "bg-secondary"}`}>
      {status}
    </span>
  );
}

/**
 * Helper function to format ISO date strings into readable localized dates.
 *
 * @param {string | null | undefined} value - ISO date string
 * @returns {string} Formatted date string or fallback em-dash
 */
function formatDate(value) {
  if (!value) return "—";
  return new Date(value).toLocaleString();
}

export default function App() {
  // ---------------------------------------------------------------------------
  // State Management
  // ---------------------------------------------------------------------------
  // URL input field state in the form
  const [url, setUrl] = useState("");

  // List of repositories fetched from the FastAPI backend
  const [repositories, setRepositories] = useState([]);

  // Loading state during initial repository list fetch
  const [loading, setLoading] = useState(true);

  // Submitting state to disable the form button and show a spinner/loading label
  const [submitting, setSubmitting] = useState(false);

  // User notification alerts
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // ---------------------------------------------------------------------------
  // Data Fetching: Load All Repositories
  // ---------------------------------------------------------------------------
  /**
   * Fetches registered repositories from GET /api/repositories.
   * Can be invoked manually (e.g. clicking 'Refresh') or automatically on mount.
   */
  const loadRepositories = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/repositories`);

      if (!response.ok) {
        throw new Error("Could not load repositories from the server.");
      }

      const data = await response.json();
      setRepositories(data);
      setError(""); // Clear any previous error state
    } catch {
      setError(
        "Cannot connect to the RepoPilot backend. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch repositories on initial component mount
  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    loadRepositories();
  }, [loadRepositories]);

  // ---------------------------------------------------------------------------
  // Form Handler: Submit New Repository
  // ---------------------------------------------------------------------------
  /**
   * Handles submission of a new GitHub repository URL.
   * Sends POST request to /api/repositories.
   */
  async function handleSubmit(event) {
    event.preventDefault();

    setSubmitting(true);
    setError("");
    setSuccess("");

    try {
      const response = await fetch(`${API_URL}/repositories`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: url.trim() }),
      });

      const data = await response.json().catch(() => ({}));

      // Handle non-2xx responses (e.g. 404 repo not found, 409 already exists, 422 invalid URL)
      if (!response.ok) {
        const message =
          typeof data.detail === "string"
            ? data.detail
            : "Could not register repository.";
        throw new Error(message);
      }

      // Prepend newly added repository to the top of the list for immediate UI feedback
      setRepositories((previous) => [data, ...previous]);
      setUrl(""); // Reset input field
      setSuccess(`Repository '${data.full_name}' was added successfully!`);
    } catch (err) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setSubmitting(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Action Handler: Update Repository Lifecycle Status
  // ---------------------------------------------------------------------------
  /**
   * Updates repository status in the database (PATCH /api/repositories/:id/status).
   *
   * @param {number} repositoryId - Database ID of the repository
   * @param {string} newStatus - 'pending' | 'indexing' | 'completed' | 'failed'
   */
  async function changeStatus(repositoryId, newStatus) {
    setError("");
    setSuccess("");

    try {
      const response = await fetch(
        `${API_URL}/repositories/${repositoryId}/status`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ status: newStatus }),
        }
      );

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Could not update status."
        );
      }

      // Update the modified repository in local state
      setRepositories((previous) =>
        previous.map((repo) => (repo.id === repositoryId ? data : repo))
      );

      setSuccess(`Updated status for ${data.full_name} to '${newStatus}'.`);
    } catch (err) {
      setError(err.message || "Could not update status.");
    }
  }

  // ---------------------------------------------------------------------------
  // Derived Metrics Computations
  // ---------------------------------------------------------------------------
  const completedCount = repositories.filter(
    (repo) => repo.status === "completed"
  ).length;

  const activeCount = repositories.filter(
    (repo) => repo.status === "pending" || repo.status === "indexing"
  ).length;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="app-shell">
      {/* Top Navigation Bar */}
      <nav className="navbar bg-white border-bottom px-4 py-3">
        <a className="navbar-brand fw-bold fs-4" href="#dashboard">
          <span className="brand-mark">R</span> RepoPilot
        </a>
        <span className="text-secondary small">
          AI Codebase Intelligence &amp; Debugging Copilot
        </span>
      </nav>

      <main id="dashboard" className="container py-5">
        {/* Header / Hero Section */}
        <section className="mb-5">
          <p className="text-primary fw-semibold mb-2">DEVELOPER WORKSPACE</p>
          <h1 className="display-6 fw-bold">Your codebase, understood.</h1>
          <p className="text-secondary mt-3">
            Connect a public GitHub repository to index its source code, build
            AST-aware embeddings, and enable evidence-based AI debugging.
          </p>
        </section>

        {/* Add Repository Card */}
        <section className="card border-0 shadow-sm mb-4">
          <div className="card-body p-4">
            <h2 className="h5 fw-bold mb-2">Add a repository</h2>
            <p className="text-secondary small">
              Enter the URL of any public GitHub repository to register it in RepoPilot.
            </p>

            <form onSubmit={handleSubmit}>
              <div className="input-group">
                <input
                  type="url"
                  className="form-control"
                  placeholder="https://github.com/owner/repository"
                  value={url}
                  onChange={(event) => setUrl(event.target.value)}
                  required
                  aria-label="GitHub repository URL"
                />

                <button
                  className="btn btn-primary px-4"
                  type="submit"
                  disabled={submitting}
                >
                  {submitting ? "Adding..." : "Add repository"}
                </button>
              </div>
            </form>

            {/* Alert Messages */}
            {error && (
              <div className="alert alert-danger mt-3 mb-0" role="alert">
                {error}
              </div>
            )}

            {success && (
              <div className="alert alert-success mt-3 mb-0" role="status">
                {success}
              </div>
            )}
          </div>
        </section>

        {/* Metrics Overview Cards */}
        <section className="row g-3 mb-4">
          <div className="col-md-4">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body p-4">
                <p className="text-secondary small mb-2">Total repositories</p>
                <p className="display-6 fw-bold mb-0">{repositories.length}</p>
              </div>
            </div>
          </div>

          <div className="col-md-4">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body p-4">
                <p className="text-secondary small mb-2">Pending / Indexing</p>
                <p className="display-6 fw-bold mb-0">{activeCount}</p>
              </div>
            </div>
          </div>

          <div className="col-md-4">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body p-4">
                <p className="text-secondary small mb-2">Completed</p>
                <p className="display-6 fw-bold mb-0">{completedCount}</p>
              </div>
            </div>
          </div>
        </section>

        {/* Repositories Table */}
        <section className="card border-0 shadow-sm">
          <div className="card-body p-4">
            <div className="d-flex flex-wrap justify-content-between align-items-center gap-3 mb-4">
              <div>
                <h2 className="h5 fw-bold mb-1">Your repositories</h2>
                <p className="text-secondary small mb-0">
                  Registered GitHub projects ready for indexing
                </p>
              </div>

              <button
                type="button"
                className="btn btn-outline-secondary btn-sm"
                onClick={loadRepositories}
              >
                Refresh
              </button>
            </div>

            {loading ? (
              <p className="text-secondary">Loading repositories...</p>
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
                      <th>Development Controls</th>
                    </tr>
                  </thead>

                  <tbody>
                    {repositories.map((repo) => (
                      <tr key={repo.id}>
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

                        <td>{repo.language || "—"}</td>

                        <td>{repo.stars.toLocaleString()}</td>

                        <td>
                          <StatusBadge status={repo.status} />
                        </td>

                        <td className="small text-secondary">
                          {formatDate(repo.created_at)}
                        </td>

                        <td>
                          {/* Manual status switcher for testing pipeline stages */}
                          <select
                            className="form-select form-select-sm"
                            value={repo.status}
                            aria-label={`Development status for ${repo.full_name}`}
                            onChange={(event) =>
                              changeStatus(repo.id, event.target.value)
                            }
                          >
                            <option value="pending">Pending</option>
                            <option value="indexing">Indexing</option>
                            <option value="completed">Completed</option>
                            <option value="failed">Failed</option>
                          </select>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center text-secondary small mt-5">
          RepoPilot · Phase 1 · Application Foundation &amp; Repository Ingestion
        </footer>
      </main>
    </div>
  );
}