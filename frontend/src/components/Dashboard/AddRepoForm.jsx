/**
 * AddRepoForm Component
 *
 * Provides a clean card interface where users can input a public GitHub repository
 * URL and submit it for registration in RepoPilot.
 *
 * Features:
 * - URL input with basic browser validation
 * - Loading indicator ("Adding…") while the backend contacts GitHub
 * - Clear error alerts (e.g. invalid repo, already registered)
 * - Clear success alerts confirming the repo was added
 */

import { useState } from "react";

export default function AddRepoForm({ onAddRepo, submitting, error, success }) {
  const [url, setUrl] = useState("");

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!url.trim()) return;
    onAddRepo(url.trim(), () => setUrl(""));
  };

  return (
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
              {submitting ? "Adding…" : "Add repository"}
            </button>
          </div>
        </form>

        {/* Dynamic Alerts */}
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
  );
}
