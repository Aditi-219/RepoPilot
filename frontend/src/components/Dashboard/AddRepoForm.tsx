/**
 * AddRepoForm Component
 *
 * Provides a clean card interface where users can input a public GitHub repository
 * URL and submit it for registration in RepoPilot.
 */

import { type FormEvent, useState } from "react";

function LinkIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
    </svg>
  );
}

function ErrorIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

function ButtonSpinner() {
  return <span className="rp-spinner" role="status" aria-label="Adding repository" />;
}

interface AddRepoFormProps {
  onAddRepo: (url: string, onSuccess?: () => void) => void | Promise<void>;
  submitting: boolean;
  error?: string;
  success?: string;
}

export default function AddRepoForm({
  onAddRepo,
  submitting,
  error,
  success,
}: AddRepoFormProps) {
  const [url, setUrl] = useState("");
  const [clientError, setClientError] = useState("");

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = url.trim();
    if (!trimmed) {
      setClientError("Please enter a GitHub repository URL.");
      return;
    }

    // Basic client-side sanity check for github.com URLs
    if (!trimmed.includes("github.com/")) {
      setClientError("Please provide a valid GitHub URL (e.g. https://github.com/owner/repo).");
      return;
    }

    setClientError("");
    onAddRepo(trimmed, () => {
      setUrl("");
      setClientError("");
    });
  };

  const displayError = clientError || error;

  return (
    <section className="rp-card add-repo-card" aria-label="Add a repository">
      <div className="rp-card-body">
        <div className="add-repo-card-header">
          <h2 className="add-repo-title">Add a repository</h2>
          <p className="add-repo-subtitle">
            Enter the URL of any public GitHub repository to register it in RepoPilot.
          </p>
        </div>

        <form onSubmit={handleSubmit} aria-label="Add repository form">
          <div className="add-repo-form-row">
            <div className="add-repo-input-wrapper">
              <span className="add-repo-input-icon">
                <LinkIcon />
              </span>
              <input
                type="url"
                className="rp-input rp-input-url add-repo-input"
                placeholder="https://github.com/owner/repository"
                value={url}
                onChange={(event) => {
                  setUrl(event.target.value);
                  if (clientError) setClientError("");
                }}
                aria-label="GitHub repository URL"
                autoComplete="url"
                spellCheck={false}
              />
            </div>
            <button
              className="rp-btn rp-btn-primary"
              type="submit"
              disabled={submitting || !url.trim()}
              aria-disabled={submitting || !url.trim()}
            >
              {submitting ? (
                <>
                  <ButtonSpinner />
                  Adding
                </>
              ) : (
                "Add repository"
              )}
            </button>
          </div>
        </form>

        {/* Dynamic Alerts */}
        <div className="add-repo-alerts" aria-live="polite" aria-atomic="true">
          {displayError && (
            <div className="rp-alert rp-alert-error" role="alert">
              <span className="rp-alert-icon"><ErrorIcon /></span>
              {displayError}
            </div>
          )}
          {success && !displayError && (
            <div className="rp-alert rp-alert-success" role="status">
              <span className="rp-alert-icon"><CheckIcon /></span>
              {success}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
