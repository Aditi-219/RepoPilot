/**
 * RepoPilot — Root Frontend Application
 *
 * This component acts as the lightweight coordinator for the RepoPilot single-page app.
 * It manages global state (repositories list, active tab) and delegates rendering
 * to specialized components:
 *
 * - Navbar: Top brand and tab switcher
 * - Dashboard:
 *     - AddRepoForm: Registering public GitHub repositories
 *     - MetricsOverview: Metric stat cards
 *     - RepoTable: Interactive repository list with actions and dev controls
 * - Chat:
 *     - ChatView: RAG-based natural-language Q&A
 */

import { useCallback, useEffect, useState } from "react";
import Navbar from "./components/Navbar";
import AddRepoForm from "./components/Dashboard/AddRepoForm";
import MetricsOverview from "./components/Dashboard/MetricsOverview";
import RepoTable from "./components/Dashboard/RepoTable";
import ChatView from "./components/ChatView";
import {
  createRepository,
  deleteRepository,
  getRepositories,
  triggerRepositoryIndexing,
  updateRepositoryStatus,
} from "./services/api";

export default function App() {
  // Navigation State: 'dashboard' | 'chat'
  const [activeView, setActiveView] = useState("dashboard");
  const [selectedChatRepoId, setSelectedChatRepoId] = useState(null);

  // Repositories State
  const [repositories, setRepositories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Track in-flight indexing button clicks
  const [indexingIds, setIndexingIds] = useState({});

  /**
   * Fetch all registered repositories from the backend.
   */
  const loadRepositories = useCallback(async () => {
    try {
      const data = await getRepositories();
      setRepositories(data);
      setError("");
    } catch {
      setError(
        "Cannot connect to the RepoPilot backend. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch repositories on initial component load
  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    loadRepositories();
  }, [loadRepositories]);

  // Automatic Polling: If any repository is currently 'indexing', poll every 3s
  useEffect(() => {
    const hasActiveIndexing = repositories.some(
      (repo) => repo.status === "indexing"
    );
    if (!hasActiveIndexing) return;

    const intervalId = setInterval(() => {
      loadRepositories();
    }, 3000);

    return () => clearInterval(intervalId);
  }, [repositories, loadRepositories]);

  /**
   * Handle registration of a new GitHub repository URL.
   */
  async function handleAddRepository(repoUrl, onSuccess) {
    setSubmitting(true);
    setError("");
    setSuccess("");

    try {
      const newRepo = await createRepository(repoUrl);
      setRepositories((prev) => [newRepo, ...prev]);
      setSuccess(`Repository '${newRepo.full_name}' was added successfully!`);
      if (onSuccess) onSuccess();
    } catch (err) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setSubmitting(false);
    }
  }

  /**
   * Handle manual status updates (Dev controls).
   */
  async function handleChangeStatus(repositoryId, newStatus) {
    setError("");
    setSuccess("");

    try {
      const updatedRepo = await updateRepositoryStatus(repositoryId, newStatus);
      setRepositories((prev) =>
        prev.map((repo) => (repo.id === repositoryId ? updatedRepo : repo))
      );
      setSuccess(`Updated status for ${updatedRepo.full_name} to '${newStatus}'.`);
    } catch (err) {
      setError(err.message || "Could not update status.");
    }
  }

  /**
   * Trigger the background indexing pipeline.
   */
  async function handleTriggerIndexing(repositoryId, fullName) {
    setError("");
    setSuccess("");
    setIndexingIds((prev) => ({ ...prev, [repositoryId]: true }));

    try {
      const res = await triggerRepositoryIndexing(repositoryId);
      setSuccess(res.message);

      // Optimistically mark as indexing in UI
      setRepositories((prev) =>
        prev.map((repo) =>
          repo.id === repositoryId ? { ...repo, status: "indexing" } : repo
        )
      );
    } catch (err) {
      setError(err.message || `Could not start indexing for '${fullName}'.`);
    } finally {
      setIndexingIds((prev) => ({ ...prev, [repositoryId]: false }));
    }
  }

  /**
   * Delete a repository from RepoPilot.
   */
  async function handleDeleteRepository(repositoryId, fullName) {
    if (!window.confirm(`Are you sure you want to delete '${fullName}'?`)) {
      return;
    }

    setError("");
    setSuccess("");

    try {
      await deleteRepository(repositoryId);
      setRepositories((prev) => prev.filter((r) => r.id !== repositoryId));
      setSuccess(`Deleted repository '${fullName}'.`);
    } catch (err) {
      setError(err.message || `Could not delete repository '${fullName}'.`);
    }
  }

  /**
   * Direct shortcut from table row to Chat tab with pre-selected repo.
   */
  function handleOpenChat(repoId) {
    setSelectedChatRepoId(repoId);
    setActiveView("chat");
  }

  // Derived metrics
  const completedCount = repositories.filter((r) => r.status === "completed").length;
  const activeCount = repositories.filter(
    (r) => r.status === "pending" || r.status === "indexing"
  ).length;

  return (
    <div className="app-shell">
      {/* Global Top Navbar */}
      <Navbar
        activeView={activeView}
        setActiveView={setActiveView}
        completedCount={completedCount}
      />

      {/* Chat View */}
      {activeView === "chat" && (
        <ChatView
          repositories={repositories}
          initialRepoId={selectedChatRepoId}
        />
      )}

      {/* Dashboard View */}
      {activeView === "dashboard" && (
        <main id="dashboard" className="container py-5">
          {/* Hero Section */}
          <section className="mb-5">
            <p className="text-primary fw-semibold mb-2">DEVELOPER WORKSPACE</p>
            <h1 className="display-6 fw-bold">Your codebase, understood.</h1>
            <p className="text-secondary mt-3">
              Connect a public GitHub repository to index its source code, build
              AST-aware embeddings, and enable evidence-based AI debugging.
            </p>
          </section>

          {/* Add Repository Card */}
          <AddRepoForm
            onAddRepo={handleAddRepository}
            submitting={submitting}
            error={error}
            success={success}
          />

          {/* Metrics Overview Cards */}
          <MetricsOverview
            totalCount={repositories.length}
            activeCount={activeCount}
            completedCount={completedCount}
          />

          {/* Repositories Table */}
          <RepoTable
            repositories={repositories}
            loading={loading}
            indexingIds={indexingIds}
            onRefresh={loadRepositories}
            onTriggerIndexing={handleTriggerIndexing}
            onChangeStatus={handleChangeStatus}
            onDeleteRepo={handleDeleteRepository}
            onOpenChat={handleOpenChat}
          />

          {/* Footer */}
          <footer className="text-center text-secondary small mt-5">
            RepoPilot · Phase 3 · RAG Codebase Q&amp;A
          </footer>
        </main>
      )}
    </div>
  );
}
