/**
 * RepoPilot — Root Frontend Application
 *
 * This component acts as the coordinator for the RepoPilot app.
 * It manages repository state and leverages React Router for navigation
 * between Dashboard and Chat views, supporting deep links (e.g. /chat/:repoId).
 */

import { useCallback, useEffect, useState } from "react";
import {
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
} from "react-router-dom";
import type { Repository, RepositoryStatus } from "./types";
import Navbar from "./components/Navbar";
import AddRepoForm from "./components/Dashboard/AddRepoForm";
import MetricsOverview from "./components/Dashboard/MetricsOverview";
import RepoTable from "./components/Dashboard/RepoTable";
import ChatView from "./components/ChatView";
import ErrorBoundary from "./components/common/ErrorBoundary";
import useTheme from "./hooks/useTheme";
import {
  createRepository,
  deleteRepository,
  getRepositories,
  triggerRepositoryIndexing,
  updateRepositoryStatus,
} from "./services/api";

function ChatRouteWrapper({
  repositories,
  onRepoSelect,
}: {
  repositories: Repository[];
  onRepoSelect: (repoId: string) => void;
}) {
  const { repoId } = useParams<{ repoId?: string }>();
  return (
    <ChatView
      repositories={repositories}
      initialRepoId={repoId ?? null}
      onRepoSelect={onRepoSelect}
    />
  );
}

export default function App() {
  const { theme, toggleTheme } = useTheme();
  const location = useLocation();
  const navigate = useNavigate();

  // Determine active view based on URL pathname
  const isChatRoute = location.pathname.startsWith("/chat");
  const activeView: "dashboard" | "chat" = isChatRoute ? "chat" : "dashboard";

  // Repositories State
  const [repositories, setRepositories] = useState<Repository[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [addError, setAddError] = useState("");
  const [addSuccess, setAddSuccess] = useState("");
  const [tableNotification, setTableNotification] = useState<{ type: "error" | "success"; message: string } | null>(null);

  // Track in-flight indexing button clicks
  const [indexingIds, setIndexingIds] = useState<Record<number, boolean>>({});

  /**
   * Fetch all registered repositories from the backend.
   */
  const loadRepositories = useCallback(async () => {
    try {
      const data = await getRepositories();
      setRepositories(data);
      setTableNotification(null);
    } catch {
      setTableNotification({
        type: "error",
        message: "Cannot connect to the RepoPilot backend. Make sure FastAPI is running on port 8000.",
      });
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch repositories on initial load
  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    loadRepositories();
  }, [loadRepositories]);

  // Automatic Polling: If any repository is currently 'indexing', poll every 3s
  const hasActiveIndexing = repositories.some((repo) => repo.status === "indexing");
  useEffect(() => {
    if (!hasActiveIndexing) return;

    const intervalId = setInterval(() => {
      loadRepositories();
    }, 3000);

    return () => clearInterval(intervalId);
  }, [hasActiveIndexing, loadRepositories]);

  /**
   * Handle registration of a new GitHub repository URL.
   */
  async function handleAddRepository(repoUrl: string, onSuccess?: () => void) {
    setSubmitting(true);
    setAddError("");
    setAddSuccess("");

    try {
      const newRepo = await createRepository(repoUrl);
      setRepositories((prev) => [newRepo, ...prev]);
      setAddSuccess(`Repository '${newRepo.full_name}' was added successfully.`);
      if (onSuccess) onSuccess();
    } catch (err: unknown) {
      setAddError(err instanceof Error ? err.message : "An unexpected error occurred.");
    } finally {
      setSubmitting(false);
    }
  }

  /**
   * Handle manual status updates (Dev controls).
   */
  async function handleChangeStatus(repositoryId: number, newStatus: RepositoryStatus) {
    try {
      const updatedRepo = await updateRepositoryStatus(repositoryId, newStatus);
      setRepositories((prev) =>
        prev.map((repo) => (repo.id === repositoryId ? updatedRepo : repo))
      );
      setTableNotification({
        type: "success",
        message: `Updated status for ${updatedRepo.full_name} to '${newStatus}'.`,
      });
    } catch (err: unknown) {
      setTableNotification({
        type: "error",
        message: err instanceof Error ? err.message : "Could not update status.",
      });
    }
  }

  /**
   * Trigger the background indexing pipeline.
   */
  async function handleTriggerIndexing(repositoryId: number, fullName: string) {
    setIndexingIds((prev) => ({ ...prev, [repositoryId]: true }));

    try {
      const res = await triggerRepositoryIndexing(repositoryId);
      setTableNotification({
        type: "success",
        message: res.message,
      });

      // Optimistically mark as indexing in UI
      setRepositories((prev) =>
        prev.map((repo) =>
          repo.id === repositoryId ? { ...repo, status: "indexing" } : repo
        )
      );
    } catch (err: unknown) {
      setTableNotification({
        type: "error",
        message: err instanceof Error ? err.message : `Could not start indexing for '${fullName}'.`,
      });
    } finally {
      setIndexingIds((prev) => ({ ...prev, [repositoryId]: false }));
    }
  }

  /**
   * Delete a repository from RepoPilot.
   */
  async function handleDeleteRepository(repositoryId: number, fullName: string) {
    if (!window.confirm(`Are you sure you want to delete '${fullName}'?`)) {
      return;
    }

    try {
      await deleteRepository(repositoryId);
      setRepositories((prev) => prev.filter((r) => r.id !== repositoryId));
      setTableNotification({
        type: "success",
        message: `Deleted repository '${fullName}'.`,
      });
    } catch (err: unknown) {
      setTableNotification({
        type: "error",
        message: err instanceof Error ? err.message : `Could not delete repository '${fullName}'.`,
      });
    }
  }

  /**
   * Direct shortcut from table row to Chat route with pre-selected repo.
   */
  function handleOpenChat(repoId: number) {
    navigate(`/chat/${repoId}`);
  }

  function handleNavbarNavigate(view: "dashboard" | "chat") {
    if (view === "dashboard") {
      navigate("/dashboard");
    } else {
      navigate("/chat");
    }
  }

  // Derived metrics
  const completedCount = repositories.filter((r) => r.status === "completed").length;
  const activeCount = repositories.filter(
    (r) => r.status === "pending" || r.status === "indexing"
  ).length;

  return (
    <div className={`app-shell${activeView === "chat" ? " app-shell--chat" : ""}`}>
      {/* Global Top Navbar */}
      <Navbar
        activeView={activeView}
        setActiveView={handleNavbarNavigate}
        completedCount={completedCount}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <ErrorBoundary>
        <Routes>
          {/* Redirect / to /dashboard */}
          <Route path="/" element={<Navigate to="/dashboard" replace />} />

          {/* Dashboard View */}
          <Route
            path="/dashboard"
            element={
              <main id="dashboard" className="dashboard-view">
                {/* Hero Section */}
                <section className="dashboard-hero">
                  <span className="dashboard-hero-eyebrow">Developer Workspace</span>
                  <h1 className="dashboard-hero-title">Your codebase, understood.</h1>
                  <p className="dashboard-hero-subtitle">
                    Connect a public GitHub repository to index its source code, build
                    AST-aware embeddings, and enable evidence-based AI debugging.
                  </p>
                </section>

                {/* Add Repository Card */}
                <AddRepoForm
                  onAddRepo={handleAddRepository}
                  submitting={submitting}
                  error={addError}
                  success={addSuccess}
                />

                {/* Metrics Overview Cards */}
                <MetricsOverview
                  totalCount={repositories.length}
                  activeCount={activeCount}
                  completedCount={completedCount}
                />

                {/* Table Notification (if any) */}
                {tableNotification && (
                  <div
                    className={`rp-alert rp-alert-${tableNotification.type}`}
                    role="alert"
                    style={{ margin: "0 0 16px 0" }}
                  >
                    {tableNotification.message}
                  </div>
                )}

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
                <footer className="dashboard-footer">
                  <span>RepoPilot</span>
                  <span className="dashboard-footer-dot" />
                  <span>Phase 5</span>
                  <span className="dashboard-footer-dot" />
                  <span>LangGraph &amp; pgvector Codebase Intelligence</span>
                </footer>
              </main>
            }
          />

          {/* Chat Views */}
          <Route
            path="/chat"
            element={
              <ChatRouteWrapper
                repositories={repositories}
                onRepoSelect={(repoId) => navigate(`/chat/${repoId}`)}
              />
            }
          />
          <Route
            path="/chat/:repoId"
            element={
              <ChatRouteWrapper
                repositories={repositories}
                onRepoSelect={(repoId) => navigate(`/chat/${repoId}`)}
              />
            }
          />

          {/* Fallback route */}
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </ErrorBoundary>
    </div>
  );
}
