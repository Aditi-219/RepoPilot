/**
 * Navbar Component
 *
 * The main header navigation bar for RepoPilot.
 *
 * Features:
 * - Brand logo and title
 * - Tab buttons to toggle between 'Dashboard' and 'Chat' views
 * - Badge indicator showing how many repositories are completed and ready for Q&A
 * - Subtitle describing RepoPilot's purpose
 */

export default function Navbar({ activeView, setActiveView, completedCount }) {
  return (
    <nav className="navbar bg-white border-bottom px-4 py-3">
      {/* Brand logo and link */}
      <a className="navbar-brand fw-bold fs-4" href="#dashboard">
        <span className="brand-mark">R</span> RepoPilot
      </a>

      {/* View switcher tabs */}
      <div className="nav-tabs-custom">
        <button
          type="button"
          className={`nav-tab ${activeView === "dashboard" ? "nav-tab--active" : ""}`}
          onClick={() => setActiveView("dashboard")}
        >
          Dashboard
        </button>
        <button
          type="button"
          className={`nav-tab ${activeView === "chat" ? "nav-tab--active" : ""}`}
          onClick={() => setActiveView("chat")}
        >
          ✨ Chat
          {completedCount > 0 && (
            <span className="nav-tab-badge">{completedCount}</span>
          )}
        </button>
      </div>

      <span className="text-secondary small d-none d-md-inline">
        AI Codebase Intelligence &amp; RAG Q&amp;A
      </span>
    </nav>
  );
}
