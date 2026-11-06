/**
 * Navbar Component
 *
 * Header navigation bar for RepoPilot with brand logo, tab switcher,
 * completed repo counter badge, and persistent light/dark theme toggle.
 */

import type { Theme } from "../types";

function DashboardIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="3" width="7" height="7" rx="1" />
      <rect x="3" y="14" width="7" height="7" rx="1" />
      <rect x="14" y="14" width="7" height="7" rx="1" />
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" />
      <line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" />
      <line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

interface NavbarProps {
  activeView: "dashboard" | "chat";
  setActiveView: (view: "dashboard" | "chat") => void;
  completedCount: number;
  theme?: Theme;
  onToggleTheme: () => void;
}

export default function Navbar({
  activeView,
  setActiveView,
  completedCount,
  theme = "light",
  onToggleTheme,
}: NavbarProps) {
  const isDark = theme === "dark";

  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      {/* Brand */}
      <a
        className="navbar-brand"
        href="#dashboard"
        onClick={(e) => {
          e.preventDefault();
          setActiveView("dashboard");
        }}
        aria-label="RepoPilot home"
      >
        <span className="brand-mark" aria-hidden="true">R</span>
        RepoPilot
      </a>

      {/* View switcher tabs */}
      <div className="nav-tabs-custom" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={activeView === "dashboard"}
          className={`nav-tab${activeView === "dashboard" ? " nav-tab--active" : ""}`}
          onClick={() => setActiveView("dashboard")}
        >
          <span className="nav-tab-icon" aria-hidden="true">
            <DashboardIcon />
          </span>
          Dashboard
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={activeView === "chat"}
          className={`nav-tab${activeView === "chat" ? " nav-tab--active" : ""}`}
          onClick={() => setActiveView("chat")}
        >
          <span className="nav-tab-icon" aria-hidden="true">
            <ChatIcon />
          </span>
          Chat
          {completedCount > 0 && (
            <span className="nav-tab-badge" aria-label={`${completedCount} repositories ready`}>
              {completedCount}
            </span>
          )}
        </button>
      </div>

      {/* Header Actions (Tagline & Theme Toggle) */}
      <div className="navbar-actions">
        <span className="navbar-tagline" aria-hidden="true">
          AI Codebase Intelligence &amp; RAG Q&amp;A
        </span>
        <button
          type="button"
          className="theme-toggle-btn"
          onClick={onToggleTheme}
          aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
          title={isDark ? "Switch to light theme" : "Switch to dark theme"}
        >
          {isDark ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>
    </nav>
  );
}
