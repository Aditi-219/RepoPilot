/**
 * MetricsOverview Component
 *
 * Displays high-level dashboard metric cards summarizing:
 * 1. Total registered repositories
 * 2. Active indexing or pending pipelines
 * 3. Repositories ready for Q&A search
 */

function DatabaseIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
    </svg>
  );
}

function ActivityIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}

function CheckCircleIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

interface MetricsOverviewProps {
  totalCount: number;
  activeCount: number;
  completedCount: number;
}

export default function MetricsOverview({
  totalCount,
  activeCount,
  completedCount,
}: MetricsOverviewProps) {
  return (
    <section className="metrics-grid" aria-label="Repository metrics">
      {/* Total Repositories */}
      <div className="rp-card metric-card">
        <div className="metric-icon metric-icon--blue" aria-hidden="true">
          <DatabaseIcon />
        </div>
        <div className="metric-content">
          <p className="metric-label">Total repositories</p>
          <p className="metric-value">{totalCount}</p>
        </div>
      </div>

      {/* Pending / Indexing */}
      <div className="rp-card metric-card">
        <div className="metric-icon metric-icon--amber" aria-hidden="true">
          <ActivityIcon />
        </div>
        <div className="metric-content">
          <p className="metric-label">Pending / Indexing</p>
          <p className="metric-value">{activeCount}</p>
        </div>
      </div>

      {/* Completed */}
      <div className="rp-card metric-card">
        <div className="metric-icon metric-icon--green" aria-hidden="true">
          <CheckCircleIcon />
        </div>
        <div className="metric-content">
          <p className="metric-label">
            Completed
            {completedCount > 0 && (
              <span className="metric-ready-badge">
                Ready
              </span>
            )}
          </p>
          <p className="metric-value">{completedCount}</p>
        </div>
      </div>
    </section>
  );
}
