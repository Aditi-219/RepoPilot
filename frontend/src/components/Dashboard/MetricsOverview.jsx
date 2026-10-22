/**
 * MetricsOverview Component
 *
 * Displays high-level dashboard metric cards summarizing:
 * 1. Total registered repositories
 * 2. Active indexing or pending pipelines
 * 3. Repositories ready for Q&A search
 */

export default function MetricsOverview({ totalCount, activeCount, completedCount }) {
  return (
    <section className="row g-3 mb-4">
      {/* Total Repositories Card */}
      <div className="col-md-4">
        <div className="card border-0 shadow-sm h-100">
          <div className="card-body p-4">
            <p className="text-secondary small mb-2">Total repositories</p>
            <p className="display-6 fw-bold mb-0">{totalCount}</p>
          </div>
        </div>
      </div>

      {/* Pending / Indexing Card */}
      <div className="col-md-4">
        <div className="card border-0 shadow-sm h-100">
          <div className="card-body p-4">
            <p className="text-secondary small mb-2">Pending / Indexing</p>
            <p className="display-6 fw-bold mb-0">{activeCount}</p>
          </div>
        </div>
      </div>

      {/* Completed Repositories Card */}
      <div className="col-md-4">
        <div className="card border-0 shadow-sm h-100">
          <div className="card-body p-4">
            <p className="text-secondary small mb-2">
              Completed
              {completedCount > 0 && (
                <span className="ms-2 badge bg-success">Ready to chat ✨</span>
              )}
            </p>
            <p className="display-6 fw-bold mb-0">{completedCount}</p>
          </div>
        </div>
      </div>
    </section>
  );
}
