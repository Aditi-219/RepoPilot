/**
 * StatusBadge Component
 *
 * Renders a color-coded Bootstrap badge to visually display the current
 * processing state of a repository in RepoPilot.
 *
 * Status Colors:
 * - 'pending'   -> Grey   (Waiting to be indexed)
 * - 'indexing'  -> Blue   (Actively chunking & computing vectors)
 * - 'completed' -> Green  (Indexed and ready for RAG Chat)
 * - 'failed'    -> Red    (Error occurred during ingestion)
 */

export default function StatusBadge({ status }) {
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
