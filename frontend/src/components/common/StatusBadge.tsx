/**
 * StatusBadge Component
 *
 * Renders a color-coded badge with an animated indicator to visually display
 * the current processing state of a repository in RepoPilot.
 */

import type { RepositoryStatus } from "../../types";

const STATUS_LABELS: Record<RepositoryStatus, string> = {
  pending: "Pending",
  indexing: "Indexing",
  completed: "Completed",
  failed: "Failed",
};

interface StatusBadgeProps {
  status?: RepositoryStatus | string;
}

export default function StatusBadge({ status }: StatusBadgeProps) {
  const safeStatus = (status ?? "pending") as RepositoryStatus;
  const label = STATUS_LABELS[safeStatus] ?? safeStatus;

  return (
    <span
      className={`status-badge status-badge--${safeStatus}`}
      title={label}
      aria-label={`Status: ${label}`}
    >
      <span className="status-dot" aria-hidden="true" />
      {label}
    </span>
  );
}
