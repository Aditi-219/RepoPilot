/**
 * ToolExecutionBadge Component
 *
 * Displays live tool execution pills with spinner or completion checks.
 */

function WrenchIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

import { formatToolName } from "./chatUtils";

interface ToolExecutionBadgeProps {
  tool: string;
  status: "running" | "completed";
}

export default function ToolExecutionBadge({ tool, status }: ToolExecutionBadgeProps) {
  const isRunning = status === "running";
  const label = formatToolName(tool);

  return (
    <div
      className={`chat-tool-pill${isRunning ? " chat-tool-pill--running" : " chat-tool-pill--completed"}`}
      role="status"
      aria-label={`${isRunning ? "Running" : "Completed"}: ${label}`}
    >
      <span className="chat-tool-icon" aria-hidden="true">
        {isRunning ? <WrenchIcon /> : <CheckIcon />}
      </span>
      <span>{label}</span>
      {isRunning && <span className="chat-tool-spinner" aria-hidden="true" />}
    </div>
  );
}
