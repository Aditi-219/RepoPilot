import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import StatusBadge from "../components/common/StatusBadge";

describe("StatusBadge", () => {
  it("renders pending status correctly", () => {
    render(<StatusBadge status="pending" />);
    const badge = screen.getByLabelText("Status: Pending");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("status-badge--pending");
    expect(screen.getByText("Pending")).toBeInTheDocument();
  });

  it("renders indexing status correctly", () => {
    render(<StatusBadge status="indexing" />);
    const badge = screen.getByLabelText("Status: Indexing");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("status-badge--indexing");
    expect(screen.getByText("Indexing")).toBeInTheDocument();
  });

  it("renders completed status correctly", () => {
    render(<StatusBadge status="completed" />);
    const badge = screen.getByLabelText("Status: Completed");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("status-badge--completed");
    expect(screen.getByText("Completed")).toBeInTheDocument();
  });

  it("renders failed status correctly", () => {
    render(<StatusBadge status="failed" />);
    const badge = screen.getByLabelText("Status: Failed");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("status-badge--failed");
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it("defaults to pending when status is undefined", () => {
    render(<StatusBadge />);
    expect(screen.getByText("Pending")).toBeInTheDocument();
  });
});
