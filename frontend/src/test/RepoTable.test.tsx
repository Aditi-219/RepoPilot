import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import RepoTable from "../components/Dashboard/RepoTable";
import type { Repository } from "../types";

const mockRepos: Repository[] = [
  {
    id: 1,
    github_id: 12345,
    full_name: "facebook/react",
    url: "https://github.com/facebook/react",
    description: "The library for web and native user interfaces",
    stars: 220000,
    language: "JavaScript",
    status: "completed",
    created_at: "2026-01-15T10:00:00Z",
  },
  {
    id: 2,
    github_id: 67890,
    full_name: "tiangolo/fastapi",
    url: "https://github.com/tiangolo/fastapi",
    description: "FastAPI framework, high performance, easy to learn",
    stars: 75000,
    language: "Python",
    status: "pending",
    created_at: "2026-02-01T12:00:00Z",
  },
];

describe("RepoTable", () => {
  it("renders empty state when no repositories exist", () => {
    render(
      <RepoTable
        repositories={[]}
        loading={false}
        indexingIds={{}}
        onRefresh={vi.fn()}
        onTriggerIndexing={vi.fn()}
        onChangeStatus={vi.fn()}
        onDeleteRepo={vi.fn()}
        onOpenChat={vi.fn()}
      />
    );
    expect(screen.getByText("No repositories yet")).toBeInTheDocument();
  });

  it("renders repositories rows when data is provided", () => {
    render(
      <RepoTable
        repositories={mockRepos}
        loading={false}
        indexingIds={{}}
        onRefresh={vi.fn()}
        onTriggerIndexing={vi.fn()}
        onChangeStatus={vi.fn()}
        onDeleteRepo={vi.fn()}
        onOpenChat={vi.fn()}
      />
    );

    expect(screen.getByText(/facebook\/react/)).toBeInTheDocument();
    expect(screen.getByText(/tiangolo\/fastapi/)).toBeInTheDocument();
    expect(screen.getByText((220000).toLocaleString())).toBeInTheDocument();
    expect(screen.getByText((75000).toLocaleString())).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Chat/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Index/i })).toBeInTheDocument();
  });

  it("triggers onOpenChat when Chat button is clicked", () => {
    const handleChat = vi.fn();
    render(
      <RepoTable
        repositories={mockRepos}
        loading={false}
        indexingIds={{}}
        onRefresh={vi.fn()}
        onTriggerIndexing={vi.fn()}
        onChangeStatus={vi.fn()}
        onDeleteRepo={vi.fn()}
        onOpenChat={handleChat}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: /Chat/i }));
    expect(handleChat).toHaveBeenCalledWith(1);
  });

  it("triggers onTriggerIndexing when Index button is clicked", () => {
    const handleIndex = vi.fn();
    render(
      <RepoTable
        repositories={mockRepos}
        loading={false}
        indexingIds={{}}
        onRefresh={vi.fn()}
        onTriggerIndexing={handleIndex}
        onChangeStatus={vi.fn()}
        onDeleteRepo={vi.fn()}
        onOpenChat={vi.fn()}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: /Index/i }));
    expect(handleIndex).toHaveBeenCalledWith(2, "tiangolo/fastapi");
  });

  it("toggles dev mode controls", () => {
    render(
      <RepoTable
        repositories={mockRepos}
        loading={false}
        indexingIds={{}}
        onRefresh={vi.fn()}
        onTriggerIndexing={vi.fn()}
        onChangeStatus={vi.fn()}
        onDeleteRepo={vi.fn()}
        onOpenChat={vi.fn()}
      />
    );

    // Dev controls header not visible initially
    expect(screen.queryByText("Dev Override")).not.toBeInTheDocument();

    // Click Dev Mode toggle using its accessible name
    fireEvent.click(screen.getByRole("button", { name: /Toggle developer controls/i }));
    expect(screen.getByText("Dev Override")).toBeInTheDocument();
  });
});
