import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import CitationCard from "../components/CitationCard";
import type { CodeCitation } from "../types";

const mockCitation: CodeCitation = {
  chunk_id: 101,
  repository_id: 1,
  file_path: "src/auth/jwt.py",
  start_line: 12,
  end_line: 35,
  content: "def verify_token(token: str):\n    return decode(token)",
  language: "python",
  score: 0.89,
};

describe("CitationCard", () => {
  it("renders citation header with file path and line numbers", () => {
    render(<CitationCard citation={mockCitation} index={1} />);

    expect(screen.getAllByText("src/auth/jwt.py")[0]).toBeInTheDocument();
    expect(screen.getByText("L12–35")).toBeInTheDocument();
    expect(screen.getByText("python")).toBeInTheDocument();
  });

  it("auto-expands first two citations by default", () => {
    render(<CitationCard citation={mockCitation} index={1} />);
    expect(screen.getByText(/def verify_token/)).toBeInTheDocument();
  });

  it("collapses when header is clicked", () => {
    render(<CitationCard citation={mockCitation} index={1} />);
    const button = screen.getByRole("button", { name: /Citation 1/i });

    // Initially expanded
    expect(screen.getByText(/def verify_token/)).toBeInTheDocument();

    // Click to collapse
    fireEvent.click(button);
    expect(screen.queryByText(/def verify_token/)).not.toBeInTheDocument();

    // Click to expand again
    fireEvent.click(button);
    expect(screen.getByText(/def verify_token/)).toBeInTheDocument();
  });

  it("copies content to clipboard when copy button clicked", async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextMock,
      },
    });

    render(<CitationCard citation={mockCitation} index={1} />);
    const copyButton = screen.getByRole("button", { name: "Copy snippet" });

    await act(async () => {
      fireEvent.click(copyButton);
    });

    expect(writeTextMock).toHaveBeenCalledWith(mockCitation.content);
  });
});
