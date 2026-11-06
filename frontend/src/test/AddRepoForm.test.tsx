import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AddRepoForm from "../components/Dashboard/AddRepoForm";

describe("AddRepoForm", () => {
  it("shows client error when submitted empty", () => {
    const handleAdd = vi.fn();
    render(<AddRepoForm onAddRepo={handleAdd} submitting={false} />);

    fireEvent.submit(screen.getByRole("form", { name: "Add repository form" }));

    expect(screen.getByText("Please enter a GitHub repository URL.")).toBeInTheDocument();
    expect(handleAdd).not.toHaveBeenCalled();
  });

  it("shows client error when submitted with non-GitHub URL", () => {
    const handleAdd = vi.fn();
    render(<AddRepoForm onAddRepo={handleAdd} submitting={false} />);

    const input = screen.getByLabelText("GitHub repository URL");
    fireEvent.change(input, { target: { value: "https://gitlab.com/owner/repo" } });
    fireEvent.submit(screen.getByRole("form", { name: "Add repository form" }));

    expect(screen.getByText(/Please provide a valid GitHub URL/)).toBeInTheDocument();
    expect(handleAdd).not.toHaveBeenCalled();
  });

  it("calls onAddRepo when valid GitHub URL is entered", () => {
    const handleAdd = vi.fn();
    render(<AddRepoForm onAddRepo={handleAdd} submitting={false} />);

    const input = screen.getByLabelText("GitHub repository URL");
    fireEvent.change(input, { target: { value: "https://github.com/facebook/react" } });
    fireEvent.submit(screen.getByRole("form", { name: "Add repository form" }));

    expect(handleAdd).toHaveBeenCalledWith("https://github.com/facebook/react", expect.any(Function));
  });

  it("displays server error and success messages", () => {
    const { rerender } = render(
      <AddRepoForm onAddRepo={vi.fn()} submitting={false} error="Repository already exists." />
    );
    expect(screen.getByText("Repository already exists.")).toBeInTheDocument();

    rerender(
      <AddRepoForm onAddRepo={vi.fn()} submitting={false} success="Added repository successfully." />
    );
    expect(screen.getByText("Added repository successfully.")).toBeInTheDocument();
  });

  it("disables button while submitting", () => {
    render(<AddRepoForm onAddRepo={vi.fn()} submitting={true} />);
    const button = screen.getByRole("button", { name: /Adding/i });
    expect(button).toBeDisabled();
  });
});
