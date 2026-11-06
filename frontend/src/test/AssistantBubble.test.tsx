import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import AssistantBubble from "../components/chat/AssistantBubble";
import type { CodeCitation } from "../types";

const mockCitations: CodeCitation[] = [
  {
    chunk_id: 1,
    repository_id: 1,
    file_path: "app/main.py",
    start_line: 1,
    end_line: 20,
    content: "app = FastAPI()",
    language: "python",
    score: 0.95,
  },
];

describe("AssistantBubble", () => {
  it("renders basic markdown headings, bold text, and lists", () => {
    const markdownContent = `### Overview\nHere is **bold** statement.\n- Item 1\n- Item 2`;
    render(<AssistantBubble content={markdownContent} />);

    expect(screen.getByRole("heading", { level: 3 })).toHaveTextContent("Overview");
    expect(screen.getByText("bold")).toBeInTheDocument();
    expect(screen.getByText("Item 1")).toBeInTheDocument();
    expect(screen.getByText("Item 2")).toBeInTheDocument();
  });

  it("renders code blocks with language indicator and copy button", () => {
    const codeBlockMarkdown = "```typescript\nconst greeting: string = 'hello';\n```";
    render(<AssistantBubble content={codeBlockMarkdown} />);

    expect(screen.getByText("typescript")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy code block" })).toBeInTheDocument();
    expect(screen.getByText(/const greeting: string/)).toBeInTheDocument();
  });

  it("displays citations when provided", () => {
    render(
      <AssistantBubble
        content="Here is the explanation."
        citations={mockCitations}
        model="gemini-flash"
      />
    );

    expect(screen.getByText(/1 code snippet referenced/)).toBeInTheDocument();
    expect(screen.getByText("gemini-flash")).toBeInTheDocument();
    expect(screen.getAllByText("app/main.py")[0]).toBeInTheDocument();
  });

  it("displays streaming cursor when isStreaming is true", () => {
    const { container } = render(
      <AssistantBubble content="Generating response..." isStreaming={true} />
    );

    expect(container.querySelector(".chat-streaming-cursor")).toBeInTheDocument();
  });
});
