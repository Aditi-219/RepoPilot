/**
 * Utility helpers and constants for the Chat module.
 */

export const EXAMPLE_QUESTIONS = [
  "Where is authentication handled in this repository?",
  "How is the database session and connection pool managed?",
  "What tools and dependencies are used across this project?",
  "Inspect the project file structure and give me an architectural overview.",
  "What happens when repository indexing fails or errors out?",
  "How are environment variables and configurations validated?",
];

export function formatToolName(toolName: string): string {
  switch (toolName) {
    case "inspect_file_tree":
      return "Inspecting repository file tree";
    case "read_source_file":
      return "Reading source file";
    case "fetch_repo_metadata":
      return "Fetching repository metadata";
    case "search_codebase":
      return "Searching hybrid index";
    default:
      return `Executing tool: ${toolName}`;
  }
}
