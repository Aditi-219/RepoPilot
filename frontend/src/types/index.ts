/**
 * Shared TypeScript definitions for RepoPilot Frontend.
 * These interfaces mirror the backend Pydantic schemas and database models.
 */

export type RepositoryStatus = "pending" | "indexing" | "completed" | "failed";

export interface Repository {
  id: number;
  github_id: number;
  full_name: string;
  url: string;
  description: string | null;
  stars: number;
  language: string | null;
  status: RepositoryStatus;
  created_at: string;
}

export interface IndexResponse {
  message: string;
  repository_id: number;
  status: RepositoryStatus;
}

export interface CodeCitation {
  chunk_id: number;
  repository_id: number;
  file_path: string;
  start_line: number;
  end_line: number;
  content: string;
  language: string | null;
  score: number;
}

export interface SearchResponse {
  query: string;
  repository_id: number;
  results: CodeCitation[];
}

export interface AskResponse {
  question: string;
  answer: string;
  citations: CodeCitation[];
  model: string;
  repository_id: number;
}

export interface ChatSession {
  id: string;
  repository_id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessageItem {
  id?: string | null;
  role: "user" | "assistant" | "tool" | "system";
  content: string;
  tool_calls?: Record<string, unknown>[] | null;
  citations?: CodeCitation[];
  model?: string;
}

export interface ChatHistoryResponse {
  session_id: string;
  repository_id: number;
  title: string;
  messages: ChatMessageItem[];
}

export interface ActiveToolExecution {
  tool: string;
  status: "running" | "completed";
}

export interface StreamHandlers {
  onToken?: (delta: string) => void;
  onToolStart?: (tool: string, args: Record<string, unknown>) => void;
  onToolEnd?: (tool: string, output: string) => void;
  onError?: (err: Error) => void;
  onDone?: () => void;
  signal?: AbortSignal;
}

export type Theme = "light" | "dark";
