/**
 * API Service for RepoPilot Frontend.
 *
 * Centralizes all HTTP network requests made to the FastAPI backend.
 * Provides typed wrappers, robust error handling, timeout / AbortSignal support,
 * and Server-Sent Events (SSE) streaming.
 */

import type {
  AskResponse,
  ChatHistoryResponse,
  ChatSession,
  IndexResponse,
  Repository,
  RepositoryStatus,
  SearchResponse,
  StreamHandlers,
} from "../types";

export const API_BASE_URL =
  (import.meta.env.VITE_API_URL as string | undefined) || "http://127.0.0.1:8000/api";

export interface ApiFetchOptions extends RequestInit {
  timeoutMs?: number;
}

/**
 * Universal wrapper around fetch() with typed responses, custom timeouts, and error parsing.
 */
export async function apiFetch<T>(
  endpoint: string,
  options: ApiFetchOptions = {}
): Promise<T> {
  const { timeoutMs = 30000, headers, ...restOptions } = options;

  const url = endpoint.startsWith("http")
    ? endpoint
    : `${API_BASE_URL}${endpoint}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  // Link external AbortSignal if provided
  if (options.signal) {
    options.signal.addEventListener("abort", () => controller.abort());
  }

  const finalHeaders: Record<string, string> = {};
  if (restOptions.body) {
    finalHeaders["Content-Type"] = "application/json";
  }

  try {
    const response = await fetch(url, {
      ...restOptions,
      headers: {
        ...finalHeaders,
        ...(headers as Record<string, string> | undefined),
      },
      signal: controller.signal,
    });

    if (response.status === 204) {
      return null as T;
    }

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const message =
        typeof data?.detail === "string" ? data.detail : `Request failed with status ${response.status}.`;
      throw new Error(message);
    }

    return data as T;
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(`Request timed out after ${timeoutMs / 1000}s.`);
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

// ---------------------------------------------------------------------------
// Repository Endpoints
// ---------------------------------------------------------------------------

export async function getRepositories(): Promise<Repository[]> {
  return apiFetch<Repository[]>("/repositories");
}

export async function createRepository(repoUrl: string): Promise<Repository> {
  return apiFetch<Repository>("/repositories", {
    method: "POST",
    body: JSON.stringify({ url: repoUrl.trim() }),
  });
}

export async function updateRepositoryStatus(
  repoId: number,
  status: RepositoryStatus
): Promise<Repository> {
  return apiFetch<Repository>(`/repositories/${repoId}/status`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
  });
}

export async function deleteRepository(repoId: number): Promise<null> {
  return apiFetch<null>(`/repositories/${repoId}`, {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Indexing Endpoints
// ---------------------------------------------------------------------------

export async function triggerRepositoryIndexing(
  repoId: number
): Promise<IndexResponse> {
  return apiFetch<IndexResponse>(`/repositories/${repoId}/index`, {
    method: "POST",
  });
}

// ---------------------------------------------------------------------------
// RAG & Search Endpoints
// ---------------------------------------------------------------------------

export async function searchRepository(
  repoId: number | string,
  query: string,
  limit = 8,
  rerank = true
): Promise<SearchResponse> {
  return apiFetch<SearchResponse>(`/repositories/${repoId}/search`, {
    method: "POST",
    body: JSON.stringify({ query, limit, rerank }),
  });
}

export async function askRepository(
  repoId: number | string,
  question: string,
  topK = 8
): Promise<AskResponse> {
  return apiFetch<AskResponse>(`/repositories/${repoId}/ask`, {
    method: "POST",
    body: JSON.stringify({ question, top_k: topK }),
  });
}

// ---------------------------------------------------------------------------
// LangGraph Chat Sessions & Streaming Endpoints
// ---------------------------------------------------------------------------

export async function listChatSessions(
  repoId: number | string
): Promise<ChatSession[]> {
  return apiFetch<ChatSession[]>(`/chat/${repoId}/sessions`);
}

export async function createChatSession(
  repoId: number | string,
  title = "New Conversation"
): Promise<ChatSession> {
  return apiFetch<ChatSession>(`/chat/${repoId}/sessions`, {
    method: "POST",
    body: JSON.stringify({ title }),
  });
}

export async function getChatHistory(
  sessionId: string
): Promise<ChatHistoryResponse> {
  return apiFetch<ChatHistoryResponse>(`/chat/sessions/${sessionId}/history`);
}

export async function deleteChatSession(sessionId: string): Promise<null> {
  return apiFetch<null>(`/chat/sessions/${sessionId}`, {
    method: "DELETE",
  });
}

/**
 * Stream conversational agent response using Server-Sent Events (SSE).
 */
export async function streamChatMessage(
  repoId: number | string,
  sessionId: string,
  message: string,
  handlers: StreamHandlers = {}
): Promise<void> {
  const { onToken, onToolStart, onToolEnd, onError, onDone, signal } = handlers;
  const url = `${API_BASE_URL}/chat/${repoId}/sessions/${sessionId}/stream`;

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal,
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.detail || `Stream failed with status ${response.status}`);
    }

    if (!response.body) {
      throw new Error("ReadableStream not supported by the response body.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let doneCalled = false;

    const safeDone = () => {
      if (!doneCalled && onDone) {
        doneCalled = true;
        onDone();
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      // Normalize line endings to support standard SSE (\r\n and \n)
      const normalized = buffer.replace(/\r\n/g, "\n");
      const lines = normalized.split("\n\n");
      // Keep incomplete trailing segment in buffer
      buffer = lines.pop() || "";

      for (const block of lines) {
        const trimmed = block.trim();
        if (!trimmed.startsWith("data:")) continue;

        const payloadStr = trimmed.replace(/^data:\s*/, "").trim();
        if (!payloadStr) continue;

        try {
          const data = JSON.parse(payloadStr);
          if (data.type === "token" && onToken) {
            onToken(data.delta);
          } else if (data.type === "tool_start" && onToolStart) {
            onToolStart(data.tool, data.args);
          } else if (data.type === "tool_end" && onToolEnd) {
            onToolEnd(data.tool, data.output);
          } else if (data.type === "error" && onError) {
            onError(new Error(data.error));
          } else if (data.type === "done") {
            safeDone();
          }
        } catch {
          // Ignore malformed JSON chunks
        }
      }
    }

    safeDone();
  } catch (err: unknown) {
    if (onError && err instanceof Error) {
      onError(err);
    } else if (err instanceof Error) {
      throw err;
    } else {
      throw new Error(String(err));
    }
  }
}
