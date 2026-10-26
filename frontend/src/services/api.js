/**
 * API Service for RepoPilot Frontend.
 *
 * This module centralizes all HTTP network requests made to the FastAPI backend.
 * By keeping network logic in one place, UI components remain clean and focused
 * strictly on user interaction and rendering.
 */

// Base URL for the backend API.
// Reads from Vite environment variable (VITE_API_URL) if configured,
// otherwise defaults to local FastAPI server running on port 8000.
export const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api";

/**
 * Universal wrapper around fetch() that parses JSON and provides clean error messages.
 *
 * @param {string} endpoint - API path (e.g. '/repositories')
 * @param {RequestInit} [options] - Standard fetch options (method, body, headers)
 * @returns {Promise<any>} The parsed JSON response body
 * @throws {Error} User-readable error message if the server responds with a non-2xx status
 */
export async function apiFetch(endpoint, options = {}) {
  const url = endpoint.startsWith("http")
    ? endpoint
    : `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  // Handle 204 No Content (e.g. successful DELETE)
  if (response.status === 204) {
    return null;
  }

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    // FastAPI returns errors under { "detail": "message" }
    const message =
      typeof data.detail === "string" ? data.detail : "Request failed.";
    throw new Error(message);
  }

  return data;
}

// ---------------------------------------------------------------------------
// Repository Endpoints
// ---------------------------------------------------------------------------

/**
 * Fetch the list of all registered repositories.
 * GET /api/repositories
 */
export async function getRepositories() {
  return apiFetch("/repositories");
}

/**
 * Register a new public GitHub repository.
 * POST /api/repositories
 *
 * @param {string} repoUrl - Valid public GitHub repository URL
 */
export async function createRepository(repoUrl) {
  return apiFetch("/repositories", {
    method: "POST",
    body: JSON.stringify({ url: repoUrl.trim() }),
  });
}

/**
 * Update a repository's processing status manually.
 * PATCH /api/repositories/:id/status
 *
 * @param {number} repoId
 * @param {'pending' | 'indexing' | 'completed' | 'failed'} status
 */
export async function updateRepositoryStatus(repoId, status) {
  return apiFetch(`/repositories/${repoId}/status`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
  });
}

/**
 * Delete a repository and its indexed vector chunks.
 * DELETE /api/repositories/:id
 *
 * @param {number} repoId
 */
export async function deleteRepository(repoId) {
  return apiFetch(`/repositories/${repoId}`, {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Indexing Endpoints
// ---------------------------------------------------------------------------

/**
 * Trigger background indexing (token chunking & vector embedding) for a repository.
 * POST /api/repositories/:id/index
 *
 * @param {number} repoId
 */
export async function triggerRepositoryIndexing(repoId) {
  return apiFetch(`/repositories/${repoId}/index`, {
    method: "POST",
  });
}

// ---------------------------------------------------------------------------
// RAG & Search Endpoints
// ---------------------------------------------------------------------------

/**
 * Perform hybrid semantic + full-text search across indexed chunks.
 * POST /api/repositories/:id/search
 *
 * @param {number|string} repoId
 * @param {string} query
 * @param {number} [limit=8]
 * @param {boolean} [rerank=true]
 */
export async function searchRepository(repoId, query, limit = 8, rerank = true) {
  return apiFetch(`/repositories/${repoId}/search`, {
    method: "POST",
    body: JSON.stringify({ query, limit, rerank }),
  });
}

/**
 * Ask a codebase question with grounded AI answer generation.
 * POST /api/repositories/:id/ask
 *
 * @param {number|string} repoId
 * @param {string} question
 * @param {number} [topK=8]
 */
export async function askRepository(repoId, question, topK = 8) {
  return apiFetch(`/repositories/${repoId}/ask`, {
    method: "POST",
    body: JSON.stringify({ question, top_k: topK }),
  });
}

// ---------------------------------------------------------------------------
// LangGraph Chat Sessions & Streaming Endpoints
// ---------------------------------------------------------------------------

/**
 * List all chat sessions for a repository.
 * GET /api/chat/:repoId/sessions
 *
 * @param {number|string} repoId
 */
export async function listChatSessions(repoId) {
  return apiFetch(`/chat/${repoId}/sessions`);
}

/**
 * Create a new chat session (thread_id) for a repository.
 * POST /api/chat/:repoId/sessions
 *
 * @param {number|string} repoId
 * @param {string} [title]
 */
export async function createChatSession(repoId, title = "New Conversation") {
  return apiFetch(`/chat/${repoId}/sessions`, {
    method: "POST",
    body: JSON.stringify({ title }),
  });
}

/**
 * Retrieve message history for a chat session from checkpoints.
 * GET /api/chat/sessions/:sessionId/history
 *
 * @param {string} sessionId
 */
export async function getChatHistory(sessionId) {
  return apiFetch(`/chat/sessions/${sessionId}/history`);
}

/**
 * Delete a chat session.
 * DELETE /api/chat/sessions/:sessionId
 *
 * @param {string} sessionId
 */
export async function deleteChatSession(sessionId) {
  return apiFetch(`/chat/sessions/${sessionId}`, {
    method: "DELETE",
  });
}

/**
 * Stream conversational agent response using Server-Sent Events (SSE).
 * POST /api/chat/:repoId/sessions/:sessionId/stream
 *
 * @param {number|string} repoId
 * @param {string} sessionId
 * @param {string} message
 * @param {Object} handlers
 * @param {(token: string) => void} handlers.onToken - Called on each streaming text token delta
 * @param {(tool: string, args: any) => void} [handlers.onToolStart] - Called when a tool begins
 * @param {(tool: string, output: string) => void} [handlers.onToolEnd] - Called when a tool finishes
 * @param {(err: Error) => void} [handlers.onError] - Called on network or generation error
 * @param {() => void} [handlers.onDone] - Called when stream completes
 */
export async function streamChatMessage(
  repoId,
  sessionId,
  message,
  { onToken, onToolStart, onToolEnd, onError, onDone } = {}
) {
  const url = `${API_BASE_URL}/chat/${repoId}/sessions/${sessionId}/stream`;

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.detail || `Stream failed with status ${response.status}`);
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
      const lines = buffer.split("\n\n");
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
  } catch (err) {
    if (onError) onError(err);
    else throw err;
  }
}
