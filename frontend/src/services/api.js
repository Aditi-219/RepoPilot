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
