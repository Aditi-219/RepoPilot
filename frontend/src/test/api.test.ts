import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  createRepository,
  deleteRepository,
  getRepositories,
  triggerRepositoryIndexing,
  updateRepositoryStatus,
} from "../services/api";

describe("api service", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("getRepositories returns list of repositories on 200", async () => {
    const mockData = [{ id: 1, full_name: "test/repo", status: "completed" }];
    (fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockData,
    });

    const result = await getRepositories();
    expect(result).toEqual(mockData);
  });

  it("createRepository sends POST with payload and returns new repo", async () => {
    const mockRepo = { id: 2, full_name: "test/new-repo", status: "pending" };
    (fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: async () => mockRepo,
    });

    const result = await createRepository("https://github.com/test/new-repo");
    expect(result).toEqual(mockRepo);
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/repositories"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ url: "https://github.com/test/new-repo" }),
      })
    );
  });

  it("updateRepositoryStatus sends PATCH with new status", async () => {
    const mockUpdated = { id: 1, status: "completed" };
    (fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockUpdated,
    });

    const result = await updateRepositoryStatus(1, "completed");
    expect(result).toEqual(mockUpdated);
  });

  it("deleteRepository handles 204 No Content cleanly", async () => {
    (fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 204,
      json: async () => ({}),
    });

    const result = await deleteRepository(1);
    expect(result).toBeNull();
  });

  it("throws error with server detail message on non-ok response", async () => {
    (fetch as any).mockResolvedValueOnce({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Repository not found." }),
    });

    await expect(getRepositories()).rejects.toThrow("Repository not found.");
  });

  it("triggerRepositoryIndexing sends POST to index endpoint", async () => {
    const mockIndexRes = { message: "Indexing started", repository_id: 1, status: "indexing" };
    (fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockIndexRes,
    });

    const result = await triggerRepositoryIndexing(1);
    expect(result).toEqual(mockIndexRes);
  });
});
