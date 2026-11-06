import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// Automatically unmount rendered components after each test
afterEach(() => {
  cleanup();
});
