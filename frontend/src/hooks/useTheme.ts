/**
 * Theme Management Hook for RepoPilot
 *
 * Supports:
 * - User preference persistence in localStorage ("light" | "dark")
 * - Default fallback to system preference (prefers-color-scheme)
 * - Dynamic system preference listener when no manual override is set
 * - Direct synchronization with document.documentElement[data-theme]
 */

import { useEffect, useState } from "react";
import type { Theme } from "../types";

const THEME_STORAGE_KEY = "repopilot-theme";

/**
 * Determine the initial theme state.
 * Priority:
 * 1. Saved preference in localStorage
 * 2. System preference via window.matchMedia
 * 3. Default to "light"
 */
function getInitialTheme(): Theme {
  if (typeof window === "undefined") {
    return "light";
  }

  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === "light" || saved === "dark") {
      return saved;
    }
  } catch {
    // localStorage may be inaccessible in private browsing or iframe
  }

  if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    return "dark";
  }

  return "light";
}

export function useTheme(): { theme: Theme; toggleTheme: () => void } {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  // Synchronize document attribute and color-scheme CSS property
  useEffect(() => {
    if (typeof document === "undefined") return;
    const root = document.documentElement;
    root.setAttribute("data-theme", theme);
    root.style.colorScheme = theme;
  }, [theme]);

  // Listen for system theme changes if user hasn't explicitly set a preference
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const handleSystemChange = (event: MediaQueryListEvent) => {
      try {
        const saved = localStorage.getItem(THEME_STORAGE_KEY);
        // Only react to OS changes if user hasn't stored a manual preference
        if (!saved) {
          const systemTheme: Theme = event.matches ? "dark" : "light";
          setTheme(systemTheme);
        }
      } catch {
        // Ignore storage errors
      }
    };

    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener("change", handleSystemChange);
    } else {
      // Fallback for older browsers
      mediaQuery.addListener(handleSystemChange);
    }

    return () => {
      if (mediaQuery.removeEventListener) {
        mediaQuery.removeEventListener("change", handleSystemChange);
      } else {
        mediaQuery.removeListener(handleSystemChange);
      }
    };
  }, []);

  /** Toggle between light and dark modes */
  const toggleTheme = () => {
    setTheme((prevTheme) => {
      const nextTheme: Theme = prevTheme === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
      } catch {
        // Ignore storage errors
      }
      return nextTheme;
    });
  };

  return { theme, toggleTheme };
}

export default useTheme;
