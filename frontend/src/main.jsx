/**
 * Client Entry Point for RepoPilot Frontend.
 *
 * This file mounts the React root component onto the DOM and imports
 * global styles:
 * 1. Bootstrap CSS: Provides base layout, cards, badges, and responsive tables.
 * 2. App.css: Custom design tokens, typography, and styling overrides.
 */

import React from "react";
import ReactDOM from "react-dom/client";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";
import App from "./App";

// Locate root DOM container in index.html and initialize React 18/19 concurrent root
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);