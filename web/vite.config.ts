import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" lets the built app be served from any sub-path. In dev, /api goes to the Python server
// (python -m server), which serves the built app itself in production.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
