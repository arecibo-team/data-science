import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" lets the built app be served from any sub-path with its data/ folder.
export default defineConfig({
  base: "./",
  plugins: [react()],
});
