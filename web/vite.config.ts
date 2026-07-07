import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// En dev, on proxifie les routes de l'API vers FastAPI (port 8000) pour rester en
// same-origin (cookies de session SameSite=Lax). En prod, FastAPI sert le build.
const target = process.env.VITE_API_TARGET || "http://127.0.0.1:8000";
const proxy = Object.fromEntries(
  ["/auth", "/collections", "/jobs", "/health"].map((p) => [
    p,
    { target, changeOrigin: true },
  ]),
);

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
});
