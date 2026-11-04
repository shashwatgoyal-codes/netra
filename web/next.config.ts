import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Fully static: the app, model and runtime are plain files any host can
  // serve, and the service worker can cache all of it for offline clinics.
  output: "export",
  trailingSlash: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
