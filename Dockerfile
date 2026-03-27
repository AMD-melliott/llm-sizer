# syntax=docker/dockerfile:1

# ─── Stage 1: Node base with ROCm ────────────────────────────────────────────
# rocm/dev-ubuntu-24.04:7.2.1 includes amd-smi, ROCm runtime, and system tools.
FROM rocm/dev-ubuntu-24.04:7.2.1 AS node-base

# Install Node.js 20 (LTS) via NodeSource
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g npm@latest \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Stage 2: Install all dependencies ───────────────────────────────────────
FROM node-base AS deps

COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

# ─── Stage 3: Build the main Vite app (UI only) ───────────────────────────────
FROM deps AS build-ui

# Build-time arg for the dashboard backend URL (embedded into the JS bundle)
ARG VITE_DASHBOARD_URL=""

COPY . .
# VITE_DASHBOARD_URL is baked into the JS bundle at build time so the tab
# appears when the env var is non-empty. VITE_BASE=/ overrides the GitHub
# Pages base path for container deployment.
RUN VITE_BASE=/ VITE_DASHBOARD_URL="${VITE_DASHBOARD_URL}" npm run build

# ─── Stage 4: Build the dashboard UI ─────────────────────────────────────────
FROM deps AS build-dashboard-ui

COPY . .
# Build the standalone dashboard UI (outputs to dist/dashboard/)
RUN npm run build:dashboard-ui

# ─── Stage 5: Build the dashboard server bundle ───────────────────────────────
FROM deps AS build-dashboard-server

COPY . .
# Bundle the Fastify dashboard server (outputs to dist/dashboard.js)
RUN npm run build:dashboard

# ─── Target: llm-sizer (main Vite app, served via vite preview) ──────────────
FROM node-base AS llm-sizer

WORKDIR /app

# vite preview only needs the dist folder and vite itself.
# --base / overrides the GitHub Pages base path baked into vite.config.ts.
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts
COPY --from=build-ui /app/dist ./dist

EXPOSE 4173

CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "4173", "--base", "/"]

# ─── Target: dashboard (Fastify backend + dashboard UI) ───────────────────────
FROM node-base AS dashboard

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts

# Pre-built dashboard server bundle and UI assets
COPY --from=build-dashboard-server /app/dist/dashboard.js ./dist/
COPY --from=build-dashboard-ui /app/dist/dashboard ./dist/dashboard

EXPOSE 3001

ENV HOST=0.0.0.0 \
    PORT=3001

CMD ["node", "dist/dashboard.js", "--port", "3001"]
