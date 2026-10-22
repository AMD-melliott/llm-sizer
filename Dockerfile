# syntax=docker/dockerfile:1

# ─── Stage 1: Build the static SPA ────────────────────────────────────────────
FROM docker.io/library/node:20-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

COPY . .
# base defaults to /llm-sizer/ (vite.config.ts) — matches the platform
# path-under-apex URL https://slai-app.amd.com/llm-sizer/.
RUN npm run build

# ─── Stage 2: Serve via hardened non-root nginx ──────────────────────────────
FROM docker.io/nginxinc/nginx-unprivileged:1.27-alpine AS runtime

# Unprivileged image already runs as a non-root user; the platform deployment
# further pins runAsUser 65534 with a read-only rootfs and in-memory mounts for
# /run and /var/cache/nginx.
COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=build /app/dist /usr/share/nginx/html/llm-sizer

EXPOSE 8080
