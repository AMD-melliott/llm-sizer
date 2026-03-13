# Server Dashboard — Design Spec

## Overview

A monitoring dashboard for vLLM inference services running on a local GPU server. Discovers running vLLM Docker containers automatically, displays their status, GPU/VRAM usage, inference metrics, and provides live container log streaming. Runs as a separate backend service within the llm-sizer repo.

**Target users:** The llm-sizer team, running vLLM on AMD Instinct GPU servers.

## Architecture

### Deployment Model

- Runs on the same machine as the GPU server and vLLM containers
- Direct access to Docker socket (`/var/run/docker.sock`) and `amd-smi` CLI
- No authentication, no remote access — local tool for the team

### System Components

```
┌─────────────────────────────────────────────────┐
│  Browser (React Dashboard UI)                   │
│  - Summary bar (instance count, VRAM, requests) │
│  - vLLM instance card grid                      │
│  - Log viewer (WebSocket)                       │
└──────────┬──────────────────┬───────────────────┘
           │ REST              │ WebSocket
           ▼                   ▼
┌─────────────────────────────────────────────────┐
│  Dashboard Backend (Fastify)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│  │ Docker   │ │ GPU      │ │ Log Stream       ││
│  │ Service  │ │ Service  │ │ Manager          ││
│  └────┬─────┘ └────┬─────┘ └────────┬─────────┘│
└───────┼─────────────┼────────────────┼──────────┘
        │             │                │
        ▼             ▼                ▼
   Docker Socket   amd-smi CLI    docker logs -f
                   Prometheus
```

### Tech Stack

- **Backend:** Fastify + @fastify/websocket + @fastify/static (TypeScript)
- **Docker API:** dockerode
- **GPU metrics:** amd-smi CLI (JSON output), with optional Prometheus scraping
- **Frontend:** React + Tailwind CSS (same stack as existing llm-sizer)
- **Build:** tsup (same as existing CLI)

## Backend Services

### DockerService

Connects to Docker socket via `dockerode`. Discovers vLLM containers by matching image names against a configurable pattern (defaults to `vllm/*`, `rocm/vllm*`).

For each discovered container, extracts:
- Container name, ID, status, creation time
- Port mappings
- Environment variables: model name, tensor parallel size, GPU assignments (`AMD_VISIBLE_DEVICES`, fallback `ROCR_VISIBLE_DEVICES` for older setups, `CUDA_VISIBLE_DEVICES` for future NVIDIA support)
- Resource limits
- Launch command / entrypoint args (vLLM parameters)

### GpuMetricsService

Uses a `GpuMetricsProvider` interface for vendor abstraction.

```typescript
interface GpuMetricsProvider {
  detect(): Promise<boolean>
  getDevices(): Promise<GpuDevice[]>
  getMetrics(): Promise<GpuMetrics[]>
  getTopology(): Promise<GpuTopology>
  getProcesses(): Promise<GpuProcess[]>
}

interface GpuDevice {
  id: string
  physicalId: number
  logicalId?: number
  name: string
  vramTotalMb: number
  partitionMode?: 'SPX' | 'DPX' | 'CPX'
  partitionIndex?: number
}

interface GpuMetrics {
  deviceId: string
  vramUsedMb: number
  vramTotalMb: number
  utilizationPercent: number
  temperatureC: number
  powerW: number
}

interface GpuProcess {
  deviceId: string
  pid: number
  vramUsedMb: number
  processName: string
}
```

**AmdSmiProvider** implements this interface:
- `detect()` — checks if `amd-smi` is on PATH
- `getDevices()` / `getMetrics()` — parses `amd-smi metric --json`
- `getTopology()` — parses `amd-smi topology --json` for partition mode detection
- `getProcesses()` — parses `amd-smi process --json`

**Container-to-GPU mapping:**
1. Get the top-level PID for each vLLM container via Docker inspect (`State.Pid`)
2. Get child PIDs of that process
3. Match against `GpuProcess.pid` from amd-smi
4. Cross-reference `AMD_VISIBLE_DEVICES` env var from container config for expected assignment

This provides both "which GPU was assigned" (env vars) and "how much VRAM is actually used" (process mapping).

NVIDIA support can be added later by implementing `NvidiaSmiProvider` against the same interface.

### VllmMetricsService

For each discovered vLLM container with an exposed HTTP port, queries:
- `GET /v1/models` — loaded model info
- `GET /metrics` — Prometheus-format text, parsed for:
  - `vllm:num_requests_running`
  - `vllm:num_requests_waiting`
  - `vllm:gpu_cache_usage_perc`

Handles containers that aren't ready yet (starting up, loading model) gracefully — marks them as "Starting" rather than erroring.

### MetricsCollector

Orchestrates polling across all three services:
- Runs on a configurable interval (default 5s)
- Each tick concurrently fetches Docker state, GPU metrics, and vLLM metrics
- Merges results into a single `DashboardSnapshot` held in memory
- Multiple browser clients share one polling cycle — no N x M amplification

### Optional Prometheus Integration

If a Prometheus endpoint is configured (via CLI flag), the GpuMetricsService also scrapes time-series GPU utilization data. This is additive — the dashboard works without Prometheus, but can display richer data when it's available.

## API Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/status` | Server health + snapshot timestamp + summary stats |
| `GET` | `/api/instances` | All vLLM instances with Docker + vLLM metrics merged |
| `GET` | `/api/instances/:id` | Single instance detail |
| `GET` | `/api/gpus` | GPU state with partition info + VRAM breakdown |
| `GET` | `/api/gpus/:id/metrics` | Time-series metrics for a GPU (requires Prometheus) |
| `WS` | `/api/logs/:containerId` | WebSocket stream of container logs |

## Log Streaming

### WebSocket Protocol

Connection: `ws://localhost:<port>/api/logs/<containerId>?tail=200`

**Lifecycle:**
1. Client connects with optional `tail` param (default 200 lines of history)
2. Server attaches to `docker logs --follow --tail=200` for that container
3. Log lines streamed as JSON: `{ "timestamp": "...", "stream": "stdout"|"stderr", "line": "..." }`
4. `stderr` lines rendered in a different color in the UI
5. Client disconnect → server detaches from Docker log stream
6. Container stops → server sends `{ "type": "closed", "reason": "container_stopped" }` and closes

**Multiple viewers:** Each WebSocket gets its own connection, but the server can share a single Docker log stream internally.

**Backpressure:** Server buffers up to 1000 undelivered lines, then drops oldest and sends `{ "type": "dropped", "count": N }`.

## Frontend

### Polling Strategy

- On load: fetches `GET /api/instances` and `GET /api/gpus`
- Then polls `GET /api/status` every 5s (lightweight — just timestamp + summary)
- If snapshot timestamp changed, fetches full data
- Poll interval auto-detected from backend response

### Layout: Summary Bar + Instance Grid

**Summary bar** (top strip, 4 stat cards):
- Instance count
- Total VRAM available
- Total VRAM used
- Total active requests

**Instance grid** (responsive card grid below):

Each card shows:
- **Header:** Model name, status badge (Running/Starting/Stopped/Error), container name
- **Info line:** GPU assignment (e.g., "GPU 0 (MI325X)"), partition info if applicable (e.g., "CPX-P0"), quantization, tensor parallel size
- **VRAM bar:** Horizontal progress bar, used/total in GB, color-coded (blue <70%, orange 70-90%, red >90%)
- **Metrics row:** KV cache usage %, running request count, waiting request count
- **Expandable: Launch parameters** — full vLLM command/env vars from container config
- **Expandable: Container logs** — live-streaming log tail in a terminal-styled panel below the card, auto-scroll with pause button

### Dashboard UI Components

```
src/dashboard/ui/
├── DashboardPage.tsx        # Main dashboard view
├── SummaryBar.tsx           # Top stats strip
├── InstanceCard.tsx         # Individual vLLM instance card
├── InstanceGrid.tsx         # Responsive grid of cards
├── LogViewer.tsx            # Terminal-styled log panel
└── hooks/
    ├── useDashboardData.ts  # REST polling hook
    └── useLogStream.ts      # WebSocket log stream hook
```

## Project Structure

```
llm-sizer/
├── src/
│   ├── dashboard/
│   │   ├── server/
│   │   │   ├── index.ts                 # Fastify server entry point
│   │   │   ├── services/
│   │   │   │   ├── DockerService.ts
│   │   │   │   ├── GpuMetricsService.ts
│   │   │   │   ├── VllmMetricsService.ts
│   │   │   │   └── MetricsCollector.ts
│   │   │   ├── providers/
│   │   │   │   ├── GpuMetricsProvider.ts   # Interface
│   │   │   │   └── AmdSmiProvider.ts
│   │   │   ├── routes/
│   │   │   │   ├── status.ts
│   │   │   │   ├── instances.ts
│   │   │   │   ├── gpus.ts
│   │   │   │   └── logs.ts
│   │   │   └── types.ts
│   │   └── ui/
│   │       ├── DashboardPage.tsx
│   │       ├── SummaryBar.tsx
│   │       ├── InstanceCard.tsx
│   │       ├── InstanceGrid.tsx
│   │       ├── LogViewer.tsx
│   │       └── hooks/
│   │           ├── useDashboardData.ts
│   │           └── useLogStream.ts
│   ├── components/          # Existing calculator UI
│   ├── cli/                 # Existing CLI tool
│   ├── utils/               # Existing calculators (shared)
│   ├── data/                # Existing GPU/model data (shared)
│   └── types/               # Existing types (shared)
```

### Running

- `npm run dashboard` — starts the Fastify backend (default port 3001), serves the dashboard UI
- Built with `tsup` (separate entry point, like the existing CLI)

### Dependencies to Add

- `fastify` + `@fastify/websocket` + `@fastify/static`
- `dockerode` + `@types/dockerode`

No new frontend dependencies — reuses existing React, Tailwind, Recharts.

## Assumptions

- Docker socket accessible at `/var/run/docker.sock`
- `amd-smi` installed and on PATH
- vLLM containers expose their HTTP port (for `/metrics` and `/v1/models`)
- vLLM container images identifiable by image name pattern (configurable, defaults to `vllm/*`, `rocm/vllm*`)

## Out of Scope

- Starting/stopping vLLM instances from docker-compose files (separate PRD)
- NVIDIA GPU provider implementation (interface is ready, implementation deferred)
- Time-series charts / historical metrics (would need Prometheus + storage)
- Authentication / multi-user access
- Log search / filtering / persistence
- Alerting on GPU utilization thresholds
- Integration as a tab in the main llm-sizer Vite app (can be added later)
