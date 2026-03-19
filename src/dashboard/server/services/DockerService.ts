import Docker from 'dockerode';
import type { VllmInstance } from '../types.js';

export interface DockerServiceOptions {
  socketPath?: string;
  imagePatterns?: string[];
}

interface DiscoveredContainer extends Omit<VllmInstance, 'vramUsedMb' | 'vramTotalMb'> {
  pid: number;
}

const DEFAULT_IMAGE_PATTERNS = ['vllm/*', 'rocm/vllm*'];
const DEFAULT_VLLM_PORT = 8000;

export class DockerService {
  private docker: Docker;
  private imagePatterns: string[];

  constructor(options: DockerServiceOptions = {}) {
    this.docker = new Docker({
      socketPath: options.socketPath ?? '/var/run/docker.sock',
    });
    this.imagePatterns = options.imagePatterns ?? DEFAULT_IMAGE_PATTERNS;
  }

  async discoverContainers(): Promise<DiscoveredContainer[]> {
    const allContainers = await this.docker.listContainers({ all: true });
    const vllmContainers = allContainers.filter((c) => this.matchesImagePattern(c.Image));

    const results: DiscoveredContainer[] = [];
    for (const container of vllmContainers) {
      const detail = await this.docker.getContainer(container.Id).inspect();
      results.push(this.parseContainer(detail));
    }
    return results;
  }

  async getContainerPids(): Promise<Record<string, number>> {
    const containers = await this.discoverContainers();
    const pids: Record<string, number> = {};
    for (const c of containers) {
      if (c.pid > 0) {
        pids[c.containerId] = c.pid;
      }
    }
    return pids;
  }

  getDocker(): Docker {
    return this.docker;
  }

  private matchesImagePattern(image: string): boolean {
    return this.imagePatterns.some((pattern) => {
      const regex = new RegExp('^' + pattern.replace(/\*/g, '.*') + '(:|$)');
      return regex.test(image);
    });
  }

  private parseContainer(detail: Docker.ContainerInspectInfo): DiscoveredContainer {
    const env = this.parseEnvVars(detail.Config.Env ?? []);
    const cmd = detail.Config.Cmd ?? [];
    const status = this.mapStatus(detail.State.Status);
    const port = this.extractPort(detail);
    const gpuIds = this.extractGpuIds(env);
    const tensorParallelSize = this.extractTensorParallel(cmd);
    const quantization = this.extractQuantization(cmd);
    const modelName = this.extractModelName(cmd, env);

    return {
      containerId: detail.Id,
      containerName: detail.Name.replace(/^\//, ''),
      status,
      modelName,
      gpuIds,
      tensorParallelSize,
      quantization: quantization ?? undefined,
      port,
      launchArgs: cmd,
      envVars: env,
      pid: detail.State.Pid,
    };
  }

  private parseEnvVars(envList: string[]): Record<string, string> {
    const env: Record<string, string> = {};
    for (const entry of envList) {
      const eqIdx = entry.indexOf('=');
      if (eqIdx > 0) {
        env[entry.substring(0, eqIdx)] = entry.substring(eqIdx + 1);
      }
    }
    return env;
  }

  private mapStatus(dockerStatus: string): VllmInstance['status'] {
    switch (dockerStatus) {
      case 'running': return 'running';
      case 'created':
      case 'restarting': return 'starting';
      case 'exited':
      case 'dead':
      case 'removing': return 'stopped';
      default: return 'error';
    }
  }

  private extractPort(detail: Docker.ContainerInspectInfo): number {
    const ports = detail.NetworkSettings?.Ports ?? {};
    for (const [, bindings] of Object.entries(ports)) {
      if (bindings && bindings.length > 0 && bindings[0].HostPort) {
        return parseInt(bindings[0].HostPort, 10);
      }
    }
    return DEFAULT_VLLM_PORT;
  }

  private extractGpuIds(env: Record<string, string>): string[] {
    const deviceVar =
      env['AMD_VISIBLE_DEVICES'] ??
      env['ROCR_VISIBLE_DEVICES'] ??
      env['CUDA_VISIBLE_DEVICES'];
    if (!deviceVar) return [];
    return deviceVar.split(',').map((s) => s.trim());
  }

  private extractTensorParallel(cmd: string[]): number {
    const idx = cmd.indexOf('--tensor-parallel-size');
    if (idx >= 0 && idx + 1 < cmd.length) {
      return parseInt(cmd[idx + 1], 10) || 1;
    }
    const tpIdx = cmd.indexOf('-tp');
    if (tpIdx >= 0 && tpIdx + 1 < cmd.length) {
      return parseInt(cmd[tpIdx + 1], 10) || 1;
    }
    return 1;
  }

  private extractQuantization(cmd: string[]): string | null {
    const idx = cmd.indexOf('--quantization');
    if (idx >= 0 && idx + 1 < cmd.length) {
      return cmd[idx + 1];
    }
    return null;
  }

  private extractModelName(cmd: string[], env: Record<string, string>): string {
    const idx = cmd.indexOf('--model');
    if (idx >= 0 && idx + 1 < cmd.length) {
      return cmd[idx + 1];
    }
    return env['MODEL_NAME'] ?? 'unknown';
  }
}
