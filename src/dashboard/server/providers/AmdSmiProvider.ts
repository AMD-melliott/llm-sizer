import { execFile as execFileCb } from 'child_process';
import { promisify } from 'util';
import { accessSync, constants } from 'fs';
import type {
  GpuMetricsProvider,
  GpuDevice,
  GpuMetrics,
  GpuProcess,
  GpuTopology,
} from '../types.js';

const execFile = promisify(execFileCb);

const KFD_DEVICE = '/dev/kfd';

function isKfdAccessible(): boolean {
  try {
    accessSync(KFD_DEVICE, constants.R_OK);
    return true;
  } catch {
    return false;
  }
}

export class AmdSmiProvider implements GpuMetricsProvider {
  async detect(): Promise<boolean> {
    // Fast pre-flight: if the KFD device isn't accessible, amd-smi will hang
    // in an uninterruptible kernel sleep (D state) that no signal can escape.
    if (!isKfdAccessible()) {
      return false;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    try {
      await execFile('amd-smi', ['version'], { signal: controller.signal });
      return true;
    } catch {
      return false;
    } finally {
      clearTimeout(timer);
    }
  }

  async getDevices(): Promise<GpuDevice[]> {
    const raw = await this.runAmdSmi(['metric', '--json']);
    const gpus = JSON.parse(raw);
    return gpus.map((g: any) => ({
      id: `gpu-${g.gpu}`,
      physicalId: g.gpu,
      name: g.name,
      vramTotalMb: g.vram_total,
    }));
  }

  async getMetrics(): Promise<GpuMetrics[]> {
    const raw = await this.runAmdSmi(['metric', '--json']);
    const gpus = JSON.parse(raw);
    return gpus.map((g: any) => ({
      deviceId: `gpu-${g.gpu}`,
      vramUsedMb: g.vram_used,
      vramTotalMb: g.vram_total,
      utilizationPercent: g.gpu_use_percent,
      temperatureC: g.temperature,
      powerW: g.power,
    }));
  }

  async getProcesses(): Promise<GpuProcess[]> {
    const raw = await this.runAmdSmi(['process', '--json']);
    const procs = JSON.parse(raw);
    return procs.map((p: any) => ({
      deviceId: `gpu-${p.gpu}`,
      pid: p.pid,
      vramUsedMb: p.vram_usage,
      processName: p.process_name,
    }));
  }

  async getTopology(): Promise<GpuTopology> {
    const raw = await this.runAmdSmi(['topology', '--json']);
    const topo = JSON.parse(raw);
    return {
      partitionMode: topo.partition_mode ?? 'unknown',
      physicalGpus: (topo.gpus ?? []).map((g: any) => ({
        physicalId: g.gpu,
        name: g.name,
        partitions: (g.partitions ?? []).map((p: any) => ({
          logicalId: p.logical_id,
          vramTotalMb: p.vram_total,
        })),
      })),
    };
  }

  private async runAmdSmi(args: string[]): Promise<string> {
    const { stdout } = await execFile('amd-smi', args, { timeout: 10000, killSignal: 'SIGKILL' });
    return stdout;
  }
}
