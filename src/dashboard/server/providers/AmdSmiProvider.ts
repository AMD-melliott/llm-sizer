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
    const [metricRaw, staticRaw] = await Promise.all([
      this.runAmdSmi(['metric', '--json']),
      this.runAmdSmi(['static', '--json']),
    ]);
    const metricData: any[] = JSON.parse(metricRaw).gpu_data ?? [];
    const staticData: any[] = JSON.parse(staticRaw).gpu_data ?? [];
    const nameMap = new Map<number, string>(
      staticData.map((g: any) => [g.gpu, g.asic?.market_name ?? `GPU ${g.gpu}`])
    );
    return metricData.map((g: any) => ({
      id: `gpu-${g.gpu}`,
      physicalId: g.gpu,
      name: nameMap.get(g.gpu) ?? `GPU ${g.gpu}`,
      vramTotalMb: g.mem_usage?.total_vram?.value ?? 0,
    }));
  }

  async getMetrics(): Promise<GpuMetrics[]> {
    const raw = await this.runAmdSmi(['metric', '--json']);
    const gpus: any[] = JSON.parse(raw).gpu_data ?? [];
    return gpus.map((g: any) => ({
      deviceId: `gpu-${g.gpu}`,
      vramUsedMb: g.mem_usage?.used_vram?.value ?? 0,
      vramTotalMb: g.mem_usage?.total_vram?.value ?? 0,
      utilizationPercent: g.usage?.gfx_activity?.value ?? 0,
      temperatureC: g.temperature?.edge?.value ?? 0,
      powerW: g.power?.socket_power?.value ?? 0,
    }));
  }

  async getProcesses(): Promise<GpuProcess[]> {
    const raw = await this.runAmdSmi(['process', '--json']);
    const gpus: any[] = JSON.parse(raw) ?? [];
    const results: GpuProcess[] = [];
    for (const gpu of gpus) {
      for (const entry of gpu.process_list ?? []) {
        const info = entry.process_info ?? entry;
        const vramBytes = info.memory_usage?.vram_mem?.value ?? info.mem_usage?.value ?? 0;
        const vramUnit = info.memory_usage?.vram_mem?.unit ?? info.mem_usage?.unit ?? 'B';
        results.push({
          deviceId: `gpu-${gpu.gpu}`,
          pid: info.pid,
          vramUsedMb: vramUnit === 'MB' ? vramBytes : Math.round(vramBytes / (1024 * 1024)),
          processName: info.name ?? 'unknown',
        });
      }
    }
    return results;
  }

  async getTopology(): Promise<GpuTopology> {
    const raw = await this.runAmdSmi(['topology', '--json']);
    const parsed = JSON.parse(raw);
    const gpus: any[] = parsed.gpu_data ?? parsed.gpus ?? [];
    return {
      partitionMode: parsed.partition_mode ?? 'unknown',
      physicalGpus: gpus.map((g: any) => ({
        physicalId: g.gpu,
        name: g.asic?.market_name ?? `GPU ${g.gpu}`,
        partitions: (g.partitions ?? []).map((p: any) => ({
          logicalId: p.logical_id,
          vramTotalMb: p.vram_total?.value ?? p.vram_total ?? 0,
        })),
      })),
    };
  }

  private async runAmdSmi(args: string[]): Promise<string> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    try {
      const { stdout } = await execFile('amd-smi', args, { signal: controller.signal });
      return stdout;
    } finally {
      clearTimeout(timer);
    }
  }
}
