import { listGPUs, showGPU, compareGPUs } from '../../../src/cli/commands/gpus';

describe('GPUs Command', () => {
  describe('listGPUs', () => {
    test('returns all GPUs when no filter', () => {
      const gpus = listGPUs({});
      expect(gpus.length).toBeGreaterThan(0);
      expect(gpus[0]).toHaveProperty('id');
      expect(gpus[0]).toHaveProperty('vram_gb');
    });

    test('filters by vendor', () => {
      const gpus = listGPUs({ vendor: 'AMD' });
      expect(gpus.length).toBeGreaterThan(0);
      for (const g of gpus) {
        expect(g.vendor).toBe('AMD');
      }
    });

    test('filters by tier', () => {
      const gpus = listGPUs({ tier: 'datacenter' });
      expect(gpus.length).toBeGreaterThan(0);
      for (const g of gpus) {
        expect(g.tier).toBe('datacenter');
      }
    });

    test('filters by minimum VRAM', () => {
      const gpus = listGPUs({ minVram: 80 });
      for (const g of gpus) {
        expect(g.vram_gb).toBeGreaterThanOrEqual(80);
      }
    });

    test('filters by partitioning support', () => {
      const gpus = listGPUs({ partitioning: true });
      expect(gpus.length).toBeGreaterThan(0);
      for (const g of gpus) {
        expect(g.partitioning?.supported).toBe(true);
      }
    });

    test('filters by vendor NVIDIA', () => {
      const gpus = listGPUs({ vendor: 'NVIDIA' });
      expect(gpus.length).toBeGreaterThan(0);
      for (const g of gpus) {
        expect(g.vendor).toBe('NVIDIA');
      }
    });

    test('returns empty array when no GPUs match filter', () => {
      const gpus = listGPUs({ minVram: 99999 });
      expect(gpus).toEqual([]);
    });

    test('combines multiple filters', () => {
      const gpus = listGPUs({ vendor: 'AMD', tier: 'datacenter', minVram: 100 });
      expect(gpus.length).toBeGreaterThan(0);
      for (const g of gpus) {
        expect(g.vendor).toBe('AMD');
        expect(g.tier).toBe('datacenter');
        expect(g.vram_gb).toBeGreaterThanOrEqual(100);
      }
    });
  });

  describe('showGPU', () => {
    test('returns GPU details for valid ID', () => {
      const gpu = showGPU('mi300x');
      expect(gpu).not.toBeNull();
      expect(gpu!.id).toBe('mi300x');
    });

    test('returns null for invalid ID', () => {
      expect(showGPU('nonexistent')).toBeNull();
    });

    test('returns full GPU details', () => {
      const gpu = showGPU('mi300x');
      expect(gpu).not.toBeNull();
      expect(gpu!).toHaveProperty('vendor');
      expect(gpu!).toHaveProperty('vram_gb');
      expect(gpu!).toHaveProperty('memory_bandwidth_gbps');
      expect(gpu!).toHaveProperty('compute_tflops_fp16');
    });
  });

  describe('compareGPUs', () => {
    test('returns comparison data for valid GPU IDs', () => {
      const result = compareGPUs(['mi300x', 'a100-80gb']);
      expect(result.length).toBe(2);
      expect(result[0].id).toBe('mi300x');
      expect(result[1].id).toBe('a100-80gb');
    });

    test('skips unknown GPU IDs', () => {
      const result = compareGPUs(['mi300x', 'nonexistent']);
      expect(result.length).toBe(1);
      expect(result[0].id).toBe('mi300x');
    });

    test('returns empty array when all IDs are unknown', () => {
      const result = compareGPUs(['nonexistent1', 'nonexistent2']);
      expect(result).toEqual([]);
    });

    test('handles single GPU ID', () => {
      const result = compareGPUs(['mi300x']);
      expect(result.length).toBe(1);
      expect(result[0].id).toBe('mi300x');
    });
  });
});
