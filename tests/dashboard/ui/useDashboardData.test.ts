import { parseDashboardStatus, shouldRefreshData } from '../../../src/dashboard/ui/hooks/useDashboardData';

describe('useDashboardData helpers', () => {
  test('parseDashboardStatus extracts fields correctly', () => {
    const raw = {
      timestamp: 1710300000000,
      pollIntervalMs: 5000,
      summary: { instanceCount: 2, totalVramMb: 393216, usedVramMb: 130000, totalActiveRequests: 5 },
      warnings: [],
    };
    const result = parseDashboardStatus(raw);
    expect(result.timestamp).toBe(1710300000000);
    expect(result.summary.instanceCount).toBe(2);
  });

  test('shouldRefreshData returns true when timestamp changes', () => {
    expect(shouldRefreshData(1000, 2000)).toBe(true);
  });

  test('shouldRefreshData returns false when timestamp is same', () => {
    expect(shouldRefreshData(1000, 1000)).toBe(false);
  });

  test('shouldRefreshData returns true on first load (null previous)', () => {
    expect(shouldRefreshData(null, 1000)).toBe(true);
  });
});
