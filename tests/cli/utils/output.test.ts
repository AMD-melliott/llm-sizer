import { formatOutput, detectOutputFormat, OutputFormat } from '../../../src/cli/utils/output';

describe('Output Utilities', () => {
  describe('detectOutputFormat', () => {
    test('returns explicit format when specified', () => {
      expect(detectOutputFormat('json')).toBe('json');
      expect(detectOutputFormat('table')).toBe('table');
    });

    test('returns table as default for undefined', () => {
      expect(detectOutputFormat(undefined)).toBe('table');
    });
  });

  describe('formatOutput', () => {
    const data = {
      model: 'llama-3-70b',
      usedVRAM: 142.3,
      totalVRAM: 384.0,
    };

    test('json format returns valid JSON string', () => {
      const output = formatOutput(data, 'json');
      const parsed = JSON.parse(output);
      expect(parsed.model).toBe('llama-3-70b');
      expect(parsed.usedVRAM).toBe(142.3);
    });

    test('json format is pretty-printed', () => {
      const output = formatOutput(data, 'json');
      expect(output).toContain('\n');
    });

    test('table format returns non-empty string', () => {
      const output = formatOutput(data, 'table');
      expect(output.length).toBeGreaterThan(0);
    });
  });
});
