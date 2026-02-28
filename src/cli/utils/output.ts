export type OutputFormat = 'json' | 'table';

export function detectOutputFormat(explicit?: string): OutputFormat {
  if (explicit === 'json') return 'json';
  if (explicit === 'table') return 'table';
  return 'table';
}

export function formatOutput(data: Record<string, unknown>, format: OutputFormat): string {
  if (format === 'json') {
    return JSON.stringify(data, null, 2);
  }

  const lines: string[] = [];
  for (const [key, value] of Object.entries(data)) {
    if (typeof value === 'object' && value !== null) {
      lines.push(`${key}:`);
      for (const [subKey, subValue] of Object.entries(value as Record<string, unknown>)) {
        lines.push(`  ${subKey}: ${subValue}`);
      }
    } else {
      lines.push(`${key}: ${value}`);
    }
  }
  return lines.join('\n');
}
