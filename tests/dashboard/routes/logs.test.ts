import { LogStreamManager } from '../../../src/dashboard/server/routes/logs';
import { Readable } from 'stream';

describe('LogStreamManager', () => {
  test('parses Docker log stream into LogMessage events', async () => {
    const manager = new LogStreamManager();
    const messages: any[] = [];

    const mockStream = new Readable({ read() {} });

    const sendFn = jest.fn((data: string) => {
      messages.push(JSON.parse(data));
    });

    manager.attachStream(mockStream, sendFn);

    mockStream.push('2026-03-13T10:00:00.000Z INFO: Server started\n');
    await new Promise((r) => setTimeout(r, 10));

    expect(messages.length).toBeGreaterThanOrEqual(1);
    expect(messages[0].type).toBe('log');
    expect(messages[0].line).toContain('Server started');
  });

  test('sends closed event when stream ends', async () => {
    const manager = new LogStreamManager();
    const messages: any[] = [];

    const mockStream = new Readable({ read() {} });
    const sendFn = jest.fn((data: string) => {
      messages.push(JSON.parse(data));
    });

    manager.attachStream(mockStream, sendFn);
    mockStream.push(null);

    await new Promise((r) => setTimeout(r, 10));

    const closedMsg = messages.find((m) => m.type === 'closed');
    expect(closedMsg).toBeDefined();
    expect(closedMsg.reason).toBe('container_stopped');
  });

  test('respects buffer limit and sends dropped message', async () => {
    const manager = new LogStreamManager({ maxBuffer: 5 });
    const messages: any[] = [];

    const mockStream = new Readable({ read() {} });
    let sendBlocked = true;
    const sendFn = jest.fn((data: string) => {
      if (sendBlocked) {
        throw new Error('socket not open');
      }
      messages.push(JSON.parse(data));
    });

    manager.attachStream(mockStream, sendFn);

    for (let i = 0; i < 10; i++) {
      mockStream.push(`line ${i}\n`);
    }

    await new Promise((r) => setTimeout(r, 50));

    sendBlocked = false;
    manager.flush(sendFn);

    const dropped = messages.find((m) => m.type === 'dropped');
    expect(dropped).toBeDefined();
    expect(dropped.count).toBeGreaterThan(0);
  });
});
