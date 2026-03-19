import type { FastifyInstance } from 'fastify';
import type { Readable } from 'stream';
import Docker from 'dockerode';
import type { LogMessage, LogControl } from '../types.js';

export interface LogStreamOptions {
  maxBuffer?: number;
}

export class LogStreamManager {
  private maxBuffer: number;
  private buffer: LogMessage[] = [];
  private droppedCount = 0;

  constructor(options: LogStreamOptions = {}) {
    this.maxBuffer = options.maxBuffer ?? 1000;
  }

  attachStream(stream: Readable, send: (data: string) => void): void {
    let partial = '';

    stream.on('data', (chunk: Buffer | string) => {
      const text = partial + chunk.toString();
      const lines = text.split('\n');
      partial = lines.pop() ?? '';

      for (const line of lines) {
        if (!line.trim()) continue;
        const msg: LogMessage = {
          type: 'log',
          timestamp: new Date().toISOString(),
          stream: 'stdout',
          line: line.replace(/^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*/, ''),
        };

        const tsMatch = line.match(/^(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s*/);
        if (tsMatch) {
          msg.timestamp = tsMatch[1];
          msg.line = line.substring(tsMatch[0].length);
        }

        if (/^(ERROR|WARN|WARNING|CRITICAL|FATAL)/i.test(msg.line)) {
          msg.stream = 'stderr';
        }

        this.enqueue(msg, send);
      }
    });

    stream.on('end', () => {
      const control: LogControl = { type: 'closed', reason: 'container_stopped' };
      send(JSON.stringify(control));
    });

    stream.on('error', () => {
      const control: LogControl = { type: 'closed', reason: 'stream_error' };
      send(JSON.stringify(control));
    });
  }

  flush(send: (data: string) => void): void {
    if (this.droppedCount > 0) {
      const dropped: LogControl = { type: 'dropped', count: this.droppedCount };
      send(JSON.stringify(dropped));
      this.droppedCount = 0;
    }
    for (const msg of this.buffer) {
      send(JSON.stringify(msg));
    }
    this.buffer = [];
  }

  private enqueue(msg: LogMessage, send: (data: string) => void): void {
    try {
      send(JSON.stringify(msg));
    } catch {
      if (this.buffer.length >= this.maxBuffer) {
        this.buffer.shift();
        this.droppedCount++;
      }
      this.buffer.push(msg);
    }
  }
}

export function registerLogsRoute(
  fastify: FastifyInstance,
  docker: Docker
) {
  fastify.get(
    '/api/logs/:containerId',
    { websocket: true },
    async (socket, request) => {
      const { containerId } = request.params as { containerId: string };
      const url = new URL(request.url, 'http://localhost');
      const tail = parseInt(url.searchParams.get('tail') ?? '200', 10);

      const container = docker.getContainer(containerId);
      let logStream: NodeJS.ReadableStream;

      try {
        logStream = await container.logs({
          follow: true,
          stdout: true,
          stderr: true,
          tail,
          timestamps: true,
        });
      } catch {
        const err: LogControl = { type: 'closed', reason: 'container_not_found' };
        socket.send(JSON.stringify(err));
        socket.close();
        return;
      }

      const manager = new LogStreamManager();
      const sendFn = (data: string) => {
        if (socket.readyState !== 1) {
          throw new Error('socket not open');
        }
        socket.send(data);
      };
      manager.attachStream(logStream as any, sendFn);

      socket.on('close', () => {
        (logStream as any).destroy?.();
      });
    }
  );
}
