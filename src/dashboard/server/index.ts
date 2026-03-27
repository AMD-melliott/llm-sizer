import Fastify from 'fastify';
import fastifyCors from '@fastify/cors';
import fastifyWebsocket from '@fastify/websocket';
import fastifyStatic from '@fastify/static';
import { resolve } from 'path';
import { DockerService } from './services/DockerService.js';
import { AmdSmiProvider } from './providers/AmdSmiProvider.js';
import { VllmMetricsService } from './services/VllmMetricsService.js';
import { MetricsCollector } from './services/MetricsCollector.js';
import { registerStatusRoute } from './routes/status.js';
import { registerInstancesRoute } from './routes/instances.js';
import { registerGpusRoute } from './routes/gpus.js';
import { registerLogsRoute } from './routes/logs.js';

interface ServerOptions {
  port?: number;
  host?: string;
  pollInterval?: number;
  socketPath?: string;
  imagePatterns?: string[];
}

async function startServer(options: ServerOptions = {}) {
  const port = options.port ?? 3001;
  const host = options.host ?? '0.0.0.0';
  const pollInterval = options.pollInterval ?? 5000;

  const fastify = Fastify({ logger: true });

  // Allow cross-origin requests from any origin so the main Vite app
  // (served from a different host/port) can reach the dashboard API.
  await fastify.register(fastifyCors, { origin: true });
  await fastify.register(fastifyWebsocket);
  await fastify.register(fastifyStatic, {
    root: resolve(import.meta.dirname, '../../..', 'dist/dashboard'),
    prefix: '/',
    wildcard: false,
  });

  const dockerService = new DockerService({
    socketPath: options.socketPath,
    imagePatterns: options.imagePatterns,
  });
  const gpuProvider = new AmdSmiProvider();
  const vllmService = new VllmMetricsService();
  const collector = new MetricsCollector(
    dockerService,
    gpuProvider,
    vllmService,
    { pollIntervalMs: pollInterval }
  );

  registerStatusRoute(fastify, collector);
  registerInstancesRoute(fastify, collector);
  registerGpusRoute(fastify, collector);
  registerLogsRoute(fastify, dockerService.getDocker());

  try {
    await fastify.listen({ port, host });
    console.log(`Dashboard running at http://${host}:${port}`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }

  collector.start();

  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.on(signal, async () => {
      collector.stop();
      await fastify.close();
      process.exit(0);
    });
  }
}

const args = process.argv.slice(2);
const portIdx = args.indexOf('--port');
const pollIdx = args.indexOf('--poll-interval');

startServer({
  port: portIdx >= 0 ? parseInt(args[portIdx + 1], 10) : undefined,
  pollInterval: pollIdx >= 0 ? parseInt(args[pollIdx + 1], 10) : undefined,
});
