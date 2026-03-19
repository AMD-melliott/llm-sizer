import type { FastifyInstance } from 'fastify';
import type { MetricsCollector } from '../services/MetricsCollector.js';

export function registerStatusRoute(
  fastify: FastifyInstance,
  collector: MetricsCollector
) {
  fastify.get('/api/status', async (_request, reply) => {
    const snapshot = collector.getSnapshot();
    if (!snapshot) {
      return reply.code(503).send({ error: 'Dashboard not ready' });
    }
    return reply.send({
      timestamp: snapshot.timestamp,
      pollIntervalMs: snapshot.pollIntervalMs,
      summary: snapshot.summary,
      warnings: snapshot.warnings,
    });
  });
}
