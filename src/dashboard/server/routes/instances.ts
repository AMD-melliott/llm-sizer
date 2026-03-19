import type { FastifyInstance } from 'fastify';
import type { MetricsCollector } from '../services/MetricsCollector.js';

export function registerInstancesRoute(
  fastify: FastifyInstance,
  collector: MetricsCollector
) {
  fastify.get('/api/instances', async (_request, reply) => {
    const snapshot = collector.getSnapshot();
    if (!snapshot) {
      return reply.code(503).send({ error: 'Dashboard not ready' });
    }
    return reply.send(snapshot.instances);
  });

  fastify.get('/api/instances/:id', async (request, reply) => {
    const snapshot = collector.getSnapshot();
    if (!snapshot) {
      return reply.code(503).send({ error: 'Dashboard not ready' });
    }
    const { id } = request.params as { id: string };
    const instance = snapshot.instances.find((i) => i.containerId === id);
    if (!instance) {
      return reply.code(404).send({ error: 'Instance not found' });
    }
    return reply.send(instance);
  });
}
