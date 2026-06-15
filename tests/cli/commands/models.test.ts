import { listModels, showModel } from '../../../src/cli/commands/models';

describe('Models Command', () => {
  describe('listModels', () => {
    test('returns all generation models when no filter', () => {
      const models = listModels({ type: 'generation' });
      expect(models.length).toBeGreaterThan(0);
      expect(models[0]).toHaveProperty('id');
      expect(models[0]).toHaveProperty('name');
      expect(models[0]).toHaveProperty('parameters_billions');
    });

    test('filters by substring match', () => {
      const models = listModels({ type: 'generation', filter: '70b' });
      expect(models.length).toBeGreaterThan(0);
      for (const m of models) {
        expect(
          m.id.toLowerCase().includes('70b') || m.name.toLowerCase().includes('70b')
        ).toBe(true);
      }
    });

    test('returns embedding models', () => {
      const models = listModels({ type: 'embedding' });
      expect(models.length).toBeGreaterThan(0);
      expect(models[0]).toHaveProperty('dimensions');
    });

    test('returns reranking models', () => {
      const models = listModels({ type: 'reranking' });
      expect(models.length).toBeGreaterThan(0);
      expect(models[0]).toHaveProperty('max_query_length');
    });

    test('filters by architecture', () => {
      const models = listModels({ type: 'generation', arch: 'moe' });
      for (const m of models) {
        expect((m as any).architecture).toBe('moe');
      }
    });

    test('defaults to generation type when no type provided', () => {
      const models = listModels({});
      expect(models.length).toBeGreaterThan(0);
      expect(models[0]).toHaveProperty('parameters_billions');
    });

    test('filters by modality', () => {
      const models = listModels({ type: 'generation', modality: 'multimodal' });
      for (const m of models) {
        expect((m as any).modality).toBe('multimodal');
      }
    });

    test('filter is case-insensitive', () => {
      const lower = listModels({ type: 'generation', filter: 'llama' });
      const upper = listModels({ type: 'generation', filter: 'LLAMA' });
      expect(lower.length).toEqual(upper.length);
      expect(lower.length).toBeGreaterThan(0);
    });

    test('returns empty array when filter matches nothing', () => {
      const models = listModels({ type: 'generation', filter: 'xyzzy-no-match-12345' });
      expect(models).toEqual([]);
    });
  });

  describe('showModel', () => {
    test('returns full model details for valid ID', () => {
      const model = showModel('llama-3-70b', 'generation');
      expect(model).not.toBeNull();
      expect(model!.id).toBe('llama-3-70b');
      expect(model!.parameters_billions).toBe(70);
    });

    test('returns null for invalid ID', () => {
      const model = showModel('nonexistent', 'generation');
      expect(model).toBeNull();
    });

    test('returns embedding model by ID', () => {
      const models = listModels({ type: 'embedding' });
      expect(models.length).toBeGreaterThan(0);
      const firstId = models[0].id;
      const model = showModel(firstId, 'embedding');
      expect(model).not.toBeNull();
      expect(model!.id).toBe(firstId);
    });

    test('returns reranking model by ID', () => {
      const models = listModels({ type: 'reranking' });
      expect(models.length).toBeGreaterThan(0);
      const firstId = models[0].id;
      const model = showModel(firstId, 'reranking');
      expect(model).not.toBeNull();
      expect(model!.id).toBe(firstId);
    });

    test('returns null when searching wrong type', () => {
      // llama-3-70b is a generation model, not embedding
      const model = showModel('llama-3-70b', 'embedding');
      expect(model).toBeNull();
    });
  });
});
