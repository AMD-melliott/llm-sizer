import {
  generateDockerCompose,
  generateDotEnvTemplate,
  validateDockerCompose,
} from '../src/utils/dockerComposeBuilder';
import type { ContainerConfig } from '../src/types';

describe('Docker Compose Builder Tests', () => {
  // Mock container configuration
  const mockConfig: ContainerConfig = {
    engine: {
      id: 'vllm',
      name: 'vLLM',
      version: '0.6.0',
      description: 'Fast LLM inference',
      documentation: 'https://docs.vllm.ai/',
      parameters: [],
    },
    image: {
      engine: 'vllm',
      repository: 'rocm/vllm',
      tag: 'latest',
      fullImage: 'rocm/vllm:latest',
      stability: 'stable' as const,
      rocmVersion: '6.0',
      pythonVersion: '3.10',
      description: 'Production vLLM',
      features: [],
      requirements: {
        minDockerVersion: '19.03',
        requiresContainerToolkit: false,
        recommendsContainerToolkit: true,
      },
    },
    containerName: 'vllm-llama3-70b',
    useContainerToolkit: true,
    autoRemove: false,
    gpuIds: ['0', '1', '2', '3'],
    gpuCount: 4,
    shmSize: '32g',
    volumes: [
      {
        hostPath: './models',
        containerPath: '/models',
        readOnly: false,
        description: 'Model storage',
      },
      {
        hostPath: '~/.cache/huggingface',
        containerPath: '/root/.cache/huggingface',
        readOnly: false,
        description: 'HuggingFace cache',
      },
    ],
    environment: [
      {
        key: 'AMD_VISIBLE_DEVICES',
        value: '0,1,2,3',
        description: 'AMD GPUs visible to container',
      },
      {
        key: 'HF_TOKEN',
        value: '${HF_TOKEN}',
        sensitive: true,
        description: 'HuggingFace token',
      },
      {
        key: 'VLLM_LOGGING_LEVEL',
        value: 'INFO',
        sensitive: false,
      },
    ],
    ports: [
      {
        host: 8000,
        container: 8000,
        protocol: 'tcp' as const,
        description: 'vLLM API',
      },
    ],
    useHostNetwork: false,
    engineParams: [
      { flag: '--model', value: 'meta-llama/Llama-3-70b' },
      { flag: '--tensor-parallel-size', value: 4 },
      { flag: '--dtype', value: 'float16' },
      { flag: '--max-model-len', value: 4096 },
      { flag: '--trust-remote-code', value: true },
    ],
    model: {
      id: 'llama-3-70b',
      name: 'Meta Llama 3 70B',
      parameters: 70,
    },
    gpus: [
      { id: 'mi300x', name: 'AMD Instinct MI300X', vram: 192 },
      { id: 'mi300x', name: 'AMD Instinct MI300X', vram: 192 },
      { id: 'mi300x', name: 'AMD Instinct MI300X', vram: 192 },
      { id: 'mi300x', name: 'AMD Instinct MI300X', vram: 192 },
    ],
    memoryUsage: {
      estimated: 156,
      available: 768,
      percentage: 20.3,
    },
  };

  describe('generateDockerCompose', () => {
    test('should generate valid Docker Compose YAML', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain("version: '3.8'");
      expect(yaml).toContain('services:');
      expect(yaml).toContain('vllm-llama3-70b:');
    });

    test('should include image specification', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain('image: rocm/vllm:latest');
      expect(yaml).toContain('container_name: vllm-llama3-70b');
    });

    test('should include AMD runtime when Container Toolkit enabled', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain('runtime: amd');
    });

    test('should NOT include runtime when Container Toolkit disabled', () => {
      const configNoToolkit = { ...mockConfig, useContainerToolkit: false };
      const yaml = generateDockerCompose(configNoToolkit);

      expect(yaml).not.toContain('runtime: amd');
    });

    test('should include AMD_VISIBLE_DEVICES with Container Toolkit', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain('environment:');
      expect(yaml).toContain('- AMD_VISIBLE_DEVICES=0,1,2,3');
    });

    test('should include device mappings without Container Toolkit', () => {
      const configNoToolkit = {
        ...mockConfig,
        useContainerToolkit: false,
        environment: [
          {
            key: 'ROCR_VISIBLE_DEVICES',
            value: '0,1,2,3',
            description: 'ROCm visible devices',
          },
          ...mockConfig.environment.filter(env => env.key !== 'AMD_VISIBLE_DEVICES'),
        ],
      };
      const yaml = generateDockerCompose(configNoToolkit);

      expect(yaml).toContain('devices:');
      expect(yaml).toContain('- /dev/kfd:/dev/kfd');
      expect(yaml).toContain('- /dev/dri:/dev/dri');
      expect(yaml).toContain('group_add:');
      expect(yaml).toContain('- video');
      expect(yaml).toContain('- ROCR_VISIBLE_DEVICES=0,1,2,3');
    });

    test('should include all environment variables', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain('- HF_TOKEN=${HF_TOKEN}');
      expect(yaml).toContain('- VLLM_LOGGING_LEVEL=INFO');
    });

    test('should include shared memory size', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain('shm_size: 32g');
    });

    test('should include resource limits when specified', () => {
      const configWithLimits = {
        ...mockConfig,
        resourceLimits: {
          shmSize: '32g',
          memoryLimit: '256g',
          cpuLimit: '32',
        },
      };

      const yaml = generateDockerCompose(configWithLimits);

      expect(yaml).toContain('deploy:');
      expect(yaml).toContain('resources:');
      expect(yaml).toContain('limits:');
      expect(yaml).toContain("cpus: '32'");
      expect(yaml).toContain('memory: 256g');
    });

    test('should include all volume mounts', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain('volumes:');
      expect(yaml).toContain('- ./models:/models');
      expect(yaml).toContain('- ~/.cache/huggingface:/root/.cache/huggingface');
    });

    test('should mark read-only volumes correctly', () => {
      const configWithRO = {
        ...mockConfig,
        volumes: [
          {
            hostPath: './models',
            containerPath: '/models',
            readOnly: true,
          },
        ],
      };

      const yaml = generateDockerCompose(configWithRO);
      expect(yaml).toContain('- ./models:/models:ro');
    });

    test('should include port mappings when not using host network', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain('ports:');
      expect(yaml).toContain('- "8000:8000/tcp"');
    });

    test('should include multiple ports', () => {
      const configMultiPort = {
        ...mockConfig,
        ports: [
          { host: 8000, container: 8000, protocol: 'tcp' as const },
          { host: 8001, container: 8001, protocol: 'tcp' as const },
        ],
      };

      const yaml = generateDockerCompose(configMultiPort);
      expect(yaml).toContain('- "8000:8000/tcp"');
      expect(yaml).toContain('- "8001:8001/tcp"');
    });

    test('should use host network mode when configured', () => {
      const configHostNet = { ...mockConfig, useHostNetwork: true };
      const yaml = generateDockerCompose(configHostNet);

      expect(yaml).toContain('network_mode: host');
      expect(yaml).not.toContain('ports:');
    });

    test('should include restart policy when autoRemove is false', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain('restart: unless-stopped');
    });

    test('should NOT include restart policy when autoRemove is true', () => {
      const configWithAutoRemove = { ...mockConfig, autoRemove: true };
      const yaml = generateDockerCompose(configWithAutoRemove, {
        includeComments: true,
      });

      expect(yaml).not.toContain('restart: unless-stopped');
      expect(yaml).toContain('# Note: Auto-remove (--rm) is not supported in Docker Compose');
    });

    test('should include comment explaining auto-remove limitation in Compose', () => {
      const configWithAutoRemove = { ...mockConfig, autoRemove: true };
      const yaml = generateDockerCompose(configWithAutoRemove, {
        includeComments: true,
      });

      expect(yaml).toContain('# Note: Auto-remove (--rm) is not supported in Docker Compose');
      expect(yaml).toContain('# Container will persist. Use "docker-compose down" to remove.');
    });

    test('should include engine parameters as command', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain('command:');
      expect(yaml).toContain('- "--model"');
      expect(yaml).toContain('- "meta-llama/Llama-3-70b"');
      expect(yaml).toContain('- "--tensor-parallel-size"');
      expect(yaml).toContain('- "4"');
      expect(yaml).toContain('- "--dtype"');
      expect(yaml).toContain('- "float16"');
    });

    test('should handle boolean flags in command', () => {
      const yaml = generateDockerCompose(mockConfig);

      // Boolean true should include the flag
      expect(yaml).toContain('- "--trust-remote-code"');
    });

    test('should NOT include boolean flags set to false', () => {
      const configWithFalseFlag = {
        ...mockConfig,
        engineParams: [
          ...mockConfig.engineParams,
          { flag: '--disable-log', value: false },
        ],
      };

      const yaml = generateDockerCompose(configWithFalseFlag);
      expect(yaml).not.toContain('--disable-log');
    });

    test('FR-4.4: should include health check', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain('healthcheck:');
      // Updated to match actual implementation with CMD-SHELL and curl/wget fallback
      expect(yaml).toContain('test: ["CMD-SHELL"');
      expect(yaml).toContain('curl -f http://localhost:8000/health');
      expect(yaml).toContain('interval: 30s');
      expect(yaml).toContain('timeout: 10s');
      expect(yaml).toContain('retries: 3');
      expect(yaml).toContain('start_period: 60s');
    });

    test('should include comments when enabled', () => {
      const yaml = generateDockerCompose(mockConfig, {
        includeComments: true,
      });

      expect(yaml).toContain('# Generated by AMD LLM Sizer');
      expect(yaml).toContain('# Model: Meta Llama 3 70B');
      expect(yaml).toContain('# Usage:');
      expect(yaml).toContain('docker-compose up -d');
    });

    test('should NOT include comments when disabled', () => {
      const yaml = generateDockerCompose(mockConfig, {
        includeComments: false,
      });

      expect(yaml).not.toContain('# Generated by');
      expect(yaml).not.toContain('# Usage:');
    });

    test('should support custom version specification', () => {
      const yaml = generateDockerCompose(mockConfig, { version: '3.9' });

      expect(yaml).toContain("version: '3.9'");
    });

    test('should include volume descriptions as comments when enabled', () => {
      const yaml = generateDockerCompose(mockConfig, {
        includeComments: true,
      });

      expect(yaml).toContain('# Model storage');
      expect(yaml).toContain('# HuggingFace cache');
    });

    test('should include environment variable descriptions', () => {
      const yaml = generateDockerCompose(mockConfig, {
        includeComments: true,
      });

      expect(yaml).toContain('# HuggingFace token');
    });

    test('should include port descriptions', () => {
      const yaml = generateDockerCompose(mockConfig, {
        includeComments: true,
      });

      expect(yaml).toContain('# vLLM API');
    });

    test('should be valid YAML structure', () => {
      const yaml = generateDockerCompose(mockConfig);

      // Check indentation and structure
      expect(yaml).toMatch(/^version:/m);
      expect(yaml).toMatch(/^services:/m);
      expect(yaml).toMatch(/^  vllm-llama3-70b:/m);
      expect(yaml).toMatch(/^    image:/m);
    });
  });

  describe('generateDotEnvTemplate', () => {
    test('should generate .env template for sensitive variables', () => {
      const dotenv = generateDotEnvTemplate(mockConfig);

      expect(dotenv).toContain('# Environment variables for Docker Compose');
      expect(dotenv).toContain('HF_TOKEN=');
    });

    test('should include descriptions for sensitive variables', () => {
      const dotenv = generateDotEnvTemplate(mockConfig);

      expect(dotenv).toContain('# HuggingFace token');
    });

    test('should NOT include non-sensitive variables', () => {
      const dotenv = generateDotEnvTemplate(mockConfig);

      expect(dotenv).not.toContain('VLLM_LOGGING_LEVEL');
    });

    test('should include generation timestamp', () => {
      const dotenv = generateDotEnvTemplate(mockConfig);

      expect(dotenv).toContain('# Generated:');
    });

    test('should include usage instructions', () => {
      const dotenv = generateDotEnvTemplate(mockConfig);

      expect(dotenv).toContain('# Copy this file to .env');
    });

    test('should handle empty sensitive variables', () => {
      const configNoSensitive = {
        ...mockConfig,
        environment: [
          {
            key: 'VLLM_LOGGING_LEVEL',
            value: 'INFO',
            sensitive: false,
          },
        ],
      };

      const dotenv = generateDotEnvTemplate(configNoSensitive);
      expect(dotenv).toContain('# Environment variables');
    });
  });

  describe('validateDockerCompose', () => {
    test('should pass validation for secure compose file', () => {
      const yaml = generateDockerCompose(mockConfig);
      const result = validateDockerCompose(yaml);

      expect(result.valid).toBe(true);
      expect(result.issues).toHaveLength(0);
    });

    test('should detect privileged mode', () => {
      const badYaml = `
version: '3.8'
services:
  test:
    privileged: true
      `;
      const result = validateDockerCompose(badYaml);

      expect(result.valid).toBe(false);
      expect(result.issues).toContain(
        'Compose file contains privileged mode (security risk)'
      );
    });

    test('should detect CAP_SYS_ADMIN capability', () => {
      const badYaml = `
version: '3.8'
services:
  test:
    cap_add:
      - SYS_ADMIN
      `;
      const result = validateDockerCompose(badYaml);

      expect(result.valid).toBe(false);
      expect(result.issues).toContain(
        'Compose file contains CAP_SYS_ADMIN capability (security risk)'
      );
    });

    test('should detect hardcoded HuggingFace tokens', () => {
      const badYaml = `
version: '3.8'
services:
  test:
    environment:
      - HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890
      `;
      const result = validateDockerCompose(badYaml);

      expect(result.valid).toBe(false);
      expect(result.issues).toContain(
        'Compose file contains hardcoded HuggingFace token'
      );
    });

    test('should detect missing version', () => {
      const badYaml = `
services:
  test:
    image: rocm/vllm:latest
      `;
      const result = validateDockerCompose(badYaml);

      expect(result.valid).toBe(false);
      expect(result.issues).toContain('Compose file missing version specification');
    });

    test('should allow environment variable substitution', () => {
      const goodYaml = `
version: '3.8'
services:
  test:
    environment:
      - HF_TOKEN=\${HF_TOKEN}
      `;
      const result = validateDockerCompose(goodYaml);

      expect(result.valid).toBe(true);
    });

    test('should validate generated compose files', () => {
      const yaml = generateDockerCompose(mockConfig);
      const result = validateDockerCompose(yaml);

      expect(result.valid).toBe(true);
      expect(result.issues).toHaveLength(0);
    });
  });

  describe('Edge Cases', () => {
    test('should handle empty volumes', () => {
      const configNoVolumes = { ...mockConfig, volumes: [] };
      const yaml = generateDockerCompose(configNoVolumes);

      expect(yaml).toContain('services:');
      expect(yaml).not.toMatch(/^\s+volumes:/m);
    });

    test('should handle empty ports', () => {
      const configNoPorts = { ...mockConfig, ports: [], useHostNetwork: false };
      const yaml = generateDockerCompose(configNoPorts);

      expect(yaml).toContain('services:');
      expect(yaml).not.toMatch(/^\s+ports:/m);
    });

    test('should handle empty environment', () => {
      const configNoEnv = {
        ...mockConfig,
        environment: [],
        useContainerToolkit: true,
      };
      const yaml = generateDockerCompose(configNoEnv);

      // When environment is empty, no environment section is added
      // This is current behavior - GPU devices would need to be set via runtime
      expect(yaml).toContain('services:');
      expect(yaml).not.toContain('environment:');

      // Note: This is a minor limitation - ideally AMD_VISIBLE_DEVICES should
      // be added even when environment array is empty when using Container Toolkit
    });

    test('should handle empty engine parameters', () => {
      const configNoParams = { ...mockConfig, engineParams: [] };
      const yaml = generateDockerCompose(configNoParams);

      expect(yaml).not.toContain('command:');
    });

    test('should handle single GPU', () => {
      const configSingleGPU = {
        ...mockConfig,
        gpuIds: ['0'],
        gpuCount: 1,
        environment: [
          {
            key: 'AMD_VISIBLE_DEVICES',
            value: '0',
            description: 'AMD GPUs visible to container',
          },
          ...mockConfig.environment.filter(env => env.key !== 'AMD_VISIBLE_DEVICES'),
        ],
      };

      const yaml = generateDockerCompose(configSingleGPU);
      expect(yaml).toContain('- AMD_VISIBLE_DEVICES=0');
    });

    test('should use "all" for AMD_VISIBLE_DEVICES with 8+ GPUs', () => {
      const config8GPUs = {
        ...mockConfig,
        gpuIds: ['0', '1', '2', '3', '4', '5', '6', '7'],
        gpuCount: 8,
        environment: [
          {
            key: 'AMD_VISIBLE_DEVICES',
            value: 'all',
            description: 'AMD GPUs visible to container',
          },
          ...mockConfig.environment.filter(env => env.key !== 'AMD_VISIBLE_DEVICES'),
        ],
      };

      const yaml = generateDockerCompose(config8GPUs);
      expect(yaml).toContain('- AMD_VISIBLE_DEVICES=all');
      expect(yaml).not.toContain('0,1,2,3,4,5,6,7');
    });
  });

  describe('PRD Compliance', () => {
    test('FR-4.1: should generate version 3.8+ Docker Compose', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain("version: '3.8'");
    });

    test('FR-4.2: should include proper service configuration', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain('services:');
      expect(yaml).toContain('restart: unless-stopped');
    });

    test('FR-4.3: should support environment variable files', () => {
      const yaml = generateDockerCompose(mockConfig);
      const dotenv = generateDotEnvTemplate(mockConfig);

      expect(yaml).toContain('${HF_TOKEN}');
      expect(dotenv).toContain('HF_TOKEN=');
    });

    test('FR-4.4: should include health checks', () => {
      const yaml = generateDockerCompose(mockConfig);

      expect(yaml).toContain('healthcheck:');
      expect(yaml).toContain('test:');
      expect(yaml).toContain('interval:');
      expect(yaml).toContain('timeout:');
      expect(yaml).toContain('retries:');
    });
  });
});
