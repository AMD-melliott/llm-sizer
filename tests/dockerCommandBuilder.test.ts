import {
  generateDockerRunCommand,
  generateShellScript,
  generateDockerCommandOnly,
  validateDockerCommand,
} from '../src/utils/dockerCommandBuilder';
import type { ContainerConfig } from '../src/types';

describe('Docker Command Builder Tests', () => {
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

  describe('generateDockerRunCommand', () => {
    test('should generate valid docker run command with Container Toolkit', () => {
      const command = generateDockerRunCommand(mockConfig, {
        includeComments: false,
        includeManagementCommands: false,
      });

      expect(command).toContain('docker run -d');
      expect(command).toContain('--name vllm-llama3-70b');
      expect(command).toContain('--runtime=amd');
      expect(command).toContain('-e AMD_VISIBLE_DEVICES=0,1,2,3');
      expect(command).toContain('--shm-size=32g');
      expect(command).toContain('rocm/vllm:latest');
    });

    test('should generate fallback command without Container Toolkit', () => {
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
      const command = generateDockerRunCommand(configNoToolkit, {
        includeComments: false,
      });

      expect(command).not.toContain('--runtime=amd');
      expect(command).not.toContain('AMD_VISIBLE_DEVICES');
      expect(command).toContain('--device=/dev/kfd');
      expect(command).toContain('--device=/dev/dri');
      expect(command).toContain('--group-add video');
      expect(command).toContain('-e ROCR_VISIBLE_DEVICES=0,1,2,3');
    });

    test('should include all environment variables', () => {
      const command = generateDockerRunCommand(mockConfig);

      expect(command).toContain('-e HF_TOKEN=${HF_TOKEN}');
    });

    test('should NOT hardcode sensitive tokens', () => {
      const command = generateDockerRunCommand(mockConfig);

      expect(command).not.toMatch(/hf_[a-zA-Z0-9]{32,}/);
      expect(command).toContain('${HF_TOKEN}');
    });

    test('should include all volume mounts', () => {
      const command = generateDockerRunCommand(mockConfig);

      expect(command).toContain('-v ./models:/models');
      expect(command).toContain('-v ~/.cache/huggingface:/root/.cache/huggingface');
    });

    test('should include read-only flag for read-only volumes', () => {
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

      const command = generateDockerRunCommand(configWithRO);
      expect(command).toContain('-v ./models:/models:ro');
    });

    test('should include port mappings when not using host network', () => {
      const command = generateDockerRunCommand(mockConfig);

      expect(command).toContain('-p 8000:8000/tcp');
      expect(command).not.toContain('--network=host');
    });

    test('should use host network when configured', () => {
      const configHostNet = { ...mockConfig, useHostNetwork: true };
      const command = generateDockerRunCommand(configHostNet);

      expect(command).toContain('--network=host');
      expect(command).not.toContain('-p 8000:8000');
    });

    test('should include all engine parameters', () => {
      const command = generateDockerRunCommand(mockConfig);

      expect(command).toContain('--model "meta-llama/Llama-3-70b"');
      expect(command).toContain('--tensor-parallel-size 4');
      expect(command).toContain('--dtype "float16"');
      expect(command).toContain('--max-model-len 4096');
    });

    test('should handle boolean engine parameters correctly', () => {
      const configWithBool = {
        ...mockConfig,
        engineParams: [
          ...mockConfig.engineParams,
          { flag: '--trust-remote-code', value: true },
          { flag: '--disable-log', value: false },
        ],
      };

      const command = generateDockerRunCommand(configWithBool);

      expect(command).toContain('--trust-remote-code');
      expect(command).not.toContain('--disable-log');
    });

    test('should include restart policy when autoRemove is false', () => {
      const command = generateDockerRunCommand(mockConfig);
      expect(command).toContain('--restart unless-stopped');
      expect(command).not.toContain('--rm');
    });

    test('should include --rm flag when autoRemove is true', () => {
      const configWithAutoRemove = { ...mockConfig, autoRemove: true };
      const command = generateDockerRunCommand(configWithAutoRemove);
      expect(command).toContain('--rm');
      expect(command).not.toContain('--restart');
    });

    test('should NOT include both --rm and --restart flags', () => {
      // Test with autoRemove false
      const commandWithRestart = generateDockerRunCommand(mockConfig);
      const hasRestart = commandWithRestart.includes('--restart');
      const hasRm = commandWithRestart.includes('--rm');
      expect(hasRestart && hasRm).toBe(false);

      // Test with autoRemove true
      const configWithAutoRemove = { ...mockConfig, autoRemove: true };
      const commandWithRm = generateDockerRunCommand(configWithAutoRemove);
      const hasRestart2 = commandWithRm.includes('--restart');
      const hasRm2 = commandWithRm.includes('--rm');
      expect(hasRestart2 && hasRm2).toBe(false);
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

      const command = generateDockerRunCommand(configWithLimits);
      expect(command).toContain('--memory=256g');
      expect(command).toContain('--cpus=32');
    });

    test('should include comments when requested', () => {
      const command = generateDockerRunCommand(mockConfig, {
        includeComments: true,
      });

      expect(command).toContain('#!/bin/bash');
      expect(command).toContain('Generated by AMD LLM Sizer');
      expect(command).toContain('Model: Meta Llama 3 70B');
      expect(command).toContain('Prerequisites:');
    });

    test('should include management commands when requested', () => {
      const command = generateDockerRunCommand(mockConfig, {
        includeComments: true,
        includeManagementCommands: true,
      });

      expect(command).toContain('docker logs -f vllm-llama3-70b');
      expect(command).toContain('docker stop vllm-llama3-70b');
      expect(command).toContain('docker rm vllm-llama3-70b');
    });

    test('should handle multiple custom ports', () => {
      const configMultiPort = {
        ...mockConfig,
        ports: [
          { host: 8000, container: 8000, protocol: 'tcp' as const },
          { host: 8001, container: 8001, protocol: 'tcp' as const },
          { host: 9000, container: 9000, protocol: 'udp' as const },
        ],
      };

      const command = generateDockerRunCommand(configMultiPort);
      expect(command).toContain('-p 8000:8000/tcp');
      expect(command).toContain('-p 8001:8001/tcp');
      expect(command).toContain('-p 9000:9000/udp');
    });

    test('should properly escape special characters in values', () => {
      const configSpecialChars = {
        ...mockConfig,
        engineParams: [
          { flag: '--model', value: 'meta-llama/Llama-3-70b' },
        ],
      };

      const command = generateDockerRunCommand(configSpecialChars);
      // Should not break on forward slashes
      expect(command).toContain('meta-llama/Llama-3-70b');
    });
  });

  describe('generateShellScript', () => {
    test('should generate complete shell script with shebang', () => {
      const script = generateShellScript(mockConfig);

      expect(script).toMatch(/^#!/);
      expect(script).toContain('#!/bin/bash');
      expect(script).toContain('docker run');
    });

    test('should include all comments and management commands', () => {
      const script = generateShellScript(mockConfig);

      expect(script).toContain('# Generated by AMD LLM Sizer');
      expect(script).toContain('Container management commands:');
      expect(script).toContain('docker logs');
    });
  });

  describe('generateDockerCommandOnly', () => {
    test('should generate command without comments', () => {
      const command = generateDockerCommandOnly(mockConfig);

      expect(command).not.toContain('#');
      expect(command).not.toContain('#!/bin/bash');
      expect(command).toContain('docker run');
    });

    test('should generate executable docker command', () => {
      const command = generateDockerCommandOnly(mockConfig);

      expect(command).toMatch(/^docker run/);
      expect(command).not.toContain('Generated by');
    });
  });

  describe('validateDockerCommand', () => {
    test('should pass validation for secure command', () => {
      const command = generateDockerRunCommand(mockConfig);
      const result = validateDockerCommand(command);

      expect(result.valid).toBe(true);
      expect(result.issues).toHaveLength(0);
    });

    test('should detect --privileged flag', () => {
      const badCommand = 'docker run --privileged rocm/vllm:latest';
      const result = validateDockerCommand(badCommand);

      expect(result.valid).toBe(false);
      expect(result.issues).toContain(
        'Command contains --privileged flag (security risk)'
      );
    });

    test('should detect CAP_SYS_ADMIN capability', () => {
      const badCommand = 'docker run --cap-add=SYS_ADMIN rocm/vllm:latest';
      const result = validateDockerCommand(badCommand);

      expect(result.valid).toBe(false);
      expect(result.issues).toContain(
        'Command contains CAP_SYS_ADMIN capability (security risk)'
      );
    });

    test('should detect disabled seccomp', () => {
      const badCommand =
        'docker run --security-opt seccomp=unconfined rocm/vllm:latest';
      const result = validateDockerCommand(badCommand);

      expect(result.valid).toBe(false);
      expect(result.issues).toContain('Command disables seccomp (security risk)');
    });

    test('should detect hardcoded HuggingFace tokens', () => {
      const badCommand =
        'docker run -e HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890 rocm/vllm:latest';
      const result = validateDockerCommand(badCommand);

      expect(result.valid).toBe(false);
      expect(result.issues).toContain('Command contains hardcoded HuggingFace token');
    });

    test('should allow environment variable substitution for tokens', () => {
      const goodCommand = 'docker run -e HF_TOKEN=${HF_TOKEN} rocm/vllm:latest';
      const result = validateDockerCommand(goodCommand);

      expect(result.valid).toBe(true);
    });

    test('should validate complex generated commands', () => {
      const command = generateDockerRunCommand(mockConfig);
      const result = validateDockerCommand(command);

      expect(result.valid).toBe(true);
      expect(result.issues).toHaveLength(0);
    });
  });

  describe('Edge Cases', () => {
    test('should handle zero GPUs gracefully', () => {
      const configNoGPU = {
        ...mockConfig,
        gpuIds: [],
        gpuCount: 0,
      };

      const command = generateDockerRunCommand(configNoGPU);
      expect(command).toContain('docker run');
    });

    test('should handle empty volumes array', () => {
      const configNoVolumes: ContainerConfig = {
        ...mockConfig,
        volumes: []
      };
      const command = generateDockerRunCommand(configNoVolumes);

      expect(command).toContain('docker run');
      // When volumes array is empty, no volume mounts should be added
      const volumeLines = command.split('\n').filter(line => line.trim().startsWith('-v '));
      expect(volumeLines.length).toBe(0);
    });

    test('should handle empty environment variables', () => {
      const configNoEnv = { ...mockConfig, environment: [] };
      const command = generateDockerRunCommand(configNoEnv);

      expect(command).toContain('docker run');
    });

    test('should handle empty engine parameters', () => {
      const configNoParams = { ...mockConfig, engineParams: [] };
      const command = generateDockerRunCommand(configNoParams);

      expect(command).toContain('rocm/vllm:latest');
      expect(command).not.toContain('--model');
    });

    test('should handle single GPU', () => {
      const configSingleGPU = {
        ...mockConfig,
        gpuIds: ['0'],
        gpuCount: 1,
      };

      const command = generateDockerRunCommand(configSingleGPU);
      expect(command).toContain('-e AMD_VISIBLE_DEVICES=0');
    });

    test('should list specific GPU IDs for fewer than 8 GPUs', () => {
      const configFewGPUs = {
        ...mockConfig,
        gpuIds: ['0', '1', '2', '3'],
        gpuCount: 4,
        environment: [
          {
            key: 'AMD_VISIBLE_DEVICES',
            value: '0,1,2,3',
            description: 'AMD GPUs visible to container',
          },
          ...mockConfig.environment.filter(env => env.key !== 'AMD_VISIBLE_DEVICES'),
        ],
      };

      const command = generateDockerRunCommand(configFewGPUs);
      expect(command).toContain('-e AMD_VISIBLE_DEVICES=0,1,2,3');
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

      const command = generateDockerRunCommand(config8GPUs);
      expect(command).toContain('-e AMD_VISIBLE_DEVICES=all');
      expect(command).not.toContain('0,1,2,3,4,5,6,7');
    });
  });

  describe('PRD Compliance', () => {
    test('FR-3.1.1: should NOT include security anti-patterns', () => {
      const command = generateDockerRunCommand(mockConfig);

      expect(command).not.toContain('--privileged');
      expect(command).not.toContain('--cap-add=CAP_SYS_ADMIN');
      expect(command).not.toContain('--cap-add=SYS_PTRACE');
      expect(command).not.toContain('--device=/dev/mem');
      expect(command).not.toContain('--security-opt seccomp=unconfined');
    });

    test('FR-3.2.1: should use AMD Container Toolkit by default', () => {
      const command = generateDockerRunCommand(mockConfig);

      expect(command).toContain('--runtime=amd');
      expect(command).toContain('AMD_VISIBLE_DEVICES');
    });

    test('FR-3.2.2: should provide fallback without Container Toolkit', () => {
      const configFallback = {
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
      const command = generateDockerRunCommand(configFallback);

      expect(command).toContain('--device=/dev/kfd');
      expect(command).toContain('--device=/dev/dri');
      expect(command).toContain('--group-add video');
    });

    test('FR-3.5.3: should use shell variable substitution for sensitive data', () => {
      const command = generateDockerRunCommand(mockConfig);

      expect(command).toContain('${HF_TOKEN}');
      expect(command).not.toMatch(/hf_[a-zA-Z0-9]+/);
    });
  });
});
