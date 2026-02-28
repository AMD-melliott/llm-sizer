#!/usr/bin/env node

import { Command } from 'commander';
import { registerCalculateCommand } from './commands/calculate.js';
import { registerModelsCommand } from './commands/models.js';
import { registerGPUsCommand } from './commands/gpus.js';

const program = new Command();

program
  .name('llm-sizer')
  .description('LLM Inference Calculator - CLI')
  .version('1.0.0');

registerCalculateCommand(program);
registerModelsCommand(program);
registerGPUsCommand(program);

program.parse();
