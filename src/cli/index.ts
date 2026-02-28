#!/usr/bin/env node

import { Command } from 'commander';
import { registerCalculateCommand } from './commands/calculate.js';

const program = new Command();

program
  .name('llm-sizer')
  .description('LLM Inference Calculator - CLI')
  .version('1.0.0');

registerCalculateCommand(program);

program.parse();
