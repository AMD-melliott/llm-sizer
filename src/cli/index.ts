#!/usr/bin/env node

import { Command } from 'commander';

const program = new Command();

program
  .name('llm-sizer')
  .description('LLM Inference Calculator - CLI')
  .version('1.0.0');

program.parse();
