import { readFileSync, writeFileSync, copyFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { ModelEntry } from './types.js';
import chalk from 'chalk';

interface ModelsData {
  models: ModelEntry[];
}

/**
 * Read the models.json file
 */
export function readModelsFile(filePath: string): ModelsData {
  if (!existsSync(filePath)) {
    throw new Error(`Models file not found: ${filePath}`);
  }

  const content = readFileSync(filePath, 'utf-8');
  return JSON.parse(content);
}

/**
 * Write the models.json file with proper formatting
 */
export function writeModelsFile(filePath: string, data: ModelsData): void {
  const json = JSON.stringify(data, null, 2);
  writeFileSync(filePath, json + '\n', 'utf-8');
}

/**
 * Create a backup of the models.json file
 */
export function createBackup(filePath: string): string {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backupPath = filePath.replace('.json', `.backup.${timestamp}.json`);
  
  copyFileSync(filePath, backupPath);
  console.log(chalk.gray(`Backup created: ${backupPath}`));
  
  return backupPath;
}

/**
 * Add a new model to the models list and sort by parameter count
 */
export function addModelToList(newModel: ModelEntry, existingModels: ModelEntry[]): ModelEntry[] {
  const updatedModels = [...existingModels, newModel];
  
  // Sort by parameter count (ascending), keeping "custom" at the end
  return updatedModels.sort((a, b) => {
    if (a.id === 'custom') return 1;
    if (b.id === 'custom') return -1;
    return a.parameters_billions - b.parameters_billions;
  });
}

/**
 * Display a diff-like comparison of what will change
 */
export function showDiff(oldModels: ModelEntry[], newModels: ModelEntry[]): void {
  console.log(chalk.bold('\nChanges to models.json:'));
  
  const added = newModels.filter(nm => !oldModels.find(om => om.id === nm.id));
  
  if (added.length > 0) {
    console.log(chalk.green('\n+ Added models:'));
    added.forEach(model => {
      console.log(chalk.green(`  + ${model.id} (${model.name}) - ${model.parameters_billions}B params`));
    });
  }
  
  console.log(chalk.gray(`\nTotal models: ${oldModels.length} → ${newModels.length}`));
}
