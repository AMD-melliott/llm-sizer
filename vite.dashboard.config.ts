import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  root: 'src/dashboard/ui',
  build: {
    outDir: '../../../dist/dashboard',
    emptyOutDir: true,
  },
});
