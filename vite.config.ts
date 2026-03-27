import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Default to GitHub Pages path; override with VITE_BASE=/ for container/local builds
  base: process.env.VITE_BASE ?? '/llm-sizer/',
})