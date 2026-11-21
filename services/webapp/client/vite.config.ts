/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  define: { 'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV ?? 'development') },
  build: { outDir: 'build' },
  server: { port: 3000, proxy: { '/api': 'http://localhost:5001' } },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/setupTests.js'],
    include: ['src/**/*.test.{js,jsx,ts,tsx}'],
  },
});
