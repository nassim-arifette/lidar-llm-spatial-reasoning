import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: '../site',
    emptyOutDir: true,
    target: 'es2020',
    chunkSizeWarningLimit: 900,
  },
});
