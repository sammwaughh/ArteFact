import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/viewer-v1/',
  plugins: [react()],
});
