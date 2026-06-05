import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// host: true exposes the dev server on the LAN (so another laptop on the
// same wifi can open it during the NUC test).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: true },
})
