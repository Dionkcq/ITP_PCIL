import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: '/dashboard/' so the production build's index.html references its
// assets at /dashboard/assets/* — the path the FastAPI app mounts them at.
// host: true exposes the dev server on the LAN (so another laptop on the
// same wifi can open it during the NUC test).
export default defineConfig({
  plugins: [react()],
  base: '/dashboard/',
  server: { port: 5173, host: true },
})
