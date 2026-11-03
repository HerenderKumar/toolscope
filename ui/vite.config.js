import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The UI only talks to the read-only API (/api). API_TARGET is the API's address
// (http://localhost:8000 locally, http://api:8000 inside Docker Compose).
const target = process.env.API_TARGET || 'http://localhost:8000'
const proxy = { '/api': { target, changeOrigin: true } }

export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173, proxy },
  preview: { host: true, port: 5173, proxy },
})
