# The React workbench, built once and served by `vite preview`, which proxies /api to the api service.
FROM node:20-alpine
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
# npm can occasionally exit successfully after an incomplete install. Do not cache
# that layer unless the build tool is actually present and runnable.
RUN npm ci && ./node_modules/.bin/vite --version
COPY ui/ ./
RUN npm run build
ENV API_TARGET=http://api:8000
EXPOSE 5173
CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "5173"]
