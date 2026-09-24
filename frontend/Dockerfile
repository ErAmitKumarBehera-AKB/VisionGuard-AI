FROM node:22-alpine AS build

WORKDIR /app
COPY frontend/artifacts/visioninspect-ai/package.json ./package.json
RUN npm install --no-audit --no-fund
COPY frontend/artifacts/visioninspect-ai/ ./
ARG VITE_API_BASE_URL=/api/v1
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist/public /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1/health || exit 1
