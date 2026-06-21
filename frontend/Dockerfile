# --- Stage 1: Build the Vite app ---
FROM node:20-slim AS build

WORKDIR /app

# Install dependencies first (better layer caching)
COPY package*.json ./
RUN rm -f package-lock.json && npm install

# Copy the rest of the source, including .env
COPY . .

# Build the static files (outputs to /app/dist by default for Vite)
RUN npm run build

# --- Stage 2: Serve with nginx ---
FROM nginx:alpine

# Remove default nginx static files
RUN rm -rf /usr/share/nginx/html/*

# Copy built files from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Custom nginx config (handles client-side routing for React Router etc.)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]