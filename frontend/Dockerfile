# --- Stage 1: Build the React application ---
FROM node:18-alpine AS builder

LABEL stage="frontend-builder"
DESCRIPTION="Builds the static frontend assets"

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock if using Yarn)
# This allows Docker to cache installed dependencies unless these files change.
COPY frontend/package.json frontend/package-lock.json* ./
# If you have a yarn.lock, ensure you copy that instead of package-lock.json and use yarn commands.
# Example for Yarn:
# COPY frontend/package.json frontend/yarn.lock ./
# RUN yarn install --frozen-lockfile

# Install dependencies
# Using --legacy-peer-deps if there are peer dependency conflicts, common in some React setups.
# Adjust if not needed or if your project requires specific flags.
RUN npm install --legacy-peer-deps

# Copy the rest of the frontend application source code
COPY frontend/ /app/
# Ensure .dockerignore is set up correctly in the project root to exclude node_modules, .git, etc. from this copy if context is project root.
# If Docker context is 'frontend' directory, then COPY . . would be used here.
# Assuming context is project root for consistency with backend Dockerfile.

# Build the application for production
# This command might differ based on your project setup (e.g., Vite, CRA, Next.js)
RUN npm run build
# Common output directories are 'dist' (Vite) or 'build' (Create React App).
# Ensure the path in the COPY command in the next stage matches your actual output directory.

# --- Stage 2: Serve the built application using Nginx ---
FROM nginx:1.25-alpine AS final

LABEL stage="frontend-server"
DESCRIPTION="Serves the static frontend assets using Nginx"

# Set working directory for Nginx
WORKDIR /usr/share/nginx/html

# Remove default Nginx static assets
RUN rm -rf ./*

# Copy the build output from the 'builder' stage to Nginx's html directory
# Adjust '/app/dist' if your build tool outputs to a different directory (e.g., /app/build for CRA)
COPY --from=builder /app/dist/ /usr/share/nginx/html/
# The trailing slash on source and destination is important for COPY to work as expected with directories.

# (Optional) Copy a custom Nginx configuration file if needed.
# This is useful for configuring routing, caching, SSL, etc.
# Example:
# COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
# A simple nginx.conf for single-page applications (SPAs) might look like:
# server {
#   listen 80;
#   server_name localhost;
#
#   root /usr/share/nginx/html;
#   index index.html index.htm;
#
#   location / {
#     try_files $uri $uri/ /index.html;
#   }
#
#   # Optional: Add headers for caching, security, etc.
#   # location ~* \.(?:ico|css|js|gif|jpe?g|png)$ {
#   #   expires 1y;
#   #   add_header Cache-Control "public";
#   # }
# }

# Expose port 80 (default HTTP port for Nginx)
EXPOSE 80

# Start Nginx when the container launches
# 'daemon off;' ensures Nginx stays in the foreground, which is standard for Docker containers.
CMD ["nginx", "-g", "daemon off;"]
