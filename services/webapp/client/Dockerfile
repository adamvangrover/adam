# Stage 1: Build the React application
FROM node:18-alpine as build

WORKDIR /app

COPY package.json ./
COPY package-lock.json ./

RUN npm install

COPY . ./

RUN npm run build

# Stage 2: Serve the application from Nginx
FROM nginx:1.21.3-alpine

COPY --from=build /app/build /usr/share/nginx/html

# Copy the nginx config file
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
