# Dockerfile for Rust Core Engine
FROM rust:1.70 as builder
WORKDIR /app

# Copy the entire backend and shared directory to ensure access to protos
# Note: We are building from the root context in docker-compose
COPY backend/core_engine ./backend/core_engine
COPY shared ./shared

WORKDIR /app/backend/core_engine
RUN cargo install --path .

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y extra-runtime-dependencies && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/cargo/bin/core_engine /usr/local/bin/core_engine
CMD ["core_engine"]
