# ==========================================
# STAGE 1: Builder (Compilers & Tools)
# ==========================================
FROM python:3.12-slim as builder

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for BUILD only
# (gcc, python3-dev for compiling; libpq-dev if using psycopg2)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev libpq-dev

# Create a virtual environment to isolate dependencies
RUN python -m venv /opt/venv
# Enable venv for the upcoming pip install
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# STAGE 2: Runtime (Production Image)
# ==========================================
FROM python:3.12-slim as runner

# Extra security: standard logic
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install only RUNTIME system dependencies 
# (e.g., libpq5 for postgres runtime, distinct from libpq-dev)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
# Change ownership of the app directory
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose the port
EXPOSE 5001

# Use Gunicorn for production (Flask/Django dev servers are not for prod)
# Make sure gunicorn is in your requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "3", "services.webapp.api:app"]
