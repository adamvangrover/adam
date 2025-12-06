# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies (gcc for compiling some python libs)
RUN apt-get update && apt-get install -y gcc python3-dev git && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set Python path
ENV PYTHONPATH=/app

# Run the Unified Server
CMD ["python", "core/api/server.py"]
