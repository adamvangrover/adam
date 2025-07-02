# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# Assuming Docker context is the project root
COPY backend/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source code into the container at /app
# This assumes the Docker build context is the project root
COPY backend/src/main/python/ /app/backend/src/main/python/
COPY data/sample_data/ /app/data/sample_data/
# Note: kg_builder.py uses a BASE_DIR calculation that might need adjustment
# if it doesn't correctly find 'data/sample_data' relative to its new position in the container.
# The current kg_builder.py calculates BASE_DIR by going up four levels from its own location.
# Inside the container, if kg_builder.py is at /app/backend/src/main/python/kg_builder.py,
# then BASE_DIR would become /app. So, data should be at /app/data/sample_data. This copy is correct.

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Set the PYTHONPATH to include the application's root directory inside the container
# This helps Python find your modules correctly when using absolute-style imports from project root.
# For `from .module` style imports, this is less critical if running as a module, but good for tools.
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Run main.py when the container launches
# The command uses module paths, which should work with PYTHONPATH set to /app.
CMD ["uvicorn", "backend.src.main.python.main:app", "--host", "0.0.0.0", "--port", "8000"]
