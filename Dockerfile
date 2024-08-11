# Use an official Python 3.11.3 runtime as a parent image
FROM python:3.11.3-slim

# Set environment variables to prevent Python from writing .pyc files to disc
ENV PYTHONUNBUFFERED=1

# Create and set the working directory
WORKDIR /app

# Install system dependencies including GDAL and its Python bindings
RUN apt-get update && \
    apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip and wheel
RUN pip install --no-cache-dir pip wheel

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install GDAL Python bindings
RUN pip install --no-cache-dir GDAL

# Copy the rest of the application code into the container
COPY . /app/

# Expose port 5000 for the Flask app
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]