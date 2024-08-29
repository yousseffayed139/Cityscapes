# Use an official Python runtime as a parent image
FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

# Set environment variables
ENV BASE_DIR=/data/base_dir
ENV MODEL_WEIGHTS_PATH=/data/model_weights.pth

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nano && \
    # Clean up to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --timeout 120

# Run main.py when the container launches
CMD ["python", "main.py"]
