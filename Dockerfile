# Use the specified base image
FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy the Python script into the container
COPY main.py .

# Install any additional Python packages if needed
RUN pip install numpy

# Set the default command to execute the Python script
CMD ["python", "main.py"]
