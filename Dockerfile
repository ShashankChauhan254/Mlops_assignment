# Use an official Python image from Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary libraries
RUN pip install --no-cache-dir numpy scikit-learn torch torchvision joblib

# Run the Python script
CMD ["python", "qml.py"]

