#!/bin/bash

# Name of the Docker image
IMAGE_NAME="diffusion-cifar10"

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Run the Docker container
echo "Running Docker container..."
docker run --rm -it \
    --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    $IMAGE_NAME
