# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    gdb \
    nodejs \
    npm \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /workspace

# Add a non-root user
RUN useradd -m -s /bin/bash cudauser && \
    echo "cudauser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Copy package info and install dependencies
COPY package.json ./
RUN npm install

# Copy project files
COPY . /workspace
RUN chown -R cudauser:cudauser /workspace

# Expose port 3000 (Playground Server)
EXPOSE 3000

# Start script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER cudauser
ENTRYPOINT ["/entrypoint.sh"]
