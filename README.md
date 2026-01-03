# CUDA Playground 🚀

A streamlined, web-based development environment for CUDA programming. Write, compile, and analyze GPU kernels directly in your browser without any local installation headaches.

## 🛠 Features

- **Zero Config**: Dockerized environment with NVIDIA GPU passthrough.
- **Minimalist IDE**: Focused Monaco-based editor for high-performance coding.
- **Instant Insights**: Automatic emission of **PTX** and **SASS** for instruction-level analysis.
- **Real-time Console**: Capture CPU and GPU output instantly.

## 🚀 Quick Start

1. **Prerequisites**: Ensure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.
2. **Launch**:
   ```bash
   docker compose up --build
   ```
3. **Open**: Navigate to **`http://localhost:3000`** in your browser.

## 📂 Project Structure

- `public/`: Frontend assets (Editor, Styles, Logic).
- `server.js`: Node.js backend handling CUDA compilation.
- `hello.cu`: The active CUDA source file (synced with the editor).
- `Makefile`: Optimized build instructions for the compiler.
- `docker-compose.yml`: GPU & Environment orchestration.

---
*Built for Parallel Programming & Performance Analysis.*
