# Lightweight Python base image
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-optimized PyTorch and Torchvision (keeps image small ~400MB instead of ~4GB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining web and application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart jinja2 pillow streamlit

# Copy project source code
COPY . .

# Expose port (default 8000, dynamically overridden by Render / Cloud Run $PORT)
EXPOSE 8000

# Start FastAPI server using dynamic PORT provided by cloud environments
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]

