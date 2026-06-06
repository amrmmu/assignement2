FROM python:3.11-slim

LABEL maintainer="MMU Assignment 2 — Vaccination Centre Assignment Benchmark"

WORKDIR /app

# Install C build tools (needed by scipy / numpy wheels on slim image)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Pre-create output directories
RUN mkdir -p results/q1 results/q2

# Set PYTHONPATH so every script can do "import utils" etc. directly
ENV PYTHONPATH=/app/src

# Generate synthetic datasets if they are not present in the image
RUN python src/generate_data.py

# Make shell scripts executable
RUN chmod +x run_q1.sh run_q2.sh run_analysis.sh run_all.sh 2>/dev/null || true

CMD ["python", "-c", "print('Container ready. Use docker compose run to execute benchmarks.')"]
