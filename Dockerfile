# ============================
# Multi-stage Optimized Dockerfile
# ============================

# -------- Stage 1: Builder --------
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Copy only requirement files to leverage Docker cache
COPY requirements.txt .

# Install dependencies (and clean up pip cache)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app code separately
COPY . .

# -------- Stage 2: Final Image --------
FROM python:3.10-slim

WORKDIR /app

# Copy only installed packages and app code from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]