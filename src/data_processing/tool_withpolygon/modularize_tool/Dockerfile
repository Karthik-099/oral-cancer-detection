
FROM python:3.10-slim


WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy all files (including models/)
COPY . /app

# Update pip and install dependencies with retries
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --retries 10 --timeout 120 --index-url https://download.pytorch.org/whl/cpu

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app with headless mode
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.enableCORS", "false", "--server.headless", "true"]