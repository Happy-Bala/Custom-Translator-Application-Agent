

FROM python:3.11-slim

# Install OS dependencies for Azure SDK, FastAPI, and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Create app directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose port for FastAPI/Uvicorn
EXPOSE 8080

# Default environment variables (can be overridden at runtime)
ENV AZURE_BLOB_STORAGE_CONNECTION_STRING="" \
    AZURE_BLOB_CONTAINER_NAME="" \
    AZURE_TRANSLATOR_API_KEY="" \
    AZURE_TRANSLATOR_ENDPOINT="" \
    AZURE_TRANSLATOR_REGION="" \
    AZURE_TRANSLATOR_TARGET_LANGUAGE="en" \
    AZURE_SEARCH_ENDPOINT="" \
    AZURE_SEARCH_API_KEY="" \
    AZURE_SEARCH_INDEX_NAME="" \
    AZURE_OPENAI_ENDPOINT="" \
    AZURE_OPENAI_API_KEY="" \
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-ada-002" \
    AZURE_OPENAI_DEPLOYMENT="gpt-4-1106-preview" \
    LLM_TEMPERATURE="0.7" \
    LLM_MAX_TOKENS="2000"

# Startup command
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8080"]

