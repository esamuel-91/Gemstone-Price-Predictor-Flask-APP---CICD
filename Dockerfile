# Stage 1: Build stage
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Set the working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy only the files needed for installation to cache layers
COPY pyproject.toml uv.lock ./

# Install dependencies (using uv for speed)
RUN uv sync --frozen --no-install-project --no-dev

# Stage 2: Final runtime stage
FROM python:3.13-slim-bookworm

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy your application code
COPY . .

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Expose the port your Flask app runs on (usually 5000 or 8080)
EXPOSE 5000

# Run the application using Gunicorn for production stability
# Install gunicorn if not in your pyproject.toml: uv pip install gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]