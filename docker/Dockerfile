# Use lightweight Python base image
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /usr/local/bin/

# Set build arguments
ARG UID=1000
ARG GID=1000
ARG APP_PORT=8000

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --force -g ${GID} appuser \
    && useradd -ms /bin/bash --no-user-group -g ${GID} -u ${UID} appuser

# Set working directory and ensure proper ownership
WORKDIR /app
RUN chown -R ${UID}:${GID} /app

# Copy only dependency files
COPY --chown=${UID}:${GID} pyproject.toml uv.lock .python-version ./

# Install dependencies
RUN uv sync --locked --no-cache --no-dev --no-install-project

# Copy application code and entrypoint script
COPY --chown=${UID}:${GID} . /app
COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user for runtime security
USER appuser

# Expose port and set entrypoint
EXPOSE ${APP_PORT}

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
