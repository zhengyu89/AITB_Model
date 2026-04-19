# Use lightweight Python base image
FROM python:3.12-slim

# Set build arguments
ARG UID=1000
ARG GID=1000
ARG APP_PORT=8000

# Set environment variables for Python and Pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies and upgrade Pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --upgrade pip

# Create non-root user
RUN groupadd --force -g ${GID} appuser \
    && useradd -ms /bin/bash --no-user-group -g ${GID} -u ${UID} appuser

# Set working directory and ensure proper ownership
WORKDIR /app
RUN chown -R ${UID}:${GID} /app

# Copy only dependency files
COPY --chown=${UID}:${GID} requirements.txt ./

# Install dependencies
RUN /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application code and entrypoint script
COPY --chown=${UID}:${GID} . /app
COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user for runtime security
USER appuser

# Expose port and set entrypoint
EXPOSE ${APP_PORT}
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
