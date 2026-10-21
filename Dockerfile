FROM python:3.12-slim

# Create non-root user matching Hugging Face Spaces security requirements
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency specifications
COPY --chown=user pyproject.toml uv.lock ./

# Install uv and sync dependencies
RUN pip install --no-cache-dir uv && uv sync --frozen

# Copy project files
COPY --chown=user . .

# Expose standard Hugging Face Spaces port
EXPOSE 7860

# Launch Gradio dashboard
CMD ["uv", "run", "python", "-m", "app.gradio_app"]

