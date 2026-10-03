# Runtime image for the FraudLens API and Streamlit app.
# The trained model is NOT baked in: mount ./artifacts (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRAUDLENS_CONFIG=/app/configs/config.yaml \
    MPLCONFIGDIR=/tmp/matplotlib

# libgomp1: OpenMP runtime required by LightGBM.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -e .

COPY configs ./configs
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 1000 fraudlens && chown -R fraudlens /app
USER fraudlens

EXPOSE 8000 8501
CMD ["uvicorn", "fraudlens.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
