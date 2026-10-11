FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# Default: deterministic SVB replay. Override with: docker run -e TRIDENT_MODE=live ...
ENV TRIDENT_MODE=replay TRIDENT_REPLAY=svb_2023-03 TRIDENT_SPEED=600
CMD ["python", "-m", "uvicorn", "trident.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
