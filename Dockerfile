# Dockerfile

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

RUN mkdir -p /app/data

# стандартный порт Gradio
EXPOSE 7860

CMD ["python", "app.py"]