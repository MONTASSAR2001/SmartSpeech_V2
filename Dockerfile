FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["/bin/bash"]