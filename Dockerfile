FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the plugin
COPY livekit-plugins-arabic-turn-detector/ ./livekit-plugins-arabic-turn-detector/
RUN pip install --no-cache-dir ./livekit-plugins-arabic-turn-detector

# Install agent deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model at build time (baked into image)
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('Moustafa3092/livekit-turn-detector-arabic', trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained('Moustafa3092/livekit-turn-detector-arabic')"

# Copy agent
COPY agent.py .
COPY .env.example .env.local

EXPOSE 8080

CMD ["python", "agent.py", "start"]
