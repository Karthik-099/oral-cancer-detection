#!/bin/bash
set -e

echo "🦷 OralAID On-Premise Setup"
echo "=========================="

# Check for Docker
if ! [ -x "$(command -v docker)" ]; then
  echo "❌ Error: docker is not installed." >&2
  exit 1
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected. Enabling GPU acceleration."
    USE_GPU=true
else
    echo "ℹ️ No GPU detected. Running on CPU."
    USE_GPU=false
fi

# Generate Hardware Fingerprint
echo "⚙️ Generating hardware fingerprint..."
FINGERPRINT=$(hostname | sha256sum | awk '{print $1}')
echo "HARDWARE_ID=$FINGERPRINT" > .env
echo "DEPLOY_MODE=onprem" >> .env
echo "SECRET_KEY=$(openssl rand -hex 24)" >> .env

echo "✅ Fingerprint generated: $FINGERPRINT"
echo "🚀 Starting services..."

docker-compose -f docker-compose.onprem.yml up -d

echo "=========================="
echo "🎉 Setup complete!"
echo "📍 Access OralAID at: http://localhost"
echo "🔑 Provide your fingerprint to OralAID for license activation."
