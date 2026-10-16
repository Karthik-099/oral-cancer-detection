#!/usr/bin/env bash
# One-time setup of an Oracle Cloud VM (Ubuntu 22.04/24.04, x86 or Ampere ARM)
# for the OralAID API. Run on the VM as the default "ubuntu" user:
#
#   curl -fsSL https://raw.githubusercontent.com/Karthik-099/oral-cancer-detection/prod/deploy/oracle/setup.sh | bash
#
# Afterwards: fill in deploy/.env, copy the model file, then run deploy.sh.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Karthik-099/oral-cancer-detection.git}"
BRANCH="${BRANCH:-prod}"
APP_DIR="${APP_DIR:-$HOME/oral-cancer-detection}"

echo "==> Installing Docker"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi
sudo systemctl enable --now docker

echo "==> Opening ports 80/443 in the OS firewall"
# Oracle's Ubuntu images ship iptables rules that block everything except SSH,
# even after you add an Ingress Rule in the VCN security list.
for port in 80 443; do
  sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null || \
    sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport "$port" -j ACCEPT
done
sudo iptables -C INPUT -p udp --dport 443 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 6 -p udp --dport 443 -j ACCEPT
if command -v netfilter-persistent >/dev/null; then sudo netfilter-persistent save; fi

echo "==> Adding 4 GB swap (helps small shapes during image builds)"
if ! sudo swapon --show | grep -q /swapfile; then
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "==> Fetching the code"
if [ ! -d "$APP_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi
mkdir -p "$APP_DIR/models"

if [ ! -f "$APP_DIR/deploy/.env" ]; then
  cp "$APP_DIR/deploy/.env.example" "$APP_DIR/deploy/.env"
  # Pre-fill the two secrets so they're never weak or reused.
  gen() { python3 -c "import secrets; print(secrets.token_urlsafe(48))"; }
  sed -i "s|^ORALAID_SECRET_KEY=.*|ORALAID_SECRET_KEY=$(gen)|; s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$(gen)|" "$APP_DIR/deploy/.env"
  PUBLIC_IP=$(curl -fsS https://ifconfig.me || true)
  [ -n "$PUBLIC_IP" ] && sed -i "s|^API_DOMAIN=.*|API_DOMAIN=api.${PUBLIC_IP}.sslip.io|" "$APP_DIR/deploy/.env"
  chmod 600 "$APP_DIR/deploy/.env"
fi

cat <<NEXT

Setup done. Next steps:
  1. Edit $APP_DIR/deploy/.env  (set ACME_EMAIL; check API_DOMAIN)
  2. Copy the model from your laptop:
       scp models/final_efficientnet_b4_oral_cancer.pth ubuntu@<vm-ip>:$APP_DIR/models/
  3. In Oracle Cloud: VCN -> Security List -> add Ingress Rules for TCP 80 and 443 (source 0.0.0.0/0)
  4. Start it:  bash $APP_DIR/deploy/oracle/deploy.sh
  (Log out and back in once so your user can run docker without sudo.)
NEXT
