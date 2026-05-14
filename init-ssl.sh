#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# SSL Certificate Bootstrap Script for AlmatyNest
# ═══════════════════════════════════════════════════════════════
# Run this ONCE to obtain the initial Let's Encrypt certificate.
#
# Usage:
#   chmod +x init-ssl.sh
#   ./init-ssl.sh YOUR_DOMAIN your@email.com
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

DOMAIN="${1:?Usage: ./init-ssl.sh DOMAIN EMAIL}"
EMAIL="${2:?Usage: ./init-ssl.sh DOMAIN EMAIL}"

echo "══════════════════════════════════════════"
echo " AlmatyNest SSL Bootstrap"
echo " Domain: $DOMAIN"
echo " Email:  $EMAIL"
echo "══════════════════════════════════════════"

# Step 1: Use the init config (HTTP only)
echo "[1/5] Switching to HTTP-only Nginx config..."
cp nginx/nginx-init.conf nginx/nginx.conf.bak
cp nginx/nginx-init.conf nginx/active.conf

# Replace YOUR_DOMAIN placeholder
sed -i "s/YOUR_DOMAIN/$DOMAIN/g" nginx/active.conf

# Step 2: Start nginx with init config
echo "[2/5] Starting Nginx (HTTP only)..."
docker compose run --rm -d \
  -v "$(pwd)/nginx/active.conf:/etc/nginx/conf.d/default.conf:ro" \
  -v "$(pwd)/certbot/www:/var/www/certbot:ro" \
  -p 80:80 \
  nginx nginx -g 'daemon off;' || {
    # If standalone doesn't work, start via compose
    cp nginx/active.conf nginx/nginx.conf
    docker compose up -d nginx
}

# Step 3: Obtain certificate
echo "[3/5] Requesting SSL certificate from Let's Encrypt..."
docker compose run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  --email "$EMAIL" \
  --agree-tos \
  --no-eff-email \
  -d "$DOMAIN"

# Step 4: Switch to full config with SSL
echo "[4/5] Switching to full SSL Nginx config..."
sed -i "s/YOUR_DOMAIN/$DOMAIN/g" nginx/nginx.conf
docker compose down

# Step 5: Start everything
echo "[5/5] Starting all services..."
docker compose up -d

echo ""
echo "══════════════════════════════════════════"
echo " ✅ Done! Your site is live at:"
echo "    https://$DOMAIN"
echo "══════════════════════════════════════════"

# Cleanup
rm -f nginx/active.conf nginx/nginx.conf.bak
