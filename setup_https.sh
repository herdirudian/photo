#!/usr/bin/env bash
# ==============================================================================
# The Lodge Maribaya - Setup HTTPS / SSL Reverse Proxy (Nginx)
# Mengaktifkan HTTPS (Port 443) dengan Sertifikat SSL untuk Akses Kamera & Web Aman
# ==============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}==========================================================${NC}"
echo -e "${BLUE}   Mengaktifkan Akses HTTPS / SSL di Server Ubuntu        ${NC}"
echo -e "${BLUE}==========================================================${NC}"

# 1. Dapatkan IP Server
IP_ADDR=$(hostname -I | awk '{print $1}')
if [ -z "$IP_ADDR" ]; then
    IP_ADDR="192.168.100.90"
fi

echo -e "\n${YELLOW}[1/4] Mendeteksi IP Server: ${CYAN}${IP_ADDR}${NC}"

# 2. Instal Nginx
echo -e "\n${YELLOW}[2/4] Menginstal Nginx Web Server...${NC}"
sudo apt-get update -y
sudo apt-get install -y nginx openssl

# 3. Buat Sertifikat SSL (Valid 10 Tahun / 3650 Hari) dengan SAN IP
echo -e "\n${YELLOW}[3/4] Membuat Sertifikat SSL untuk IP ${IP_ADDR}...${NC}"
SSL_DIR="/etc/ssl/sistemphoto"
sudo mkdir -p "$SSL_DIR"

sudo openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
  -keyout "$SSL_DIR/server.key" \
  -out "$SSL_DIR/server.crt" \
  -subj "/C=ID/ST=West Java/L=Bandung/O=The Lodge Maribaya/OU=IT/CN=${IP_ADDR}" \
  -addext "subjectAltName=IP:${IP_ADDR},IP:127.0.0.1,DNS:localhost"

# Salin juga sertifikat ke folder proyek agar mudah disalin ke Windows jika ingin di-import
sudo cp "$SSL_DIR/server.crt" "/opt/SistemPhoto/sistemphoto-ssl.crt" 2>/dev/null || true
sudo chmod 644 "/opt/SistemPhoto/sistemphoto-ssl.crt" 2>/dev/null || true

# 4. Konfigurasi Nginx Reverse Proxy
echo -e "\n${YELLOW}[4/4] Mengonfigurasi Nginx Reverse Proxy (Port 80 & 443)...${NC}"
NGINX_CONF="/etc/nginx/sites-available/sistemphoto"

sudo bash -c "cat > $NGINX_CONF" << EOL
# HTTP: Redirect otomatis ke HTTPS
server {
    listen 80;
    server_name _;
    return 301 https://\$host\$request_uri;
}

# HTTPS: SSL Termination & Reverse Proxy ke FastAPI Docker (:8000)
server {
    listen 443 ssl;
    server_name _;

    ssl_certificate $SSL_DIR/server.crt;
    ssl_certificate_key $SSL_DIR/server.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Kapasitas upload maksimal foto DSLR resolusi tinggi (hingga 50MB)
    client_max_body_size 50M;

    # Endpoint download sertifikat SSL untuk Windows Trusted Root Store
    location = /sistemphoto-ssl.crt {
        alias $SSL_DIR/server.crt;
        default_type application/x-x509-ca-cert;
        add_header Content-Disposition 'attachment; filename="sistemphoto-ssl.crt"';
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Dukungan streaming gambar & koneksi cepat
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
EOL

# Aktifkan konfigurasi di Nginx
sudo ln -sf /etc/nginx/sites-available/sistemphoto /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Uji konfigurasi Nginx
sudo nginx -t

# Buka firewall untuk port 80 & 443
sudo ufw allow 80/tcp 2>/dev/null || true
sudo ufw allow 443/tcp 2>/dev/null || true

# Restart Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx

echo -e "\n${GREEN}==========================================================${NC}"
echo -e "${GREEN}  HTTPS BERHASIL DIAKTIFKAN DENGAN SUKSES!                ${NC}"
echo -e "${GREEN}==========================================================${NC}"
echo -e "Sekarang aplikasi dapat diakses dengan aman via HTTPS:"
echo -e "  🌐 URL Utama (HTTPS) : ${GREEN}https://${IP_ADDR}${NC}"
echo -e "  ⚙️ Admin Dashboard  : ${GREEN}https://${IP_ADDR}/admin${NC}"
echo -e "\nKeuntungan HTTPS aktif:"
echo -e "  1. Akses Webcam di Chrome/Edge langsung aktif TANPA flags browser!"
echo -e "  2. Semua port 80 (HTTP) otomatis dialihkan ke HTTPS."
echo -e "  3. Sertifikat tersimpan di: /opt/SistemPhoto/sistemphoto-ssl.crt\n"

