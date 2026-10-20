#!/usr/bin/env bash
# ==============================================================================
# The Lodge Maribaya - Sistem Photo Automated Deployment Script
# Kompatibel dengan Ubuntu Server 20.04 / 22.04 / 24.04 LTS
# ==============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================================${NC}"
echo -e "${BLUE}   The Lodge Maribaya - Setup & Deploy Sistem Photo       ${NC}"
echo -e "${BLUE}==========================================================${NC}"

# 1. Update package list & install essentials
echo -e "\n${YELLOW}[1/5] Menginstal dependensi sistem dasar (curl, git, cifs-utils)...${NC}"
sudo apt-get update -y
sudo apt-get install -y curl git cifs-utils ca-certificates gnupg lsb-release

# 2. Install Docker & Docker Compose if not present
if ! command -v docker &> /dev/null; then
    echo -e "\n${YELLOW}[2/5] Memasang Docker Engine resmi...${NC}"
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER" || true
else
    echo -e "\n${GREEN}[2/5] Docker sudah terpasang.${NC}"
fi

# 3. Prepare data directories & Samba share
echo -e "\n${YELLOW}[3/5] Menyiapkan struktur folder data & Windows File Share (Samba)...${NC}"
mkdir -p data/raw data/results data/db data/models/insightface img static templates
sudo chmod -R 777 data

# Install & configure Samba automatically so operators can open \\IP\park-photos\raw
sudo apt-get install -y samba
DATA_DIR="$(pwd)/data"
if ! grep -q "\[park-photos\]" /etc/samba/smb.conf 2>/dev/null; then
    echo -e "${YELLOW}Mengonfigurasi share folder [park-photos] di Samba...${NC}"
    sudo bash -c "cat >> /etc/samba/smb.conf" << EOL

[park-photos]
   comment = The Lodge Maribaya Photos & Results
   path = $DATA_DIR
   browseable = yes
   read only = no
   guest ok = yes
   create mask = 0777
   directory mask = 0777
   follow symlinks = yes
   wide links = yes
   unix extensions = no
EOL
    sudo systemctl restart smbd || true
    sudo systemctl enable smbd || true
fi

# 4. Build and start container
echo -e "\n${YELLOW}[4/5] Membangun image Docker & mengunduh model AI...${NC}"
sudo docker compose build

echo -e "\n${YELLOW}[5/5] Menyalakan container Sistem Photo...${NC}"
sudo docker compose down 2>/dev/null || true
sudo docker compose up -d

echo -e "\n${GREEN}==========================================================${NC}"
echo -e "${GREEN}  DEPLOYMENT BERHASIL SELESAI!                             ${NC}"
echo -e "${GREEN}==========================================================${NC}"
sudo docker compose ps

IP_ADDR=$(hostname -I | awk '{print $1}')
echo -e "\nSistem Photo siap digunakan:"
echo -e "  🌐 Layar Tamu / Kiosk : ${GREEN}http://${IP_ADDR}:8000/${NC}"
echo -e "  ⚙️  Dashboard Admin    : ${GREEN}http://${IP_ADDR}:8000/admin${NC}"
echo -e "  🔑 PIN Keamanan Default : ${YELLOW}BI5mill4h@@@${NC} atau ${YELLOW}1234${NC}\n"

