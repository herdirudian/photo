#!/usr/bin/env bash
# ==============================================================================
# The Lodge Maribaya - Samba File Share Setup Script
# Membuka folder \\<IP-SERVER>\park-photos untuk akses Windows File Explorer
# ==============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}==========================================================${NC}"
echo -e "${BLUE}   Mengaktifkan Windows File Sharing (Samba) di Server    ${NC}"
echo -e "${BLUE}==========================================================${NC}"

# 1. Install Samba
echo -e "\n${YELLOW}[1/3] Menginstal paket Samba...${NC}"
sudo apt-get update -y
sudo apt-get install -y samba

# 2. Pastikan direktori data ada dan memiliki izin 777
DATA_DIR="/opt/SistemPhoto/data"
mkdir -p "$DATA_DIR/raw" "$DATA_DIR/results" "$DATA_DIR/db"
sudo chmod -R 777 "$DATA_DIR"

# Backup smb.conf jika belum ada
if [ ! -f /etc/samba/smb.conf.backup ]; then
    sudo cp /etc/samba/smb.conf /etc/samba/smb.conf.backup
fi

# 3. Tambahkan konfigurasi [park-photos] jika belum ada
if ! grep -q "\[park-photos\]" /etc/samba/smb.conf; then
    echo -e "\n${YELLOW}[2/3] Mendaftarkan share folder [park-photos] ke /etc/samba/smb.conf...${NC}"
    sudo bash -c "cat >> /etc/samba/smb.conf" << EOL

# =========================================================
# The Lodge Maribaya - Sistem Photo Share
# =========================================================
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
else
    echo -e "\n${GREEN}[2/3] Konfigurasi [park-photos] sudah ada di /etc/samba/smb.conf.${NC}"
fi

# 4. Restart layanan Samba
echo -e "\n${YELLOW}[3/3] Merestart layanan Samba...${NC}"
sudo systemctl restart smbd
sudo systemctl enable smbd

IP_ADDR=$(hostname -I | awk '{print $1}')
echo -e "\n${GREEN}==========================================================${NC}"
echo -e "${GREEN}  SAMBA SHARING BERHASIL DIAKTIFKAN!                       ${NC}"
echo -e "${GREEN}==========================================================${NC}"
echo -e "Sekarang Anda dapat membuka langsung dari Windows File Explorer:"
echo -e "  📂 Folder Masuk Foto : ${GREEN}\\\\${IP_ADDR}\\park-photos\\raw${NC}"
echo -e "  📁 Folder Hasil Tamu : ${GREEN}\\\\${IP_ADDR}\\park-photos\\results${NC}"
echo -e "\nSilakan salin/paste folder foto wahana ke: ${GREEN}\\\\${IP_ADDR}\\park-photos\\raw${NC}"
echo -e "Lalu klik tombol 'Pindai Ulang' pada web browser.\n"
