#!/usr/bin/env bash
# ==============================================================================
# The Lodge Maribaya - Samba File Share Setup Script
# Mengaktifkan sharing folder \\<IP-SERVER>\park-photos untuk Windows File Explorer
# ==============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}==========================================================${NC}"
echo -e "${BLUE}   Mengaktifkan Windows File Sharing (Samba) di Server    ${NC}"
echo -e "${BLUE}==========================================================${NC}"

# 1. Pastikan paket Samba terpasang
echo -e "\n${YELLOW}[1/4] Menginstal paket Samba...${NC}"
sudo apt-get update -y
sudo apt-get install -y samba

# 2. Buat direktori data fisik dan pastikan izin 777
DATA_DIR="/opt/SistemPhoto/data"
echo -e "\n${YELLOW}[2/4] Menyiapkan direktori fisik $DATA_DIR...${NC}"
sudo mkdir -p "$DATA_DIR/raw" "$DATA_DIR/results" "$DATA_DIR/db"
sudo chmod -R 777 "$DATA_DIR"
CURRENT_USER="${SUDO_USER:-$USER}"

# 3. Konfigurasi /etc/samba/smb.conf
echo -e "\n${YELLOW}[3/4] Mengonfigurasi /etc/samba/smb.conf...${NC}"
if [ ! -f /etc/samba/smb.conf.backup ]; then
    sudo cp /etc/samba/smb.conf /etc/samba/smb.conf.backup
fi

# Pastikan map to guest = Bad User ada di [global] agar guest login diizinkan
if ! grep -q "map to guest" /etc/samba/smb.conf; then
    sudo sed -i '/\[global\]/a \   map to guest = Bad User\n   usershare allow guests = yes' /etc/samba/smb.conf
fi

# Tambahkan atau perbarui konfigurasi [park-photos]
if ! grep -q "\[park-photos\]" /etc/samba/smb.conf; then
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
   force user = $CURRENT_USER
   create mask = 0777
   directory mask = 0777
   follow symlinks = yes
   wide links = yes
   unix extensions = no
EOL
fi

# Izinkan firewall Ubuntu jika UFW aktif
sudo ufw allow samba 2>/dev/null || true

# Daftarkan user server ke Samba dengan password default '1234'
# (Agar jika Windows 11 menolak Guest Login, operator bisa login dengan user: server, pass: 1234)
(echo "1234"; echo "1234") | sudo smbpasswd -a -s "$CURRENT_USER" 2>/dev/null || true

# 4. Restart layanan Samba
echo -e "\n${YELLOW}[4/4] Memulai ulang layanan Samba...${NC}"
sudo systemctl restart smbd
sudo systemctl enable smbd

IP_ADDR=$(hostname -I | awk '{print $1}')
echo -e "\n${GREEN}==========================================================${NC}"
echo -e "${GREEN}  SAMBA SHARING BERHASIL DIAKTIFKAN!                       ${NC}"
echo -e "${GREEN}==========================================================${NC}"
echo -e "Sekarang folder jaringan sudah AKTIF dan SIAP DIAKSES di Windows:"
echo -e "  📂 Folder Upload Foto : ${GREEN}\\\\${IP_ADDR}\\park-photos\\raw${NC}"
echo -e "  📁 Folder Hasil Tamu  : ${GREEN}\\\\${IP_ADDR}\\park-photos\\results${NC}"
echo -e "\nCatatan jika Windows meminta Username & Password:"
echo -e "  Username : ${YELLOW}${CURRENT_USER}${NC}"
echo -e "  Password : ${YELLOW}1234${NC} (atau password akun Ubuntu Anda)\n"
