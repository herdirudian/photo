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
CURRENT_USER="${SUDO_USER:-$USER}"

echo -e "\n${YELLOW}[2/4] Menyiapkan direktori fisik $DATA_DIR...${NC}"
# Lepas mount CIFS lama jika masih terpasang agar tidak me-lock folder raw
if mount | grep -q "$DATA_DIR/raw"; then
    echo -e "${YELLOW}Terdeteksi mount CIFS lama pada data/raw, melepas mount secara bersih...${NC}"
    sudo umount -f -l "$DATA_DIR/raw" 2>/dev/null || true
fi

sudo mkdir -p "$DATA_DIR/raw" "$DATA_DIR/results" "$DATA_DIR/db"
sudo chown -R "$CURRENT_USER":"$CURRENT_USER" "$DATA_DIR"
sudo chmod -R 777 "$DATA_DIR"

# Buat symlink kompatibilitas /app di host sebagai backup kompatibilitas untuk path absolut container
if [ ! -d "/app" ] && [ ! -L "/app" ]; then
    sudo ln -s /opt/SistemPhoto /app 2>/dev/null || true
fi

# 3. Konfigurasi /etc/samba/smb.conf
echo -e "\n${YELLOW}[3/4] Mengonfigurasi /etc/samba/smb.conf untuk file besar original...${NC}"
if [ ! -f /etc/samba/smb.conf.backup ]; then
    sudo cp /etc/samba/smb.conf /etc/samba/smb.conf.backup
fi

# Pastikan map to guest = Bad User dan unix extensions = no ada di [global] agar guest login & symlinks diizinkan
if ! grep -q "map to guest" /etc/samba/smb.conf; then
    sudo sed -i '/\[global\]/a \   map to guest = Bad User\n   usershare allow guests = yes\n   unix extensions = no' /etc/samba/smb.conf
fi

# Bersihkan blok [park-photos] lama jika ada agar tidak duplikat/konflik
if grep -q "\[park-photos\]" /etc/samba/smb.conf; then
    sudo sed -i '/\[park-photos\]/,$d' /etc/samba/smb.conf
fi

# Tulis konfigurasi baru yang mendukung transfer file DSLR besar tanpa batas ukuran
sudo bash -c "cat >> /etc/samba/smb.conf" << EOL

# =========================================================
# The Lodge Maribaya - Sistem Photo Share (Original Large Files)
# =========================================================
[park-photos]
   comment = The Lodge Maribaya Photos & Results
   path = $DATA_DIR
   browseable = yes
   read only = no
   writable = yes
   writeable = yes
   guest ok = yes
   force user = $CURRENT_USER
   create mask = 0777
   directory mask = 0777
   force create mode = 0777
   force directory mode = 0777
   follow symlinks = yes
   wide links = yes
   unix extensions = no
   strict allocate = no
   allocation roundup size = 0
   aio read size = 1
   aio write size = 1
EOL

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
