#!/usr/bin/env bash
# ==============================================================================
# The Lodge Maribaya - Persistent Permissions Auto-Fix Script
# Memastikan folder /opt/SistemPhoto/data/raw SELALU Read, Write & Delete (777)
# secara otomatis bahkan setelah server Ubuntu di-restart / reboot.
# ==============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}==========================================================${NC}"
echo -e "${BLUE}   Mengaktifkan Izin Otomatis Permanen Folder RAW (777)   ${NC}"
echo -e "${BLUE}==========================================================${NC}"

DATA_DIR="/opt/SistemPhoto/data"
CURRENT_USER="${SUDO_USER:-$USER}"

# 1. Bersihkan auto-mount CIFS lama di /etc/fstab jika ada
echo -e "\n${YELLOW}[1/4] Memeriksa /etc/fstab dari auto-mount lama...${NC}"
if grep -q "$DATA_DIR/raw" /etc/fstab 2>/dev/null; then
    echo -e "${YELLOW}Menonaktifkan auto-mount lama pada /etc/fstab...${NC}"
    sudo sed -i "\|${DATA_DIR}/raw|s|^|# |" /etc/fstab
fi

# 2. Paksa lepas mount yang masih menggantung
echo -e "\n${YELLOW}[2/4] Melepas mount lama jika masih aktif...${NC}"
if mount | grep -q "$DATA_DIR/raw"; then
    sudo umount -f -l "$DATA_DIR/raw" 2>/dev/null || true
fi

# 3. Buat folder dan terapkan izin 777
echo -e "\n${YELLOW}[3/4] Menerapkan izin kepemilikan dan 777 ke $DATA_DIR...${NC}"
sudo mkdir -p "$DATA_DIR/raw" "$DATA_DIR/results" "$DATA_DIR/db"
sudo chown -R "$CURRENT_USER":"$CURRENT_USER" "$DATA_DIR"
sudo chmod -R 777 "$DATA_DIR"

# 4. Pasang Systemd Service agar otomatis 777 setiap server booting/restart
echo -e "\n${YELLOW}[4/4] Memasang Systemd Service (Otomatis saat Restart)...${NC}"
SERVICE_FILE="/etc/systemd/system/sistemphoto-permissions.service"

sudo bash -c "cat > $SERVICE_FILE" << EOL
[Unit]
Description=Ensure SistemPhoto Data Directory is 777 Read/Write
After=network.target local-fs.target

[Service]
Type=oneshot
ExecStartPre=-/bin/umount -f -l $DATA_DIR/raw
ExecStart=/bin/chmod -R 777 $DATA_DIR
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOL

sudo systemctl daemon-reload
sudo systemctl enable sistemphoto-permissions.service
sudo systemctl restart sistemphoto-permissions.service

echo -e "\n${GREEN}==========================================================${NC}"
echo -e "${GREEN}  IZIN PERMANEN BERHASIL DIAKTIFKAN!                      ${NC}"
echo -e "${GREEN}==========================================================${NC}"
echo -e "Folder ${GREEN}$DATA_DIR/raw${NC} sekarang:"
echo -e "  ✅ Izin 777 aktif (Bebas Copy, Paste, Rename, Delete)."
echo -e "  ✅ Otomatis tetap 777 setiap kali server Ubuntu di-restart."
echo -e "  ✅ Service systemd: sistemphoto-permissions.service (Active & Enabled)\n"

