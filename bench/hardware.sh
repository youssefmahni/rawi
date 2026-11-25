#!/usr/bin/env bash
# Read-only hardware report for the Rawi mini PC. Prints to stdout; capture with:
#   bash bench/hardware.sh > bench/results/hardware.txt
set -u

echo "== uname =="
uname -a

echo
echo "== OS =="
grep -E '^PRETTY_NAME=' /etc/os-release 2>/dev/null || echo "PRETTY_NAME unavailable"

echo
echo "== CPU =="
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket' || true
echo "flags:"
lscpu | grep -o 'avx2\|avx-512\|amx\|avx512' | sort -u | tr '\n' ' ' || true
echo

echo
echo "== RAM =="
free -g

echo
echo "== Disk =="
df -h ~ /tmp 2>/dev/null

echo
echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  echo "no NVIDIA GPU"
fi
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi
else
  echo "no ROCm (rocm-smi) present"
fi

echo
echo "== lspci (VGA / 3D / NPU) =="
lspci 2>/dev/null | grep -Ei 'vga|3d|npu|display' || echo "no VGA/3D/NPU lines (or lspci missing)"

echo
echo "== toolchain =="
python3 --version 2>/dev/null || echo "no python3"
ffmpeg -version 2>/dev/null | head -1 || echo "no ffmpeg"
if systemctl --user >/dev/null 2>&1; then
  echo "systemctl --user: works"
else
  echo "systemctl --user: NOT available"
fi
