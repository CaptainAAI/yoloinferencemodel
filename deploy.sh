#!/usr/bin/env bash
# Deploys the YOLO inference server (server.py) on this Jetson.
#
# Run it ON the Jetson, from a clone of this repository:
#
#   git clone https://github.com/CaptainAAI/yoloinferencemodel.git
#   cd yoloinferencemodel
#   bash deploy.sh             install and restart in the background (nohup)
#   bash deploy.sh --service   install as a systemd service that also starts after a
#                              reboot (asks for your sudo password)
#   bash deploy.sh --status    only show what is running
#
# To update later: git pull && bash deploy.sh
#
# Settings (environment variables):
#   INFER_DIR   folder server.py runs in     default ~/yoloinference (model files in model/)
#   INFER_PORT  port                         default 6767
#
# Left untouched: model/, models.json, settings.json. The previous server.py is
# kept as server.py.bak-<date>.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_DIR="${INFER_DIR:-$HOME/yoloinference}"
INFER_PORT="${INFER_PORT:-6767}"
MODE="${1:-}"
UNIT=yolo-inference

say()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }
fail() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

port_pid() { ss -ltnp 2>/dev/null | grep ":$1 " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true; }
has_unit() { [ -f "/etc/systemd/system/$1.service" ]; }

wait_http() {  # url seconds
  for _ in $(seq 1 "$2"); do curl -fs -m 2 -o /dev/null "$1" && return 0; sleep 1; done
  return 1
}

stop_port() {  # stops the process (of this user) listening on port $1
  local pid; pid="$(port_pid "$1")"
  [ -z "$pid" ] && return 0
  echo "stopping pid $pid on port $1: $(ps -o cmd= -p "$pid")"
  kill "$pid"
  for _ in $(seq 1 30); do [ -z "$(port_pid "$1")" ] && return 0; sleep 1; done
  fail "port $1 is still in use"
}

status() {
  say "Status"
  ss -ltn | grep -E ":$INFER_PORT " || echo "nothing listening on $INFER_PORT"
  has_unit "$UNIT" && echo "systemd $UNIT: $(systemctl is-active "$UNIT" || true)"
  if curl -fs -m 3 "http://127.0.0.1:$INFER_PORT/models" -o /tmp/yolo_models.json; then
    python3 - <<'PY'
import json
for name, m in json.load(open("/tmp/yolo_models.json"))["models"].items():
    print(f"  {name:<16} {m['status']:<8} {m['file']}  save_images={m.get('save_images', False)}")
PY
  else
    warn "server.py does not answer on port $INFER_PORT"
  fi
}

if [ "$MODE" = "--status" ]; then status; exit 0; fi
[ -z "$MODE" ] || [ "$MODE" = "--service" ] || fail "unknown option $MODE (use --service or --status)"

# ----- 1. Check what is needed -----
say "Checking requirements"
[ -f "$SRC/server.py" ] || fail "$SRC/server.py not found"
command -v python3 >/dev/null || fail "python3 not found"
if ! python3 -c "import fastapi, uvicorn, multipart, requests, PIL, ultralytics" 2>/dev/null; then
  fail "Python packages missing. Install them with:
  pip3 install --user --break-system-packages -r $SRC/requirements.txt
(ultralytics, PyTorch and TensorRT must already be installed for the models)"
fi
echo "python3 $(python3 -c 'import sys; print(sys.version.split()[0])')"

# ----- 2. Install the file -----
say "Installing server.py to $INFER_DIR"
mkdir -p "$INFER_DIR/model"
if [ -f "$INFER_DIR/server.py" ] && ! cmp -s "$SRC/server.py" "$INFER_DIR/server.py"; then
  backup="$INFER_DIR/server.py.bak-$(date +%Y%m%d-%H%M%S)"
  cp "$INFER_DIR/server.py" "$backup"
  echo "previous server.py kept as $backup"
fi
cp -f "$SRC/server.py" "$INFER_DIR/server.py"

# ----- 3. (Re)start -----
if [ "$MODE" = "--service" ]; then
  say "Installing systemd service $UNIT (sudo)"
  stop_port "$INFER_PORT"
  sudo tee "/etc/systemd/system/$UNIT.service" >/dev/null <<UNIT
[Unit]
Description=YOLO inference server (server.py, port $INFER_PORT)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$INFER_DIR
Environment=PORT=$INFER_PORT
ExecStart=$(command -v python3) $INFER_DIR/server.py
Restart=on-failure
RestartSec=5
StandardOutput=append:$INFER_DIR/server.log
StandardError=append:$INFER_DIR/server.log

[Install]
WantedBy=multi-user.target
UNIT
  sudo systemctl daemon-reload
  sudo systemctl enable "$UNIT" >/dev/null
  sudo systemctl restart "$UNIT"
elif has_unit "$UNIT"; then
  say "Restarting the systemd service (installed earlier with --service)"
  sudo systemctl restart "$UNIT"
else
  say "Restarting server.py on port $INFER_PORT (background)"
  stop_port "$INFER_PORT"
  # Redirect the whole background subshell, so it never holds this terminal (or an ssh session) open
  (cd "$INFER_DIR" && PORT="$INFER_PORT" exec nohup python3 server.py) >> "$INFER_DIR/server.log" 2>&1 < /dev/null &
fi

echo "waiting for the models to load (up to 5 minutes)..."
wait_http "http://127.0.0.1:$INFER_PORT/models" 300 || fail "server.py did not start; see $INFER_DIR/server.log"

# ----- 4. Check -----
status
if [ -z "$MODE" ] && ! has_unit "$UNIT"; then
  warn "Running in the background with nohup: after a reboot run this script again, or use --service."
fi
