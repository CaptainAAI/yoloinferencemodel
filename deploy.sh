#!/usr/bin/env bash
# deploy.sh - installs the YOLO inference server (server.py) on this Jetson as the
# systemd service "yolo-inference", and manages that service.
#
# Run it ON the Jetson, as your normal user (not with sudo), from a clone of this
# repository. It asks for your sudo password once, to install system packages and
# the service:
#
#   git clone https://github.com/CaptainAAI/yoloinferencemodel.git
#   cd yoloinferencemodel
#   bash deploy.sh
#
# To update later:  git pull && bash deploy.sh
#
# Commands:
#   bash deploy.sh [install]     install or update everything, then (re)start the service
#   bash deploy.sh check         only check the requirements; installs and changes nothing
#   bash deploy.sh status        service state, port, loaded models, recent log lines
#   bash deploy.sh logs [N|-f]   the last N log lines (default 50), or -f to follow the log
#   bash deploy.sh start | stop | restart
#   bash deploy.sh unit          print the systemd unit file this script installs
#   bash deploy.sh uninstall     stop and remove the service (files and models stay)
#   bash deploy.sh help          this text
#
# What "install" does, step by step:
#   1. Requirements: installs what is missing. python3, pip and curl with apt; the
#      packages in requirements.txt with pip (--user); ultralytics with pip. PyTorch
#      and TensorRT come with the Jetson setup (JetPack) and are only checked:
#      PyPI has no CUDA build of PyTorch for Jetson.
#   2. Files: copies server.py to INFER_DIR. A different previous version is kept as
#      server.py.bak-<date>. Lists the model files and models.json entries.
#   3. Service: writes /etc/systemd/system/yolo-inference.service (prints the diff
#      when it changes) and enables it, so it starts at every boot.
#   4. Start: stops a copy of server.py started by hand (nohup) on the same port,
#      then restarts the service.
#   5. Wait: until GET /models answers. Loading TensorRT engines takes up to minutes.
#   6. Report: the same output as "bash deploy.sh status".
#
# The service runs as your user, starts after the network (and after yolo-images
# when that is installed), restarts 5 s after a crash, and appends its output to
# INFER_DIR/server.log.
#
# Settings (environment variables), e.g.  INFER_PORT=6800 bash deploy.sh
#   INFER_DIR       folder server.py runs in          default ~/yoloinference
#   INFER_PORT      port                              default 6767
#   START_TIMEOUT   seconds to wait for the models    default 300
# Settings given once are remembered in ~/.config/yolo-deploy/yolo-inference.env and
# used by later runs.
#
# Never touched: model/, models.json, settings.json and the log.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=yolo-inference
UNIT_FILE="/etc/systemd/system/$UNIT.service"
CONF="$HOME/.config/yolo-deploy/$UNIT.env"
SETTINGS="INFER_DIR INFER_PORT START_TIMEOUT"

# Remembered settings fill in what the environment does not set.
if [ -f "$CONF" ]; then
  while IFS='=' read -r key value; do
    case " $SETTINGS " in *" $key "*) if [ -z "${!key:-}" ]; then printf -v "$key" '%s' "$value"; fi ;; esac
  done < "$CONF"
fi
INFER_DIR="${INFER_DIR:-$HOME/yoloinference}"
INFER_PORT="${INFER_PORT:-6767}"
START_TIMEOUT="${START_TIMEOUT:-300}"

PORT="$INFER_PORT"
LOG="$INFER_DIR/server.log"
HEALTH_URL="http://127.0.0.1:$PORT/models"
RUN_USER="$(id -un)"
RUN_GROUP="$(id -gn)"
PYTHON="$(command -v python3 || echo /usr/bin/python3)"

# ---------------------------------------------------------------- output helpers
if [ -t 1 ]; then G=$'\033[1;32m' Y=$'\033[1;33m' R=$'\033[1;31m' N=$'\033[0m'; else G='' Y='' R='' N=''; fi
STEP=0
STEPS=6
step() { STEP=$((STEP + 1)); printf '\n%s[%d/%d] %s%s\n' "$G" "$STEP" "$STEPS" "$*" "$N"; }
head2() { printf '\n%s==> %s%s\n' "$G" "$*" "$N"; }
info() { printf '    %s\n' "$*"; }
ok() { printf '    %sok%s    %s\n' "$G" "$N" "$*"; }
warn() { printf '    %swarn%s  %s\n' "$Y" "$N" "$*"; }
fail() { printf '\n%sERROR: %s%s\n' "$R" "$*" "$N" >&2; exit 1; }
kv() { printf '    %-11s %s\n' "$1" "$2"; }

usage() { awk 'NR > 1 { if (/^set -euo/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }

# ---------------------------------------------------------------- sudo and apt
SUDO=sudo
need_sudo() {  # $1 = what sudo is needed for
  if [ "$(id -u)" -eq 0 ]; then SUDO=''; return; fi
  command -v sudo >/dev/null || fail "sudo not found; it is needed to $1"
  if ! sudo -n true 2>/dev/null; then
    info "sudo is needed to $1; enter your password if asked"
    [ -t 0 ] || fail "sudo needs your password but there is no terminal: run this in a terminal on the Jetson (or over ssh -t)"
    sudo -v || fail "sudo failed"
  fi
}

not_root() {
  if [ "$(id -u)" -eq 0 ]; then
    fail "run this as your normal user, without sudo (the service runs as the user who installs it); it asks for sudo itself"
  fi
}

APT_UPDATED=''
apt_install() {  # packages...
  need_sudo "install $*"
  if [ -z "$APT_UPDATED" ]; then
    info "apt-get update"
    $SUDO apt-get update -qq
    APT_UPDATED=1
  fi
  info "apt-get install $*"
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@" >/dev/null
}

CHECK_ONLY=''
MISSING=0
missing() {  # $1 = what is missing, $2 = what install would do
  warn "$1 missing (install would: $2)"
  MISSING=1
}

ensure_cmd() {  # command apt-package label
  if command -v "$1" >/dev/null; then ok "$3: $(command -v "$1")"; return; fi
  if [ -n "$CHECK_ONLY" ]; then missing "$3" "apt-get install $2"; return; fi
  apt_install "$2"
  command -v "$1" >/dev/null || fail "$3 still missing after apt-get install $2"
  ok "$3 installed: $(command -v "$1")"
}

# ---------------------------------------------------------------- python helpers
py_has() { python3 -c "import $1" 2>/dev/null; }
py_ver() { python3 -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null || echo "?"; }

PIP=(python3 -m pip install --user --disable-pip-version-check --no-warn-script-location -q)
pip_install() {  # pip arguments...
  local flags=()
  if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    flags=(--break-system-packages)  # Ubuntu 24.04 marks its Python "externally managed"
  fi
  info "pip install $*"
  "${PIP[@]}" "${flags[@]}" "$@" || fail "pip install $* failed (is the Jetson online?)"
}

# Python packages of requirements.txt, as "pip-name:import-name".
PY_PACKAGES="fastapi:fastapi uvicorn:uvicorn python-multipart:multipart requests:requests pillow:PIL"

check_python_packages() {
  local pair name module absent=''
  for pair in $PY_PACKAGES; do
    module="${pair#*:}"
    if ! py_has "$module"; then absent="$absent ${pair%%:*}"; fi
  done
  if [ -n "$absent" ]; then
    if [ -n "$CHECK_ONLY" ]; then
      missing "python packages:$absent" "pip install -r requirements.txt"
      return
    fi
    pip_install -r "$SRC/requirements.txt"
  fi
  for pair in $PY_PACKAGES; do
    name="${pair%%:*}"
    module="${pair#*:}"
    if py_has "$module"; then ok "$name $(py_ver "$name")"; else fail "$name still missing after pip install"; fi
  done
}

# ---------------------------------------------------------------- systemd helpers
prop() { systemctl show -p "$1" --value "$UNIT" 2>/dev/null || true; }
in_service() { grep -q "/$UNIT.service" "/proc/$1/cgroup" 2>/dev/null; }
port_pids() { ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true; }
port_busy() { [ -n "$(ss -ltnH "sport = :$PORT" 2>/dev/null)" ]; }
log_lines() { if [ -f "$LOG" ]; then wc -l < "$LOG"; else echo 0; fi; }

stop_stray() {  # stops a copy started by hand (nohup) that holds the port, so the service can bind it
  local pid
  for pid in $(port_pids); do
    if in_service "$pid"; then continue; fi
    info "stopping pid $pid on port $PORT, started by hand: $(ps -o cmd= -p "$pid" || true)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "$pid" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      warn "pid $pid did not stop after 30 s, killing it"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  if port_busy && [ -z "$(port_pids)" ] && [ "$(prop ActiveState)" != active ]; then
    fail "port $PORT is used by a process of another user; see: sudo ss -ltnp 'sport = :$PORT'"
  fi
}

wait_up() {  # $1 = log line count before the start; waits until HEALTH_URL answers
  local from="$1" t0=$SECONDS last=$SECONDS restarts0
  restarts0="$(prop NRestarts)"
  info "waiting for $HEALTH_URL (up to $START_TIMEOUT s)"
  while ! curl -fs -m 2 -o /dev/null "$HEALTH_URL"; do
    if [ "$(prop ActiveState)" = failed ] || [ "$(prop NRestarts)" != "$restarts0" ]; then
      printf '\n'
      tail -n +"$((from + 1))" "$LOG" 2>/dev/null | tail -n 25 | sed 's/^/    | /'
      fail "the service stopped while starting (log above; more: bash deploy.sh logs)"
    fi
    if [ $((SECONDS - t0)) -ge "$START_TIMEOUT" ]; then
      fail "no answer from $HEALTH_URL after $START_TIMEOUT s; see: bash deploy.sh logs"
    fi
    if [ $((SECONDS - last)) -ge 15 ]; then
      info "still starting, $((SECONDS - t0)) s; log: $(tail -n 1 "$LOG" 2>/dev/null | cut -c1-100)"
      last=$SECONDS
    fi
    sleep 1
  done
  ok "answers after $((SECONDS - t0)) s"
}

src_version() {
  if git -C "$SRC" rev-parse HEAD >/dev/null 2>&1; then
    local dirty=''
    if [ -n "$(git -C "$SRC" status --porcelain 2>/dev/null)" ]; then dirty=' + local changes'; fi
    echo "commit $(git -C "$SRC" log -1 --format='%h (%cs) %s')$dirty"
  else
    echo "$SRC (not a git clone)"
  fi
}

save_settings() {
  local key
  mkdir -p "$(dirname "$CONF")"
  for key in $SETTINGS; do echo "$key=${!key}"; done > "$CONF"
}

render_unit() {
  cat <<UNIT
# Written by deploy.sh of https://github.com/CaptainAAI/yoloinferencemodel
# Changes made here are overwritten by the next deploy; change the settings instead.
[Unit]
Description=YOLO inference server (server.py, port $PORT)
Documentation=https://github.com/CaptainAAI/yoloinferencemodel
After=network-online.target yolo-images.service
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
Group=$RUN_GROUP
WorkingDirectory=$INFER_DIR
Environment=PORT=$PORT
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON $INFER_DIR/server.py
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
StandardOutput=append:$LOG
StandardError=append:$LOG

[Install]
WantedBy=multi-user.target
UNIT
}

install_unit() {
  local tmp
  tmp="$(mktemp)"
  render_unit > "$tmp"
  if [ -f "$UNIT_FILE" ] && cmp -s "$tmp" "$UNIT_FILE"; then
    ok "$UNIT_FILE unchanged"
  else
    if [ -f "$UNIT_FILE" ]; then
      info "changes to $UNIT_FILE:"
      diff -u "$UNIT_FILE" "$tmp" | tail -n +3 | sed 's/^/      /' || true
    fi
    $SUDO install -m 644 "$tmp" "$UNIT_FILE"
    $SUDO systemctl daemon-reload
    ok "wrote $UNIT_FILE"
  fi
  rm -f "$tmp"
  if [ "$(prop UnitFileState)" != enabled ]; then
    $SUDO systemctl enable "$UNIT" 2>&1 | sed 's/^/    /'
  fi
  ok "enabled: starts at every boot, restarts 5 s after a crash"
}

install_file() {  # source destination
  local name backup
  name="$(basename "$2")"
  if [ -f "$2" ] && cmp -s "$1" "$2"; then ok "$name unchanged"; return; fi
  if [ -f "$2" ]; then
    backup="$2.bak-$(date +%Y%m%d-%H%M%S)"
    cp -p "$2" "$backup"
    info "previous $name kept as $backup"
  fi
  cp -f "$1" "$2"
  ok "$name installed ($(wc -l < "$2") lines)"
}

# ---------------------------------------------------------------- this server's checks
check_requirements() {
  ensure_cmd curl curl "curl"
  ensure_cmd python3 python3 "python3"
  if ! command -v python3 >/dev/null; then return; fi
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || fail "Python 3.10 or newer is needed, found $(python3 -V 2>&1)"
  ok "python $(python3 -c 'import sys; print(sys.version.split()[0])')"
  if python3 -m pip --version >/dev/null 2>&1; then
    ok "pip $(python3 -m pip --version | awk '{print $2}')"
  elif [ -n "$CHECK_ONLY" ]; then
    missing "pip" "apt-get install python3-pip"
    return
  else
    apt_install python3-pip
    ok "pip installed"
  fi
  check_python_packages

  # PyTorch and TensorRT belong to the Jetson setup; installing them from PyPI would
  # replace the CUDA build with one that cannot use the GPU.
  if py_has torch; then
    ok "torch $(py_ver torch), CUDA $(python3 -c 'import torch; print("available" if torch.cuda.is_available() else "NOT available")' 2>/dev/null || echo '?')"
  else
    fail "PyTorch is missing. On a Jetson it comes from NVIDIA's Jetson wheels, not from PyPI:
  https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
Install it, then run this script again."
  fi
  if py_has tensorrt; then
    ok "tensorrt $(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null)"
  elif [ -n "$CHECK_ONLY" ]; then
    missing "TensorRT for Python (needed for .engine models)" "apt-get install python3-libnvinfer"
  elif apt-cache show python3-libnvinfer >/dev/null 2>&1; then
    apt_install python3-libnvinfer
    if py_has tensorrt; then ok "tensorrt installed"; else warn "tensorrt still not importable; .engine models will not load"; fi
  else
    warn "TensorRT for Python not found and not in apt (JetPack); .engine models will not load, .pt models will"
  fi
  if py_has ultralytics; then
    ok "ultralytics $(py_ver ultralytics)"
  elif [ -n "$CHECK_ONLY" ]; then
    missing "ultralytics" "pip install ultralytics"
  else
    pip_install ultralytics   # keeps the installed torch, which already satisfies it
    ok "ultralytics $(py_ver ultralytics) installed"
  fi
  local free_gb
  free_gb="$(df -BG --output=avail "$HOME" | tail -n 1 | tr -dc 0-9)"
  if [ "$free_gb" -lt 5 ]; then warn "only $free_gb GB free on the disk"; else ok "disk: $free_gb GB free"; fi
}

list_models() {
  local files
  files="$(find "$INFER_DIR/model" -maxdepth 1 -type f \( -name '*.engine' -o -name '*.pt' -o -name '*.onnx' \) 2>/dev/null | wc -l)"
  info "model files in $INFER_DIR/model: $files"
  if [ -f "$INFER_DIR/models.json" ]; then
    python3 - "$INFER_DIR" <<'PY'
import json, os, sys
base = sys.argv[1]
for name, entry in json.load(open(os.path.join(base, "models.json"))).items():
    found = "found" if os.path.isfile(os.path.join(base, "model", entry["file"])) else "MISSING in model/"
    print("      %-14s %-34s %s, threads %s, save images %s" % (
        name, entry["file"], found, entry.get("workers", 1), "on" if entry.get("save_images") else "off"))
PY
  else
    info "no models.json yet: server.py creates it, empty, on its first start; add models from the dashboard"
  fi
}

app_status() {
  head2 "server.py"
  local json
  if ! json="$(curl -fs -m 5 "http://127.0.0.1:$PORT/monitor")"; then
    warn "no answer on http://127.0.0.1:$PORT"
    return
  fi
  python3 -c '
import json, sys
d = json.loads(sys.argv[1])
s = d.get("server", {})
print("    versions    ultralytics %s, torch %s, python %s" % (s.get("ultralytics"), s.get("torch"), s.get("python")))
print("    gpu         %s" % s.get("gpu"))
print("    uptime      %d s" % s.get("uptime_s", 0))
print("    models      %d (name, status, file, threads, requests, failed, save images)" % len(d.get("models", {})))
for name, m in d.get("models", {}).items():
    st = m.get("stats") or {}
    print("      %-14s %-9s %-34s %-2s %7s %5s  %s" % (name, m.get("status"), m.get("file"), m.get("workers"),
          st.get("requests", "-"), st.get("failed", "-"), "on" if m.get("save_images") else "off"))
    if m.get("error"):
        print("        error: %s" % m["error"])
' "$json" || warn "could not read GET /monitor"
}

# ---------------------------------------------------------------- commands
do_status() {
  head2 "Service $UNIT"
  if [ -f "$UNIT_FILE" ]; then
    local since mem
    since="$(prop ActiveEnterTimestamp)"
    kv state "$(prop ActiveState) ($(prop SubState))${since:+, since $since}"
    kv "at boot" "$(prop UnitFileState)"
    kv "main pid" "$(prop MainPID)"
    mem="$(prop MemoryCurrent)"
    case "$mem" in '' | '[not set]' | 18446744073709551615) ;; *) kv memory "$(numfmt --to=iec --suffix=B "$mem")" ;; esac
    kv restarts "$(prop NRestarts) automatic restart(s) after a crash"
    kv "unit file" "$UNIT_FILE"
  else
    warn "not installed; install it with: bash deploy.sh"
  fi
  kv folder "$INFER_DIR"
  if [ -f "$LOG" ]; then kv log "$LOG ($(du -h "$LOG" | cut -f1))"; else kv log "$LOG (not created yet)"; fi
  if [ -f "$INFER_DIR/.deployed" ]; then kv deployed "$(cat "$INFER_DIR/.deployed")"; fi
  kv settings "INFER_DIR=$INFER_DIR INFER_PORT=$INFER_PORT START_TIMEOUT=$START_TIMEOUT"

  head2 "Port $PORT"
  local line pid
  line="$(ss -ltnpH "sport = :$PORT" 2>/dev/null | head -n 1)"
  if [ -z "$line" ]; then
    warn "nothing listens on port $PORT"
  else
    pid="$(grep -o 'pid=[0-9]*' <<< "$line" | head -n 1 | cut -d= -f2 || true)"
    if [ -z "$pid" ]; then
      info "$(awk '{print $4}' <<< "$line"), a process of another user (sudo ss -ltnp 'sport = :$PORT')"
    elif in_service "$pid"; then
      ok "$(awk '{print $4}' <<< "$line"), pid $pid (the service)"
    else
      warn "$(awk '{print $4}' <<< "$line"), pid $pid, started by hand, not by the service: $(ps -o cmd= -p "$pid" || true)"
      info "bash deploy.sh replaces it with the service"
    fi
  fi

  app_status

  head2 "Last 8 log lines"
  if [ -f "$LOG" ]; then tail -n 8 "$LOG" | sed 's/^/    | /'; else info "(no log yet)"; fi
}

do_check() {
  CHECK_ONLY=1
  STEPS=1
  step "Checking requirements (nothing is installed or changed)"
  check_requirements
  head2 "Files"
  info "source: $(src_version)"
  if [ -f "$INFER_DIR/server.py" ]; then
    if cmp -s "$SRC/server.py" "$INFER_DIR/server.py"; then ok "installed server.py is up to date"; else info "installed server.py differs: install would update it"; fi
  else
    info "server.py not installed yet in $INFER_DIR"
  fi
  if [ -d "$INFER_DIR/model" ]; then list_models; fi
  if [ -f "$UNIT_FILE" ]; then ok "service $UNIT installed ($(prop ActiveState))"; else info "service $UNIT not installed yet"; fi
  if [ "$MISSING" -ne 0 ]; then
    printf '\n'
    warn "something is missing; bash deploy.sh installs it"
    exit 1
  fi
  printf '\n'
  ok "all requirements are there"
}

do_install() {
  not_root
  [ -f "$SRC/server.py" ] || fail "$SRC/server.py not found; run deploy.sh from its repository"
  printf '%sDeploying the YOLO inference server%s from %s\n' "$G" "$N" "$(src_version)"
  info "to $INFER_DIR, port $PORT, service $UNIT, as user $RUN_USER"

  step "Requirements"
  need_sudo "install missing packages and the systemd service"
  check_requirements
  PYTHON="$(command -v python3)"

  step "Files"
  mkdir -p "$INFER_DIR/model"
  install_file "$SRC/server.py" "$INFER_DIR/server.py"
  echo "$(src_version), deployed $(date '+%Y-%m-%d %H:%M')" > "$INFER_DIR/.deployed"
  save_settings
  ok "settings remembered in $CONF"
  list_models

  step "systemd service $UNIT"
  install_unit

  step "Starting"
  stop_stray
  local from
  from="$(log_lines)"
  $SUDO systemctl restart "$UNIT"
  ok "systemctl restart $UNIT"

  step "Waiting for server.py to load the models"
  wait_up "$from"

  step "Report"
  do_status
  printf '\n%sDone.%s server.py runs as the service %s on port %s.\n' "$G" "$N" "$UNIT" "$PORT"
  info "logs: bash deploy.sh logs -f    status: bash deploy.sh status    update: git pull && bash deploy.sh"
}

do_ctl() {  # start | stop | restart
  not_root
  [ -f "$UNIT_FILE" ] || fail "the service is not installed; install it with: bash deploy.sh"
  need_sudo "$1 the service"
  local from
  from="$(log_lines)"
  if [ "$1" != stop ]; then stop_stray; fi
  $SUDO systemctl "$1" "$UNIT"
  ok "systemctl $1 $UNIT"
  if [ "$1" != stop ]; then wait_up "$from"; fi
  info "state: $(prop ActiveState) ($(prop SubState))"
}

do_uninstall() {
  not_root
  if [ ! -f "$UNIT_FILE" ]; then ok "the service is not installed"; return; fi
  need_sudo "remove the service"
  $SUDO systemctl disable --now "$UNIT" 2>&1 | sed 's/^/    /'
  $SUDO rm -f "$UNIT_FILE"
  $SUDO systemctl daemon-reload
  ok "service $UNIT stopped and removed"
  info "kept: $INFER_DIR (server.py, model/, models.json, settings.json, server.log)"
}

do_logs() {
  [ -f "$LOG" ] || fail "no log yet: $LOG"
  case "${1:-50}" in
    -f) tail -n 30 -f "$LOG" ;;
    *[!0-9]*) fail "usage: bash deploy.sh logs [N|-f]" ;;
    *) tail -n "${1:-50}" "$LOG" ;;
  esac
}

CMD="${1:-install}"
if [ $# -gt 0 ]; then shift; fi
case "$CMD" in
  install | --service | update) do_install ;;
  check) do_check ;;
  status | --status) do_status ;;
  logs) do_logs "$@" ;;
  start | stop | restart) do_ctl "$CMD" ;;
  unit) render_unit ;;
  uninstall) do_uninstall ;;
  help | -h | --help) usage ;;
  *) usage; fail "unknown command: $CMD" ;;
esac
