#!/usr/bin/env bash
# startup.sh — GCE startup script for the cortex runner VM.
#
# Runs on EVERY boot (including after a Spot preemption), so everything here is idempotent.
# It provisions the host and installs the schedule; it does not run the pipeline itself and
# does not shut the machine down — this VM stays up as a shared host for other cortex work.
#
# Delivered via instance metadata by ops/vm.sh, along with the compose file and daily-run.sh,
# so the repo stays the single source of truth and the VM needs no git clone or repo creds.

set -euo pipefail

# Mirror to the serial console so a failed boot is diagnosable with
# `gcloud compute instances get-serial-port-output` — no SSH required.
exec > >(tee -a /var/log/cortex-startup.log > /dev/console) 2>&1
echo "=== cortex startup $(date -Is) ==="

APP_DIR=/opt/wikipedia
REGION=us-east4
SECRET_NAME=anthropic-api-key

md() {  # read an instance metadata attribute
  curl -fsS -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

# ── Docker ───────────────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  echo "==> Installing Docker"
  curl -fsSL https://get.docker.com | sh
else
  echo "==> Docker already present"
fi
systemctl enable --now docker

# ── App directory + files from metadata ──────────────────────────────────────
echo "==> Writing $APP_DIR from instance metadata"
mkdir -p "$APP_DIR/data" "$APP_DIR/reports"
md docker-compose > "$APP_DIR/docker-compose.yml"
md daily-run      > "$APP_DIR/daily-run.sh"
chmod +x "$APP_DIR/daily-run.sh"

# ── Secret → .env ────────────────────────────────────────────────────────────
# Secret Manager rather than a file that only ever exists on this disk. Warp learned this the
# hard way: its gateway .env lives solely on the VM boot disk and nothing in the repo can
# reconstruct it. On a Spot VM that can be recreated, a disk-only secret is a trap.
echo "==> Fetching $SECRET_NAME from Secret Manager"
umask 077
if gcloud secrets versions access latest --secret="$SECRET_NAME" \
     > "$APP_DIR/.env.tmp" 2>/dev/null; then
  { printf 'ANTHROPIC_API_KEY='; cat "$APP_DIR/.env.tmp"; } > "$APP_DIR/.env"
  rm -f "$APP_DIR/.env.tmp"
  echo "    ok"
else
  rm -f "$APP_DIR/.env.tmp"
  # Not fatal: without a key, narratives degrade to placeholder text but the pipeline still runs.
  echo "    WARNING: secret unavailable — narratives will be unavailable this run"
  touch "$APP_DIR/.env"
fi
chmod 600 "$APP_DIR/.env"
umask 022

# ── Artifact Registry auth ───────────────────────────────────────────────────
echo "==> Configuring Docker for Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ── systemd unit + timer ─────────────────────────────────────────────────────
echo "==> Installing systemd units"
cat > /etc/systemd/system/wikipedia-pipeline.service <<'UNIT'
[Unit]
Description=Wikipedia daily pageviews pipeline
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
WorkingDirectory=/opt/wikipedia
ExecStartPre=/usr/bin/docker compose pull --quiet
ExecStart=/usr/bin/docker compose run --rm pipeline
TimeoutStartSec=3600
UNIT

cat > /etc/systemd/system/wikipedia-pipeline.timer <<'UNIT'
[Unit]
Description=Run the Wikipedia pipeline daily

[Timer]
OnCalendar=*-*-* 06:00:00 America/New_York
# If the VM was down at trigger time — a Spot preemption, say — run on the next boot instead
# of silently skipping the day.
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now wikipedia-pipeline.timer

echo "=== startup complete $(date -Is) ==="
systemctl list-timers wikipedia-pipeline.timer --no-pager || true
