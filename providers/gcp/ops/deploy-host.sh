#!/usr/bin/env bash
# deploy-host.sh — install the daily pipeline on an EXISTING host over SSH.
#
# Sibling to vm.sh, deliberately not a replacement for it. vm.sh owns cortex-runner: it
# provisions the VM and delivers the compose file + daily-run.sh through *instance metadata*.
# That is the right shape for a VM this repo owns end to end, and the wrong shape for a shared
# host somebody else provisioned — warp-host already carries a startup-script in metadata for
# Warp's own boot provisioning, and `vm.sh update` would overwrite it. So this script delivers
# the same two files over SSH and never touches metadata.
#
# Usage:
#   ./providers/gcp/ops/deploy-host.sh --instance warp-host \
#       --project terra-491618 --zone us-central1-a
#
# Idempotent: safe to re-run to push a changed daily-run.sh or compose file.
#
# Cross-project note. The host's ATTACHED service account must be able to reach the wikipedia
# project's resources, because the container authenticates via ADC off the metadata server and
# there is no key file anywhere. Grant these in wikipedia-cortex before the first run:
#   roles/storage.objectAdmin        on gs://wikipedia-cortex-data
#   roles/artifactregistry.reader    on the wikipedia AR repo (us-east4)
#   roles/secretmanager.secretAccessor on the anthropic-api-key secret
# `--check-iam` verifies all three from the host and exits.

set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────
INSTANCE="${WIKI_HOST_INSTANCE:-warp-host}"
PROJECT="${WIKI_HOST_PROJECT:-terra-491618}"
ZONE="${WIKI_HOST_ZONE:-us-central1-a}"
APP_DIR="${WIKI_HOST_APP_DIR:-/opt/wikipedia}"

# Where the pipeline's own resources live — these stay in wikipedia-cortex regardless of
# which project the host itself sits in.
WIKI_PROJECT="${WIKI_GCP_PROJECT:-wikipedia-cortex}"
AR_REGION="${WIKI_AR_REGION:-us-east4}"
SECRET_NAME="${WIKI_SECRET_NAME:-anthropic-api-key}"

CHECK_IAM_ONLY=0

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OPS_DIR="$REPO_ROOT/providers/gcp/ops"
COMPOSE_FILE="$REPO_ROOT/providers/gcp/compose/docker-compose.yml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance)   INSTANCE="$2";  shift 2 ;;
    --project)    PROJECT="$2";   shift 2 ;;
    --zone)       ZONE="$2";      shift 2 ;;
    --app-dir)    APP_DIR="$2";   shift 2 ;;
    --check-iam)  CHECK_IAM_ONLY=1; shift ;;
    -h|--help)    sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

log() { echo "▶ $*"; }
on_host() { gcloud compute ssh "$INSTANCE" --zone "$ZONE" --project "$PROJECT" --command "$1"; }

# ── IAM preflight ─────────────────────────────────────────────────────────────
# Run FROM the host so it exercises the attached service account's real credentials rather
# than the operator's, which are almost always more privileged and would pass regardless.
check_iam() {
  log "Checking the host's attached SA against wikipedia-cortex resources"
  on_host "
    set -u; rc=0
    sa=\$(curl -fsS -H 'Metadata-Flavor: Google' \
      http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email)
    echo \"    attached SA: \$sa\"
    if gcloud storage ls gs://wikipedia-cortex-data/ >/dev/null 2>&1; then
      echo '    ok   storage.objectAdmin  gs://wikipedia-cortex-data'
    else
      echo '    FAIL storage.objectAdmin  gs://wikipedia-cortex-data'; rc=1
    fi
    if gcloud artifacts docker images list \
         ${AR_REGION}-docker.pkg.dev/${WIKI_PROJECT}/wikipedia >/dev/null 2>&1; then
      echo '    ok   artifactregistry.reader'
    else
      echo '    FAIL artifactregistry.reader'; rc=1
    fi
    if gcloud secrets versions access latest --secret=${SECRET_NAME} \
         --project=${WIKI_PROJECT} >/dev/null 2>&1; then
      echo '    ok   secretmanager.secretAccessor'
    else
      echo '    FAIL secretmanager.secretAccessor (narratives degrade, pipeline still runs)'
    fi
    exit \$rc
  "
}

if [[ $CHECK_IAM_ONLY -eq 1 ]]; then
  check_iam
  exit $?
fi

# ── Stage the two delivered files ─────────────────────────────────────────────
log "Copying compose file and daily-run.sh to $INSTANCE"
gcloud compute scp "$COMPOSE_FILE" "$OPS_DIR/daily-run.sh" \
  "$INSTANCE:/tmp/" --zone "$ZONE" --project "$PROJECT"

# ── Provision ─────────────────────────────────────────────────────────────────
# Everything below is idempotent. Docker is assumed present on a shared host but installed if
# not; the systemd units are rewritten every run so an edit here always lands.
log "Provisioning $APP_DIR on $INSTANCE"
on_host "sudo env APP_DIR='$APP_DIR' WIKI_PROJECT='$WIKI_PROJECT' \
  AR_REGION='$AR_REGION' SECRET_NAME='$SECRET_NAME' bash -s" <<'REMOTE'
set -euo pipefail

command -v docker >/dev/null 2>&1 || { echo '==> Installing Docker'; curl -fsSL https://get.docker.com | sh; }
systemctl enable --now docker

mkdir -p "$APP_DIR/data" "$APP_DIR/reports"
install -m 644 /tmp/docker-compose.yml "$APP_DIR/docker-compose.yml"
install -m 755 /tmp/daily-run.sh       "$APP_DIR/daily-run.sh"
rm -f /tmp/docker-compose.yml /tmp/daily-run.sh

# Secret → .env. Not fatal if unavailable: narratives degrade to placeholder text but every
# other stage still runs, and a failed key should not cost a day of archive coverage.
echo "==> Fetching $SECRET_NAME from Secret Manager ($WIKI_PROJECT)"
umask 077
if gcloud secrets versions access latest --secret="$SECRET_NAME" \
     --project="$WIKI_PROJECT" > "$APP_DIR/.env.tmp" 2>/dev/null; then
  { printf 'ANTHROPIC_API_KEY='; cat "$APP_DIR/.env.tmp"; } > "$APP_DIR/.env"
  echo "    ok"
else
  echo "    WARNING: secret unavailable — narratives will be unavailable this run"
  : > "$APP_DIR/.env"
fi
rm -f "$APP_DIR/.env.tmp"
chmod 600 "$APP_DIR/.env"
umask 022

# Registry auth must land in ROOT's docker config — systemd runs the unit as root, not as the
# SSH user, so a `gcloud auth configure-docker` in the login shell would not be seen.
echo "==> Configuring Docker for Artifact Registry"
gcloud auth configure-docker "${AR_REGION}-docker.pkg.dev" --quiet

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
# Persistent so a host that was down at trigger time runs on next boot rather than silently
# skipping the day — the archive's value is that it has no holes.
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now wikipedia-pipeline.timer
echo '==> Installed'
systemctl list-timers wikipedia-pipeline.timer --no-pager || true
REMOTE

log "Done. Trigger a run with:"
echo "    gcloud compute ssh $INSTANCE --zone $ZONE --project $PROJECT \\"
echo "      --command 'sudo systemctl start wikipedia-pipeline.service'"
