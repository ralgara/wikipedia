#!/usr/bin/env bash
# vm.sh — provision and manage the cortex runner VM that hosts the daily Wikipedia pipeline.
#
# Modelled on projects/warp/gcp/warp-gcp.sh in the cortex workspace — same action structure,
# same gc() helper, same flag-or-env config block. Reusing that shape rather than inventing a
# second convention for the same job.
#
# Usage:
#   ./providers/gcp/ops/vm.sh <action> [flags]
#
# Actions:
#   create    Provision the Spot VM (idempotent — skips if it already exists)
#   update    Push current repo copies of startup/compose/daily-run to metadata, re-run startup
#   start     Start a stopped/preempted instance
#   stop      Stop the instance (disk and archive persist)
#   run-now   Trigger the pipeline immediately instead of waiting for the timer
#   logs      Tail the pipeline's most recent run
#   ssh       Open an SSH session
#   status    Show instance status, external IP, and next scheduled run
#   destroy   Delete the VM and its boot disk
#
# The VM is Spot for cost, but stays up 24/7 as a shared host for other cortex processes.
# --instance-termination-action=STOP means a preemption stops it and preserves the disk (and
# the ~925MB archive) rather than deleting it; `start` brings it back.

set -euo pipefail

# ── CONFIG — override via env vars or flags ───────────────────────────────────
PROJECT="${WIKI_GCP_PROJECT:-wikipedia-cortex}"
ZONE="${WIKI_GCP_ZONE:-us-east4-c}"          # same region as gs://wikipedia-cortex-data
MACHINE_TYPE="${WIKI_GCP_MACHINE:-e2-medium}" # 4GB — a full all-time load peaks at ~2.05GB RSS,
                                              # so e2-micro (1GB) and e2-small (2GB) both OOM
INSTANCE_NAME="${WIKI_GCP_INSTANCE:-cortex-runner}"
DISK_SIZE="${WIKI_GCP_DISK:-30}"
DISK_TYPE="${WIKI_GCP_DISK_TYPE:-pd-standard}"
OS_IMAGE_FAMILY="${WIKI_GCP_IMAGE_FAMILY:-debian-12}"
OS_IMAGE_PROJECT="${WIKI_GCP_IMAGE_PROJECT:-debian-cloud}"
SA="${WIKI_GCP_SA:-wikipedia-pipeline@wikipedia-cortex.iam.gserviceaccount.com}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OPS_DIR="$REPO_ROOT/providers/gcp/ops"
COMPOSE_FILE="$REPO_ROOT/providers/gcp/compose/docker-compose.yml"

# ── Argument parsing ──────────────────────────────────────────────────────────
ACTION="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)  PROJECT="$2";       shift 2 ;;
    --zone)     ZONE="$2";          shift 2 ;;
    --machine)  MACHINE_TYPE="$2";  shift 2 ;;
    --name)     INSTANCE_NAME="$2"; shift 2 ;;
    --disk)     DISK_SIZE="$2";     shift 2 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "▶ $*"; }
die() { echo "✗ $*" >&2; exit 1; }

gc() { gcloud compute "$@" --project="$PROJECT"; }

metadata_args() {
  echo "--metadata-from-file=startup-script=${OPS_DIR}/startup.sh,docker-compose=${COMPOSE_FILE},daily-run=${OPS_DIR}/daily-run.sh"
}

exists() { gc instances describe "$INSTANCE_NAME" --zone="$ZONE" &>/dev/null; }

remote() { gc ssh "$INSTANCE_NAME" --zone="$ZONE" --command "$1"; }

# ── Actions ───────────────────────────────────────────────────────────────────
case "$ACTION" in

  create)
    if exists; then
      log "$INSTANCE_NAME already exists — use 'update' to refresh its config"
      exit 0
    fi
    log "Creating Spot $MACHINE_TYPE '$INSTANCE_NAME' in $PROJECT/$ZONE"
    # No key file anywhere: attaching the service account makes Application Default
    # Credentials work inside the container via the metadata server.
    gc instances create "$INSTANCE_NAME" \
      --zone="$ZONE" \
      --machine-type="$MACHINE_TYPE" \
      --provisioning-model=SPOT \
      --instance-termination-action=STOP \
      --service-account="$SA" \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --image-family="$OS_IMAGE_FAMILY" \
      --image-project="$OS_IMAGE_PROJECT" \
      --boot-disk-size="${DISK_SIZE}GB" \
      --boot-disk-type="$DISK_TYPE" \
      --labels=project=wikipedia,managed-by=cortex \
      $(metadata_args)
    log "Created. First boot installs Docker and the timer; follow along with:"
    echo "    $0 logs --boot"
    ;;

  update)
    exists || die "$INSTANCE_NAME does not exist — run 'create' first"
    log "Pushing repo copies of startup-script, compose file, and daily-run to metadata"
    gc instances add-metadata "$INSTANCE_NAME" --zone="$ZONE" $(metadata_args)
    log "Re-running startup script on the instance"
    remote "sudo google_metadata_script_runner startup"
    ;;

  start)
    log "Starting $INSTANCE_NAME"
    gc instances start "$INSTANCE_NAME" --zone="$ZONE"
    ;;

  stop)
    log "Stopping $INSTANCE_NAME (boot disk and archive persist)"
    gc instances stop "$INSTANCE_NAME" --zone="$ZONE"
    ;;

  run-now)
    log "Triggering the pipeline immediately"
    remote "sudo systemctl start wikipedia-pipeline.service"
    log "Started. Follow with: $0 logs"
    ;;

  logs)
    if [[ "${1:-}" == "--boot" ]]; then
      gc instances get-serial-port-output "$INSTANCE_NAME" --zone="$ZONE" | tail -60
    else
      remote "sudo journalctl -u wikipedia-pipeline.service -n 100 --no-pager"
    fi
    ;;

  ssh)
    exec gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT"
    ;;

  status)
    log "Instance: $INSTANCE_NAME ($PROJECT / $ZONE)"
    gc instances describe "$INSTANCE_NAME" --zone="$ZONE" \
      --format="table(status, scheduling.provisioningModel:label=MODEL, networkInterfaces[0].accessConfigs[0].natIP:label=EXTERNAL_IP, machineType.basename():label=MACHINE)"
    echo
    remote "systemctl list-timers wikipedia-pipeline.timer --no-pager" 2>/dev/null \
      || echo "  (instance not reachable — is it stopped?)"
    ;;

  destroy)
    log "Deleting $INSTANCE_NAME and its boot disk"
    gc instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
    ;;

  *)
    sed -n '2,30p' "$0"
    exit 1
    ;;
esac
