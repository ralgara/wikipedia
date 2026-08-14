#!/usr/bin/env bash
# grant-builder-access.sh — one-time IAM setup so images can be pushed from the pipeline
# host without giving the pipeline itself the ability to publish them.
#
# Run this from a machine authenticated as a principal with IAM admin on wikipedia-cortex
# (i.e. your own account on the Mac) — NOT from warp-host, whose attached SA deliberately
# cannot edit IAM.
#
#   ./providers/gcp/ops/grant-builder-access.sh
#
# Idempotent: re-running is safe. Bindings are additive and the SA create is guarded.
#
# WHY A SEPARATE SA. The obvious move is to grant roles/artifactregistry.writer directly to
# warp-host's attached SA, which is what `docker push` runs as there. Don't. The systemd unit
# runs the pulled image as ROOT on a host that also carries Warp's Postgres, gateway .env and
# Tailscale cert. Writer on the runtime identity closes a loop: anything executing inside the
# daily pipeline container could overwrite the image root executes on the next tick. The
# pipeline's job is to download pageviews and render HTML; publishing container images is not
# in that job description.
#
# The split below keeps push rights on a builder identity that nothing runs as. The host SA is
# only allowed to MINT A TOKEN for it, which means an operator at a shell can push (by asking
# for that token explicitly) while the unattended container cannot (it never asks).
#
# No key files are created. Impersonation is the whole point — a JSON key here would reintroduce
# exactly the credential-on-disk problem the metadata-server design removed.

set -euo pipefail

WIKI_PROJECT="${WIKI_GCP_PROJECT:-wikipedia-cortex}"
AR_REGION="${WIKI_AR_REGION:-us-east4}"
AR_REPO="${WIKI_AR_REPO:-wikipedia}"

BUILDER_NAME="${WIKI_BUILDER_NAME:-wikipedia-builder}"
BUILDER_SA="${BUILDER_NAME}@${WIKI_PROJECT}.iam.gserviceaccount.com"

# warp-host's attached service account — the identity a shell on that host runs as.
HOST_SA="${WIKI_HOST_SA:-956908985786-compute@developer.gserviceaccount.com}"

log() { echo "▶ $*"; }

log "Acting as: $(gcloud config get-value account 2>/dev/null)"
log "Target project: ${WIKI_PROJECT}"
echo

# ── 1. The builder identity ───────────────────────────────────────────────────
if gcloud iam service-accounts describe "$BUILDER_SA" --project="$WIKI_PROJECT" >/dev/null 2>&1; then
  log "Builder SA already exists: ${BUILDER_SA}"
else
  log "Creating builder SA: ${BUILDER_SA}"
  gcloud iam service-accounts create "$BUILDER_NAME" \
    --project="$WIKI_PROJECT" \
    --display-name="Wikipedia pipeline image builder" \
    --description="Pushes providers/gcp images to Artifact Registry. Nothing runs as this SA; it is assumed by operators via impersonation."
fi

# ── 2. Push rights, scoped to the ONE repository ──────────────────────────────
# Bound on the repo rather than the project: this identity has no business writing to any
# other Artifact Registry repo that may exist later.
log "Granting artifactregistry.writer on ${AR_REPO} (${AR_REGION}) to the builder SA"
gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" \
  --location="$AR_REGION" \
  --project="$WIKI_PROJECT" \
  --member="serviceAccount:${BUILDER_SA}" \
  --role="roles/artifactregistry.writer" \
  --quiet >/dev/null

# ── 3. Let the host SA assume the builder, and nothing more ───────────────────
# Bound on the builder SA itself, not project-wide, so this grants exactly one capability:
# "mint tokens for wikipedia-builder". It confers no rights over any other service account.
log "Granting serviceAccountTokenCreator on the builder SA to ${HOST_SA}"
gcloud iam service-accounts add-iam-policy-binding "$BUILDER_SA" \
  --project="$WIKI_PROJECT" \
  --member="serviceAccount:${HOST_SA}" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --quiet >/dev/null

echo
log "Done. Push from warp-host with:"
echo "    ./providers/gcp/iac/deploy-vm.sh --impersonate"
echo
log "Verify the bindings:"
echo "    gcloud artifacts repositories get-iam-policy ${AR_REPO} \\"
echo "      --location=${AR_REGION} --project=${WIKI_PROJECT}"
echo "    gcloud iam service-accounts get-iam-policy ${BUILDER_SA} \\"
echo "      --project=${WIKI_PROJECT}"
echo
log "IAM enforcement lags policy reads by a few minutes — if the first push still"
log "returns PERMISSION_DENIED, wait and retry rather than re-granting."
