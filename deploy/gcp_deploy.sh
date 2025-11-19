#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   PROJECT_ID=my-gcp-project REGION=us-central1 SERVICE_NAME=zoomcamp-ml ./deploy/gcp_deploy.sh
#
# This script builds the project's Docker image with Cloud Build, pushes it to Container Registry
# and deploys the image to Cloud Run.

PROJECT_ID=${PROJECT_ID:-}
REGION=${REGION:-us-central1}
SERVICE_NAME=${SERVICE_NAME:-zoomcamp-ml}
IMAGE_NAME=${IMAGE_NAME:-${SERVICE_NAME}}

if [ -z "$PROJECT_ID" ]; then
  echo "ERROR: PROJECT_ID environment variable must be set."
  echo "Example: PROJECT_ID=my-project REGION=us-central1 SERVICE_NAME=zoomcamp-ml ./deploy/gcp_deploy.sh"
  exit 1
fi

FULL_IMAGE=gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest

echo "Building and pushing image to ${FULL_IMAGE} using Cloud Build..."
gcloud builds submit --project="$PROJECT_ID" --tag "$FULL_IMAGE" .

echo "Deploying to Cloud Run: service=${SERVICE_NAME}, region=${REGION}"
gcloud run deploy "$SERVICE_NAME" \
  --project="$PROJECT_ID" \
  --image="$FULL_IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --allow-unauthenticated \
  --port=9696

echo "Deployment finished. You can get the service URL with:"
echo "  gcloud run services describe ${SERVICE_NAME} --project=${PROJECT_ID} --region=${REGION} --format='value(status.url)'"
