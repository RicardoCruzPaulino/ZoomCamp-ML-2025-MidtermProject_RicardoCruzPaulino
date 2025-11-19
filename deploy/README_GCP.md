# Deploy to Google Cloud Run

This directory contains helper files for deploying the project to Google Cloud Run.

Requirements
- `gcloud` CLI installed and authenticated
- Billing enabled on the GCP project
- APIs enabled: Cloud Run, Cloud Build, Container Registry (or Artifact Registry)

Quick deploy (local) — using `gcloud builds submit` and `gcloud run deploy`

1. Set environment variables and run the script (example):

```bash
export PROJECT_ID="my-gcp-project"
export REGION="us-central1"
export SERVICE_NAME="zoomcamp-ml"
export IMAGE_NAME="zoomcamp-ml"
PROJECT_ID=$PROJECT_ID REGION=$REGION SERVICE_NAME=$SERVICE_NAME IMAGE_NAME=$IMAGE_NAME ./deploy/gcp_deploy.sh
```

The script does two things:
- submits a Cloud Build to build the Docker image and push it to Container Registry
- deploys the pushed image to Cloud Run (managed)

Cloud Build automated deploy

You can also use Cloud Build directly with the included `cloudbuild.yaml`. Use substitutions when invoking Cloud Build:

```bash
gcloud builds submit --config=deploy/cloudbuild.yaml \
  --substitutions=_IMAGE_NAME=zoomcamp-ml,_SERVICE_NAME=zoomcamp-ml,REGION=us-central1
```

Notes & permissions
- Ensure the Cloud Build service account and the user running the commands have the following roles:
  - Cloud Run Admin
  - Storage Admin (or roles needed to push to Container Registry/Artifact Registry)
  - Cloud Build Editor/Worker

- If you want to use Artifact Registry instead of Container Registry, change the image destination and Cloud Build steps accordingly.

Security
- The `gcloud run deploy` call above uses `--allow-unauthenticated`. Remove that flag to require authentication, or configure IAM to allow specific principals.

If you want, I can:
- Add a Terraform module that provisions Cloud Run + Cloud Build triggers + Artifact Registry, or
- Add a GitHub Actions workflow that calls `gcloud` to run the `gcloud builds submit` step on push to `main`.
