# ZoomCamp ML — Midterm Project

This repository contains a small machine learning project for predicting whether an individual's income is greater than 50K using the UCI Adult dataset. It includes training code, a saved model artifact, and a small FastAPI prediction endpoint.

**Contents**
- `train.py` — Training script: loads data, preprocesses, trains models and saves the best model as `model.bin`.
- `predict.py` — FastAPI app that loads `model.bin` and exposes a `/predict` endpoint.
- `adult.data.csv`, `adult.test.csv` — Dataset CSVs (original UCI dataset; consider excluding from Git).
- `model.bin` — Trained model artifact (binary). Consider excluding from Git and storing artifacts elsewhere.
- `results_models.csv` — Training results / metrics.

**Requirements**
- Python 3.8+
- See `requirements.txt` for pinned package versions.

Quick start
1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

How to train

```bash
python train.py
```

This will:
- Download and preprocess the UCI Adult dataset (if not using the local CSVs).
- Train candidate models defined in `train.py` (XGBoost is selected by default in the script).
- Save the best trained pipeline to `model.bin` and write `results_models.csv`.

Run the prediction API

```bash
python predict.py
```

The FastAPI app exposes `POST /predict` and runs by default on `http://0.0.0.0:9696`.
Visit `http://127.0.0.1:9696/docs` for interactive API docs (if FastAPI is installed).

Example request

```bash
curl -X POST "http://127.0.0.1:9696/predict" \
	-H "Content-Type: application/json" \
	-d '{
		"workclass": "Private",
		"education": "HS-grad",
		"marital_status": "Married-civ-spouse",
		"occupation": "Prof-specialty",
		"relationship": "Husband",
		"race": "White",
		"sex": "Male",
		"native_country": "United-States",
		"age": 35,
		"fnlwgt": 200000,
		"education_num": 10,
		"capital_gain": 0,
		"capital_loss": 0,
		"hours_per_week": 40
	}'
```

The response contains `income_probability` (float) and `income` (boolean for predicted class >50K).

Notes & recommendations
- Large files such as datasets and model binaries should typically not be checked into Git. Use `.gitignore` to exclude them and store artifacts in a dedicated storage (S3, model registry, etc.).
- I can add a `.gitignore` and remove tracked large files from the current commit if you want (this repo already included CSVs and `model.bin`).
- For reproducibility, pin exact package versions in `requirements.txt` (created in this repo).

Contributing
- Fork, create a branch, and open a pull request. Include tests for new functionality when possible.

Contact
- If you want, I can also:
	- create a `.gitignore` and remove large files from tracking (recommended),
	- produce a `requirements.txt` from the current environment (already generated), or
	- add a small `Makefile` / helper scripts for running the API and training.

**Makefile**

This project includes a `Makefile` with convenient targets for building and running the Docker image, training locally, and quick checks.

- `make build` — Build the Docker image (`zoomcamp-ml:latest`).
- `make compose-up` — Run `docker compose up --build`.
- `make run` — Run the Docker image and (optionally) mount your local `model.bin` to skip retraining.
- `make train` — Run `python train.py` locally to produce a real `model.bin`.
- `make shell` — Open a shell inside the image (project mounted).
- `make logs` — Follow logs of the `zoomcamp_test` container (if used).
- `make test` — Quick HTTP check for the `/docs` endpoint (returns HTTP status code).
- `make clean` — Remove test container and image.

Examples

Build and run with Docker Compose:

```bash
docker compose up --build
```

Build image locally:

```bash
make build
```

Run image and mount `model.bin` (skip training):

```bash
make run
```

Train locally to create a real `model.bin`:

```bash
make train
```

If you want, I can update this README further to include `make` examples that automatically mount volumes for development, or add CI steps to build and smoke-test the container.

**Cloud Deployment (GCP)**

Cloud deployment helpers and instructions are in `deploy/README_GCP.md`. That file explains how to use the included helper script `deploy/gcp_deploy.sh` and the `deploy/cloudbuild.yaml` configuration to build and deploy the application to Google Cloud Run (Cloud Build + Cloud Run). See that file for required IAM roles, APIs to enable, and example commands.

