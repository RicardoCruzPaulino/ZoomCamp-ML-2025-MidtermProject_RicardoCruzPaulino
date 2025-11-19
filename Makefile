IMAGE = zoomcamp-ml:latest

.PHONY: build compose-up run train shell logs test clean

build:
	docker build -t $(IMAGE) .

compose-up:
	docker compose up --build

run:
	# Run container and mount local model.bin if present (avoids retraining)
	docker run --rm -p 9696:9696 -v $(PWD)/model.bin:/app/model.bin $(IMAGE)

train:
	# Train locally (useful to produce a real model.bin)
	python train.py

shell:
	# Start a shell inside the image with project mounted
	docker run --rm -it -v $(PWD):/app -w /app $(IMAGE) /bin/bash

logs:
	# Follow logs of the test container (if named zoomcamp_test)
	docker logs -f zoomcamp_test || true

test:
	# Quick HTTP check for the docs endpoint (returns HTTP status code)
	curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9696/docs

clean:
	# Remove test container and image (best-effort)
	docker rm -f zoomcamp_test || true
	docker rmi $(IMAGE) || true
