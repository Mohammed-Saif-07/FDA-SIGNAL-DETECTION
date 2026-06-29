# =====================================================================
# FDA Drug Safety Signal Detection — convenience targets
# =====================================================================

.PHONY: help up down logs ps hive-init docker-smoke smoke-docker smoke-local research-eval report download parse hdfs hive signals features train predict backtest dashboard api clean local-quarter local-backtest-2020 local-backtest-2020-wide local-finish-2020

CONDA_RUN ?= conda run -n fda
SPARK_HOME ?= /opt/anaconda3/envs/fda/lib/python3.11/site-packages/pyspark
SPARK_OPTS ?= --driver-memory 4g --conf spark.driver.maxResultSize=1g
SPARK_SUBMIT ?= $(CONDA_RUN) env SPARK_HOME=$(SPARK_HOME) spark-submit $(SPARK_OPTS)
QUARTER ?= 2020q3
QUARTERS ?= 2020q1 2020q2 2020q3 2020q4
LIMIT ?= 20
BACKTEST_CUTOFF ?= 2020-12-31

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	    awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'

# ---------------------- Docker lifecycle ----------------------
up:             ## Bring everything up
	docker compose up -d

down:           ## Stop everything (preserve volumes)
	docker compose down

logs:           ## Tail Airflow + Spark logs
	docker compose logs -f airflow-scheduler spark-master hive-server

ps:             ## Show running services
	docker compose ps

hive-init:      ## Initialize Hive metastore schema after a fresh Docker volume
	docker compose up -d hive-metastore-db namenode datanode
	sleep 8
	docker compose run --rm --no-deps hive-metastore /opt/hive/bin/schematool -dbType postgres -initSchema
	docker compose up -d hive-metastore hive-server

docker-smoke:   ## Smoke-test Docker API, HDFS, Hive schema, and Hive PRR/ROR HQL
	./scripts/smoke_docker.sh

smoke-docker: docker-smoke ## Alias for Docker smoke test

smoke-local:   ## Smoke-test local conda dependencies and backtest
	./scripts/smoke_local.sh

research-eval: ## Run paper-style baseline, case-study, and false-positive evaluation
	./scripts/research_eval.sh

report: research-eval ## Regenerate research evaluation artifacts used by docs/research_report.md

# ---------------------- Smoke-test pipeline -------------------
download:       ## Download a single quarter (smoke test)
	python ingestion/download_faers.py --quarter 2024q1

parse:          ## Parse downloaded JSON -> Parquet
	spark-submit ingestion/parse_faers.py

hdfs:           ## Push Parquet to HDFS
	python ingestion/load_to_hdfs.py

hive:           ## Create Hive schema
	beeline -u jdbc:hive2://localhost:10000 -f hive/create_tables.hql

signals:        ## Run PRR + ROR signal detection
	beeline -u jdbc:hive2://localhost:10000 -f hive/signal_detection.hql
	beeline -u jdbc:hive2://localhost:10000 -f hive/signal_trends.hql

features:       ## Build ML feature matrix
	spark-submit spark/feature_engineering.py

train:          ## Train the XGBoost model
	python ml/train_model.py --train-cutoff 2020-12-31

predict:        ## Score the latest features with the model
	python ml/predictor.py

backtest:       ## Run the backtest and print the headline
	python ml/evaluate.py

dashboard:      ## Start Streamlit locally
	streamlit run dashboard/app.py

api:            ## Start FastAPI locally
	uvicorn api.main:app --reload --port 8000

clean:          ## Remove processed parquet + models (keep raw)
	rm -rf data/processed/* ml/models/*

local-quarter:  ## Local no-Docker run for one FAERS quarter; deletes raw only after Spark succeeds
	rm -rf data/raw/* data/processed/* ml/models/*
	$(CONDA_RUN) python ingestion/download_faers.py --quarter $(QUARTER) --limit $(LIMIT)
	$(SPARK_SUBMIT) ingestion/parse_faers.py --quarter $(QUARTER)
	$(SPARK_SUBMIT) spark/data_cleaning.py
	rm -rf data/raw/*
	$(SPARK_SUBMIT) spark/feature_engineering.py --train-cutoff $(BACKTEST_CUTOFF)
	$(CONDA_RUN) python ml/train_model.py --train-cutoff $(BACKTEST_CUTOFF)
	$(CONDA_RUN) python ml/predictor.py --no-write-pg
	$(CONDA_RUN) python ml/evaluate.py --cutoff $(BACKTEST_CUTOFF)

local-backtest-2020: ## Local 2020-cutoff backtest sample; override QUARTER/LIMIT as needed
	$(MAKE) local-quarter QUARTER=$(QUARTER) LIMIT=$(LIMIT) BACKTEST_CUTOFF=2020-12-31

local-backtest-2020-wide: ## Local 2020-cutoff backtest across several quarters, parsing/deleting raw one quarter at a time
	rm -rf data/raw/* data/processed/* ml/models/*
	for q in $(QUARTERS); do \
		$(CONDA_RUN) python ingestion/download_faers.py --quarter $$q --limit $(LIMIT); \
		$(SPARK_SUBMIT) ingestion/parse_faers.py --quarter $$q; \
		rm -rf data/raw/*; \
	done
	$(SPARK_SUBMIT) spark/data_cleaning.py
	$(SPARK_SUBMIT) spark/feature_engineering.py --train-cutoff 2020-12-31
	$(CONDA_RUN) python ml/train_model.py --train-cutoff 2020-12-31
	$(CONDA_RUN) python ml/predictor.py --no-write-pg
	$(CONDA_RUN) python ml/evaluate.py --cutoff 2020-12-31

local-finish-2020: ## Resume from clean Parquet: features, train, predict, evaluate
	$(SPARK_SUBMIT) spark/feature_engineering.py --train-cutoff 2020-12-31
	$(CONDA_RUN) python ml/train_model.py --train-cutoff 2020-12-31
	$(CONDA_RUN) python ml/predictor.py --no-write-pg
	$(CONDA_RUN) python ml/evaluate.py --cutoff 2020-12-31
