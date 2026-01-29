.PHONY: pipeline clean help

help:
	@echo "Available commands:"
	@echo "  make pipeline    - Run the newspaper parsing pipeline"
	@echo "  make clean       - Remove preprocessed files and output JSON"
	@echo "  make clean-all   - Remove all generated files including maps"

pipeline:
	python -m src.pipeline

clean:
	rm -rf data/preprocessed/*
	rm -rf data/out_json/*.json

clean-all: clean
	rm -rf data/maps/*
	rm -rf data/tiles/*
