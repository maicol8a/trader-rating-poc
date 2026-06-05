.PHONY: all clean figures paper

DATA_FILE := data/traders_data.csv

all: preprocess models fusion validation baseline sensitivity figures
	@echo "Pipeline complete. Results in results/ and figures/"

preprocess: $(DATA_FILE)
	python src/01_preprocessing.py

models: preprocess
	python src/02_models.py

fusion: models
	python src/03_fusion_score.py

validation: fusion
	python src/04_validation.py

baseline: fusion
	python src/05_baseline.py

sensitivity: fusion
	python src/06_sensitivity.py

figures: fusion validation baseline sensitivity
	python src/07_visualization.py

paper:
	cd paper && pdflatex working_paper_v6.tex && pdflatex working_paper_v6.tex

clean:
	rm -f results/traders_rated.csv figures/*.png

$(DATA_FILE):
	@echo "ERROR: traders_data.csv not found in data/"
	@echo "See data/README_data.md for collection instructions."
	@exit 1
