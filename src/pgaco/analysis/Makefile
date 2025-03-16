# Define variables
PYTHON := python3

PLOT_RUN_SRC := plot_run.py        # Script for plot_run
TSP_SRC := tsp_plot.py             # Script for tsp

OUTPUT_DIR := ../results/pgtests               # Directory containing output images

PLOT_RUN_IMAGE := $(OUTPUT_DIR)/test1.png
TSP_IMAGE := $(OUTPUT_DIR)/tsp1.png
DEFAULT_IMAGE := $(OUTPUT_DIR)/result.png # Default image to open

# Phony targets
.PHONY: plot_run tsp run clean open open-all run-open

all: run-open

# Target to run plot_run.py
plot_run:
	@echo "Running $(PLOT_RUN_SRC)..."
	$(PYTHON) $(PLOT_RUN_SRC)
	@echo "plot_run completed."
	$(IMAGE_VIEWER) $(OUTPUT_DIR)/test1.png

# Target to run tsp_plot.py
tsp:
	@echo "Running $(TSP_SRC)..."
	$(PYTHON) $(TSP_SRC)
	@echo "tsp completed."
	$(IMAGE_VIEWER) $(OUTPUT_DIR)/tsp1.png

# Open the default image
open: $(DEFAULT_IMAGE)
	@echo "Opening $(DEFAULT_IMAGE) with feh..."
	feh $(DEFAULT_IMAGE)

# Open all images in the output directory
open-all:
	@echo "Opening all images in $(OUTPUT_DIR) with feh..."
	feh $(OUTPUT_DIR)/*.png

# Clean generated files
clean:
	@echo "Cleaning output directory..."
	rm -f $(OUTPUT_DIR)/*.png
	@echo "Clean completed."

# Combined target to run and open default image
run-open: plot_run open



