.PHONY: install setup figures figures-fast clean

install: setup

setup:
	python3 -m pip install -r requirements.txt

figures:
	python3 -m figures.make_figures
	python3 code/simulations/replicate_figures.py

figures-fast:
	python3 -m figures.make_figures --fast
	python3 code/simulations/replicate_figures.py --fast

clean:
	rm -f paper/figures/*.png paper/figures/build_metadata.txt
	rm -f code/simulations/figures/*.png code/simulations/figures/build_metadata.txt
