PYTHON = python3
SOURCES = client/main.py
TESTS = test_subset_generator test_preprocessing test_interpretation
EXAMPLES =

# ----------------------------------------------------------------------------------

run:
	python3 -m client.main

test:
	$(foreach test,$(TESTS),python3 -m unittest tests.$(test);)

test_verbose:
	$(foreach test,$(TESTS),python3 -m unittest tests.$(test) -v;)

demo:
	$(foreach example,$(EXAMPLES),python3 -m examples.$(example).$(example);)

# ----------------------------------------------------------------------------------

conda_install:
	conda env create -f environment.yml

conda_remove:
	conda remove --name information-noise-reduction-for-investors --all

# ----------------------------------------------------------------------------------

docker_build:
	docker build -t information-noise-reduction .

docker_run:
	docker run -d -p 8888:8888 --name information-noise-reduction-container information-noise-reduction
	sleep 2
	xdg-open http://127.0.0.1:8888

# ----------------------------------------------------------------------------------

.PHONY: run