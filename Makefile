PYTHON = python3
SOURCES = client/main.py
TESTS = test_x
EXAMPLES =

run:
	python3 -m client.main

test:
	$(foreach test,$(TESTS),python3 -m unittest tests.$(test);)

test_verbose:
	$(foreach test,$(TESTS),python3 -m unittest tests.$(test) -v;)

demo:
	$(foreach example,$(EXAMPLES),python3 -m examples.$(example).$(example);)

conda_install:
	conda env create -f environment.yml

conda_remove:
	conda remove --name information-noise-reduction-for-investors --all

.PHONY: run