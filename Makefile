CLEANUP :=
TEST_MODELS = model1 model2 model3 model4 model5
TEST_TARGETS = $(patsubst %,examples/sim.filt.ss-0.fit-%.world.nc,${TEST_MODELS})
EXAMPLES_IN_ORDER := examples/simulate_data examples/filter_data examples/fit_metagenotype examples/fit_metagenotype_advanced examples/evaluate_simulation_fit

include examples/Makefile

test: ${TEST_TARGETS}

clean:
	rm -f ${CLEANUP}

tags:
	ctags -R

.git_init:
	git config --local filter.dropoutput_ipynb.clean scripts/ipynb_output_filter.py
	git config --local filter.dropoutput_ipynb.smudge cat

.conda:
	conda env create -n sfacts-dev -f envs/sfacts-dev.yaml

.conda_test_clean:
	conda env remove -n sfacts-dev-test

.conda_test:
	conda env create -n sfacts-dev-test -f envs/sfacts-dev.yaml

start_jupyter:
	jupyter lab --port=8888 --notebook-dir examples

.PHONY: .git_init .conda test start_jupyter clean
.SECONDARY:

%.ipynb.html: %.ipynb
	jupyter nbconvert $< --execute --to=html --stdout > $@

%.html: %.md
	pandoc -o $@ $<
