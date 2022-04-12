.PHONY: initialize_git start_jupyter

initialize_git:
	git config --local filter.dropoutput_ipynb.clean scripts/ipynb_output_filter.py
	git config --local filter.dropoutput_ipynb.smudge cat

initialize_dev_env:
	conda env create -n sfacts-dev -f envs/sfacts-dev.yaml

start_jupyter:
	jupyter lab --port=8888 --notebook-dir examples

build_example_data: examples/example.mgen.tsv

examples/example.sim.nc:
	sfacts simulate \
	    --model-structure simplest_simulation \
	    --num-strains 10 --num-samples 50 --num-positions 500 \
	    --hyperparameters pi_hyper=0.4 mu_hyper_mean=10.0 epsilon_hyper_mode=0.01 \
	    --random-seed 0 \
	    --outpath $@

examples/example.mgen.tsv: examples/example.sim.nc
	sfacts dump $< --tsv --metagenotype $@
