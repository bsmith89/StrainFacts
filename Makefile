CLEANUP := examples/sim.*
EXAMPLES_IN_ORDER := examples/simulate_data examples/fit_metagenotype examples/evaluate_simulation_fit

test: examples/sim.filt.fit.refit.eval.tsv

clean:
	rm -f ${CLEANUP}

.git_init:
	git config --local filter.dropoutput_ipynb.clean scripts/ipynb_output_filter.py
	git config --local filter.dropoutput_ipynb.smudge cat

.conda:
	conda env create -n sfacts-dev -f envs/sfacts-dev.yaml

start_jupyter:
	jupyter lab --port=8888 --notebook-dir examples


%.html: %.ipynb
	jupyter nbconvert $< --execute --to=html --stdout > $@

.compile_examples: ${addsuffix .ipynb,${EXAMPLES_IN_ORDER}}
	for example in ${EXAMPLES_IN_ORDER} ; \
	do \
	    echo $${example} ; \
	    ${MAKE} $${example}.html ; \
	done


.PHONY: .git_init .conda test start_jupyter clean .compile_examples
.SECONDARY:

examples/sim.world.nc:
	sfacts simulate \
	    --model-structure=ssdd3_with_error \
	    --num-strains=10 --num-samples=50 --num-positions=5000 \
	    --hyperparameters
	        gamma_hyper=1e-5 rho_hyper=10.0 pi_hyper=0.1 \
	        mu_hyper_mean=10.0 mu_hyper_scale=3.0 m_hyper_concentration=3.0 \
	        epsilon_hyper_mode=0.01 epsilon_hyper_spread=1.5 \
	        alpha_hyper_mean=100 alpha_hyper_scale=0.5 \
	    --random-seed=0 \
	    --outpath $@

%.mgen.tsv: %.world.nc
	sfacts dump --tsv $< --metagenotype $@

%.mgen.nc: %.mgen.tsv
	sfacts load_mgen $< $@

%.filt.mgen.nc: %.mgen.nc
	sfacts filter_mgen \
	    --min-minor-allele-freq 0.05 \
	    --min-horizontal-cvrg 0.1 \
	    --random-seed 0 $< $@

%.mgen.tsv: %.mgen.nc
	sfacts dump_mgen $< $@

%.fit.world.nc: %.mgen.nc
	sfacts fit \
	    --verbose \
	    --model-structure ssdd3_with_error \
	    --num-strains 15 --num-positions 500 \
	    --random-seed 0 \
	    $< $@

%.fit.refit.geno.nc: %.fit.world.nc %.mgen.nc
	sfacts fit_genotype \
	    --verbose \
	    --model-structure ssdd3_with_error \
	    --num-positionsB 1000 \
	    --hyperparameters gamma_hyper=1.0 \
	    --block-number 0 \
	    --random-seed=0 \
	    $^ $@

%.fit.refit.world.nc: %.fit.world.nc %.mgen.nc %.fit.refit.geno.nc
	sfacts concatenate_genotype_chunks \
	    --metagenotype $*.mgen.nc \
	    --community $*.fit.world.nc \
	    --outpath $@ \
	    $*.fit.refit.geno.nc

%.filt.fit.refit.eval.tsv: %.world.nc %.filt.fit.refit.world.nc
	sfacts evaluate_fit $^ $@
