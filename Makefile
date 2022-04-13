.PHONY: .init .conda .example start_jupyter clean
.SECONDARY:

CLEANUP := examples/*.sim.*

clean:
	rm -f ${CLEANUP}

.init:
	git config --local filter.dropoutput_ipynb.clean scripts/ipynb_output_filter.py
	git config --local filter.dropoutput_ipynb.smudge cat

.conda:
	conda env create -n sfacts-dev -f envs/sfacts-dev.yaml

.example: examples/example0.sim.mgen.tsv

start_jupyter:
	jupyter lab --port=8888 --notebook-dir examples

%.sim.world.nc: %.params
	sfacts simulate @$< \
	    --outpath $@

%.mgen.nc: %.world.nc
	sfacts dump $< --metagenotype $@

%.filt.mgen.nc: %.mgen.nc
	sfacts filter_mgen --min-minor-allele-freq 0.05 --min-horizontal-cvrg 0.1 --random-seed 0 $< $@

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

%.sim.filt.fit.refit.eval.tsv: %.sim.world.nc %.sim.filt.fit.refit.world.nc
	sfacts evaluate_fit $^ $@
