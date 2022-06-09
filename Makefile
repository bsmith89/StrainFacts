CLEANUP := examples/sim.* examples/*.html
EXAMPLES_IN_ORDER := examples/simulate_data examples/filter_data examples/fit_metagenotype examples/fit_metagenotype_advanced examples/evaluate_simulation_fit
TEST_MODELS = model1 model2 model3 model4 model5

test: examples/sim.eval_all_fits.tsv $(patsubst %,examples/sim.filt.ss-0.fit-%.world.nc,${TEST_MODELS})

clean:
	rm -f ${CLEANUP}

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


%.ipynb.html: %.ipynb
	jupyter nbconvert $< --execute --to=html --stdout > $@

%.html: %.md
	pandoc -o $@ $<

.compile_examples: ${addsuffix .ipynb,${EXAMPLES_IN_ORDER}}
	for example in ${EXAMPLES_IN_ORDER} ; \
	do \
	    echo $${example} ; \
	    ${MAKE} $${example}.ipynb.html ; \
	done


.PHONY: .git_init .conda test start_jupyter clean .compile_examples
.SECONDARY:

examples/sim.world.nc:
	sfacts simulate \
	    --num-strains=10 --num-samples=50 --num-positions=1000 \
	    --hyperparameters pi_hyper=0.1 mu_hyper_mean=10.0 epsilon_hyper_mode=0.01 \
	    --random-seed=0 \
	    $@

%.mgen.tsv: %.world.nc
	sfacts dump $< --metagenotype $@

%.mgen.nc: %.mgen.tsv
	sfacts load --metagenotype $< $@

%.filt.mgen.nc: %.mgen.nc
	sfacts filter_mgen \
	    --min-minor-allele-freq 0.05 \
	    --min-horizontal-cvrg 0.15 \
	    --random-seed 0 \
	    $< $@

%.ss-0.mgen.nc: %.mgen.nc
	sfacts sample_mgen --verbose \
	    --num-positions 500 \
	    --block-number 0 \
	    --random-seed 0 \
	    $< $@

%.ss-1.mgen.nc: %.mgen.nc
	sfacts sample_mgen --verbose \
	    --num-positions 500 \
	    --block-number 1 \
	    --random-seed 0 \
	    $< $@

%.approx-nmf.world.nc: %.mgen.nc
	sfacts nmf_init \
	    --verbose \
	    --num-strains 15 \
	    --random-seed 0 \
	    $< $@

%.fit.world.nc: %.mgen.nc
	sfacts fit \
	    --verbose \
	    --num-strains 15 \
	    --random-seed 0 \
	    $< $@

%.fit2.world.nc: %.mgen.nc %.approx-nmf.world.nc
	sfacts fit \
	    --verbose \
	    --init-from $*.approx-nmf.world.nc --init-vars genotype \
	    --num-strains 15 \
	    --random-seed 0 \
	    $*.mgen.nc $@

%.ss-0.fit3.geno.nc: %.ss-0.fit2.world.nc %.ss-0.mgen.nc
	sfacts fit_geno \
	    --verbose \
	    --hyperparameters gamma_hyper=1.01 \
	    --chunk-size=250 \
	    --random-seed=0 \
	    $*.ss-0.fit2.world.nc $*.ss-0.mgen.nc $@

%.ss-1.fit3.geno.nc: %.ss-0.fit2.world.nc %.ss-1.mgen.nc
	sfacts fit_geno \
	    --verbose \
	    --hyperparameters gamma_hyper=1.01 \
	    --chunk-size=250 \
	    --random-seed=0 \
	    $*.ss-0.fit2.world.nc $*.ss-1.mgen.nc $@

%.fit3.world.nc: %.mgen.nc %.ss-0.fit2.world.nc %.ss-0.fit3.geno.nc %.ss-1.fit3.geno.nc
	sfacts concat_geno \
	    --metagenotype $*.mgen.nc \
	    --community $*.ss-0.fit2.world.nc \
	    --outpath $@ \
	    $*.ss-0.fit3.geno.nc $*.ss-1.fit3.geno.nc

examples/sim.filt.ss-0.fit-%.world.nc: examples/sim.filt.ss-0.mgen.nc
	sfacts fit \
	    --verbose \
	    --model-structure $* \
	    --num-strains 15 \
	    --random-seed 0 \
	    $< $@

%.eval_all_fits.tsv: %.world.nc %.world.nc %.filt.ss-0.fit.world.nc %.filt.ss-0.fit2.world.nc %.filt.fit3.world.nc
	sfacts evaluate_fit --outpath $@ $*.world.nc $*.world.nc $*.filt.ss-0.fit.world.nc $*.filt.ss-0.fit2.world.nc $*.filt.fit3.world.nc | tee $@ | column -t

examples/sim.filt.fit-%.eval.tsv: examples/sim.world.nc examples/sim.filt.fit-%.world.nc
	sfacts evaluate_fit examples/sim.world.nc examples/sim.filt.fit-$*.world.nc | tee $@ | column -t

%.geno.tsv: %.world.nc
	sfacts dump --genotype $@ $<

%.comm.tsv: %.world.nc
	sfacts dump --community $@ $<
