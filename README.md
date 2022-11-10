# NLR

Code for the EMNLP 2022 paper "Can Transformers Reason in Fragments of Natural Language?".

## Setup
Setup the environment correspondingly, e.g.

```bash
conda create -n nlr python=3.8 anaconda
pip install -f requirements.txt
```

Experiments referred to in the paper are in the folder `./experiments`.

To replicate our experiments, simply run

```bash
PYTHONPATH='.' python scripts/classify.py experiments/$EXPERIMENT_TO_REPLICATE
```

Note that the evaluation experiments (e.g. in folder `./experiments/eval-across`) rely on the existence of trained models. In this case, the corresponding models need to be trained first (e.g. by replicating `./experiments/final`).

## Generation script for random problems
The script to run is `scripts/verify_vampire.py` -- it generates a problem set then classifies it with Vampire
before turning it to a `JSON` file. Also generates some statistics. NOTE: requires a [Vampire](https://github.com/vprover/vampire) executable called `vampire`
in the main project directory.

- Parameters:
	- `na` number of atoms
	- `nb` number of binary atoms
	- `nf` number of formulae
	- `pr` probability of relational (set to 0 to get problems in S)
	- `pus` probability of universal in s-formula
	- `puf` probability of universal in r-formula
	- `pue` probability of universal in e-term
	- `pns` probability of negated subject (set to 0 to get problems in R)
	- `pno` probability of negated object (set to 0 to get problems in R)
	- `pnp` probability of negated predicate (set to 0 to get problems in R)
	- `e` true for random generation, false to employ removal of trivial absurdities and universal chain blowup
	- `ch` length of chain blowup in hard generation
	- `s` how many problems to generate
	- `sd` seed
    - `nf_min`  min number of formulae if generating a range
    - `nf_max`  max number of formulae if generating a range
    - `af_ratio` ratio of atoms to formulae (can be used instead of na, when generating range)
    - `bf_ratio` ratio of binary atoms to formulae (can be used instead when generating range)
    
The script `comp_subj_verify_vampire.py` is used for generating SRel and SRelNeg fragments, and `generate.py` and `generate_all_all.py` are used to generate constructed S+/R problems, respectively. Refer to scripts for more details.

## Data and models
You can get the data from [here](https://kant.cs.man.ac.uk/data/public/nlr/data.tar.gz) and the pretrained models from [here](https://kant.cs.man.ac.uk/data/public/nlr/models.tar.gz).
