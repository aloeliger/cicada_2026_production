# CICADA 2026 Production & Testing
## Installation
I recommend running this via virtual environment inside of CMSSW
```
cmsrel CMSSW_15_1_0_pre1
cd CMSSW_15_1_0_pre1/src/
cmsenv && git cms-init
git clone git@github.com:aloeliger/cicada_2026_production.git
cd cicada_2026_production
python3 -m venv cicada_env
source cicada_env/bin/activate
python3 -m pip install . --no-cache-dir
```

## Pipeline
This repo uses DVC (Data Version Control) for pipeline infrastructure and for version control of models.
To quickly rerun the pipeline or any elements that have changed, use `dvc repro`.

### params.yaml
Major parameters and configuration elements are kept in `params.yaml`.
This is recognized by DVC and can be used in DVC experiments, or can be edited by hand.
However, major configuration elements should be written here, whenever possible

### dvc.yaml
This is used by DVC to control the complete pipeline flow.
Any and all steps from making files, to making models, to running evaluations,
to making the firmware should be put in this file.
Dependencies on code and parameters should be specified in their specific sections, and data based outputs (data files, models, logs, but not code and firmware) should be listed here.
