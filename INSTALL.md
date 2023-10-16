# Installation

## Dependencies Installation
This repository is bult in Pytorch 2.0.1 and tested on Ubuntu 20.04.4 LTS5.4.0-148-generic environment (Python 3.8, CUDA 11.7, cuDNN 8.5). Follow these instructions:

1. Clone our repository:
```bash
git clone https://github.com/hellopipu/PromptMR.git
cd PromptMR
```
2. Create conda environment
The Conda environment used can be recreated using the env.yml file:
```bash
conda env create -f env.yml
```

## Dataset Download and Preparation

For CMRxRecon dataset, follow the instruction [here](promptmr_examples/cmrxrecon/README.md).

For FastMRI knee dataset, follow the instruction [here](promptmr_examples/fastmri/README.md).