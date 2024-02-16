#!/bin/bash
#SBATCH --account=macondo --partition=macondo
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=jphwa@cs.stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --output=preprocess_homage_%j.out
panasonic-venv/bin/python videocompare/dataset/preprocessing/preprocess_homage.py
