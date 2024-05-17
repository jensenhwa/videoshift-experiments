#!/bin/bash
#SBATCH --account=macondo --partition=macondo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=jphwa@cs.stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --output=preprocess_homage_%j.out
~/vision2/sail_panasonic/panasonic-venv/bin/python ~/vision2/sail_panasonic/videocompare/dataset/preprocessing/resize_interactadl.py
