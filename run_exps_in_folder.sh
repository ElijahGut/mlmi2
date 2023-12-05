#!/bin/bash

EXP_DIR="$1"

echo "folder is $EXP_DIR..."

for file in "$EXP_DIR/slurm_mlmi2_*"; do
	sbatch $file
done

echo "done!"
    
