#!/bin/bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/dataNAS/people/krmarus/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/dataNAS/people/krmarus/miniconda/etc/profile.d/conda.sh" ]; then
        . "/dataNAS/people/krmarus/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/dataNAS/people/krmarus/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate c2c_env
echo "Job started at: $(date)"
#  spine_muscle_adipose_tissue -i /bmrNAS/people/krmarus/abct_lbp_data/ismrm-katie/additional/unzipped/ --save_segmentations
python muscle_seg_addUnet_yaml.py model_config.yaml

echo "Job ended at: $(date)"
echo "Completed"

# Activate nnunet base environment
