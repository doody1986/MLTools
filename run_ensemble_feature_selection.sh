#! /bin/bash

PWD="$(pwd)"
SCRIPT="TreeEnsembleFeatureSelection.py"
DATA="data_preprocessing/combined_v1_v2_missing_handling_before_merge.csv"
LABEL="PPTERM"
NUM_ENSEMBLE=50
NUM_FEATS=30
RESULTS="feature_selection_ensemble_results.txt"

METHOD_LIST=(CLA WMA OFA CAA)
for method in ${METHOD_LIST[@]}
do
  ${PWD}/${SCRIPT} ${DATA} ${LABEL} ${NUM_ENSEMBLE} ${method} ${NUM_FEATS} 2>&1 | tee ${PWD}/${RESULTS}
done
