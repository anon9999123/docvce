#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=doclaynet python src/docvce/sfid_split_generator.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
    $@
