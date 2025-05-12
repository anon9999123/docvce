#!/bin/bash -l
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYTHONPATH=$SCRIPT_DIR/src:$SCRIPT_DIR/external/atria/src:$PYTHONPATH python ./src/docvce/conf/build_configurations.py
