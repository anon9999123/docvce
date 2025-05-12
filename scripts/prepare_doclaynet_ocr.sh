#!/bin/bash
apt update
apt install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd poppler-utils
pip install pytesseract
LOG_LEVEL=DEBUG PYTHONPATH=./external/atria/src/:./external/docsets/src:./src/ \
    python ./external/atria/src/atria/core/task_runners/atria_data_processor.py \
    hydra.searchpath=[pkg://docsets/conf,pkg://docvce/conf] \
    data_module=document_classification/doclaynet_with_ocr_override \
    +image_size=256 \
    +gray_to_rgb=True \
    +train_batch_size=32 \
    +eval_batch_size=32 \
    _zen_exclude=[image_size,gray_to_rgb,train_batch_size,eval_batch_size]
