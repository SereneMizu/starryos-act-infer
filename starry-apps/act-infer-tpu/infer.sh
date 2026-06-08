#!/bin/sh
set -e

MODEL=/opt/act-infer/model.cvimodel
DIR=/opt/act-infer/frames
STATS=/opt/act-infer/stats.json
REF=/opt/act-infer/reference.json

export LD_LIBRARY_PATH=/usr/lib

echo "=== ACT TPU Inference ==="
echo "model:  $MODEL"
echo "frames: $DIR"
echo "stats:  $STATS"

if [ -f "$REF" ]; then
    exec act-infer-tpu \
        --model "$MODEL" \
        --dir "$DIR" \
        --stats "$STATS" \
        --reference "$REF" \
        --track-mem
else
    exec act-infer-tpu \
        --model "$MODEL" \
        --dir "$DIR" \
        --stats "$STATS" \
        --track-mem
fi
