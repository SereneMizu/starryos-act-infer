#!/bin/sh
set -e

MODEL=/opt/act-infer/model.cvimodel
DIR=/opt/act-infer/frames
STATS=/opt/act-infer/stats.json
REF=/opt/act-infer/reference.json

echo "=== ACT TPU Inference ==="
echo "model:  $MODEL"
echo "frames: $DIR"
echo "stats:  $STATS"
echo "ref:    $REF ($(if [ -f "$REF" ]; then echo "exists"; else echo "missing"; fi))"

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
