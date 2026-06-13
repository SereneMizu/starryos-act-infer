#!/bin/sh
set -e

MODEL=/opt/act-infer/model.rknn
DIR=/opt/act-infer/frames
STATS=/opt/act-infer/stats.json
REF=/opt/act-infer/reference.json

echo "=== ACT RKNN Inference (RK3588) ==="
echo "model:  $MODEL"
echo "frames: $DIR"
echo "stats:  $STATS"
echo "ref:    $REF ($(if [ -f "$REF" ]; then echo "exists"; else echo "missing"; fi))"

if [ -f "$REF" ]; then
    exec act-infer-rknn \
        --model "$MODEL" \
        --dir "$DIR" \
        --stats "$STATS" \
        --reference "$REF" \
        --track-mem
else
    exec act-infer-rknn \
        --model "$MODEL" \
        --dir "$DIR" \
        --stats "$STATS" \
        --track-mem
fi
