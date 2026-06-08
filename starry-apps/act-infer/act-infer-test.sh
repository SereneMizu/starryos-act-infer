#!/bin/sh
export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH:-}
exec /opt/act-infer/act-infer-ort \
    --model /opt/act-infer/model.onnx \
    --dir /opt/act-infer/frames \
    --stats /opt/act-infer/stats.json
