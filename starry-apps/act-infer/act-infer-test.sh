#!/bin/sh
export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH:-}
exec /opt/act-infer/act-infer-ort \
    --model /opt/act-infer/model.onnx \
    --left /opt/act-infer/frame_000000.jpg \
    --right /opt/act-infer/frame_000227.jpg
