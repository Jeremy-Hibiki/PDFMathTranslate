#!/usr/bin/env bash
python -m pdf2zh.pdf2zh \
  ./pdfs/in/pg059-axi-interconnect.pdf \
  --onnx /home/vscode/.cache/babeldoc/models/doclayout_yolo_docstructbench_imgsz1280_2501.onnx \
  -s openailiked:glm-4-flash openailiked:glm-4-9b-chat \
  -t 8 \
  -w 8 \
  -f '(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Sym|.*Math)' \
  --max-retries 1 \
  --output ./pdfs/out --debug --ignore-cache
