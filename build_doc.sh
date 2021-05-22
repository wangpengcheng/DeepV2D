#!/bin/bash
#cd ./docs

echo "start build doc in docs dir"

sphinx-apidoc -o source/modules/deepv2d ../deepv2d/
sphinx-apidoc -o source/modules/netadapt ../netadapt/
sphinx-apidoc -o source/modules/training ../training/
# sphinx-apidoc -o source/modules/vot_siamban ../vot_siamban/
# sphinx-apidoc -o source/modules/tools ../tools/

make clean
make html