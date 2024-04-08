#!/bin/bash

# compile custom operators
export TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6"
cd libs/pointops2
rm -rf build
python setup.py install
cd -
cd odin/modeling/pixel_decoder/ops
rm -rf build
sh make.sh
cd -

