#!/bin/bash

CAFFE_EXE_DIR=$(pwd)/build/tools

echo $CAFFE_EXE_DIR

$CAFFE_EXE_DIR/caffe train -solver training/solver_lenet.prototxt -weights training/models/GoogleNet_SOD_finetune.caffemodel 
