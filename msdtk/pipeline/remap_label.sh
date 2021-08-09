#!/bin/bash

## Purpose
##    This bash script remap the msdtk step 1 and step 2 output to a single label and evalute the performance if ground-
##    truth is available.
## Requirements
##    This script requires pytorch-medical-imaging package to perform deep learning inference.
##


help_msg() {
  echo "Usage:"
  echo "  remap_label.sh -i INPUT_DIR"
  echo ""
  echo "Variables:"
  echo "  -i  INPUT_DIR     The image inputs."
  echo "  -n  NUM_THREAD    (Optional) Number of thread used."
  echo "  -v  VERBOSE       (Optional) Verbosity."
}

## Useful variables
SCRIPT_PATH="`dirname \"$0\"`"
SCRIPT_PATH="`realpath $SCRIPT_PATH`"

## Default values
INPUT_DIR=""
NUM_THREAD=16
_V_FLAG=""
GROUND_TRUTH=""
while getopts 'hi:o:m:n:c:s:b:g:vks' OPT; do
  case $OPT in
    i) INPUT_DIR=`realpath "$OPTARG"`;;
    v) _V_FLAG="--verbose";;
    n) NUM_THREAD="$OPTARG";;
    g) GROUND_TRUTH=`realpath "$OPTARG"`;;
    h) help_msg
       exit 0;;
    ?) help_msg
       exit 0;;
  esac
done

## Variables
S2_TESTING_DATA=${INPUT_DIR}/Cropped_S2_PP
S2_GT_DATA=${INPUT_DIR}/Cropped_S2_GT

if [[ ! -d ${S2_GT_DATA} ]]
then
  echo "Cannot find ground-truth data!"
  exit 1
fi

pmi-labels_remap -i ${S2_TESTING_DATA} -o ${INPUT_DIR}/Cropped_S2_PP_OneClass -n ${NUM_THREAD} ${_V_FLAG} -m "{1:1, 2:1, 3:1}"
pmi-labels_remap -i ${S2_GT_DATA} -o ${INPUT_DIR}/Cropped_S2_GT_OneClass -n ${NUM_THREAD} ${_V_FLAG} -m "{1:1, 2:1, 3:1}"

pmi-analysis_segment --test-data ${INPUT_DIR}/Cropped_S2_PP_OneClass --gt-data ${INPUT_DIR}/Cropped_S2_GT_OneClass -a ${_V_FLAG} --save "${INPUT_DIR}/Oneclass_results.xlsx" --id-globber "^[0-9]+_(L|R)"
exit 0

