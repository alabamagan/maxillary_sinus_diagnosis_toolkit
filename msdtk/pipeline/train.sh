#!/bin/bash

## Purpose
##    This bash scripts train the models for step 3 of the whole algorithm
## Requirements
##    - msdtk
##    - scikit-learn



help_msg() {
  echo "Usage:"
  echo "  inference.sh -i INPUT_DIR -o OUTPUT_DIR [-g GT_DIR]"
  echo ""
  echo "Variables:"
  echo "  -i  INPUT_DIR     The image inputs."
  echo "  -o  OUTPUT_DIR    The output directory."
  echo "  -c  CHECKPOINTS   The directory of the deep learning checkpoints."
  echo "  -k  NORMALIZE     Whether normlization is needed."
  echo "  -g  GROUND_TRUTH  The directory where the ground-truth labels are."
  echo "  -n  NUM_THREAD    (Optional) Number of thread used."
  echo "  -m  MASK_DIR      (Optional) Manually specify mask directory. The mask should be under MASK_DIR/Mask-80 for "
  echo "                               step 1 and MASK_DIR/Air-space for step 2."
  echo "  -s  STEP          (Optional) Specify which step(s) to perform. e.g. '123'"
  echo "  -v  VERBOSE       (Optional) Verbosity."
  exit 0
}

## Useful variables
SCRIPT_PATH="`dirname \"$0\"`"
SCRIPT_PATH="`realpath $SCRIPT_PATH`"

## Default values
S1_INPUT_DIR=""
OUTPUT_DIR=""
CHECKPOINTS_DIR=""
INTER_FILES_DIR=/tmp/intermediate_files
NUM_THREAD=16
STEP="123"
_V_FLAG=""
GROUND_TRUTH=""
while getopts 'hi:o:n:m:c:s:b:g:vks' OPT; do
  case $OPT in
    i) S1_INPUT_DIR=$(realpath "$OPTARG");;
    o) OUTPUT_DIR=$(realpath "$OPTARG");;
    c) CHECKPOINTS_DIR=$(realpath "$OPTARG");;
    b) BATCH_SIZE="$OPTARG";;
    k) NORMALIZE_FLAG=1;;
    s) STEP="$OPTARG";;
    v) _V_FLAG="--verbose";;
    n) NUM_THREAD="$OPTARG";;
    m) MASK_DIR=$(realpath "$OPTARG");;
    g) GROUND_TRUTH=$(realpath "$OPTARG");;
    h) help_msg;;
    ?) help_msg;;
  esac
done

## Value checks are mostly done in the python codes. So not going to it now.
if [ $# -eq 0 ]
then
  echo "No arguments were provided"
  echo ""
  help_msg
  exit 0
fi



#==================================
# Preparations
#==================================

# make directories
mkdir -p ${INTER_FILES_DIR}
mkdir -p ${OUTPUT_DIR}

# Normalize the input if requested
if [ $NORMALIZE_FLAG ]
then
  NORMALIZED_INPUT=${INTER_FILES_DIR}/normalized
  mkdir -p $NORMALIZED_INPUT
  msdtk-pre_processing -i ${S1_INPUT_DIR} -o ${NORMALIZED_INPUT} -g ${GROUND_TRUTH}
  S1_INPUT_DIR=${NORMALIZED_INPUT}
  GROUND_TRUTH=${NORMALIZED_INPUT}/Seg
fi
#==================================
# Stage 1 Air-Segmentation
#==================================

step1() {
  toilet -f slant "=== Step 1 ==="

  # Check if checkpoints are found
  S1_CHECKPOINT="${CHECKPOINTS_DIR}/s1_airspace.pt"

  if ! [[ -f $S1_CHECKPOINT ]]
  then
    >&2 echo "Cannot open checkpoints! Put the checkpoint under: ${S1_CHECKPOINT}"
    return 1 # Failed
  fi

  cd "${SCRIPT_PATH}/../../" || return 1

  # If mask directory is specified use it instead
  if [[ -d ${MASK_DIR} ]]
  then
    MASK=${MASK_DIR}/Mask-80
  else
    MASK=${OUTPUT_DIR}/Mask-80
  fi

  pmi-main --config=pmi_config/Seg_S1.ini ${_V_FLAG} --override="(Checkpoint,cp_load_dir)=$S1_CHECKPOINT;(Data,input_dir)=${S1_INPUT_DIR};(Data,prob_map_dir)=${MASK};(Data,output_dir)=${OUTPUT}/S1_output_train;(Data,target_dir)=${GROUND_TRUTH}/Air-space;(General,force_train_data)=Yes" --inference --batch-size=$BATCH_SIZE
  return 0
}

#==================================
# Stage 2 Segment lesions
#==================================

step2() {
  toilet -f slant "=== Step 2 ==="

  # Check if checkpoints are found
  S2_CHECKPOINT="${CHECKPOINTS_DIR}/s2_lesions.pt"

  if ! [[ -f $S2_CHECKPOINT ]]
  then
    >&2 echo "Cannot open checkpoints! Put the checkpoint under: ${S2_CHECKPOINT}"
    return 1 # Failed
  fi

  cd "${SCRIPT_PATH}/../../" || return 1

  # If mask directory is specified use it instead
  if [[ -d ${MASK_DIR} ]]
  then
    MASK=${MASK_DIR}/Air-space
  else
    MASK=${OUTPUT_DIR}/S1_output_train
  fi

  pmi-main --config=pmi_config/Seg_S2.ini ${_V_FLAG} --override="(Checkpoint,cp_load_dir)=$S2_CHECKPOINT;(Data,input_dir)=${S1_INPUT_DIR};(Data,prob_map_dir)=${MASK};(Data,output_dir)=${OUTPUT_DIR}/S2_output_train;(Data,target_dir)=${GROUND_TRUTH}/Lesion-only;(General,force_train_data)=Yes" --inference
  return 0
}

#==================================
# Stage 3 Classify based on segment
#==================================

step3(){
  toilet -f slant "=== Step 3 ==="

  # Crop stage 1 and 2 output into left and right sinuses
  echo "Cropping the input into left and right sinuses"

  ## Prepare arguments
  S1_OUTPUT=${OUTPUT_DIR}/S1_output
  S2_OUTPUT=${OUTPUT_DIR}/S2_output
  S1_OUTPUT_CROPPED=${OUTPUT_DIR}/Cropped_S1
  S2_OUTPUT_CROPPED=${OUTPUT_DIR}/Cropped_S2
  S1_OUTPUT_CROPPED_POST_PROC=${S1_OUTPUT_CROPPED}_PP
  S2_OUTPUT_CROPPED_POST_PROC=${S2_OUTPUT_CROPPED}_PP
  S1_CROP_PAIR="-i ${S1_OUTPUT} -o ${S1_OUTPUT_CROPPED}"
  S2_CROP_PAIR="-i ${S2_OUTPUT} -o ${S2_OUTPUT_CROPPED}"
  S1_CROPPED_GT="${OUTPUT_DIR}/Cropped_S1_GT"
  S2_CROPPED_GT="${OUTPUT_DIR}/Cropped_S2_GT"
  IMG_CROP_PAIR="-i ${S1_INPUT_DIR} -o ${OUTPUT_DIR}/Cropped_IMG"
  GT_S1_CROP_PAIR=""
  GT_S2_CROP_PAIR=""
  ## If ground truth is provided, also crop the manually segmented labels
  ### For Step 1
  if [[ -d ${GROUND_TRUTH}/Air-space ]]
  then
    echo "Also crop ground-truth for step 1"
    GT_S1_CROP_PAIR="-i ${GROUND_TRUTH}/Air-space -o ${S1_CROPPED_GT}"
  fi
  ### For Step 2
  if [[ -d ${GROUND_TRUTH}/Lesion-only ]]
  then
    echo "Also crop ground-truth for step 2"
      GT_S2_CROP_PAIR="-i ${GROUND_TRUTH}/Lesion-only -o ${S2_CROPPED_GT}"
  fi

  msdtk-crop_sinuses ${S1_CROP_PAIR} ${S2_CROP_PAIR} ${GT_S1_CROP_PAIR} ${GT_S2_CROP_PAIR} ${IMG_CROP_PAIR} --save-bounds -g "^[0-9]+" -n $NUM_THREAD ${_V_FLAG} --skip

  ## Postprocessing of the lesion label
  msdtk-post_processing -s1 ${S1_OUTPUT_CROPPED} -s2 ${S2_OUTPUT_CROPPED} -o ${OUTPUT_DIR}/Cropped_PP/ -n ${NUM_THREAD} ${_V_FLAG}
  msdtk-post_processing -s1 ${S1_CROPPED_GT} -s2 ${S2_CROPPED_GT} -o ${OUTPUT_DIR}/Cropped_GT/ -n ${NUM_THREAD} ${_V_FLAG} --skip-proc

  ## create label statistics in the testing data
  ##---------------------------------------------
  msdtk-label_statistics -i ${OUTPUT_DIR}/Cropped_GT -o ${OUTPUT_DIR}/Cropped_GT/lab_stat.csv -g "^[0-9]+_(L|R)" -n $NUM_THREAD ${_V_FLAG}

  ## Compute segmentation result
  ##----------------------------
  pmi-analysis_segment -a --test-data=${OUTPUT_DIR}/Cropped_PP --gt-data=${OUTPUT_DIR}/Cropped_GT --id-globber="^[0-9]+_(L|R)" ${_V_FLAG} --save ${OUTPUT_DIR}/Cropped_AllClasses_result.xlsx

  # load trained model and perform inference and save the results
  S3_MODEL_DIR=${CHECKPOINTS_DIR}/s3_seg2diag.msdtks2d
  S3_GT_DIR=${GROUND_TRUTH}/datasheet.csv

  # If ground truth is provided, generate performance report for classification
  msdtk-seg2diag_inference -s1 ${OUTPUT_DIR}/Cropped_PP/lab_stat.csv -s2 -i ${S3_MODEL_DIR} -v -o ${OUTPUT_DIR}/S3_results.xlsx -gt ${S3_GT_DIR}
  return 0
}


# Actually run code
while read -n1 character;
do
  case $character in
    "1") step1 || >&2 echo "Error when performing step 1!";;
    "2") step2 || >&2 echo "Error when performing step 2!";;
    "3") step3 || >&2 echo "Error when performing step 3!";;
    ?) echo "Wrong argument for -s option: ${_STEP}"
  esac
done < <(echo -n "$STEP")