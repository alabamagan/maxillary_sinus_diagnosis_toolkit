#!/bin/bash

## Purpose
##    This bash script
## Requirements
##    This script requires pytorch-medical-imaging package to perform deep learning inference.
##


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
  echo "  -s  STEP          (Optional) Specify which step(s) to perform. e.g. '123'"
  echo "  -v  VERBOSE       (Optional) Verbosity."
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
while getopts 'hi:o:n:c:s:b:g:vks' OPT; do
  case $OPT in
    i) S1_INPUT_DIR=`realpath "$OPTARG"`;;
    o) OUTPUT_DIR=`realpath "$OPTARG"`;;
    c) CHECKPOINTS_DIR=`realpath "$OPTARG"`;;
    b) BATCH_SIZE="$OPTARG";;
    k) NORMALIZE_FLAG=1;;
    s) STEP="$OPTARG";;
    v) _V_FLAG="--verbose";;
    n) NUM_THREAD="$OPTARG";;
    g) GROUND_TRUTH=`realpath "$OPTARG"`;;
    h) help_msg
       exit 0;;
    ?) help_msg
       exit 0;;
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
  pmi-main --config=pmi_config/Seg_S1.ini ${_V_FLAG} --override="(Checkpoint,cp_load_dir)=$S1_CHECKPOINT;(Data,input_dir)=${S1_INPUT_DIR};(Data,prob_map_dir)="${S1_INPUT_DIR}/Mask-80/";(Data,output_dir)=${OUTPUT_DIR}/S1_output;(Data,target_dir)=${GROUND_TRUTH}/Air-space;(General,force_train_data)=No" --inference
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
  pmi-main --config=pmi_config/Seg_S2.ini ${_V_FLAG} --override="(Checkpoint,cp_load_dir)=$S2_CHECKPOINT;(Data,input_dir)=${S1_INPUT_DIR};(Data,prob_map_dir)=${OUTPUT_DIR}/S1_output/;(Data,output_dir)=${OUTPUT_DIR}/S2_output;(Data,target_dir)=${GROUND_TRUTH}/Lesion-only;(General,force_train_data)=No;" --inference

  # Post processing


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
  S1_CROP_PAIR="-i ${OUTPUT_DIR}/S1_output -o ${OUTPUT_DIR}/Cropped_S1"
  S2_CROP_PAIR="-i ${OUTPUT_DIR}/S2_output -o ${OUTPUT_DIR}/Cropped_S2"
  GT_S1_CROP_PAIR=""
  GT_S2_CROP_PAIR=""
  ## If ground truth is provided, also crop the manually segmented labels
  ### For Step 1
  if [[ -d ${GROUND_TRUTH}/Air-space ]]
  then
    echo "Also crop ground-truth for step 1"
    GT_S1_CROP_PAIR="-i ${GROUND_TRUTH}/Air-space -o ${OUTPUT_DIR}/Cropped_S1_GT"
  fi
  ### For Step 2
  if [[ -d ${GROUND_TRUTH}/Lesion-only ]]
  then
    echo "Also crop ground-truth for step 2"
      GT_S2_CROP_PAIR="-i ${GROUND_TRUTH}/Lesion-only -o ${OUTPUT_DIR}/Cropped_S2_GT"
  fi

  msdtk-crop_sinuses ${S1_CROP_PAIR} ${S2_CROP_PAIR} ${GT_S1_CROP_PAIR} ${GT_S2_CROP_PAIR} --save-bounds -g "^[0-9]+" -n $NUM_THREAD ${_V_FLAG}

  ## create label statistics in the testing data
  ##---------------------------------------------
  msdtk-label_statistics -i ${OUTPUT_DIR}/Cropped_S1 -o ${OUTPUT_DIR}/Cropped_S1/lab_stat.csv -g "^[0-9]+_(L|R)" -n $NUM_THREAD ${_V_FLAG}
  msdtk-label_statistics -i ${OUTPUT_DIR}/Cropped_S2 -o ${OUTPUT_DIR}/Cropped_S2/lab_stat.csv -g "^[0-9]+_(L|R)" -n $NUM_THREAD ${_V_FLAG}

  ## Compute segmentation result
  ##----------------------------
  if [[ -d ${OUTPUT_DIR}/Cropped_S1 && -d ${OUTPUT_DIR}/Cropped_S1_GT ]]
  then
    pmi-analysis_segment -a --test-data=${OUTPUT_DIR}/Cropped_S1 --gt-data=${OUTPUT_DIR}/Cropped_S1_GT --id-globber="^[0-9]+_(L|R)" ${_V_FLAG} --save ${OUTPUT_DIR}/Cropped_S1_result.csv
  fi
  if [[ -d ${OUTPUT_DIR}/Cropped_S2 && -d ${OUTPUT_DIR}/Cropped_S2_GT ]]
  then
    pmi-analysis_segment -a --test-data=${OUTPUT_DIR}/Cropped_S2 --gt-data=${OUTPUT_DIR}/Cropped_S2_GT --id-globber="^[0-9]+_(L|R)" ${_V_FLAG} --save ${OUTPUT_DIR}/Cropped_S2_result.csv
  fi

  return 0
  # load trained model and perform inference and save the results
  S3_MODEL_FILE_MT=${TRAINED_MODELS_DIR}/seg2deg_mt.msdtk
  S3_MODEL_FILE_MRC=${TRAINED_MODELS_DIR}/seg2deg_mrc.msdtk
  S3_MODEL_FILE_HEALTHY=${TRAINED_MODELS_DIR}/seg2deg_healthy.msdtk

  # If ground truth is provided, generate performance report for classification

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