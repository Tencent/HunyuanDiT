# ==============================================================================
# Description: Export ONNX model and build TensorRT engine.
# ==============================================================================

# Check if the model root path is exists or provided.
if [ -z "$1" ]; then
  if [ -d "ckpts" ]; then
    echo "The model root directory is not provided. Use the default path 'ckpts'."
    export MODEL_ROOT=ckpts
  else
    echo "Default model path 'ckpts' does not exist. Please provide the path of the model root directory."
    exit 1
  fi
elif [ ! -d "$1" ]; then
  echo "The model root directory ($1) does not exist."
  exit 1
else
  export MODEL_ROOT=$(cd "$1"; pwd)
fi

export ONNX_WORKDIR=${MODEL_ROOT}/onnx_model
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "ONNX_WORKDIR=${ONNX_WORKDIR}"

# Remove old directories.
if [ -d "${ONNX_WORKDIR}" ]; then
  echo "Remove old ONNX directories..."
  rm -r ${ONNX_WORKDIR}
fi

# Inspect the project directory.
SCRIPT_PATH="$( cd "$( dirname "$0" )" && pwd )"
PROJECT_DIR=$(dirname "$SCRIPT_PATH")
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"
cd ${PROJECT_DIR}
echo "Change directory to ${PROJECT_DIR}"

# ----------------------------------------
# 1. Export ONNX model.
# ----------------------------------------

# Sleep for reading the message.
sleep 2s

echo "Exporting ONNX model..."
python trt/export_onnx.py --model-root ${MODEL_ROOT} --onnx-workdir ${ONNX_WORKDIR}
echo "Exporting ONNX model finished"

# ----------------------------------------
# 2. Build TensorRT engine.
# ----------------------------------------

echo "Building TensorRT engine..."
ENGINE_DIR="${MODEL_ROOT}/t2i/model_trt/engine"
mkdir -p ${ENGINE_DIR}
ENGINE_PATH=${ENGINE_DIR}/model_onnx.plan
PLUGIN_PATH=${MODEL_ROOT}/t2i/model_trt/fmha_plugins/9.2_plugin_cuda11/fMHAPlugin.so

trtexec \
  --onnx=${ONNX_WORKDIR}/export_modified_fmha/model.onnx \
  --fp16 \
  --saveEngine=${ENGINE_PATH} \
  --minShapes=x:2x4x90x90,t:2,encoder_hidden_states:2x77x1024,text_embedding_mask:2x77,encoder_hidden_states_t5:2x256x2048,text_embedding_mask_t5:2x256,image_meta_size:2x6,style:2,cos_cis_img:2025x88,sin_cis_img:2025x88 \
  --optShapes=x:2x4x128x128,t:2,encoder_hidden_states:2x77x1024,text_embedding_mask:2x77,encoder_hidden_states_t5:2x256x2048,text_embedding_mask_t5:2x256,image_meta_size:2x6,style:2,cos_cis_img:4096x88,sin_cis_img:4096x88 \
  --maxShapes=x:2x4x160x160,t:2,encoder_hidden_states:2x77x1024,text_embedding_mask:2x77,encoder_hidden_states_t5:2x256x2048,text_embedding_mask_t5:2x256,image_meta_size:2x6,style:2,cos_cis_img:6400x88,sin_cis_img:6400x88 \
  --shapes=x:2x4x128x128,t:2,encoder_hidden_states:2x77x1024,text_embedding_mask:2x77,encoder_hidden_states_t5:2x256x2048,text_embedding_mask_t5:2x256,image_meta_size:2x6,style:2,cos_cis_img:4096x88,sin_cis_img:4096x88 \
  --verbose \
  --builderOptimizationLevel=4  \
  --staticPlugins=${PLUGIN_PATH}
