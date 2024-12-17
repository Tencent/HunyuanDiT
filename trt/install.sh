# ==============================================================================
# Description: Install TensorRT and prepare the environment for TensorRT.
# ==============================================================================

# ----------------------------------------
# Check the system, tools and arguments
# ----------------------------------------

# Check system. Only support TensorRT on Linux (MacOS is not supported.)
if [ "$(uname)" != "Linux" ]; then
    echo "Only support TensorRT on Linux"
    exit 1
fi

# Check if the model_trt path is provided. If not, use the default path.
if [ -z "$1" ]; then
    MODEL_TRT_DIR=$(cd ckpts/t2i/model_trt; pwd)
else
    MODEL_TRT_DIR=$(cd "$1"; pwd)
fi

# Check if the model_trt path exists.
if [ ! -d "${MODEL_TRT_DIR}" ]; then
    echo "The model_trt directory (${MODEL_TRT_DIR}) does not exist. Please specify the path by:"
    echo "    sh trt/install.sh <model_trt_dir>"
    exit 1
fi

# Check if ldconfig exists.
if [ ! -x "$(command -v ldconfig)" ]; then
    echo "ldconfig is not installed. Please install it first."
    exit 1
fi

export TENSORRT_VERSION='10.1.0.27'
TENSORRT_PACKAGE="${MODEL_TRT_DIR}/TensorRT-${TENSORRT_VERSION}.tar.gz"

# Check if the TensorRT package is downloaded.
if [ ! -f "${TENSORRT_PACKAGE}" ]; then
    echo "The TensorRT package (${TENSORRT_PACKAGE}) does not exist. Please download it first with following steps:"
    echo "1. cd HunyuanDiT"
    echo "2. huggingface-cli download Tencent-Hunyuan/HunyuanDiT-TensorRT --local-dir ./ckpts/t2i/model_trt"
    exit 1
else
    echo "Found TensorRT package: ${TENSORRT_PACKAGE}"
fi

# ----------------------------------------
# Start to install TensorRT
# ----------------------------------------

# Extract the TensorRT package.
echo "Extracting the TensorRT package..."
tar xf "${TENSORRT_PACKAGE}" -C "${MODEL_TRT_DIR}"
TENSORRT_DIR="${MODEL_TRT_DIR}/TensorRT-${TENSORRT_VERSION}"
echo "Extracting the TensorRT package finished"

# Add the TensorRT library path to the system library path.
echo "${MODEL_TRT_DIR}/lib/" >> /etc/ld.so.conf.d/nvidia.conf && ldconfig

# Install the TensorRT Python wheel.
echo "Installing the TensorRT Python wheel..."
# Get python version, e.g., cp38 for Python 3.8; cp310 for Python 3.10
PYTHON_VERSION=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
python -m pip install --no-cache-dir ${TENSORRT_DIR}/python/tensorrt*-${PYTHON_VERSION}*
echo "Installing the TensorRT Python wheel finished"

# Prepare activate.sh and deactivate.sh
{
  echo "TENSORRT_DIR=${TENSORRT_DIR}"
  echo 'export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib/:$LD_LIBRARY_PATH'
  echo 'export LIBRARY_PATH=${TENSORRT_DIR}/lib/:$LIBRARY_PATH'
  echo 'export PATH=${TENSORRT_DIR}/bin/:$PATH'
} > $(dirname "$0")/activate.sh
{
  echo "TENSORRT_DIR=${TENSORRT_DIR}"
  echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH/${TENSORRT_DIR}\/lib\/:}'
  echo 'export LIBRARY_PATH=${LIBRARY_PATH/${TENSORRT_DIR}\/lib\/:}'
  echo 'export PATH=${PATH/${TENSORRT_DIR}\/bin\/:}'
} > $(dirname "$0")/deactivate.sh
