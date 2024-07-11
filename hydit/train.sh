training_type=""
all_params=("$@")

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --training-type)
            training_type="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# check --training-type
if [ -z "$training_type" ]; then
    echo "Please input --training-type"
    exit 1
fi

case $training_type in
    full|lora)
        python hydit/train_deepspeed.py "${all_params[@]}"
        ;;
    controlnet)
        python hydit/train_deepspeed_controlnet.py "${all_params[@]}"
        ;;
    *)
        echo "Invalid --training-type: $training_type"
        exit 1
        ;;
esac
