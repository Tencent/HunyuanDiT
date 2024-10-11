# =========================
#   Controlnet Config
# =========================

control_type = 'canny'  # Controlnet condition type,  choices=['canny', 'depth', 'pose']
control_weight = '1.0' # Controlnet weight, You can use a float to specify the weight for all layers, or use a list to separately specify the weight for each layer, for example, '[1.0 * (0.825 ** float(19 _ i)) for i in range(19)]
# Inference condition image path
condition_image_path = None
