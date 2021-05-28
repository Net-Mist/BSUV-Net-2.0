import torch


# model_name = "bs8_unetsmall" 
model_name = "bs8_ddrnet_23_slim_2"
# model_name = "bs8_no_segmentation_number2"
# model_name = "bs8_no_segmentation_number2_mobilenetv3"
# model_name = "BSUVNet-emptyBG-recentBG"

MODEL_PATH = f"{model_name}.mdl"
IMG_H=240
IMG_W=320


model = torch.load(MODEL_PATH).cuda().eval()

input_names = ["input"]
output_names = ["output"]


dummy_input = torch.randn(2, 9, IMG_H,  IMG_W, device='cuda')

torch.onnx.export(model, dummy_input, f"{model_name}_{IMG_W}x{IMG_H}.onnx", verbose=True,
                  input_names=input_names, output_names=output_names)
