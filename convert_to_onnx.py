import torch


# model_name = "bs8_unetsmall" 
model_name = "bs8_mobilenetsmall"
# model_name = "bs8_no_segmentation_number2"
# model_name = "bs8_no_segmentation_number2_mobilenetv3"
# model_name = "BSUVNet-emptyBG-recentBG"

MODEL_PATH = f"{model_name}.mdl"

model = torch.load(MODEL_PATH).cuda().eval()

input_names = ["input"]
output_names = ["output"]


dummy_input = torch.randn(2, 9, 480, 640, device='cuda')

torch.onnx.export(model, dummy_input, f"{model_name}_640x480.onnx", verbose=True,
                  input_names=input_names, output_names=output_names)
