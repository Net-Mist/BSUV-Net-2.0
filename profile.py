import numpy as np
import torch
import time
import torch.autograd.profiler as profiler
import torchvision.transforms as tvtf

# model_name = "bs8_unetsmall" 
# model_name = "bs8_mobilenetsmall"
model_name = "BSUVNet-emptyBG-recentBG"
# model_name = "bs8_no_segmentation_number2"
# model_name = "bs8_no_segmentation_number2_mobilenetv3"

MODEL_PATH = f"{model_name}.mdl"

model = torch.load(MODEL_PATH)
model.cuda().eval()

inputs = tvtf.ToTensor()(np.random.randn(240, 320, 9).astype(np.float32))
inputs = torch.stack([inputs, inputs], dim=0)

start = time.time()
num_inf = 10
with torch.no_grad():
    out = model(inputs.cuda().float()).cpu().numpy()
    for _ in range(num_inf):
        out = model(inputs.cuda().float()).cpu().numpy()

print(f"Mean inference time {(time.time()-start)/num_inf:.3f}")
print(inputs.shape)
print(inputs.dtype)
print(out.shape)
print(out.dtype)
