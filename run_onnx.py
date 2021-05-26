import numpy as np
import onnx
import onnxruntime as ort
import time

# model_name = "bs8_unetsmall" 
model_name = "bs8_mobilenetsmall"
# model_name = "bs8_no_segmentation_number2"
# model_name = "bs8_no_segmentation_number2_mobilenetv3"

ort_session = ort.InferenceSession(f"{model_name}.onnx")
inputs = np.random.randn(2, 9, 240, 320).astype(np.float32)
outputs = ort_session.run(None, {'input': inputs})


start = time.time()
num_inf = 1000
for _ in range(num_inf):
    out = ort_session.run(None, {'input': inputs})

print(f"Mean inference time {(time.time()-start)/num_inf:.3f}")
print(inputs.shape)
print(inputs.dtype)
print(out[0].shape)
print(out[0].dtype)
