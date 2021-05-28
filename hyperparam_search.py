import os



cmd = "CUDA_VISIBLE_DEVICES={cuda} python train.py --batch_size {bs}" \
      "--seg_ch 0 --model_name {name} --set_number 2 --network ddrnet_23_slim --inp_size {inp_size}"


print(cmd.format(cuda=1,bs=8, inp_size=224, name="coucou"))

with os.popen('sleep 5') as stream:
    out = stream.read()
