import os


input_sizes = [224, 240, 320]
batch_sizes = [8, 16, 32]
learning_rates = [1e-4]
optimizers = ['adam', 'sgd']


cmd_template = "CUDA_VISIBLE_DEVICES={cuda} python train.py --batch_size {bs}" \
               " --seg_ch 0 --model_name {name} --set_number 2 --network ddrnet_23_slim --inp_size {inp_size}" \
               " --opt {opt}"

list_args = []

for inp_size in input_sizes:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for opt in optimizers:
                name = f"ddrnet_23_slim_bs{batch_size}_lr{learning_rate}_inpsize{inp_size}_opt{opt}"
                d = {
                    "bs": batch_size,
                    "opt": opt,
                    "inp_size": inp_size,
                    "lr": learning_rate,
                    "name": name
                }
                list_args.append(d)

print(len(list_args))
streams = [None for _ in range(4)]
cuda_id = 0

for args in list_args:
    if cuda_id > 3:
        for stream in streams:
            out = stream.read()
            print(out)
        cuda_id = 0

    cmd = cmd_template.format(**args, cuda=cuda_id)
    streams[cuda_id] = os.popen(cmd)

    cuda_id += 1

for stream in streams:
    out = stream.read()
    print(out)

