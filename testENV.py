import torch
print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Version: ", torch.version.cuda)
print("cuDNN Version: ", torch.backends.cudnn.version())

#CUDA Available:  True
#CUDA Version:  12.1
#cuDNN Version:  8902