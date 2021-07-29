import torch
import time
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes

ctypes.cdll.LoadLibrary('./build_nano/DCNv2PluginDyn_nano.so')

im_blob = torch.randn([1,3,608,1088]).cuda().float()

from trt_lite import TrtLite
import numpy as np
import os

class PyTorchTensorHolder(pycuda.driver.PointerHolderBase):
    def __init__(self, tensor):
        super(PyTorchTensorHolder, self).__init__()
        self.tensor = tensor
    def get_pointer(self):
        return self.tensor.data_ptr()

for engine_file_path in ['../../weights/fairmot_nano.trt']:
    if not os.path.exists(engine_file_path):
        print('Engine file', engine_file_path, 'doesn\'t exist. Please run trtexec and re-run this script.')
        exit(1)
    
    print('====', engine_file_path, '===')
    trt = TrtLite(engine_file_path=engine_file_path)
    trt.print_info()
    i2shape = {0: (1, 3, 608, 1088)}
    io_info = trt.get_io_info(i2shape)
    d_buffers = trt.allocate_io_buffers(i2shape, True)
    output1_data_trt = np.zeros(io_info[1][2], dtype=np.float32)
    output2_data_trt = np.zeros(io_info[2][2], dtype=np.float32)
    output3_data_trt = np.zeros(io_info[3][2], dtype=np.float32)

    # input from device to device
    cuda.memcpy_dtod(d_buffers[0], PyTorchTensorHolder(im_blob), im_blob.nelement() * im_blob.element_size())
    trt.execute(d_buffers, i2shape)
    
    cuda.memcpy_dtoh(output1_data_trt, d_buffers[1])
    cuda.memcpy_dtoh(output2_data_trt, d_buffers[2])
    cuda.memcpy_dtoh(output3_data_trt, d_buffers[3])

    cuda.Context.synchronize()
    t0 = time.time()
    for i in range(nRound):
        trt.execute(d_buffers, i2shape)
    cuda.Context.synchronize()
    time_trt = (time.time() - t0) / nRound
    print('TensorRT time:', time_trt)
