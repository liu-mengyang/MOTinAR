SHELL=/bin/bash
CUDA_PATH = /usr/local/cuda-10.2
# TRT_PATH = /usr/local/tensorrt
TORCH_PATH = /usr/local/lib/python3.6/dist-packages/torch
GENCODE = -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75

GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
CCFLAGS = -g -DEBUG -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ 
INCLUDES := -I../include -isystem $(CUDA_PATH)/include -isystem $(TORCH_PATH)/include
LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(TORCH_PATH)/lib/ -Wl,-rpath=$(CUDA_PATH)/lib64
# INCLUDES := -I../include -isystem $(CUDA_PATH)/include -isystem $(TRT_PATH)/include -isystem $(TORCH_PATH)/include
# LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(TORCH_PATH)/lib/ -L$(TRT_PATH)/lib -Wl,-rpath=$(CUDA_PATH)/lib64
LDFLAGS += -lcudart -lnvinfer -lnvonnxparser -ldl -lpthread -lcuda -ltorch -lc10 -ltorch_cuda -lc10_cuda -ltorch_cpu -ltorch_python

OBJ_DIR = obj
OBJ = $(shell find $(OBJ_DIR) -name *.o 2>/dev/null)
DEP = $(OBJ:.o=.d)

BIN = 
SO = DCNv2PluginDyn_nano.so

all: $(BIN) $(SO)

# DCNv2Plugin.so: $(addprefix $(OBJ_DIR)/, DCNv2Plugin.o DeformConv.o)
# DCNv2PluginDyn.so: $(addprefix $(OBJ_DIR)/, DCNv2PluginDyn.o DeformConv.o)
DCNv2PluginDyn_nano.so: $(addprefix $(OBJ_DIR)/, DCNv2PluginDyn_nano.o DeformConv_nano.o)

-include $(DEP)

clean:
	rm -rf $(BIN) $(SO) $(DEP) $(OBJ_DIR) out.trt int8_cache.*

VPATH = ../include:../plugins

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CCFLAGS) -M -MT $@ $(INCLUDES) -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAGS) $(INCLUDES) -Xcompiler -fPIC $(GENCODE) -o $@ -c $<

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(GCC) $(CCFLAGS) -fPIC -MD -MP $(INCLUDES) -o $@ -c $<


$(SO):
	$(GCC) $(CCFLAGS) -fPIC -shared -o $@ $+ $(LDFLAGS)

$(CUDA_BIN):
	$(NVCC) $(NVCCFLAGS) -o $@ $+ $(LDFLAGS)

$(BIN):
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)


