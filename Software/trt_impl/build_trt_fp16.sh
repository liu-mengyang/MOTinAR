trtexec --onnx=fairmot_plugin_dyn.onnx\
        --explicitBatch \
        --minShapes="input":1x3x608x1088\
        --optShapes="input":1x3x608x1088\
        --maxShapes="input":1x3x608x1088\
        --shapes="input":1x3x608x1088\
        --saveEngine=../../weights/fairmot_dyn_fp16.trt\
        --plugins=./build/DCNv2PluginDyn.so\
        --fp16
        --verbose
        

