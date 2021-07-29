trtexec --onnx=fairmot_plugin_dyn.onnx\
        --explicitBatch \
        --minShapes="input":1x3x32x576\
        --optShapes="input":1x3x320x576\
        --maxShapes="input":1x3x608x1088\
        --shapes="input":1x3x608x1088\
        --saveEngine=../../weights/fairmot_dyn.trt\
        --plugins=./build/DCNv2PluginDyn.so\
        --verbose
