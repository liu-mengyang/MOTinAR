trtexec --onnx=fairmot_plugin_nano.onnx\
        --explicitBatch \
        --minShapes="input":1x3x320x576\
        --optShapes="input":1x3x320x576\
        --maxShapes="input":1x3x608x1088\
        --shapes="input":1x3x608x1088\
        --saveEngine=../../weights/fairmot_nano.trt\
        --plugins=./build_nano/DCNv2PluginDyn_nano.so\
        --verbose
