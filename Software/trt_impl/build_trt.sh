trtexec --onnx=fairmot_plugin.onnx\
        --explicitBatch \
        --minShapes="input":1x3x608x1088\
        --optShapes="input":1x3x608x1088\
        --maxShapes="input":1x3x608x1088\
        --shapes="input":1x3x608x1088\
        --saveEngine=../../weights/fairmot.trt\
        --plugins=./build/DCNv2PluginDyn.so\
        --verbose
