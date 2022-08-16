# MOTinAR
 **Multi-Object Tracking in Adaptive Region (tentative)**

该项目实现了一个面向物联网端边环境下的多目标跟踪应用并行加速系统，它于内部集成了我所设计开发的三种算法加速方案，并给出了一套带有用户交互界面的客户端程序以测试与查看加速方案的执行效果。

除了本README外，可以阅读项目中的 [paper](https://github.com/liu-mengyang/MOTinAR/blob/main/lmy_CBD2021_final.pdf) 获取更多信息。

# News

- (2022.6.29):3rd_place_medal:本项目工作被江苏省教育厅评选为江苏省优秀毕业论文（设计），获三等奖！🎉🎉🎉
- (2022.3.27):mega:本项目的工作在[CBD2021](https://ist.nwu.edu.cn/dfiles/CBD/index.html)会议上进行了线上报告。[slides](https://github.com/liu-mengyang/MOTinAR/blob/main/slides.pdf)

# 可运行的环境

本项目涉及边缘和终端两台设备，另外你可能还需要一台路由器以提供Wifi连接服务。

## 本README所展示加速效果的硬件测试平台

### 边缘

- GPU：NVIDIA GTX1050
- CPU：Intel i5-7300HQ
- 内存：8G

### 终端

**NVIDIA Jetson Nano设备**

- GPU：NVIDIA Maxwell 128Core
- CPU：ARM A57 4Core
- 内存：2G

## 一些重要的软件环境

### 边缘

- Ubuntu 20.04
- CUDA 11.2
- TensorRT 7.2.3
- Python 3.8.5
- PyTorch 1.8.1
- GCC 9.3.0

### 终端

- JetPack 4.5
- CUDA 10.2
- Python 3.6.9
- PyTorch 1.8.0

# 特性

- 基于FairMOT算法
- 针对多目标跟踪算法特性所设计出了一种时空间执行优化策略
- 在边缘设备上实现了基于TensorRT的模型加速
- 设计使用了一种自适应端边算力和网络动态环境的端边协同机制

# 能够达到的效果

1. 使用时空间执行优化策略，能够在原始处理仅能达到3.05fps的机器上也能获得24.05fps，基本达到实时分析需求，代价是损失12.8%绝对跟踪精度
2. TensorRT引擎能达到1.53的预测模型推理加速比以及1.16的应用处理加速比
3. 自适应端边协同机制的使用相对终端本地执行能达到4.42的加速比，同时具备网络环境适应性
4. 结合使用三种加速方案，通过分别损失3%、7.1%和10.4%的MOTA准确度，能够在本地执行仅能达到0.26FPS的Jetson Nano 2G设备上达到4.58、7.7和9.91FPS的处理吞吐率。

![image-20210919155933612](https://images.liumengyang.xyz/image-20210919155933612.png)

# 安装

### 边缘

1. 安装好CUDA、CuDNN等深度学习底层支持组件

2. 安装一些来自第三方的软件库

   ```
   sh install.sh
   ```

3. 安装一些必要的Python包

   ```
   pip install -r requirements.txt
   ```

4. 下载原版权重 这里给出原版仓库所给出的一些链接 fairmot_dla34.pth [Google] [Baidu, 提取码:uouv] [Onedrive]。下载完成后将权重文件放到weights文件夹下。

   ```
   mv fairmot_dla34.pth ./weights/
   ```

### 终端

1. 在Jetson Nano上安装好JetPack4.5 OS包 [JetPack4.5 Archive](https://developer.nvidia.com/jetpack-sdk-45-archive)

2. 安装一些来自第三方的软件库

   ```
   sh install.sh
   ```

3. 安装一些必要的Python包

   ```
   pip install -r requirements.txt
   ```

4. 下载原版权重 这里给出原版仓库所给出的一些链接 fairmot_dla34.pth [Google](https://drive.google.com/file/d/1SFOhg_vos_xSYHLMTDGFVZBYjo8cr2fG/view?usp=sharing) [Baidu, 提取码:uouv](https://pan.baidu.com/share/init?surl=H1Zp8wrTKDk20_DSPAeEkg) [Onedrive](https://microsoftapc-my.sharepoint.com/:u:/g/personal/v-yifzha_microsoft_com/EUsj0hkTNuhKkj9bo9kE7ZsBpmHvqDz6DylPQPhm94Y08w?e=3OF4XN)。下载完成后将权重文件放到weights文件夹下。

   ```
   mv fairmot_dla34.pth ./weights/
   ```

# 使用

### 边缘

#### 构建TensorRT Engine

1. 编译Plugin

   ```
   cd Software/trt_impl/
   make -j
   ```

2. 导出PyTorch模型为ONNX模型

   ```
   python build_onnx_fairmot.py
   python replace_dcn_plugin.py
   ```

3. 将ONNX模型转化为TensorRT Engine

   ```
   sh build_trt_dyn.sh
   ```

#### 启用客户端程序

1. 运行程序

   ```
   cd ../MOTsc
   python main.py
   ```

2. 跟随[演示视频]()进行交互操作

### 终端

终端上的使用需要配合边缘客户端程序进行，具体交互操作详见[演示视频]()

1. 进入终端程序文件夹

   ```
   cd Software/MOTsc/end
   ```

2. 启动终端client

   ```
   python client.py
   ```

# 相关的项目

**基础优化对象FairMOT**

- 本项目中的PyTorch版本FairMOT代码大部分来自于它的原版作者[ifzhang](https://github.com/ifzhang)的[FairMOT](https://github.com/ifzhang/FairMOT)
- 由于在 PyTorch1.8.1+cu11 版本中不能成功运行来自FairMOT原版仓库的DCNv2算子了，所以使用了一份来自[tteepe](https://github.com/tteepe)的JIT版本DCNv2算子[DCNv2](https://github.com/tteepe/DCNv2)代替它

**TensorRT工作**

- 本项目中的TensorRT加速相关工作也可详见本人的竞赛项目[trt-fairmot](https://github.com/liu-mengyang/trt-fairmot)

# Citation

If this work helps your research or work, please consider citing the following works:

```BibTex
@INPROCEEDINGS{MOTinAR,
   author={Liu, Mengyang and Tang, Anran and Wang, Huitian and Shen, Lin and Chang, Yunhan and Cai, Guangxing and Yin, Daheng and Dong, Fang and Zhao, Wei},
   booktitle={2021 Ninth International Conference on Advanced Cloud and Big Data (CBD)},
   title={Accelerating Multi-Object Tracking in Edge Computing Environment with Time-Spatial Optimization},
   year={2022},
   volume={},
   number={},
   pages={279-284},
   doi={10.1109/CBD54617.2021.00055}
}
```
