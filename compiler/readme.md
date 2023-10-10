# compiler
## 1. Introduction
compiler将一个神经网络模型转换成适配于目标数据流平台的数据流图（dataflow graph, DFG).
compiler的输入是一个模型的既定格式的描述文件，输出是一个NetworkX的Graph对象(nn.Graph)。
DFG的每个节点（node）代表一个算子op，边（edge）代表一个张量。

## 2. Implementation
### 2.1 Model description

Pytorch模型通过`torch.onnx.export`方法可以转换成onnx表示。onnx
所描述的模型已经包含了模型的原始DFG，但还无法直接适配目标平台：DFG的op
是一层完整的网络算子，例如卷积层，全连接层。张量也为一层网络的完整输入，
例如卷积层的权重矩阵。数据流架构所要求的最大op对应于一个tile的单次最大算力。
例如tile单次可完成25 MACs/cycle的计算，且输入张量最大宽度为5 elements,
而某一层卷积层op的输入张量size为(1, 3, 32, 32). 则该op无法直接映射到一个
tile上，需要划分为更小的uop，这些uop或空间并行(spatial)映射多个tile上，
或时间顺序(temporal)映射到单个tile上，或spatial和temporal结合。

### 2.2 Graph partitioning

目标是将ONNX模型描述中的DFG切分成细粒度的、适配目标平台的sub-DFG. 
sub-DFG在Tangram中被称为segment.

加载得到一个onnx model后, step 1: 提取model的原始DFG。 step 2: 切分DFG

#### step1：生成原始DFG

可通过`netron`工具在线可视化model的DFG. 

代码方式：通过`<onnx_model>.graph`方法直接得到model的原始DFG为GraphProto
对象。其中包含所有node（为NodeProto)对象和输入输出张量（为ValueInfoProto)对象。
这些元素共同组成原始DFG。通过`graph.node`，`graph.input`和`graph.output`可
分别获取DFG的节点，输入张量和输出张量列表。节点列表中的每个元素包含的信息类型如下：
1. Input list，其中每个元素包括输入张量的唯一名称
2. Output list, 同上
3. name, 本节点唯一名称
4. op_type, 算子类型，如Conv, Relu
5. attribute（有的节点如Conv具有，有些节点如Relu没有）, 节点属性包括dilation（默认为1，可忽略），group, kernel_shape, pads, strides

通过实例化`class DFGGenerator`对象创建DFG生成器，该生成器接收onnx模型路径，
自动将其转换成Networkx图。通过`Fuse_layers`方法进行层融合。目前支持CONV+ReLU融合。

#### step2: Segment划分

实例化`class SegGenerator`对象创建Segment划分器，接收step1生成的DFG和硬件平台配置，
采用穷举法产生所有可行的Segment划分: 


step1: 根据硬件平台配置实例化`class RegionPartitioner`对象创建Region生成器。Region
是硬件架构上的一块可执行一个segment的区域，即一个segment每次只在 一个Region中运行。



Segment的具体生成方法：

1. 首先找到DFG的关键节点。关键节点定义如下：如果一个节点有多个后继节点，说明其ofmap可以被
复用多次，则这个节点为关键节点。Note: 有的模型可能没有关键节点，例如简单模型(LeNet5)
2. 输入和输出层标记为关键节点。
3. 将每两个相邻的关键节点之间的所有网络层划分到一个segment中。
4. 遍历每个seg，对于每个seg，遍历每个region，如果seg在这个region中可直接运行，
计算该seg的推理延时（一次片外读，一次片外写，计算，片上传输）和energy cost，然后将
其放入seg字典中。如果无法一次运行，需要将seg平均拆分成多个sub-segments，同样计算
推理延时（多次片外访存，计算，片上传输），然后放入seg字典中。

#### RegionPartition
`class RegionPartitioner`接受硬件架构配置`conf.ini`，生成所有有效的Region

Region的基本组成单位是Tile，因此需要从配置文件`conf.ini`解析出硬件平台在X，Y两个
维度上的Tile数量(M, N).

为蒸馏搜索空间，我们规定：
1. 将硬件架构划分成一个或多个互不交叠的region.
2. 每个region的形状为一个矩形。最小为一个tile，最大为所有tile.
3. 每个region至少有一个off-chip 访存通道，用于传输在其中运行的segment的输入、输出数据。

因此只需按照上述规则将(M, N)大小的大矩形简单分割成若干小矩形。收集所有小矩形region
和region内的OCM容量。