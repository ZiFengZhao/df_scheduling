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

目标是将ONNX模型描述中的DFG切分成细粒度的、适配目标平台的DFG.

