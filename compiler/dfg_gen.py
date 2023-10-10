import networkx as nx
import os
import matplotlib.pyplot as plt
from utils import *


class DFGGenerator:
    def __init__(self, onnx_model_path):
        self.onnx_model_path = onnx_model_path
        self.onnx_model = get_onnx_model(onnx_model_path)
        self.nx_graph = self.create_nx_graph()

    def create_nx_graph(self):
        # 创建一个NetworkX图
        nx_graph = nx.DiGraph()

        graph = self.onnx_model.graph

        # 创建一个NetworkX图
        nx_graph = nx.DiGraph()

        # 遍历Onnx graph中的节点，将其添加到nx图中
        for node in graph.node:
            node_name = node.name
            nx_graph.add_node(node_name)

            # 为节点添加属性
            nx_graph.nodes[node_name]["op_type"] = node.op_type
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    nx_graph.nodes[node_name]["kernel_size"] = list(attr.ints)[0]
                elif attr.name == "pads":
                    nx_graph.nodes[node_name]["pad"] = list(attr.ints)[0]
                elif attr.name == "strides":
                    nx_graph.nodes[node_name]["stride"] = list(attr.ints)[0]

            # 为每个节点添加模型参数（权重和偏置）
            for input_name in node.input:
                for tensor in graph.initializer:
                    if tensor.name == input_name:
                        # 二维字典，索引方式：节点名称-模型参数类型。存储二元组（张量，张量维度)
                        if 'Conv' in input_name:
                            if len(tensor.dims) == 1:
                                nx_graph.nodes[node_name]['Weight'] = (tensor, tensor.dims)
                            else:
                                assert len(tensor.dims) == 4
                                nx_graph.nodes[node_name]['Bias'] = (tensor, tensor.dims)

            # 此时还未添加ifmap和ofmap (前面只为每个节点添加了模型参数）
            # 根据一个节点(node)的输出张量和图中其他节点(dest node)的输入张量是否相同，创建edge代表张量
            for out_tensor_name in node.output:
                for dest_node in graph.node:
                    for in_tensor_name in dest_node.input:
                        if out_tensor_name == in_tensor_name:
                            dest_node_name = dest_node.name
                            nx_graph.add_edge(node_name, dest_node_name)

        # 添加graph的输入输出节点，这两个节点没有模型参数，属性只有op_type"Input/Output"
        assert len(graph.input) == 1  # Resnet50只有1个输入，暂不考虑其他模型
        for i in graph.input:
            nx_graph.add_node(i.name)
            # 为节点添加属性
            nx_graph.nodes[i.name]["op_type"] = 'Input'
            out_tensor_name = i.name  # 输入节点名称等于其输出张量名称
            for j in graph.node:  # 添加edge
                for in_tensor_name in j.input:
                    if out_tensor_name == in_tensor_name:
                        dest_node_name = j.name
                        nx_graph.add_edge(i.name, dest_node_name)
                        break

        assert len(graph.output) == 1  # Resnet50只有1个输输出，暂不考虑其他模型
        for o in graph.output:
            nx_graph.add_node(o.name)
            nx_graph.nodes[o.name]["op_type"] = "Output"
            in_tensor_name = o.name  # 输出节点名称等于其输入张量名称
            for j in graph.node:  # 添加edge
                for out_tensor_name in j.output:
                    if out_tensor_name == in_tensor_name:
                        src_node_name = j.name
                        nx_graph.add_edge(src_node_name, o.name)
                        break

        return nx_graph

    def get_nx_graph(self):
        """
        :return: 返回生成的DFG
        """
        return self.nx_graph

    def fuse_layers(self):
        """
        目前只支持将相邻的CONV和ReLU层融合
        """
        G = self.nx_graph
        nodes_to_remove = []  # 记录需要删除的节点
        nodes_to_add = []  # 记录需要添加的节点
        in_edges_to_add = {}  # 记录需要添加的入边
        out_edges_to_add = {}  # 记录需要添加的出边

        for idx, (node_name, node_dict) in enumerate(G.nodes(data=True)):
            op_type = node_dict["op_type"]
            if op_type == 'Conv':
                # 获取该节点的直接后继节点
                succ_nodes = list(G.successors(node_name))

                # 考察该节点的所有直接后继节点（即下一层网络）是否有且只有一个Relu层
                succ_node_name = succ_nodes[0]
                if len(succ_nodes) == 1 and G.nodes[succ_node_name]['op_type'] == 'Relu':
                    relu_node = succ_node_name  # 获取当前ReLU节点

                    # 执行融合操作
                    # 添加新的融合层的信息到 nodes_to_add
                    fused_node = ('FusedConvRelu_' + str(idx), {
                        'op_type': 'FusedConvRelu',
                        'kernel_size': node_dict['kernel_size'],
                        'pad': node_dict['pad'],
                        'stride': node_dict['stride']
                    })
                    nodes_to_add.append(fused_node)

                    # 记录需要删除的节点
                    nodes_to_remove.append(node_name)
                    nodes_to_remove.append(relu_node)

                    # 记录需要添加的边
                    in_edges_to_add[fused_node[0]] = list(G.predecessors(node_name))  # 入边
                    out_edges_to_add[fused_node[0]] = list(G.successors(relu_node))  # 出边

        # 在循环外执行添加和删除操作
        for node_name, node_attrs in nodes_to_add:
            G.add_node(node_name, **node_attrs)
            pred_nodes_lst = in_edges_to_add[node_name]
            for pred_node in pred_nodes_lst:
                G.add_edge(pred_node, node_name)  # 入边
            succ_nodes_lst = out_edges_to_add[node_name]
            for succ_node in succ_nodes_lst:
                G.add_edge(node_name, succ_node)  # 出边

        for node_to_remove in nodes_to_remove:
            G.remove_node(node_to_remove)


if __name__ == '__main__':
    model_name_lst = ['resnet50.onnx']
    path2onnx_model = '../model_zoo/'
    model_path = path2onnx_model + model_name_lst[0]

    dfg_generator = DFGGenerator(model_path)
    G = dfg_generator.get_nx_graph()

    # 获取节点信息
    for idx, (node_name, node_dict) in enumerate(G.nodes(data=True)):
        if idx == len(G.nodes) - 1:
            # 提取第一个节点的kernel_shape、pads和strides
            if 'kernel_size' in node_dict:
                kernel_shape = node_dict['kernel_size']
            else:
                kernel_shape = None
            if 'pad' in node_dict:
                pads = node_dict['pad']
            else:
                pads = None
            if 'stride' in node_dict:
                strides = node_dict['stride']
            else:
                strides = None
            op = node_dict['op_type']

            enable_debug_output = False
            if enable_debug_output:
                # 输出提取的信息
                print("Node name:", node_name)
                print("OP type:", op)
                print("Kernel Shape:", kernel_shape)
                print("Pads:", pads)
                print("Strides:", strides)

    # 检查是否启用调试输出的标志
    enable_debug_output = bool(os.environ.get("ENABLE_DEBUG_OUTPUT", False))
    # 如果启用了调试输出标志，添加调试输出
    if enable_debug_output:
        for node in G.nodes():
            print("Node:", node)

            # 获取节点的输入边
            in_edges = list(G.in_edges(node))
            if in_edges:
                print("  Input Edges:", in_edges)

            # 获取节点的输出边
            out_edges = list(G.out_edges(node))
            if out_edges:
                print("  Output Edges:", out_edges)

    # 测试fuse_layers
    dfg_generator.fuse_layers()
    enable_debug_output = True
    if enable_debug_output:
        for node_name, node_dict in G.nodes(data=True):
            print("Node:", node_name)
            if node_name == 'FusedConvRelu_0':
                pred_node = list(G.predecessors(node_name))
                succ_node = list(G.successors(node_name))
                print("FusedConvRelu_0's pred, succ node:", pred_node, succ_node)
