import configparser
import logging

import networkx as nx
from networkx import DiGraph
from region_partition import RegionPartitioner
from dfg_gen import DFGGenerator
from utils import *


class SegGenerator:
    def __init__(self, config_file='conf.ini', G=None, verbose=False, energy_costs=None):
        self.energy_cost = energy_costs
        self.config_file = config_file
        self.G = G
        # 将DFG的所有节点按照拓扑顺序放入列表
        topo_order_gen = nx.topological_sort(G)
        self.sorted_node_lst = list(topo_order_gen)
        self.verbose = verbose

        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        if verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        self.logger = logging.getLogger('{}.{}'.format(__name__, 'SegGenerator'))
        self.logger.setLevel(log_level)
        console_handler = logging.StreamHandler()
        self.logger.addHandler(console_handler)

        self.logger.debug("Creating SegGenerator Object")

    def gen_seg_v1(self):
        """
        生成segments，每个segment里包括至少一个节点
        """

        # G = self.G
        segs_dict = {}

        # 收集所有关键节点，创建相应的segments
        critical_nodes_lst = self.collect_critical_nodes()
        # 使用相等性检查确定关键节点列表是否在元素种类、排序上完全被包含于DFG list
        # print(critical_nodes_lst)
        # print("************")
        # print(self.sorted_node_lst[:len(critical_nodes_lst)])

        assert check_sub_lst(critical_nodes_lst, self.sorted_node_lst)
        assert len(critical_nodes_lst) >= 2  # 至少有输入和输出层是关键节点

        for idx in range(0, len(critical_nodes_lst) - 1):
            seg_start_layer_name = critical_nodes_lst[idx]
            seg_end_layer_name = critical_nodes_lst[idx + 1]
            flg = False
            contained_nodes = []

            if idx == 0:
                assert self.sorted_node_lst[0] == seg_start_layer_name

            for node in self.sorted_node_lst:
                if node == seg_start_layer_name:
                    flg = True
                    contained_nodes.append(node)
                    continue
                if node == seg_end_layer_name:
                    flg = False
                    break
                if flg:
                    contained_nodes.append(node)

            seg_name = 'seg' + str(idx)
            segs_dict[seg_name] = contained_nodes
            if self.verbose:
                self.logger.debug("Creating segment {}, which contains the following nodes {}".format(seg_name,
                                                                                                      segs_dict[
                                                                                                          seg_name]))

        return segs_dict

        # Creating a RegionPartitioner object
        # reg_partitioner = RegionPartitioner(config_file=self.config_file, verbose=self.verbose)
        # avail_region_lst = reg_partitioner.collect_regions()

    def collect_critical_nodes(self):

        G = self.G
        # 找到所有关键节点
        critical_nodes_lst = []
        for node_name, node_dict in G.nodes(data=True):
            # 获取该节点的直接后继节点
            succ_nodes = list(G.successors(node_name))
            if len(succ_nodes) > 1:
                critical_nodes_lst.append(node_name)
                if self.verbose:
                    self.logger.debug("Adding critical node {}".format(node_name))

            if G.nodes[node_name]["op_type"] == 'Input' or G.nodes[node_name]["op_type"] == 'Output':
                if node_name not in critical_nodes_lst:
                    critical_nodes_lst.append(node_name)  # 将输入和输出层标记为关键节点
                    if self.verbose:
                        self.logger.debug("Adding critical node {}".format(node_name))

        # 将列表中的关键节点按照DFG的节点前后顺序排序
        sorted_critical_nodes_lst = self.sort_nodes_in_DFG_order(critical_nodes_lst)

        return sorted_critical_nodes_lst

    def sort_nodes_in_DFG_order(self, nodes_lst):
        sorted_nodes_lst = []
        sorted_G = list(nx.topological_sort(self.G))

        for node in sorted_G:
            if node in nodes_lst:
                sorted_nodes_lst.append(node)
                if self.verbose:
                    self.logger.debug("[Sorting] Critical node: {}".format(node))

        return sorted_nodes_lst


if __name__ == '__main__':
    model_name_lst = ['resnet50.onnx']
    path2onnx_model = '../model_zoo/'
    model_path = path2onnx_model + model_name_lst[0]

    config_file = '../configs/hw_cfg.ini'
    verbose = True

    dfg_generator = DFGGenerator(model_path)
    G = dfg_generator.get_nx_graph()

    seg_generator = SegGenerator(config_file, G, verbose)
    segs_dict = seg_generator.gen_seg_v1()
