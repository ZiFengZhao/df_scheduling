import onnx
import copy
from onnx import ModelProto


def get_onnx_model(model_name: str):
    model = onnx.load_model(model_name)
    return model


def get_origin_dfg(model: ModelProto):
    """This func extracts the original dfg from an onnx model"""
    graph = model.graph
    nodes = graph.node
    inputs = graph.input
    outputs = graph.output

    dfg = (graph, nodes, inputs, outputs)
    return dfg


def check_sub_lst(lst_a: list, lst_b: list):
    """This func checks whether lst_a is totally contained by lst_b in the respect of
    elements and their order in the list"""
    result = True
    flg = 0
    for i in range(0, len(lst_a)):
        for j in range(flg, len(lst_b)):
            if lst_a[i] == lst_b[j]:
                flg = j + 1
                break
        else:
            result = False
            break

    return result


if __name__ == '__main__':
    #  test check_sub_lst
    lst_a = [1, 2, 3]
    lst_b = [1, 5, 2, 4, 3, 6]  # totally contains lst_a
    lst_c = [1, 3, 2, 4]

    rst_ab = check_sub_lst(lst_a, lst_b)
    rst_ac = check_sub_lst(lst_a, lst_c)

    if rst_ab:
        print("lst a is totally contained in lst b")
    else:
        print("lst a is not totally contained in lst b")

    if rst_ac:
        print("lst a is totally contained in lst c")
    else:
        print("lst a is not totally contained in lst c")
