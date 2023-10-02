import torch
import torchvision.models as cv_models
from torchvision.models import ResNet50_Weights

# Resnet
resnet = cv_models.resnet50(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))


def convert2onnx(model_name, model_dict, input_tensor):
    model = model_dict[model_name]
    new_model = model_name + '.onnx'
    with torch.no_grad():
        torch.onnx.export(model, input_tensor, new_model)

    return new_model


def get_input_tensor(model_name: str):
    print('Generating {}\'s input tensor:'.format(model_name))
    if model_name == 'resnet50':
        tensor_size = [1, 3, 224, 224]
        x = torch.randn(tensor_size)
    else:
        raise RuntimeError("Unsupported model!")

    return x


def main():
    model_name_lst = ['resnet50']
    model_dict = {'resnet50': resnet}
    input_tensor_dict = {}

    for n in model_name_lst:
        input_tensor = get_input_tensor(n)
        input_tensor_dict[n] = input_tensor

    onnx_model_dict = {}
    for name, model in model_dict:
        onnx_model = convert2onnx(name, model_dict, input_tensor_dict[name])
        onnx_model_dict[name] = onnx_model


if '__name__' == 'main()':
    main()
