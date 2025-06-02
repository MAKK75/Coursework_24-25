import torch
import torchvision

device = torch.device("cpu")

#Предобученный Resnet18: если не нашли веса в torch, то берем веса, предобученные на ImageNet.
try:
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
except AttributeError:
    model = torchvision.models.resnet18(pretrained=True)
model.eval().to(device)

#Фиктивные входные данные
dummy_input = torch.randn(1, 3, 224, 224, device=device)

#Параметры экспорта
output_onnx_file = "resnet18.onnx"
input_names = ["input_tensor"]
output_names = ["output_tensor"]
dynamic_axes_config = {
    input_names[0]: {0: 'batch_size'},
    output_names[0]: {0: 'batch_size'}
}

#Экспорт
print(f"Экспорт Resnet18 в {output_onnx_file}...")
torch.onnx.export(model,
                  dummy_input,
                  output_onnx_file,
                  export_params=True,        
                  opset_version=13,          
                  do_constant_folding=True,  
                  input_names=input_names,
                  output_names=output_names,
                  dynamic_axes=dynamic_axes_config)

print(f"Экспорт Resnet18 успешно завершен. Файл сохранен как {output_onnx_file}")

# Опциональная проверка
# try:
#     import onnxruntime
#     sess = onnxruntime.InferenceSession(output_onnx_file, providers=['CPUExecutionProvider'])
#     print("Проверка ONNX файла с помощью onnxruntime прошла успешно.")
# except ImportError:
#     print("Библиотека onnxruntime не найдена. Пропустите проверку ONNX файла.")
# except Exception as e:
#     print(f"Ошибка при проверке ONNX файла: {e}")