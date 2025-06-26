import openvino as ov
import numpy as np
import argparse
from transformers import AutoTokenizer, BertTokenizer
import os
import nncf

def create_calibration_dataset(model, model_type, num_samples=300):
    dataset = []
    print(f"Создание {num_samples} калибровочных сэмплов для {model_type.upper()}...")
    
    input_names = [inp.get_any_name() for inp in model.inputs]
    
    if model_type == "resnet":
        shape = [1, 3, 224, 224]
        input_name = input_names[0] 
        for _ in range(num_samples):
            dataset.append({input_name: np.random.rand(*shape).astype(np.float32)}) 
            
    elif model_type in ["bert", "smollm"]:
        if model_type == "bert": tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./.cache')
        else:
            try: tokenizer = AutoTokenizer.from_pretrained('../models/smollm_local_model')
            except Exception as e: print(f"Ошибка токенизатора: {e}"); return None
        
        for i in range(num_samples):
            inputs_dict = tokenizer(f"Sample text for calibration {i}", return_tensors='np', padding='max_length', truncation=True, max_length=128)
            sample = {key: tensor for key, tensor in inputs_dict.items() if key in input_names}
            dataset.append(sample)
            
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Квантизация моделей."); parser.add_argument("--model_type", type=str, required=True, choices=["resnet", "bert", "smollm"])
    args = parser.parse_args()
    model_name_map = {"resnet": "resnet18/resnet18_fp32", "bert": "bert/bert_fp32", "smollm": "smollm/smollm_fp32"}
    model_short_name = model_name_map[args.model_type].split('/')[0]
    fp32_model_path = f"../models/ir_fp32/{model_name_map[args.model_type]}.xml"
    int8_model_dir = f"../models/ir_int8/{model_short_name}"
    int8_model_path = os.path.join(int8_model_dir, f"{model_short_name}_int8.xml")
    print(f"--- Запуск INT8 квантизации для {args.model_type.upper()} ---")
    core = ov.Core()
    model = core.read_model(fp32_model_path)
    calibration_dataset = create_calibration_dataset(model, args.model_type)
    if not calibration_dataset: exit("Не удалось создать датасет.")
    print("\nЗапуск квантизации...")
    nncf_calibration_dataset = nncf.Dataset(calibration_dataset)
    quantization_kwargs = {}
    if args.model_type in ["bert", "smollm"]:
        print("Применение параметров для трансформеров..."); quantization_kwargs['model_type'] = nncf.ModelType.TRANSFORMER
    quantized_model = nncf.quantize(model, nncf_calibration_dataset, preset=nncf.QuantizationPreset.PERFORMANCE, **quantization_kwargs)
    print("Квантизация завершена!")
    os.makedirs(int8_model_dir, exist_ok=True)
    ov.save_model(quantized_model, int8_model_path)
    print(f"\n--- УСПЕХ! Квантованная модель сохранена в {int8_model_path} ---")