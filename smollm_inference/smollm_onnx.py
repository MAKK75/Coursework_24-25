import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "./smollm_local_model" 
OUTPUT_ONNX_FILE = "smollm_135m.onnx"
OPSET_VERSION = 11

# Загрузка
print(f"Загрузка модели и токенизатора из: {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
pt_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)


# ПРИВЕДЕНИЕ МОДЕЛИ К FLOAT32 
print("Приведение модели к типу float32...")
pt_model = pt_model.to(torch.float32)
pt_model.eval()

# Отключаем использование KV-кэша, чтобы избежать ошибки с 'DynamicCache'
pt_model.config.use_cache = False


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask):

        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        return outputs.hidden_states[-1]

wrapped_model = ModelWrapper(pt_model)
wrapped_model.eval()

# Фиктивные входные данные
text = "Пример текста для SmolLM."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

dummy_inputs_dict = {
    'input_ids': inputs['input_ids'],
    'attention_mask': inputs['attention_mask']
}

# Входы, выходы, динамические оси
input_names = list(dummy_inputs_dict.keys())
output_names = ['last_hidden_state'] 

dynamic_axes_config = {
    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
    output_names[0]: {0: 'batch_size', 1: 'sequence_length'}
}

# Экспорт
print(f"Экспорт модели из {MODEL_PATH} в {OUTPUT_ONNX_FILE} (opset {OPSET_VERSION})...")
torch.onnx.export(
    wrapped_model,
    args=tuple(dummy_inputs_dict.values()),
    f=OUTPUT_ONNX_FILE,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes_config,
    opset_version=OPSET_VERSION,
    do_constant_folding=True,
    export_params=True
)
print(f"Модель ONNX сохранена как {OUTPUT_ONNX_FILE}")