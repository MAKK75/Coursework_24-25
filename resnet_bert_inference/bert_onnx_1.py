import torch
from transformers import BertTokenizer, BertModel

MODEL_NAME = "bert-base-uncased"
OUTPUT_ONNX_FILE = "bert_model.onnx"
OPSET_VERSION = 14 

#Загрузка
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
pt_model = BertModel.from_pretrained(MODEL_NAME)
pt_model.eval() 

# Фиктивные входные данные
texts = ["Пример текста для BERT.", "Еще один пример."]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)


dummy_input_tuple_for_export = (
    inputs['input_ids'],
    inputs['attention_mask'],
    inputs['token_type_ids']
)

#Входы, выходы, динамические оси
input_names = ['input_ids', 'attention_mask', 'token_type_ids']
output_names = ['last_hidden_state', 'pooler_output']

dynamic_axes_config = {}
for name in input_names:
    dynamic_axes_config[name] = {0: 'batch_size', 1: 'sequence_length'}

dynamic_axes_config['last_hidden_state'] = {0: 'batch_size', 1: 'sequence_length'}
dynamic_axes_config['pooler_output'] = {0: 'batch_size'}


#Экспорт
print(f"Экспорт модели {MODEL_NAME} в {OUTPUT_ONNX_FILE} (opset {OPSET_VERSION})...")
torch.onnx.export(
    pt_model,
    dummy_input_tuple_for_export,
    OUTPUT_ONNX_FILE,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes_config,
    opset_version=OPSET_VERSION,
    do_constant_folding=True, 
    export_params=True        
)
print(f"Модель ONNX сохранена как {OUTPUT_ONNX_FILE}")

# Опционально: проверить, что выходы модели соответствуют ожидаемым output_names
# with torch.no_grad():
#     pt_outputs = pt_model(**inputs)
# print("PyTorch model output keys:", pt_outputs.keys())
# убедимся, что 'last_hidden_state' и 'pooler_output' действительно есть.