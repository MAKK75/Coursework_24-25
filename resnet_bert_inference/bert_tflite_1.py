import tensorflow as tf
from transformers import TFBertModel

MODEL_NAME = 'bert-base-uncased'
SAVED_MODEL_DIR = "./bert_tf_savedmodel"
TFLITE_FP32_PATH = 'bert_model_fp32.tflite'
TFLITE_FP16_PATH = 'bert_model_fp16.tflite'

#Загрузка
print(f"Загрузка TF модели: {MODEL_NAME}...")
tf_model = TFBertModel.from_pretrained(MODEL_NAME)
print("Модель загружена.")

#Сигнатура
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="input_ids"),
    tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="attention_mask"),
    tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="token_type_ids")
])
def serving_fn(input_ids, attention_mask, token_type_ids):
    output = tf_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        training=False, 
        return_dict=True
    )
    return {"last_hidden_state": output.last_hidden_state}

#Saved_model
print(f"Сохранение модели в SavedModel: {SAVED_MODEL_DIR}...")
tf.saved_model.save(
    tf_model,
    SAVED_MODEL_DIR,
    signatures={'serving_default': serving_fn}
)
print("SavedModel сохранена.")

#TFLite FP32
print(f"Конвертация в TFLite FP32: {TFLITE_FP32_PATH}...")
converter_fp32 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
tflite_model_fp32 = converter_fp32.convert()
with open(TFLITE_FP32_PATH, 'wb') as f:
    f.write(tflite_model_fp32)
print(f"Модель TFLite FP32 сохранена.")

#TFLite FP16
print(f"Конвертация в TFLite Float16: {TFLITE_FP16_PATH}...")
converter_fp16 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]
tflite_model_fp16 = converter_fp16.convert()
with open(TFLITE_FP16_PATH, 'wb') as f:
    f.write(tflite_model_fp16)
print(f"Модель TFLite Float16 сохранена.")

print("\nВсе конвертации завершены.")