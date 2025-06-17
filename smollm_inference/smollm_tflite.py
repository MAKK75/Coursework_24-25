import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import os
import subprocess 

ONNX_PATH = "smollm_135m.onnx"
ONNX_SIMPLIFIED_PATH = "smollm_135m.simplified.onnx" 
SAVED_MODEL_DIR = "./smollm_135m_tf_savedmodel"
TFLITE_FP32_PATH = 'smollm_135m_fp32.tflite'
TFLITE_FP16_PATH = 'smollm_135m_fp16.tflite'


#Упрощение ONNX модели
print(f"Проверка наличия onnx-simplifier...")
try:
    subprocess.run(["onnxsim", "-h"], check=True, capture_output=True)
    print("onnx-simplifier найден.")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Ошибка: onnx-simplifier не найден или не работает.")
    print("Пожалуйста, установите его: pip install onnx-simplifier")
    exit()

print(f"Упрощение ONNX модели: {ONNX_PATH} -> {ONNX_SIMPLIFIED_PATH}")
if not os.path.exists(ONNX_PATH):
    print(f"Ошибка: Файл {ONNX_PATH} не найден. Сначала запустите smollm_onnx.py")
    exit()


result = subprocess.run(
    ["onnxsim", ONNX_PATH, ONNX_SIMPLIFIED_PATH],
    capture_output=True, text=True
)

if result.returncode != 0:
    print("Ошибка при упрощении ONNX модели:")
    print(result.stderr)
    exit()
print("ONNX модель успешно упрощена.")


print(f"Загрузка УПРОЩЕННОЙ ONNX модели из: {ONNX_SIMPLIFIED_PATH}...")
onnx_model = onnx.load(ONNX_SIMPLIFIED_PATH)

print("Подготовка к конвертации ONNX в TensorFlow...")
tf_rep = prepare(onnx_model)

print(f"Экспорт в формат SavedModel: {SAVED_MODEL_DIR}...")
tf_rep.export_graph(SAVED_MODEL_DIR)
print("SavedModel успешно сохранена.")



# TFLite FP32
print(f"Конвертация в TFLite FP32: {TFLITE_FP32_PATH}...")
converter_fp32 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter_fp32.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS    
]
tflite_model_fp32 = converter_fp32.convert()
with open(TFLITE_FP32_PATH, 'wb') as f:
    f.write(tflite_model_fp32)
print(f"Модель TFLite FP32 сохранена.")

# TFLite FP16
print(f"Конвертация в TFLite Float16: {TFLITE_FP16_PATH}...")
converter_fp16 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]
converter_fp16.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model_fp16 = converter_fp16.convert()
with open(TFLITE_FP16_PATH, 'wb') as f:
    f.write(tflite_model_fp16)
print(f"Модель TFLite Float16 сохранена.")

print("\nВсе конвертации завершены.")