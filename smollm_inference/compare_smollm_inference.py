import time
import numpy as np
import torch
import onnxruntime as ort
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Глобальные настройки для ограничения потоков (важно для честного сравнения на CPU)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
torch.set_num_threads(1)

#Конфигурация
MODEL_PATH = './smollm_local_model'
ONNX_PATH = 'smollm_135m.simplified.onnx' 
TFLITE_FP32_PATH = 'smollm_135m_fp32.tflite'
TFLITE_FP16_PATH = 'smollm_135m_fp16.tflite'
MODEL_NAME_FOR_PLOT = 'SmolLM-135M'
DEVICE = torch.device('cpu')
REPEATS = 20
PLOT_FILENAME = 'inference_times_smollm_comparison.png'


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

print(f"Загрузка токенизатора из {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
sample_text = "Hello, this is a test input for benchmarking SmolLM inference!"
numpy_inputs_dict = tokenizer(sample_text, return_tensors='np', padding='max_length', truncation=True, max_length=128)
if 'token_type_ids' in numpy_inputs_dict:
    del numpy_inputs_dict['token_type_ids']


def prepare_torch_inputs(inputs_dict_np):
    return {k: torch.tensor(v).to(DEVICE) for k, v in inputs_dict_np.items()}

def run_torch(model, inputs_dict_np, repeats=REPEATS):
    model.to(DEVICE).eval()
    torch_inputs = prepare_torch_inputs(inputs_dict_np)

    with torch.no_grad():
        _ = model(**torch_inputs) # Warm-up

    outputs_list = []
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            start_time = time.perf_counter()
            out_tensor = model(**torch_inputs)
            out = out_tensor.cpu().numpy()
            times.append(time.perf_counter() - start_time)
            if not outputs_list:
                outputs_list.append(out)
    return outputs_list[0], np.mean(times), np.std(times)

def run_onnx(session, inputs_dict_np, repeats=REPEATS):
    input_feed = {}
    for onnx_input in session.get_inputs():
        name = onnx_input.name
        if name in inputs_dict_np:
            data = inputs_dict_np[name]
            if onnx_input.type == 'tensor(int64)' and data.dtype != np.int64:
                 input_feed[name] = data.astype(np.int64)
            else:
                 input_feed[name] = data
        else:
            print(f"Warning for ONNX: Could not map input '{name}'.")
    
    if len(input_feed) != len(session.get_inputs()):
         print(f"CRITICAL for ONNX: Mismatch in mapped inputs. Expected {len(session.get_inputs())}, got {len(input_feed)}. Cannot proceed.")
         return None, np.nan, np.nan

    output_names = [out.name for out in session.get_outputs()]
    _ = session.run(output_names, input_feed) # Warm-up

    outputs_list = []
    times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        out_run = session.run(output_names, input_feed)
        out = out_run[0] 
        times.append(time.perf_counter() - start_time)
        if not outputs_list:
            outputs_list.append(out)
    return outputs_list[0], np.mean(times), np.std(times)

def run_tflite(interpreter, inputs_dict_np, repeats=REPEATS):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    

    tflite_input_map = {}
    for detail in input_details:
        tflite_name = detail['name']
        matched_key = None
        for np_key in inputs_dict_np.keys():
            if np_key in tflite_name: 
                matched_key = np_key
                break
        if matched_key:
            tflite_input_map[detail['index']] = inputs_dict_np[matched_key]
        else:
            print(f"Warning TFLite: Could not map input '{tflite_name}'.")

    if len(tflite_input_map) != len(input_details):
        print(f"CRITICAL for TFLite: Mismatch in mapped inputs. Expected {len(input_details)}, got {len(tflite_input_map)}. Cannot proceed.")
        return None, np.nan, np.nan
    

    for detail in input_details:
        input_data = tflite_input_map[detail['index']]
        if not np.array_equal(detail['shape'], input_data.shape):
            interpreter.resize_tensor_input(detail['index'], input_data.shape, strict=False)

    interpreter.allocate_tensors()
    
    # Warm-up
    for detail in input_details:
        interpreter.set_tensor(detail['index'], tflite_input_map[detail['index']].astype(detail['dtype']))
    interpreter.invoke() 

    outputs_list = []
    times = []
    for _ in range(repeats):
        for detail in input_details:
             interpreter.set_tensor(detail['index'], tflite_input_map[detail['index']].astype(detail['dtype']))
        
        start_time = time.perf_counter()
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.perf_counter() - start_time)
        if not outputs_list:
            outputs_list.append(out)
            
    return outputs_list[0], np.mean(times), np.std(times)

def compare_outputs(a, b, model_name_a, model_name_b):
    if a is None or b is None: return np.inf, np.inf, False

    a_cmp = a.astype(np.float32) if hasattr(a, 'dtype') and a.dtype == np.float16 else a
    b_cmp = b.astype(np.float32) if hasattr(b, 'dtype') and b.dtype == np.float16 else b

    if a_cmp.shape != b_cmp.shape:
        print(f"Shape mismatch! {model_name_a}: {a_cmp.shape}, {model_name_b}: {b_cmp.shape}")
        return np.inf, np.inf, False

    diff = np.abs(a_cmp - b_cmp)
    max_d = np.max(diff)
    mean_d = np.mean(diff)

    is_fp16 = "FP16" in model_name_a.upper() or "FP16" in model_name_b.upper()
    atol, rtol = (1e-2, 1e-2) if is_fp16 else (5e-6, 1e-5) 
    is_close = np.allclose(a_cmp, b_cmp, atol=atol, rtol=rtol)

    return max_d, mean_d, is_close

def plot_inference_times(results, filename=PLOT_FILENAME, model_name_str=MODEL_NAME_FOR_PLOT):
    labels = [k for k, v in results.items() if not np.isnan(v[0])]
    means = [v[0] for k, v in results.items() if not np.isnan(v[0])]
    stds = [v[1] if not np.isnan(v[1]) else 0 for k, v in results.items() if not np.isnan(v[0])]

    if not labels:
        print("Нет данных для построения графика.")
        return

    x = np.arange(len(labels))
    width = 0.5 

    fig, ax = plt.subplots(figsize=(12, 7))
    rects = ax.bar(x, means, width, yerr=stds, capsize=5, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])

    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + stds[i]*1.1, f'{height:.4f}s', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Среднее время инференса (секунды)')
    ax.set_title(f'Сравнение скорости инференса {model_name_str} (CPU, {REPEATS} повторов)\nДлина последовательности: {numpy_inputs_dict["input_ids"].shape[1]}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.set_ylim(0, max(means) * 1.3)
    
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"\nГрафик сохранен в файл: {filename}")
    except Exception as e:
        print(f"Не удалось сохранить график: {e}")


if __name__ == '__main__':
    print(f"Benchmarking: {MODEL_NAME_FOR_PLOT}. Device: {DEVICE} for PyTorch. Repeats: {REPEATS}")
    print(f"Input sequence length: {numpy_inputs_dict['input_ids'].shape[1]}")
    print("-" * 30)

    #PyTorch
    print("Загрузка PyTorch модели для бенчмаркинга...")
    pt_base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    pt_base_model = pt_base_model.to(torch.float32) # Приведение к FP32, как в скрипте экспорта
    pt_base_model.config.use_cache = False 
    torch_model = ModelWrapper(pt_base_model) 
    torch_out, torch_mean, torch_std = run_torch(torch_model, numpy_inputs_dict)
    
    #ONNX
    onnx_out, onnx_mean, onnx_std = None, np.nan, np.nan
    if os.path.exists(ONNX_PATH):
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            ort_session = ort.InferenceSession(ONNX_PATH, sess_options=sess_options, providers=['CPUExecutionProvider'])
            onnx_out, onnx_mean, onnx_std = run_onnx(ort_session, numpy_inputs_dict)
        except Exception as e:
            print(f"Ошибка при запуске ONNX: {e}")
    else:
        print(f"Файл ONNX модели не найден: {ONNX_PATH}. Запустите smollm_onnx.py и smollm_tflite.py (для упрощения).")

    #TFLite FP32
    tflite_fp32_out, tflite_fp32_mean, tflite_fp32_std = None, np.nan, np.nan
    if os.path.exists(TFLITE_FP32_PATH):
        try:

            tflite_fp32_interpreter = tf.lite.Interpreter(model_path=TFLITE_FP32_PATH, num_threads=1)
            tflite_fp32_out, tflite_fp32_mean, tflite_fp32_std = run_tflite(tflite_fp32_interpreter, numpy_inputs_dict)
        except Exception as e:
            print(f"Ошибка при запуске TFLite FP32: {e}")
    else:
        print(f"Файл TFLite FP32 модели не найден: {TFLITE_FP32_PATH}. Запустите smollm_tflite.py.")

    #TFLite FP16
    tflite_fp16_out, tflite_fp16_mean, tflite_fp16_std = None, np.nan, np.nan
    if os.path.exists(TFLITE_FP16_PATH):
        try:
            tflite_fp16_interpreter = tf.lite.Interpreter(model_path=TFLITE_FP16_PATH, num_threads=1)
            tflite_fp16_out, tflite_fp16_mean, tflite_fp16_std = run_tflite(tflite_fp16_interpreter, numpy_inputs_dict)
        except Exception as e:
            print(f"Ошибка при запуске TFLite FP16: {e}")
    else:
        print(f"Файл TFLite FP16 модели не найден: {TFLITE_FP16_PATH}. Запустите smollm_tflite.py.")


    print("\n--- Скорость инференса (среднее ± ст. отклонение) в секундах ---")
    print(f"PyTorch:       {torch_mean:.6f} ± {torch_std:.6f}")
    if onnx_out is not None: print(f"ONNX:          {onnx_mean:.6f} ± {onnx_std:.6f}")
    if tflite_fp32_out is not None: print(f"TFLite FP32:   {tflite_fp32_mean:.6f} ± {tflite_fp32_std:.6f}")
    if tflite_fp16_out is not None: print(f"TFLite FP16:   {tflite_fp16_mean:.6f} ± {tflite_fp16_std:.6f}")

    print("\n--- Разница в выходах (относительно PyTorch) ---")
    if torch_out is not None:
        if onnx_out is not None:
            d_max, d_mean, close = compare_outputs(torch_out, onnx_out, "PyTorch", "ONNX")
            print(f"ONNX vs PyTorch:          MaxAbsDiff: {d_max:.2e}, MeanAbsDiff: {d_mean:.2e}, Close: {close}")
        if tflite_fp32_out is not None:
            d_max, d_mean, close = compare_outputs(torch_out, tflite_fp32_out, "PyTorch", "TFLite FP32")
            print(f"TFLite FP32 vs PyTorch:   MaxAbsDiff: {d_max:.2e}, MeanAbsDiff: {d_mean:.2e}, Close: {close}")
        if tflite_fp16_out is not None:
            d_max, d_mean, close = compare_outputs(torch_out, tflite_fp16_out, "PyTorch", "TFLite FP16")
            print(f"TFLite FP16 vs PyTorch:   MaxAbsDiff: {d_max:.2e}, MeanAbsDiff: {d_mean:.2e}, Close: {close}")

    if tflite_fp32_out is not None and tflite_fp16_out is not None:
        print("\n--- Разница в выходах (TFLite FP32 vs TFLite FP16) ---")
        d_max, d_mean, close = compare_outputs(tflite_fp32_out, tflite_fp16_out, "TFLite FP32", "TFLite FP16")
        print(f"TFLite FP32 vs FP16:      MaxAbsDiff: {d_max:.2e}, MeanAbsDiff: {d_mean:.2e}, Close: {close}")


    results_for_plot = {
        "PyTorch": (torch_mean, torch_std),
        "ONNX": (onnx_mean, onnx_std),
        "TFLite FP32": (tflite_fp32_mean, tflite_fp32_std),
        "TFLite FP16": (tflite_fp16_mean, tflite_fp16_std)
    }
    plot_inference_times(results_for_plot)