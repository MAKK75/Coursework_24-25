import time
import numpy as np
import torch
import onnxruntime as ort
import tensorflow as tf
from transformers import BertTokenizer, BertModel
import os
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker 

# Глобальные настройки для ограничения потоков
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
torch.set_num_threads(1)

# Конфигурация
MODEL_NAME = 'bert-base-uncased'
ONNX_PATH = 'bert_model.onnx'
TFLITE_FP32_PATH = 'bert_model_fp32.tflite'
TFLITE_FP16_PATH = 'bert_model_fp16.tflite'
DEVICE = torch.device('cpu')
REPEATS = 20
PLOT_FILENAME = 'inference_times_bert_comparison.png' 

# Инициализация токенизатора и входных данных
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
sample_text = "Hello, this is a test input for benchmarking BERT inference!"
# Используем return_tensors='np' напрямую для numpy_inputs_dict
numpy_inputs_dict = tokenizer(sample_text, return_tensors='np', padding=True, truncation=True, max_length=128)

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
            out_tuple = model(**torch_inputs)
            # Для BertModel основной выход - last_hidden_state
            out = out_tuple.last_hidden_state.cpu().numpy() # Добавил .cpu() на всякий случай
            times.append(time.perf_counter() - start_time)
            if not outputs_list:
                outputs_list.append(out)
    return outputs_list[0], np.mean(times), np.std(times)

def run_onnx(session, inputs_dict_np, repeats=REPEATS):
    onnx_input_details = session.get_inputs()
    input_feed = {}

    for detail in onnx_input_details:
        onnx_name = detail.name
        matched_key = None
        for np_key in inputs_dict_np.keys():
            if np_key in onnx_name:
                matched_key = np_key
                break

        if matched_key:
            data = inputs_dict_np[matched_key]
            # ONNX ожидает int64 для ids/mask, float32 для других float-тензоров
            # Проверяем тип данных в ONNX модели
            if detail.type == 'tensor(int64)' and data.dtype != np.int64:
                input_feed[onnx_name] = data.astype(np.int64)
            elif 'float' in detail.type and data.dtype != np.float32: # e.g. 'tensor(float)'
                input_feed[onnx_name] = data.astype(np.float32)
            else:
                input_feed[onnx_name] = data
        else:
            print(f"Warning for ONNX: Could not map input '{onnx_name}'.")

    if len(input_feed) != len(onnx_input_details):
         print(f"CRITICAL for ONNX: Mismatch in mapped inputs. Expected {len(onnx_input_details)}, got {len(input_feed)}. Cannot proceed.")
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
    initial_input_details = interpreter.get_input_details()

    # Сопоставление имен входов TFLite с ключами в numpy_inputs_dict
    # TFLite часто добавляет префиксы, например, 'serving_default_input_ids:0'
    tflite_input_map = {}
    for detail in initial_input_details:
        tflite_name = detail['name']
        matched_key = None
        # Ищем точное соответствие или частичное (input_ids в serving_default_input_ids)
        for np_key in inputs_dict_np.keys():
            if np_key == tflite_name or np_key in tflite_name:
                matched_key = np_key
                break
        if matched_key:
            tflite_input_map[detail['index']] = (matched_key, inputs_dict_np[matched_key])
        else:
            print(f"Warning TFLite: Could not map input '{tflite_name}' to numpy_inputs_dict keys.")

    if len(tflite_input_map) != len(initial_input_details):
        print(f"CRITICAL for TFLite: Mismatch in mapped inputs. Expected {len(initial_input_details)}, got {len(tflite_input_map)}. Cannot proceed.")
        return None, np.nan, np.nan

    # Resize inputs, если необходимо (обычно для BERT длина последовательности может меняться)
    for detail in initial_input_details:
        if detail['index'] in tflite_input_map:
            _, data_for_shape = tflite_input_map[detail['index']]
            current_model_shape = list(detail['shape'])
            data_shape = list(data_for_shape.shape)

            # Если модель имеет динамические размеры (-1 или None) или форма не совпадает
            is_dynamic_shape = any(s == -1 or s is None for s in current_model_shape)
            if (is_dynamic_shape and current_model_shape != data_shape) or \
               (not is_dynamic_shape and current_model_shape != data_shape):
                try:
                    #print(f"TFLite: Resizing input '{detail['name']}' from {current_model_shape} to {data_shape}")
                    interpreter.resize_tensor_input(detail['index'], data_shape, strict=True)
                except (TypeError, ValueError):
                    try:
                        interpreter.resize_tensor_input(detail['index'], data_shape)
                    except ValueError as e_resize:
                         print(f"Warning TFLite: Resize failed for {detail['name']} to {data_shape}. Error: {e_resize}")

    interpreter.allocate_tensors()
    # Получаем детали после allocate_tensors, так как форма может измениться
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    # Устанавливаем тензоры для warm-up
    for detail in input_details:
        if detail['index'] in tflite_input_map:
            np_key, data = tflite_input_map[detail['index']]
            # Убедимся, что тип данных совпадает с ожидаемым моделью TFLite
            interpreter.set_tensor(detail['index'], data.astype(detail['dtype']))
        else:
             # Эта ситуация не должна возникнуть, если предыдущее сопоставление прошло успешно
             raise ValueError(f"Error TFLite (warm-up): Data for input '{detail['name']}' (index {detail['index']}) not found in tflite_input_map.")


    interpreter.invoke() # Warm-up

    outputs_list = []
    times = []
    for _ in range(repeats):
        for detail in input_details:
            if detail['index'] in tflite_input_map:
                np_key, data = tflite_input_map[detail['index']]
                interpreter.set_tensor(detail['index'], data.astype(detail['dtype']))
            else:
                raise ValueError(f"Error TFLite (loop): Data for input '{detail['name']}' (index {detail['index']}) not found in tflite_input_map.")

        start_time = time.perf_counter()
        interpreter.invoke()
        # Предполагаем, что интересующий выход (last_hidden_state) первый
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
        slicers = tuple(slice(0, min(s_a, s_b)) for s_a, s_b in zip(a_cmp.shape, b_cmp.shape))
        a_sliced, b_sliced = a_cmp[slicers], b_cmp[slicers]
        if a_sliced.size == 0: return np.inf, np.inf, False
        diff = np.abs(a_sliced - b_sliced)
    else:
        diff = np.abs(a_cmp - b_cmp)

    max_d = np.max(diff) if diff.size > 0 else 0.0
    mean_d = np.mean(diff) if diff.size > 0 else 0.0

    is_fp16 = ("FP16" in model_name_a.upper() or "FP16" in model_name_b.upper() or \
               (hasattr(a, 'dtype') and a.dtype == np.float16) or \
               (hasattr(b, 'dtype') and b.dtype == np.float16))

    atol, rtol = (1e-2, 1e-2) if is_fp16 else (5e-6, 1e-5) 
    is_close = np.allclose(a_cmp, b_cmp, atol=atol, rtol=rtol)

    return max_d, mean_d, is_close


def plot_inference_times(results, filename=PLOT_FILENAME, model_name_str=MODEL_NAME):

    labels = []
    means = []
    stds = []

    for framework, (mean, std) in results.items():
        if not np.isnan(mean): 
            labels.append(framework)
            means.append(mean)
            stds.append(std if not np.isnan(std) else 0) 

    if not labels:
        print("Нет данных для построения графика.")
        return

    x = np.arange(len(labels))
    width = 0.5 

    fig, ax = plt.subplots(figsize=(12, 7)) 
    rects = ax.bar(x, means, width, yerr=stds, label='Mean Time', capsize=5,
                   color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(labels)])

    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + (stds[i] if stds[i] else 0) * 0.1 + (0.001 * max(means, default=1)), # Адаптивный отступ
                f'{height:.4f}s',
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Mean Inference Time (seconds)')
    ax.set_title(f'{model_name_str} Inference Speed Comparison (CPU, {REPEATS} repeats)\nMax Seq Length: {numpy_inputs_dict["input_ids"].shape[1]}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

    if means:
        max_val_with_std = max(m + (s if s else 0) for m, s in zip(means, stds))
        ax.set_ylim(0, max_val_with_std * 1.25) 

    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"\nГрафик сохранен в файл: {filename}")
    except Exception as e:
        print(f"Не удалось сохранить график: {e}")
    # plt.show() 

if __name__ == '__main__':
    print(f"Benchmarking: {MODEL_NAME}. Device: {DEVICE} for PyTorch. Repeats: {REPEATS}")
    print(f"Input sequence length: {numpy_inputs_dict['input_ids'].shape[1]}")


    torch_model = BertModel.from_pretrained(MODEL_NAME)
    torch_out, torch_mean, torch_std = run_torch(torch_model, numpy_inputs_dict)

    onnx_out, onnx_mean, onnx_std = None, np.nan, np.nan
    if os.path.exists(ONNX_PATH):
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            ort_session = ort.InferenceSession(ONNX_PATH, sess_options=sess_options, providers=['CPUExecutionProvider'])
            onnx_out, onnx_mean, onnx_std = run_onnx(ort_session, numpy_inputs_dict)
        except Exception as e:
            print(f"Error ONNX: {e}")
    else:
        print(f"ONNX model file not found: {ONNX_PATH}")


    tflite_fp32_out, tflite_fp32_mean, tflite_fp32_std = None, np.nan, np.nan
    if os.path.exists(TFLITE_FP32_PATH):
        try:
            tflite_fp32_interpreter = tf.lite.Interpreter(model_path=TFLITE_FP32_PATH, num_threads=1)
            tflite_fp32_out, tflite_fp32_mean, tflite_fp32_std = run_tflite(tflite_fp32_interpreter, numpy_inputs_dict)
        except Exception as e:
            print(f"Error TFLite FP32: {e}")
    else:
        print(f"TFLite FP32 model file not found: {TFLITE_FP32_PATH}")

    tflite_fp16_out, tflite_fp16_mean, tflite_fp16_std = None, np.nan, np.nan
    if os.path.exists(TFLITE_FP16_PATH):
        try:
            tflite_fp16_interpreter = tf.lite.Interpreter(model_path=TFLITE_FP16_PATH, num_threads=1)
            tflite_fp16_out, tflite_fp16_mean, tflite_fp16_std = run_tflite(tflite_fp16_interpreter, numpy_inputs_dict)
        except Exception as e:
            print(f"Error TFLite FP16: {e}")
    else:
        print(f"TFLite FP16 model file not found: {TFLITE_FP16_PATH}")


    print("\n--- Inference Speed (mean ± std) seconds ---")
    print(f"PyTorch:       {torch_mean:.6f} ± {torch_std:.6f}")
    if onnx_out is not None: print(f"ONNX:          {onnx_mean:.6f} ± {onnx_std:.6f}")
    if tflite_fp32_out is not None: print(f"TFLite FP32:   {tflite_fp32_mean:.6f} ± {tflite_fp32_std:.6f}")
    if tflite_fp16_out is not None: print(f"TFLite FP16:   {tflite_fp16_mean:.6f} ± {tflite_fp16_std:.6f}")

    print("\n--- Output Differences (vs PyTorch baseline) ---")
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
        print("\n--- Output Differences (TFLite FP32 vs TFLite FP16) ---")
        d_max, d_mean, close = compare_outputs(tflite_fp32_out, tflite_fp16_out, "TFLite FP32", "TFLite FP16")
        print(f"TFLite FP32 vs FP16:      MaxAbsDiff: {d_max:.2e}, MeanAbsDiff: {d_mean:.2e}, Close: {close}")

    results_for_plot = {
        "PyTorch": (torch_mean, torch_std),
        "ONNX": (onnx_mean, onnx_std),
        "TFLite FP32": (tflite_fp32_mean, tflite_fp32_std),
        "TFLite FP16": (tflite_fp16_mean, tflite_fp16_std)
    }
    plot_inference_times(results_for_plot, model_name_str=MODEL_NAME)