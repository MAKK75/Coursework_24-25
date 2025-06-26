import time
import numpy as np
import torch
import torchvision.models as models
import onnxruntime as ort
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Глобальные настройки для ограничения потоков
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Конфигурация
ONNX_PATH = 'resnet18.onnx'
TFLITE_FP32_PATH = 'resnet18_tf/resnet18_float32.tflite'
TFLITE_FP16_PATH = 'resnet18_tf/resnet18_float16.tflite'
DEVICE = torch.device('cpu')
REPEATS = 30
DUMMY_INPUT_SHAPE = (1, 3, 224, 224) 
MODEL_NAME_FOR_PLOT = 'resnet'
PLOT_FILENAME = 'inference_times_resnet_comparison.png' 

dummy_image_np = np.random.rand(*DUMMY_INPUT_SHAPE).astype(np.float32)

def run_torch(model, image_np_array, repeats=REPEATS):
    model.to(DEVICE).eval()
    torch_input = torch.tensor(image_np_array).to(DEVICE)
    with torch.no_grad():
        _ = model(torch_input) # Warm-up

    outputs = []
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            start_time = time.perf_counter()
            out_tensor = model(torch_input)
            out = out_tensor.cpu().numpy()
            times.append(time.perf_counter() - start_time)
            if not outputs:
                outputs.append(out)
    return outputs[0], np.mean(times), np.std(times)

def run_onnx(session, inputs_dict_np, repeats=REPEATS):
    onnx_input_details = session.get_inputs()
    model_input_name = onnx_input_details[0].name
    data = list(inputs_dict_np.values())[0]

    expected_type_str = onnx_input_details[0].type
    if expected_type_str == 'tensor(int64)' and data.dtype != np.int64:
        input_feed = {model_input_name: data.astype(np.int64)}
    elif ('tensor(float)' in expected_type_str) and data.dtype != np.float32:
        input_feed = {model_input_name: data.astype(np.float32)}
    else:
        input_feed = {model_input_name: data}

    output_names = [out.name for out in session.get_outputs()]
    _ = session.run(output_names, input_feed) # Warm-up

    outputs = []
    times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        out_list = session.run(output_names, input_feed)
        times.append(time.perf_counter() - start_time)
        if not outputs:
            outputs.append(out_list[0])
    return outputs[0], np.mean(times), np.std(times)

def run_tflite(interpreter, inputs_dict_np, repeats=REPEATS):
    initial_input_detail = interpreter.get_input_details()[0]
    data_to_use = list(inputs_dict_np.values())[0]

    current_model_shape = initial_input_detail['shape']
    if not np.array_equal(current_model_shape, data_to_use.shape) and \
       (any(s == -1 or s is None for s in current_model_shape) or \
        (initial_input_detail.get('shape_signature') is not None and \
         not np.array_equal(initial_input_detail['shape_signature'], data_to_use.shape))):
        try:
            interpreter.resize_tensor_input(initial_input_detail['index'], list(data_to_use.shape), strict=True)
        except (TypeError, ValueError):
             try:
                interpreter.resize_tensor_input(initial_input_detail['index'], list(data_to_use.shape))
             except ValueError as e:
                print(f"Warning: TFLite resize_tensor_input failed for {initial_input_detail['name']} to shape {list(data_to_use.shape)}. Error: {e}")

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], data_to_use.astype(input_details[0]['dtype']))
    interpreter.invoke() # Warm-up

    outputs = []
    times = []
    for _ in range(repeats):
        interpreter.set_tensor(input_details[0]['index'], data_to_use.astype(input_details[0]['dtype']))
        start_time = time.perf_counter()
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.perf_counter() - start_time)
        if not outputs:
            outputs.append(out)
    return outputs[0], np.mean(times), np.std(times)

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

    atol, rtol = (1e-2, 1e-2) if is_fp16 else (5e-5, 1e-4)
    is_close = np.allclose(a_cmp, b_cmp, atol=atol, rtol=rtol)

    return max_d, mean_d, is_close

def plot_inference_times(results, filename=PLOT_FILENAME):

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

    fig, ax = plt.subplots(figsize=(10, 6)) 
    rects = ax.bar(x, means, width, yerr=stds, label='Mean Time', capsize=5,
                   color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(labels)]) 

   
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + (stds[i] if stds[i] else 0) * 0.1 + 0.001, 
                f'{height:.4f}s',
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Mean Inference Time (seconds)')
    ax.set_title(f'Inference Speed Comparison (CPU, {REPEATS} repeats)\nInput: {DUMMY_INPUT_SHAPE}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    
    if means:
        max_val_with_std = max(m + (s if s else 0) for m, s in zip(means, stds))
        ax.set_ylim(0, max_val_with_std * 1.2) 

    plt.tight_layout() 
    try:
        plt.savefig(filename)
        print(f"\nГрафик сохранен в файл: {filename}")
    except Exception as e:
        print(f"Не удалось сохранить график: {e}")
    # plt.show()

if __name__ == '__main__':
    print(f"Using PyTorch device: {DEVICE}. Input: {DUMMY_INPUT_SHAPE}, {REPEATS} repeats.")
    dummy_image_tflite_fmt_np = dummy_image_np.transpose(0, 2, 3, 1) # NHWC for TFLite

    torch_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    torch_out, torch_mean, torch_std = run_torch(torch_model, dummy_image_np)

    onnx_out, onnx_mean, onnx_std = None, np.nan, np.nan
    try:
        
        if not os.path.exists(ONNX_PATH):
            print(f"Error ONNX: Файл {ONNX_PATH} не найден.")
        else:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            ort_session = ort.InferenceSession(ONNX_PATH, sess_options=sess_options, providers=['CPUExecutionProvider'])
            onnx_input_name = ort_session.get_inputs()[0].name
            onnx_out, onnx_mean, onnx_std = run_onnx(ort_session, {onnx_input_name: dummy_image_np})
    except Exception as e: print(f"Error ONNX: {e}")

    tflite_fp32_out, tflite_fp32_mean, tflite_fp32_std = None, np.nan, np.nan
    try:
        if not os.path.exists(TFLITE_FP32_PATH):
            print(f"Error TFLite FP32: Файл {TFLITE_FP32_PATH} не найден.")
        else:
            tflite_fp32_interpreter = tf.lite.Interpreter(model_path=TFLITE_FP32_PATH, num_threads=1)
            tflite_fp32_input_name = tflite_fp32_interpreter.get_input_details()[0]['name']
            tflite_fp32_out, tflite_fp32_mean, tflite_fp32_std = run_tflite(tflite_fp32_interpreter, {tflite_fp32_input_name: dummy_image_tflite_fmt_np})
    except Exception as e: print(f"Error TFLite FP32: {e}")

    tflite_fp16_out, tflite_fp16_mean, tflite_fp16_std = None, np.nan, np.nan
    try:
        if not os.path.exists(TFLITE_FP16_PATH):
            print(f"Error TFLite FP16: Файл {TFLITE_FP16_PATH} не найден.")
        else:
            tflite_fp16_interpreter = tf.lite.Interpreter(model_path=TFLITE_FP16_PATH, num_threads=1)
            tflite_fp16_input_name = tflite_fp16_interpreter.get_input_details()[0]['name']
            tflite_fp16_out, tflite_fp16_mean, tflite_fp16_std = run_tflite(tflite_fp16_interpreter, {tflite_fp16_input_name: dummy_image_tflite_fmt_np})
    except Exception as e: print(f"Error TFLite FP16: {e}")

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
    plot_inference_times(results_for_plot)

    print("\nСохранение эталонного выхода PyTorch...")
    os.makedirs('./pytorch_outputs', exist_ok=True) 
    np.save(f'./pytorch_outputs/torch_output_{MODEL_NAME_FOR_PLOT}.npy', torch_out) 
    print(f"Эталонный выход сохранен в ./pytorch_outputs/torch_output_{MODEL_NAME_FOR_PLOT}.npy")
