import time
import numpy as np
import openvino as ov
import os
import argparse
from transformers import AutoTokenizer, BertTokenizer
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def compare_outputs(base_output, other_output, base_name="FP32", other_name="Other"):
    if base_output is None or other_output is None:
        return np.nan, np.nan
    
    base_fp32 = base_output.astype(np.float32)
    other_fp32 = other_output.astype(np.float32)
    
    if base_fp32.shape != other_fp32.shape:
        print(f"ОШИБКА: Расхождение в формах тензоров! {base_name}: {base_fp32.shape}, {other_name}: {other_fp32.shape}")
        return np.nan, np.nan
        
    diff = np.abs(base_fp32 - other_fp32)
    max_abs_diff = np.max(diff)
    mean_abs_diff = np.mean(diff)
    
    return max_abs_diff, mean_abs_diff

def plot_benchmark_results(results, filename, title):
    labels = list(results.keys())
    means = [res['time'][0] for res in results.values()]
    stds = [res['time'][1] for res in results.values()]
    
    valid_indices = [i for i, m in enumerate(means) if m is not None and not np.isnan(m)]
    if not valid_indices:
        print("Нет данных для построения графика.")
        return

    labels = [labels[i] for i in valid_indices]
    means = [means[i] for i in valid_indices]
    stds = [stds[i] if stds[i] is not None else 0 for i in valid_indices]

    x = np.arange(len(labels))
    width = 0.45
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    rects = ax1.bar(x, means, width, yerr=stds, capsize=5, color=colors[:len(labels)], zorder=3, label='Время инференса')
    
    ax1.set_ylabel('Среднее время инференса (секунды)', fontsize=12, color='darkblue')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)

    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2., height + (stds[i] or 0) * 1.1,
                 f'{height:.4f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    max_diffs = [results[label]['diff'][0] for label in labels]
    ax2 = ax1.twinx()
    ax2.plot(x, max_diffs, 'o-', color='crimson', markersize=8, lw=2, label='Max Abs Diff (vs FP32)')
    ax2.set_ylabel('Максимальное абсолютное расхождение', fontsize=12, color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_yscale('log') 
    
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, fontsize=12)
    if means:
        ax1.set_ylim(0, max(m + (s or 0) for m, s in zip(means, stds)) * 1.4)
    
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

    fig.tight_layout()
    try:
        plt.savefig(filename)
        print(f"\nГрафик сохранен в файл: {filename}")
    except Exception as e:
        print(f"Не удалось сохранить график: {e}")

def run_openvino_benchmark(model_path, inputs_dict, repeats):
    print(f"--- Бенчмарк OpenVINO для: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"ОШИБКА: Файл модели не найден: {model_path}"); return (None, None), None
        
    try:
        core = ov.Core()
        core.set_property("CPU", {"INFERENCE_NUM_THREADS": 1, "AFFINITY": "CORE"})
        compiled_model = core.compile_model(model_path, "CPU")
        output_node = compiled_model.outputs[0]

        print("Прогрев модели..."); [compiled_model(inputs_dict) for _ in range(10)]
        
        print(f"Запуск бенчмарка ({repeats} повторов)...")
        times, final_output = [], None
        for i in range(repeats):
            start_time = time.perf_counter()
            results = compiled_model(inputs_dict)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            if i == repeats - 1: 
                final_output = results[output_node]
            
        mean_time, std_time = np.mean(times), np.std(times)
        print(f"Результат: {mean_time*1000:.3f} ± {std_time*1000:.3f} мс")
        print("-" * 50)
        return (mean_time, std_time), final_output
        
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА во время бенчмарка: {e}"); return (None, None), None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Бенчмарк OpenVINO моделей (скорость и точность).")
    parser.add_argument("--model_type", type=str, required=True, choices=["resnet", "bert", "smollm"], help="Тип модели.")
    args = parser.parse_args()
    
    # Конфигурация путей
    model_path_map = {"resnet": "resnet18", "bert": "bert", "smollm": "smollm"}
    folder_name = model_path_map[args.model_type]
    OV_FP32_PATH = f'../models/ir_fp32/{folder_name}/{folder_name}_fp32.xml'
    OV_FP16_PATH = f'../models/ir_fp16/{folder_name}/{folder_name}_fp16.xml'
    OV_INT8_PATH = f'../models/ir_int8/{folder_name}/{folder_name}_int8.xml'


    pytorch_output_path = f'../data/pytorch_outputs/torch_output_{args.model_type}.npy' 
    if os.path.exists(pytorch_output_path):
        print(f"Загрузка эталонного выхода PyTorch из: {pytorch_output_path}")
        baseline_output = np.load(pytorch_output_path)
    else:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Эталонный выход PyTorch не найден по пути {pytorch_output_path}. Сравнение будет производиться с OpenVINO FP32.")
        baseline_output = None

    # Подготовка входных данных
    if args.model_type == "resnet":
        print("Подготовка данных для ResNet...")
        REPEATS = 200
        input_name = ov.Core().read_model(OV_FP32_PATH).inputs[0].get_any_name()
        inputs_dict = {input_name: np.random.rand(1, 3, 224, 224).astype(np.float32)}
    else: # bert, smollm
        print(f"Подготовка данных для {args.model_type.upper()}...")
        REPEATS = 100
        text = "This is a sample sentence for benchmarking the language model inference time and accuracy."
        if args.model_type == "bert": tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else: tokenizer = AutoTokenizer.from_pretrained('../models/smollm_local_model')

        inputs_dict = dict(tokenizer(text, return_tensors='np', padding='max_length', truncation=True, max_length=128))
        model_input_names = {inp.get_any_name() for inp in ov.Core().read_model(OV_FP32_PATH).inputs}
        if 'token_type_ids' not in model_input_names and 'token_type_ids' in inputs_dict:
            del inputs_dict['token_type_ids']
    
    # Запуск бенчмарков и сбор результатов
    print(f"\n====== ЗАПУСК БЕНЧМАРКА ДЛЯ {args.model_type.upper()} ======")
    all_results = {}
    
    time_fp32, out_fp32 = run_openvino_benchmark(OV_FP32_PATH, inputs_dict, REPEATS)
    if baseline_output is None:
        baseline_output = out_fp32
    all_results["FP32"] = {'time': time_fp32, 'output': out_fp32, 'diff': compare_outputs(baseline_output, out_fp32, "PyTorch", "OpenVINO FP32")}


    time_fp16, out_fp16 = run_openvino_benchmark(OV_FP16_PATH, inputs_dict, REPEATS)
    all_results["FP16"] = {'time': time_fp16, 'output': out_fp16, 'diff': compare_outputs(baseline_output, out_fp16, "PyTorch", "OpenVINO FP16")}
    
    time_int8, out_int8 = run_openvino_benchmark(OV_INT8_PATH, inputs_dict, REPEATS)
    all_results["INT8"] = {'time': time_int8, 'output': out_int8, 'diff': compare_outputs(baseline_output, out_int8, "PyTorch", "OpenVINO INT8")}

    # Итоговая таблица
    print("\n=============================== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===============================")
    print(f"Модель: {args.model_type.upper()}")
    print("Формат          | Время (мс)         | Max Abs Diff (vs FP32) | Mean Abs Diff (vs FP32)")
    print("----------------|--------------------|------------------------|--------------------------")
    for name, res in all_results.items():
        if res and res['time'] and res['time'][0] is not None:
            mean_t, std_t = res['time']
            max_d, mean_d = res['diff']
            time_str = f"{mean_t*1000:.2f} ± {std_t*1000:.2f}"
            max_d_str = f"{max_d:.2e}" if not np.isnan(max_d) else "N/A"
            mean_d_str = f"{mean_d:.2e}" if not np.isnan(mean_d) else "N/A"
            print(f"{name:<15} | {time_str:<18} | {max_d_str:<22} | {mean_d_str:<24}")
        else:
            print(f"{name:<15} | {'НЕ ВЫПОЛНЕНО':<18} | {'N/A':<22} | {'N/A':<24}")
    print("=====================================================================================")

    # Построение графика
    PLOT_FILENAME = f'benchmark_full_results_{args.model_type}.png'
    plot_title = f'Сравнение производительности и точности {args.model_type.upper()} (CPU, 1 ядро)'
    plot_benchmark_results(all_results, PLOT_FILENAME, plot_title)