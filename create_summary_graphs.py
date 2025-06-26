import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def plot_performance_chart(data, title, filename):

    labels = list(data.keys())
    means = [val[0] for val in data.values()]
    stds = [val[1] for val in data.values()]

    x = np.arange(len(labels))
    width = 0.6  

    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = [
        '#87CEEB', # PyTorch
        '#F08080', # ONNX
        '#90EE90', '#FFD700', # TFLite
        '#1f77b4', '#ff7f0e', '#2ca02c' 
    ]

    rects = ax.bar(x, means, width, yerr=stds, capsize=5, color=colors[:len(labels)], label='Mean Time')

    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.,
            height + (stds[i] if stds[i] else 0) * 0.1,  
            f'{height:.4f}s',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_ylabel('Среднее время инференса (секунды)')
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right") 
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

    if means:
        max_val = max(m + (s or 0) for m, s in zip(means, stds))
        ax.set_ylim(0, max_val * 1.25)

    plt.tight_layout()  
    plt.savefig(filename, dpi=150)
    print(f"График сохранен как: {filename}")
    plt.close(fig)

if __name__ == '__main__':
    # --- Данные для ResNet-18 (в секундах) ---
    data_resnet = {
        'PyTorch':      (0.0724, 0.0076),
        'ONNX Runtime': (0.0340, 0.0017),
        'TFLite FP32':  (0.0448, 0.0016),
        'TFLite FP16':  (0.0446, 0.0023),
        'OpenVINO FP32':(0.0471, 0.0168),
        'OpenVINO FP16':(0.0364, 0.0032),
        'OpenVINO INT8':(0.0194, 0.0022),
    }
    plot_performance_chart(
        data_resnet,
        'Сводное сравнение производительности ResNet-18 (CPU, 1 ядро)',
        'summary_performance_resnet.png'
    )

    # --- Данные для BERT (в секундах) ---
    data_bert = {
        'PyTorch':      (0.3089, 0.0292),
        'ONNX Runtime': (0.2337, 0.0213),
        'TFLite FP32':  (0.2646, 0.0099),
        'TFLite FP16':  (0.3399, 0.0560),
        'OpenVINO FP32':(0.2530, 0.0177),
        'OpenVINO FP16':(0.2673, 0.0138),
        'OpenVINO INT8':(0.1560, 0.0055),
    }
    plot_performance_chart(
        data_bert,
        'Сводное сравнение производительности BERT (CPU, 1 ядро)',
        'summary_performance_bert.png'
    )

    # --- Данные для SmolLM-135M (в секундах) ---
    data_smollm = {
        'PyTorch':      (0.4875, 0.0516),
        'ONNX Runtime': (0.3026, 0.0260),
        'TFLite FP32':  (0.6206, 0.0274),
        'TFLite FP16':  (0.6643, 0.0550),
        'OpenVINO FP32':(0.2951, 0.0210),
        'OpenVINO FP16':(0.3371, 0.0231),
        'OpenVINO INT8':(0.1962, 0.0067),
    }
    plot_performance_chart(
        data_smollm,
        'Сводное сравнение производительности SmolLM-135M (CPU, 1 ядро)',
        'summary_performance_smollm.png'
    )