import matplotlib.pyplot as plt
import numpy as np

def create_table_image(data, title, col_labels, row_labels, filename="table.png"):

    fig, ax = plt.subplots(figsize=(16, 4)) 
    ax.axis('tight')
    ax.axis('off')


    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        rowLabels=row_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12] 
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8) 


    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor("#f0f0f0") 
        table[(0, i)].set_text_props(weight='bold')

    for i in range(len(row_labels)):
         table[(i + 1, -1)].set_facecolor("#f0f0f0")
         table[(i + 1, -1)].set_text_props(weight='bold')


    plt.title(title, fontsize=16, weight='bold', pad=20)
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    print(f"Таблица сохранена как: {filename}")
    plt.close(fig)


col_headers = [
    'Точность', 'Время, мс', 'Ускорение\n(отн. PyTorch)', 'Размер, МБ',
    'Сокращение\nразмера', 'Max Abs Diff\n(отн. PyTorch)', 'Mean Abs Diff\n(отн. PyTorch)'
]
row_headers_all = ['PyTorch', 'ONNX Runtime', 'TFLite', 'TFLite', 'OpenVINO', 'OpenVINO', 'OpenVINO']

# --- Таблица 1: ResNet-18 ---
data_resnet = [
    ['FP32', '72.4', '1.00x', '44.6', '0.0%', '0.00e+00', '0.00e+00'],
    ['FP32', '34.0', '2.13x', '44.6', '0.0%', '6.79e-06', '1.15e-06'],
    ['FP32', '44.8', '1.62x', '44.6', '0.0%', '6.52e-06', '1.74e-06'],
    ['FP16', '44.6', '1.62x', '22.3', '-50.0%', '1.03e-02', '2.65e-03'],
    ['FP32', '47.1', '1.54x', '44.6', '0.0%', '7.64e-01', '1.66e-01'],
    ['FP16', '36.4', '1.99x', '22.3', '-50.0%', '7.61e-01', '1.66e-01'],
    ['INT8', '19.4', '3.73x', '11.2', '-74.9%', '1.12e+00', '2.27e-01']
]
create_table_image(data_resnet, 'Таблица Б.1. Сводные результаты для модели ResNet-18', col_headers, row_headers_all, "table_resnet18.png")

# --- Таблица 2: BERT ---
data_bert = [
    ['FP32', '308.9', '1.00x', '417.9', '0.0%', '0.00e+00', '0.00e+00'],
    ['FP32', '233.7', '1.32x', '417.9', '0.0%', '2.91e-05', '5.22e-07'],
    ['FP32', '264.6', '1.17x', '415.6', '-0.5%', '3.34e-05', '6.23e-07'],
    ['FP16', '339.9', '0.91x', '207.9', '-50.2%', '4.29e-03', '2.45e-04'],
    ['FP32', '253.0', '1.22x', '417.6', '-0.1%', '9.24e-06', '3.04e-07'],
    ['FP16', '267.3', '1.16x', '208.8', '-50.0%', '4.29e-03', '2.45e-04'],
    ['INT8', '156.0', '1.98x', '104.9', '-74.9%', '8.31e+00', '1.44e-01']
]
create_table_image(data_bert, 'Таблица Б.2. Сводные результаты для модели BERT', col_headers, row_headers_all, "table_bert.png")

# --- Таблица 3: SmolLM-135M ---
data_smollm = [
    ['FP32', '487.5', '1.00x', '515.3', '0.0%', '0.00e+00', '0.00e+00'],
    ['FP32', '302.6', '1.61x', '515.3', '0.0%', '1.29e-04', '9.88e-07'],
    ['FP32', '620.6', '0.79x', '515.1', '0.0%', '7.34e-05', '7.83e-07'],
    ['FP16', '664.3', '0.73x', '258.1', '-49.9%', '3.10e-02', '2.96e-04'],
    ['FP32', '295.1', '1.65x', '514.1', '-0.2%', '5.71e+01', '2.48e-01'],
    ['FP16', '337.1', '1.45x', '257.1', '-50.1%', '5.71e+01', '2.48e-01'],
    ['INT8', '196.2', '2.48x', '129.4', '-74.9%', '5.55e+01', '3.45e-01']
]
create_table_image(data_smollm, 'Таблица Б.3. Сводные результаты для модели SmolLM-135M', col_headers, row_headers_all, "table_smollm.png")