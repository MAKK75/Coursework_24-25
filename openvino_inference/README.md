Конечно. Вот финальная, выверенная версия `README.md` для папки `openvino_inference`. Она предполагает, что скрипты уже исправлены и готовы к запуску из подпапки `scripts/`.

---

# Конвейер OpenVINO: Конвертация, Квантизация и Бенчмаркинг

Этот проект является завершающим этапом, на котором модели, ранее конвертированные в ONNX, проходят полный цикл обработки с помощью инструментов OpenVINO.

**Цели конвейера:**
1.  **Конвертация** из ONNX в OpenVINO Intermediate Representation (IR) в форматах FP32 и FP16.
2.  **Квантизация** моделей FP32 в INT8 с использованием NNCF.
3.  **Бенчмаркинг** всех версий (FP32, FP16, INT8) для сравнения скорости и точности.

## Структура папки

Проект имеет модульную структуру: скрипты отделены от данных и моделей.

```
openvino_inference/
├── models/
│   ├── onnx/
│   └── smollm_local_model/
├── data/
│   └── pytorch_outputs/
└── scripts/
    ├── convert_to_ir.sh
    ├── run_nncf_quantization.py
    └── run_full_benchmark.py
```

## Предварительные требования

1.  **Выполнены предыдущие этапы:** У вас должны быть результаты работы Docker-конвейеров из папок `resnet_bert_inference` и `smollm_inference`.
2.  **Установлена среда Python** и необходимые библиотеки:
    ```bash
    pip install "openvino>=2023.2" "nncf>=2.8" transformers torch matplotlib numpy
    ```
    *Рекомендуется использовать виртуальное окружение.*

## Шаг 1: Подготовка рабочего пространства

Скопируйте все необходимые артефакты с предыдущих этапов в соответствующие папки **внутри `openvino_inference`**.

1.  **Скопируйте ONNX-модели** в `./models/onnx/`:
    ```bash
    # Команды выполняются из папки openvino_inference/

    # Копируем из папки resnet_bert_inference/docker_results/
    cp ../resnet_bert_inference/docker_results/resnet18.onnx ./models/onnx/
    cp ../resnet_bert_inference/docker_results/bert_model.onnx ./models/onnx/

    # Копируем из папки smollm_inference_pipeline/docker_results/
    cp ../smollm_inference_pipeline/docker_results/smollm_135m.onnx ./models/onnx/smollm_135m.onnx
    ```

2.  **Скопируйте эталонные выходы PyTorch** в `./data/pytorch_outputs/`:
    ```bash
    # Папку pytorch_outputs можно взять из результатов любого из предыдущих шагов
    cp ../smollm_inference_pipeline/docker_results/pytorch_outputs/*.npy ./data/pytorch_outputs/
    ```

3.  **Скопируйте токенизатор SmolLM** в `./models/`:
    ```bash
    # Исходная папка с токенизатором должна находиться в smollm_inference/
    cp -r ../smollm_inference_pipeline/smollm_local_model/ ./models/
    ```

## Шаг 2: Запуск конвейера OpenVINO

Все скрипты запускаются из папки `scripts`.

1.  **Перейдите в папку со скриптами:**
    ```bash
    cd scripts
    ```

2.  **Конвертируйте модели в OpenVINO IR (FP32/FP16):**
    ```bash
    bash convert_to_ir.sh
    ```
    *Этот скрипт создаст папки `ir_fp32` и `ir_fp16` в директории `../models/`.*

3.  **Выполните INT8-квантизацию:**
    Запустите скрипт три раза, по одному для каждой модели. Этот шаг может занять некоторое время.
    ```bash
    python run_nncf_quantization.py --model_type resnet
    python run_nncf_quantization.py --model_type bert
    python run_nncf_quantization.py --model_type smollm
    ```
    *Этот скрипт создаст папку `ir_int8` в директории `../models/`.*

4.  **Запустите полный бенчмарк:**
    Снова запустите скрипт три раза для каждой модели.
    ```bash
    python run_full_benchmark.py --model_type resnet
    python run_full_benchmark.py --model_type bert
    python run_full_benchmark.py --model_type smollm
    ```

5.  **Вернитесь в основную папку (опционально):**
    ```bash
    cd ..
    ```

## Результаты

Для каждой модели будут сгенерированы результаты:

*   **Таблица в консоли:** Показывает среднее время инференса, стандартное отклонение и метрики расхождения по сравнению с эталоном PyTorch.
*   **График `.png`:** В основной папке `openvino_inference/` появятся графики (например, `benchmark_full_results_resnet.png`), которые визуализируют производительность (столбчатая диаграмма) и потерю точности (линейный график на логарифмической шкале) для версий FP32, FP16 и INT8.

![Пример графика для ResNet18](./benchmark_full_results_resnet.png)
