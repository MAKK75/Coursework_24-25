# Конвейер ResNet-BERT с использованием Docker

Эта часть проекта демонстрирует конвейер для конвертации моделей ResNet18 и BERT в форматы ONNX и TFLite, а также сравнивает их скорость инференса (вывода) на CPU. Вся обработка, включая загрузку предобученных моделей, установку зависимостей, конвертацию и сравнение, происходит внутри Docker-контейнера на этапе его сборки.

Все необходимые файлы для этого Docker-проекта находятся в папке `resnet_bert_inference` репозитория `Coursework_24-25`.

## Предварительные требования

*   Установленный и запущенный [Docker](https://www.docker.com/get-started).
*   Установленный [Git](https://git-scm.com/downloads).

## Инструкция по запуску

1.  **Клонируйте репозиторий:**
    Откройте терминал или командную строку и выполните:
    ```bash
    git clone https://github.com/MAKK75/Coursework_24-25.git
    ```

2.  **Перейдите в папку проекта:**
    После клонирования перейдите в директорию, где находится `Dockerfile` и скрипты:
    ```bash
    cd Coursework_24-25/resnet_bert_inference
    ```
    *Все последующие команды должны выполняться из этой директории (`Coursework_24-25/resnet_bert_inference`)*.

3.  **Соберите Docker-образ:**
    Эта команда запустит процесс сборки образа `resnet_bert_pipe`. Во время сборки будут выполнены все шаги, описанные в `Dockerfile`: установка зависимостей, загрузка моделей, их конвертация и запуск скриптов сравнения производительности.

    ```bash
    docker build -t resnet_bert_pipe .
    ```
    *   **Важно:** Внимательно следите за выводом в консоли во время выполнения этой команды. Вы увидите логи выполнения Python-скриптов, включая информацию о времени инференса, этапы конвертации и сообщения о сохранении файлов. Этот процесс может занять некоторое время.

4.  **(Опционально) Запустите контейнер для подтверждения:**
    Эта команда просто выведет сообщение о том, что все скрипты были выполнены во время сборки образа.
    ```bash
    docker run -it --rm resnet_bert_pipe
    ```
    Вы должны увидеть: `All scripts executed during Docker image build. Models and comparison results are part of the image or were printed to logs.`

5.  **Получение результатов (модели):**
    Все сгенерированные файлы (модели `.onnx`, `.tflite`, папки с TensorFlow SavedModel и графики `.png`) находятся внутри собранного Docker-образа в директории `/app`. Чтобы получить к ним доступ на вашем локальном компьютере (они будут скопированы в папку `docker_results` внутри текущей директории `resnet_bert_inference`):

    *   Создайте папку на вашем компьютере, куда будут скопированы результаты (если она еще не существует):
        ```bash
        mkdir -p ./docker_results
        ```
    *   Создайте временный контейнер и скопируйте из него файлы:
        ```bash
        ID=$(docker create resnet_bert_pipe)

        echo "Копирование результатов ResNet18..."
        docker cp $ID:/app/resnet18.onnx ./docker_results/
        docker cp $ID:/app/resnet18_tf ./docker_results/
        docker cp $ID:/app/inference_times_resnet_comparison.png ./docker_results/

        echo "Копирование результатов BERT..."
        docker cp $ID:/app/bert_model.onnx ./docker_results/
        docker cp $ID:/app/bert_tf_savedmodel ./docker_results/
        docker cp $ID:/app/bert_model_fp32.tflite ./docker_results/
        docker cp $ID:/app/bert_model_fp16.tflite ./docker_results/
        docker cp $ID:/app/inference_times_bert_comparison.png ./docker_results/

        docker rm -v $ID
        echo "Все файлы скопированы в папку ./docker_results (внутри Coursework_24-25/resnet_bert_inference)"
        ```
    После выполнения этих команд, вы найдете все указанные файлы и папки в директории `Coursework_24-25/resnet_bert_inference/docker_results` на вашем компьютере.

## Структура проекта (внутри `resnet_bert_inference`)

*   `Dockerfile`: Определяет процесс сборки Docker-образа.
*   `requirements.txt`: Список Python-зависимостей.
*   `resnet_onnx_1.py`: Скрипт для конвертации ResNet18.
*   `compare_resnet_inference_1.py`: Скрипт для сравнения производительности ResNet18.
*   `bert_onnx_1.py`: Скрипт для конвертации BERT в ONNX.
*   `bert_tflite_1.py`: Скрипт для конвертации BERT в TFLite.
*   `compare_bert_inference_1.py`: Скрипт для сравнения производительности BERT.

## Примечание
Все модели загружаются и обрабатываются с использованием CPU.
