# Код
Код использует библиотеку transformers и pytorch для обучения моделей.
Для обучения использовался сервис vast.ai и модели обучались на 4 GeForce RTX 3090.

Зависимости указаны в файле requirements.txt, в качестве образа был взят 
докер-образ `nvcr.io/nvidia/pytorch:22.12-py3`.

Для обучения моделей нужно сделать препроцессинг данных - разбить train.parquet на 
обучение и валидацию.

# Запуск обучения
train.py - код для запуска обучения модели-классификатора. Пример запуска:
```bash
deepspeed train.py --model MODEL_NAME --data DATA_PATH --name CHECKPOINT_PATH 
```
где:
* MODEL_NAME - имя модели, например `roberta-base`
* DATA_PATH - путь с данными, описанными выше. Там должны быть файлы `train_split.parquet` и `val_split.parquet`
* CHECKPOINT_PATH - имя папки, куда сохранять модель

train_mlm.py - код для обучения MLM модели на неразмеченных данных. Код запуска:
```bash
deepspeed train_mlm.py --model MODEL_NAME --data DATA_PATH --name CHECKPOINT_PATH
```
аргументы аналогичны.


convert_to_mlm.py - скрипт, который превращает unlabeled.snappy.parquet в формат для
обучения MLM-модели из предыдущего шага.


predict.py - код для получения предсказания.
```bash
python predict.py CHECKPOINT_PATH DATA_PATH FILENAME.csv
```

* CHECKPOINT - чекпоинт модели из скрипта train.py
* DATA_PATH - путь к директории с файлом `test.parquet`
* FILENAME - файл, куда записать предсказание


markup.py - код для генерации разметки на неразмеченном корпусе.
К сожалению, не помогло :(

