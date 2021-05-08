# PyTorch PlayGround

Основа для этого проекта взята с этого [репозитория](https://github.com/victoresque/pytorch-template "Github")

-----

Реализовать с помощью pytorch аналог https://playground.tensorflow.org/
Фичи, которые нужно реализовать
1) Генерация данных, датасет и даталоадер (можно добавить подгрузку своих csv)
2) Класс генерирующий сеть по заданной архитектуре
3) Класс обучающий сеть
4) Сделать визаулизацию

-----

## Installation

Для начала создайте виртуальную среду с необходимыми библиотеками:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py -c config.py
```

## Evaluation

После обучения у вас будет сохранена модель в папке `saved/models/HomeTask/{current_date}`. Этот путь надо будет указать ниже

```bash
python test.py --resume saved/models/HomeTask/{current_date}
```