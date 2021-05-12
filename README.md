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

### Вариант 1: Docker

Соберите и запустите контейнер
```bash
sudo docker-compose up -d
```
Войдите в него
```bash
sudo docker exec -it deep-learning-template_main_1 bash
```

### Вариант 2: Python Environment

Создайте виртуальную среду с необходимыми библиотеками:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py -c config.json
```

## Evaluation

После обучения у вас будет сохранена модель в папке `saved/models/HomeTask/{current_date}`. Этот путь надо будет указать ниже

```bash
python test.py --resume saved/models/HomeTask/{current_date}/model_best.pth
```