# Image Recognizer | Распознаватель изображений

## Цель приложения
Предоставить пользователю приложение с GUI, распознающее сущность на изображениях.

**Пример:** Изображению цифры 8 соответсвует вывод "восемь", а изображению цифры 9 соответствует вывод "девять".

## Как это работает?
Распознавание изображений выполняется с использованием заранее обученных, либо созданных пользователем нейронных сетей.
Пользователь выбирает существующую нейронную сеть, загружает изображение для распознавания и (после анализа изображения нейронной сетью) получает наименование некоторого класса, которым нейронная сеть наделила данное изображение.

**Пример:** Пользователь выбрал заранее обученную нейронную сеть *"распознаватель цифр от '0' до '9'"*.
Затем пользователь загружает изображение рукописной цифры '4'. Нейросеть проводит анализ изображения и выводит результат: *"цифра '4'"*.

**Пример:** Пользователь создаёт новую нейронную сеть *"распознаватель котов и собак"*.
Затем он вводит характеристики создаваемой нейронной сети: количество скрытых слоёв (например, 1), количество входных нейронов (количество пикселей в изображениях из тренировочных данных. Например, 20x20 = 400), количество выходных нейронов (количество классов. Например, |{кот', 'собака}| = 2).
После этого пользователь указывает расположение тренировочных данных -- путь до файла *"кошки-и-собаки-тренировочные-данные.csv"*.
Создание сети завершается началом её обучения на загруженных пользователем тренировочных данных (При условии корректности введённых пользователем данных).
Далее пользователь загружает изображение своего питомца, и нейросеть определяет к какому классу ('кот', 'собака') он/она принадлежит.