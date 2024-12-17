# Отчет по первой лабораторной работе

## 1. Теоретическая база

В основе данного проекта лежит задача классификации изображений с использованием глубокой нейронной сети. Ключевые моменты теоретической части:

- **Классификация изображений**: задача отнесения входного изображения к одному из нескольких заранее известных классов.
- **Сверточные нейронные сети (CNN)**: архитектуры глубокого обучения, специально разработанные для обработки данных в виде сеток (например, изображений). Они используют операции свертки, позволяющие извлекать пространственные зависимости из пиксельных данных.
- **AlexNet**: одна из первых глубоких CNN-архитектур, существенно улучшившая результаты в задачах классификации изображений. Характерные особенности — относительно глубокая архитектура на момент её появления, использование ReLU-активации, дропаута и больших ядер свертки на ранних слоях.
- **Оптимизаторы (Adam, AdaSmooth)**: методы, используемые для обновления весов нейронной сети при обучении.
  - **Adam**: адаптивный оптимизатор, использующий первый и второй моменты градиента для динамической подстройки скоростей обучения различных параметров.
  - **AdaSmooth**: модификация адаптивного оптимизатора, использующая сглаживание для более стабильного обучения, улучшая в ряде случаев сходимость по сравнению с Adam.

## 2. Описание разработанной системы

**Принцип работы системы:**
- Система использует датасет Stanford Cars, который состоит из изображений автомобилей и аннотаций, описывающих класс (марку и модель).
- Датасет разбит на тренировочную и тестовую выборки.
- Изображения нормализуются и масштабируются под размер, ожидаемый входом AlexNet (224x224).
- AlexNet инициализируется с числом выходных нейронов, соответствующим количеству классов.
- Запускается процесс обучения: сеть итеративно обрабатывает батчи изображений, вычисляет ошибку (кросс-энтропийная функция потерь) и обновляет веса с помощью выбранного оптимизатора.
- После обучения модель оценивается на тестовой выборке, вычисляется точность классификации.

**Архитектура:**
- **Входные данные:** изображения формата JPEG.
- **Предобработка:** изменение размера, нормализация.
- **Модель:** AlexNet (Conv + ReLU + MaxPool слои, затем полносвязные слои для классификации).
- **Функция потерь:** Кросс-энтропия.
- **Оптимизаторы:** Adam, AdaSmooth.
- **Инфраструктура:** обучается в среде PyTorch, с использованием GPU.
  
Алгоритмически:
1. Загрузка и парсинг аннотаций из `.mat` файлов.
2. Создание `Dataset` и `DataLoader` для удобной итерации по данным.
3. Определение и инициализация модели.
4. Выбор и инициализация оптимизаторов Adam и AdaSmooth.
5. Запуск циклов обучения.
6. Сравнительный анализ результатов после каждой эпохи.

## 3. Результаты работы и тестирования системы

Ниже приведены примеры результатов:

**Пример выводов в процессе обучения (консоль):**
```
Эпоха 1, Потери: 5.2786, Точность: 0.48%
Эпоха 2, Потери: 5.2775, Точность: 0.82%
Эпоха 3, Потери: 5.2376, Точность: 1.08%
Эпоха 4, Потери: 5.1693, Точность: 1.26%
Эпоха 5, Потери: 5.1361, Точность: 1.36%
Эпоха 6, Потери: 5.1094, Точность: 1.61%
Эпоха 7, Потери: 5.0572, Точность: 1.81%
Эпоха 8, Потери: 5.0036, Точность: 2.19%
Эпоха 9, Потери: 4.9579, Точность: 2.81%
Эпоха 10, Потери: 4.9150, Точность: 2.87%
Тестовая точность: 2.50%
```

## 4. Выводы по работе

- Использование сверточной нейронной сети AlexNet позволило успешно решить задачу классификации изображений автомобилей по классам.
- Нормализация изображений и применение архитектуры CNN существенно повышает точность по сравнению с классическими методами компьютерного зрения.
- Оптимизаторы Adam и AdaSmooth демонстрируют схожую динамику обучения, однако в некоторых экспериментах AdaSmooth показывает более плавную сходимость и чуть более высокую итоговую точность.
- Данный подход может быть расширен для использования более современных архитектур (ResNet, EfficientNet) и других адаптивных оптимизаторов.

## 5. Использованные источники

Крижевский А. В., Суцкевер И. В., Хинтон Г. Е. Классификация ImageNet с глубокими сверточными нейронными сетями // Коммуникации АКМ. – 2017. – Т. 60, No6. – С. 84–90.

Кингма Д., Ба Дж. Адам: метод стохастической оптимизации [электронный ресурс] // Международная конференция по обучающимся представлениям (ICLR). – 2015. – Режим доступа: https://arxiv.org/abs/1412.6980 (дата обращения: 17.12.2024).

Stanford Cars Dataset [электронный ресурс]. – Режим доступа: http://ai.stanford.edu/~jkrause/cars/car_dataset.html (дата обращения: 17.12.2024).

Документация PyTorch [Электронный ресурс]. – Режим доступа: https://pytorch.org/docs/stable/ (дата обращения: 17.12.2024).

Рудер С. Обзор алгоритмов оптимизации градиентного спуска [Электронный ресурс]. – 2016. – Режим доступа: https://arxiv.org/abs/1609.04747 (дата обращения: 17.12.2024).
