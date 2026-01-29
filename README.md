# Sentiment Analysis on E-commerce Reviews

Този проект представлява система за **Анализ на настроенията (Sentiment Analysis)** върху потребителски ревюта на продукти. Целта е автоматично да се класифицира дали даден коментар е **Positive** (Позитивен) или **Negative** (Негативен).

Проектът сравнява два подхода:
1. **Класически ML (Machine Learning):** Logistic Regression, Naive Bayes, Linear SVM (реализирани в `main.py`).
2. **Deep Learning:** Bidirectional LSTM невронна мрежа (реализирана в `ltsm.py`).

---

## 1. Настройка на данните (Dataset)

Проектът използва **"Synthetic E-commerce Product Reviews Dataset"** от Kaggle. Тъй като файлът е голям, той не е включен директно в това хранилище.

**Инструкции за изтегляне:**

1. Изтеглете данните от следния линк:
   [https://www.kaggle.com/datasets/aryan208/synthetic-e-commerce-product-reviews-dataset](https://www.kaggle.com/datasets/aryan208/synthetic-e-commerce-product-reviews-dataset)

2. Създайте папка с име **data** в основната директория на проекта.

3. Разархивирайте изтегления файл и поставете файла `ecommerce_product_reviews_dataset.csv` вътре в папка `data`.

Пътят до файла трябва да изглежда така:
`data/ecommerce_product_reviews_dataset.csv`

---

## 2. Инсталация

За да стартирате проекта, ви е необходим Python 3.x и следните библиотеки:

pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow
spacy

**Стъпка 1: Инсталиране на библиотеките**
Можете да инсталирате всичко с командата:

    pip install -r requirements.txt

**Стъпка 2: Изтегляне на езиковия модел за SpaCy**
Проектът използва библиотеката `spaCy` за лематизация. Трябва да свалите английския модел с тази команда:

    python -m spacy download en_core_web_sm

---

## 3. Как се стартира

### Вариант A: Класически модели (main.py)
Този скрипт сравнява три модела (Logistic Regression, Naive Bayes, SVM), избира най-добрия и генерира визуализации.

За стартиране:

    python main.py

*Резултат:* Ще видите точността на моделите в конзолата, а в папка `plots/` ще се генерират графики.

### Вариант Б: Deep Learning модел (ltsm.py)
Този скрипт обучава LSTM мрежа, която разбира контекста на изреченията.

За стартиране:

    python ltsm.py

*Резултат:* Обучението отнема няколко минути. Накрая ще видите тест с трудни изречения в реално време.

---

## 4. Структура на файловете

* **data/** - Папка за данните (създава се от потребителя)
* **plots/** - Папка за генерираните графики (автоматично)
* **main.py** - Основен код с класически ML модели
* **ltsm.py** - Код с Deep Learning (LSTM) модел
* **requirements.txt** - Списък с библиотеки

---

## Автор
Студент: Явор Йорданов Чамов
Специалност: Изкуствен интелект