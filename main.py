import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- 1. ЗАРЕЖДАНЕ НА ДАННИТЕ ---
import os
file_name = os.path.join("data", "ecommerce_product_reviews_dataset.csv")

try:
    # Опитваме се да заредим твоя файл
    # Ако е 400MB, може да отнеме няколко секунди
    df = pd.read_csv(file_name)
    print(f"Успешно зареден файл: {file_name} с {len(df)} реда.")
except FileNotFoundError:
    print(f"Файлът '{file_name}' не е намерен. Генерирам тестови данни за демонстрация...")
    # Генериране на фиктивни данни, за да работи кода
    data = {
        'review_text': [
            "This product is amazing! I love it.", "Terrible quality, broke immediately.",
            "It's okay, not bad.", "Waste of money, do not buy.",
            "Highly recommended, great value.", "Customer service was rude.",
            "Five stars! Exceeded my expectations.", "I regret buying this.",
            "Fast shipping and good packaging.", "Very disappointed, will return."
        ] * 100,
        'sentiment': [
            "Positive", "Negative", "Neutral", "Negative", "Positive",
            "Negative", "Positive", "Negative", "Positive", "Negative"
        ] * 100
    }
    df = pd.DataFrame(data)

# --- 2. ПОДГОТОВКА НА ДАННИТЕ (PREPROCESSING) ---

# Филтрираме само Positive и Negative (бинарна класификация)
df_filtered = df[df['sentiment'].isin(['Positive', 'Negative'])].copy()

# Превръщаме етикетите в цифри: Positive -> 1, Negative -> 0
df_filtered['label'] = df_filtered['sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)

# Функция за почистване на текст
def clean_text(text):
    text = str(text).lower()                 # Малки букви
    text = re.sub(r'[^\w\s]', '', text)      # Махане на препинателни знаци
    return text

print("Обработка на текста...")
df_filtered['clean_text'] = df_filtered['review_text'].apply(clean_text)

# Разделяне на Training (80%) и Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    df_filtered['clean_text'], 
    df_filtered['label'], 
    test_size=0.2, 
    random_state=42
)

# --- 3. ВЕКТОРИЗАЦИЯ (TF-IDF) ---
# Превръщаме думите в матрица от цифри
# ngram_range=(1,2) хваща и фрази като "not good"
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- 4. ОБУЧЕНИЕ И СРАВНЕНИЕ НА МОДЕЛИ ---

# Модел 1: Logistic Regression (Логистична регресия)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)

# Модел 2: Naive Bayes (Наивен Бейс)
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
nb_acc = accuracy_score(y_test, nb_pred)

print(f"\n--- РЕЗУЛТАТИ ОТ СРАВНЕНИЕТО ---")
print(f"Точност на Logistic Regression: {lr_acc*100:.2f}%")
print(f"Точност на Naive Bayes:        {nb_acc*100:.2f}%")

# --- 5. АНАЛИЗ НА ТЕЖЕСТТА НА ДУМИТЕ (FEATURE IMPORTANCE) ---
# Това е частта за "по-висока оценка" - обясняваме какво гледа моделът
feature_names = vectorizer.get_feature_names_out()
coefs = lr_model.coef_[0]

# Създаваме таблица с думите и тяхната сила
word_importance = pd.DataFrame({'word': feature_names, 'weight': coefs})
word_importance = word_importance.sort_values(by='weight', ascending=True)

print("\n--- КАКВО НАУЧИ МОДЕЛЪТ? ---")
print("Топ думи за НЕГАТИВНО ревю:")
print(word_importance.head(10)['word'].tolist())

print("\nТоп думи за ПОЗИТИВНО ревю:")
print(word_importance.tail(10)['word'].tolist())

# --- 6. ТЕСТ С ТВОИ ИЗРЕЧЕНИЯ ---
print("\n--- ТЕСТ В РЕАЛНО ВРЕМЕ ---")
my_reviews = [
    "This is the best purchase I have ever made!",
    "Total waste of money, very disappointed.",
    "It arrived late but the quality is okay."
]

my_reviews_vec = vectorizer.transform([clean_text(r) for r in my_reviews])
predictions = lr_model.predict(my_reviews_vec)

for review, pred in zip(my_reviews, predictions):
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    print(f"Ревю: '{review}' -> {sentiment}")
