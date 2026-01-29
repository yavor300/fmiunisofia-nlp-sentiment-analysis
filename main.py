import pandas as pd
import re
import os
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

print("Loading spaCy (NLP)...")
try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
except OSError:
    print("ERROR: Please run in terminal: python -m spacy download en_core_web_sm")
    exit()

file_name = os.path.join("data", "ecommerce_product_reviews_dataset.csv")

try:
    df = pd.read_csv(file_name, nrows=100000)
    print(f"Successfully uploaded file: {file_name} with {len(df)} rows.")
except FileNotFoundError:
    print(f"File not found. Generating test data...")
    data = {
        'review_text': [
            "This product is amazing! I loved it.", "Terrible quality, broke immediately.",
            "It's okay, not bad.", "Waste of money, do not buy.",
            "Highly recommended, great value.", "Customer service was rude.",
            "Five stars! Exceeded my expectations.", "I regret buying this.",
            "Fast shipping and better packaging.", "Very disappointed, will return."
        ] * 100,
        'sentiment': ["Positive", "Negative", "Neutral", "Negative", "Positive",
                      "Negative", "Positive", "Negative", "Positive", "Negative"] * 100
    }
    df = pd.DataFrame(data)

df_filtered = df[df['sentiment'].isin(['Positive', 'Negative'])].copy()
df_filtered['label'] = df_filtered['sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) # Removing punctuation marks
    return text

def advanced_clean(text):
    # 1. Remove special characters
    text = re.sub(r'\W', ' ', str(text))
    # 2. Remove words containing numbers (e.g. user123)
    text = re.sub(r'\w*\d\w*', ' ', text)
    # 3. Removing single letters (e.g. " a ", " s ")
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    # 4. Remove unnecessary spaces
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # 5. Lowercase
    text = text.lower()

    doc = nlp(text)
    # 6. Lemmatization
    # "better" -> "good", "running" -> "run"
    tokens = []
    for token in doc:
        if (not token.is_stop) or (token.text in ['not', 'no', 'never', 'n\'t']):
            tokens.append(token.lemma_)

    return " ".join(tokens)

print("Processing and lemmatization of the text (this takes a little time)...")
df_filtered['clean_text'] = df_filtered['review_text'].apply(advanced_clean)

print("\n" + "="*80)
print(f"--- [DEMO 1] EFFECT OF CLEANING AND LEMMATIZATION (First 5 lines) ---")
print("="*80)
# We show a table: Original vs Processed
for i in range(5):
    original = df_filtered['review_text'].iloc[i]
    cleaned = df_filtered['clean_text'].iloc[i]
    print(f"Original:  {original}")
    print(f"Processed: {cleaned}")
    print("-" * 80)

# Splitting into Training (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    df_filtered['clean_text'], 
    df_filtered['label'], 
    test_size=0.2, 
    random_state=42
)

print("Vectorization (TF-IDF)...")
# IMPORTANT: We removed stop_words='english' to keep "not"
# We added min_df=3 to remove words that occur too rarely
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\n--- WHAT THE MATRIX LOOKS LIKE (First 5 rows) ---")
feature_names = vectorizer.get_feature_names_out()
dense_matrix = X_train_vec[:5].toarray()
df_matrix = pd.DataFrame(dense_matrix, columns=feature_names)
print(df_matrix)

print("\n" + "="*80)
print("--- [DEMO 2] HOW MATHEMATICS SEES TEXT (TF-IDF) ---")
print("="*80)
sample_text = "Top-notch build and performance"
sample_vec = vectorizer.transform([sample_text])
feature_names = vectorizer.get_feature_names_out()

print(f"Entry phrase: '{sample_text}'")
print(f"Found tokens (words/phrases) in the dictionary:\n")
print(f"{'ID':<10} | {'TOKEN':<20} | {'TF-IDF WEIGHT'}")
print("-" * 50)

for idx, score in sorted_items:
    print(f"{idx:<10} | {feature_names[idx]:<20} | {score:.4f}")

if len(sorted_items) == 0:
    print("(No words found in the dictionary - maybe they are too rare or stop words)")
print("="*80 + "\n")

print("\n--- MODEL COMPARISON ---")

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test_vec))
print(f"1. Logistic Regression: {lr_acc*100:.2f}%")

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_acc = accuracy_score(y_test, nb_model.predict(X_test_vec))
print(f"2. Naive Bayes:        {nb_acc*100:.2f}%")

svm_model = LinearSVC(dual='auto', random_state=42)
svm_model.fit(X_train_vec, y_train)
svm_acc = accuracy_score(y_test, svm_model.predict(X_test_vec))
print(f"3. Linear SVM:         {svm_acc*100:.2f}%")

if svm_acc >= lr_acc and svm_acc >= nb_acc:
    best_model = svm_model
    best_name = "Linear SVM"
    print("\n>>> Most accurate: Linear SVM")
elif lr_acc >= nb_acc:
    best_model = lr_model
    best_name = "Logistic Regression"
    print("\n>>> Most accurate: Logistic Regression")
else:
    best_model = nb_model
    best_name = "Naive Bayes"
    print("\n>>> Most accurate: Naive Bayes")

# Works for SVM and Logistic Regression (Naive Bayes uses probabilities, not coefficients)
if hasattr(best_model, 'coef_'):
    feature_names = vectorizer.get_feature_names_out()
    coefs = best_model.coef_[0]

    word_importance = pd.DataFrame({'word': feature_names, 'weight': coefs})
    word_importance = word_importance.sort_values(by='weight', ascending=True)

    print("\n--- WHAT DID THE MODEL LEARN? ---")
    print("Top words for NEGATIVE review:")
    print(word_importance.head(10)['word'].tolist())

    print("\nTop words for a POSITIVE review:")
    print(word_importance.tail(10)['word'].tolist())

print("\n--- TEST WITH OUR SENTENCES ---")
my_reviews = [
    "This is the best purchase I have ever made!",
    "Total waste of money, very disappointed.",
    "It arrived late but the quality is okay.",
    "Amazing product",
    "Late shipping and terrible quality",
    "Late delivery and terrible quality"
]

my_reviews_clean = [advanced_clean(r) for r in my_reviews]
my_reviews_vec = vectorizer.transform(my_reviews_clean)
predictions = best_model.predict(my_reviews_vec)

for review, pred in zip(my_reviews, predictions):
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    print(f"Review: '{review}' -> {sentiment}")

def explain_prediction(text, model, vectorizer):
    clean = advanced_clean(text)
    vec = vectorizer.transform([clean])
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        word_indices = vec.nonzero()[1]
        
        print(f"\n--- ANALYSIS OF: '{text}' ---")
        print(f"(After cleaning: '{clean}')")
        total_score = 0
        
        for idx in word_indices:
            word = feature_names[idx]
            weight = coefs[idx]
            total_score += weight
            sentiment = "POS (+)" if weight > 0 else "NEG (-)"
            print(f"Word: '{word:20}' | Тежест: {weight:6.3f} | {sentiment}")
            
        print(f"--------------------------------------------------")
        print(f"TOTAL RESULT: {total_score:.3f}")
        prediction = "POSITIVE" if total_score > 0 else "NEGATIVE"
        print(f"FORECAST: {prediction}")
    else:
        print("Model doesn't support explanations.")

explain_prediction("Late shipping and terrible quality",  best_model, vectorizer)
explain_prediction("Late delivery and terrible quality",  best_model, vectorizer)

print("\n" + "="*80)
print("--- GRAPHIC GENERATION ---")
print("="*80)
if not os.path.exists('plots'):
    os.makedirs('plots')
sns.set_theme(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df_filtered, palette='viridis')
plt.title('Review distribution (Positive vs Negative)')
plt.xlabel('Sentiment')
plt.ylabel('Number of reviews')
plt.savefig('plots/1_balance.png')
plt.close()
print("-> Saved graphics: plots/1_balance.png")

model_names = ['Logistic Regression', 'Naive Bayes', 'Linear SVM']
accuracies = [lr_acc, nb_acc, svm_acc]

plt.figure(figsize=(8, 5))
barplot = sns.barplot(x=model_names, y=accuracies, palette='magma')
plt.ylim(0, 1.05) # Скала от 0 до 100%
plt.title('Comparison of model accuracy')
plt.ylabel('Accuracy')
for i, v in enumerate(accuracies):
    barplot.text(i, v + 0.01, f"{v*100:.1f}%", ha='center', fontweight='bold')

plt.savefig('plots/2_models_comparison.png')
plt.close()
print("-> Saved graphics: plots/2_models_comparison.png")

y_pred = best_model.predict(X_test_vec)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion matrix ({best_name})')
plt.xlabel('Predicted by the model')
plt.ylabel('True meaning')

plt.savefig('plots/3_confusion_matrix.png')
plt.close()
print("-> Saved graphics: plots/3_confusion_matrix.png")

if hasattr(best_model, 'coef_'):
    # We take the top 10 negatives and the top 10 positives
    top_neg = word_importance.head(10)
    top_pos = word_importance.tail(10)
    combined = pd.concat([top_neg, top_pos])
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in combined['weight']]
    
    sns.barplot(x='weight', y='word', data=combined, palette=colors)
    plt.title(f'Top 20 most influential words ({best_name})')
    plt.xlabel('Word weight (More to the right = More positive)')
    plt.ylabel('Word')

    plt.savefig('plots/4_feature_importance.png')
    plt.close()
    print("-> Saved graphics: plots/4_feature_importance.png")

print("\nThe graphs have been generated! Check the Plots window/panel.")
