import pandas as pd
import re
import os
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

MAX_VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 64
EPOCHS = 3
BATCH_SIZE = 32

if not os.path.exists('plots'):
    os.makedirs('plots')

print("Loading spaCy (NLP)...")
try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
except OSError:
    print("ERROR: Please run in terminal: python -m spacy download en_core_web_sm")
    exit()

file_name = os.path.join("data", "ecommerce_product_reviews_dataset.csv")

try:
    df = pd.read_csv(file_name, nrows=50000)
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

def advanced_clean(text):
    # 1. Regex
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = text.lower()
    
    # 2. SpaCy Lemmatization
    doc = nlp(text)
    tokens = []
    for token in doc:
        # Запазваме отрицанията!
        if (not token.is_stop) or (token.text in ['not', 'no', 'never', 'n\'t']):
            tokens.append(token.lemma_)
            
    return " ".join(tokens)

print("Processing and lemmatization of the text (this takes a little time)...")
df_filtered['clean_text'] = df_filtered['review_text'].apply(advanced_clean)

print("\n" + "="*60)
print("--- [DEMO 1] CLEANING EFFECT ---")
for i in range(3):
    print(f"Original: {df_filtered['review_text'].iloc[i]}")
    print(f"Cleaned:  {df_filtered['clean_text'].iloc[i]}")
    print("-" * 30)
print("="*60 + "\n")

X = df_filtered['clean_text'].values
y = df_filtered['label'].values

X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Tokenization (Dictionary Creation)...")
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print("\n" + "="*60)
print("--- [DEMO 2] HOW LSTM SEES TEXT (Sequences) ---")
demo_text = "Top-notch build and performance"
demo_clean = advanced_clean(demo_text)
demo_seq = tokenizer.texts_to_sequences([demo_clean])
demo_pad = pad_sequences(demo_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"Phrase: '{demo_text}' -> Clean: '{demo_clean}'")
print(f"Numerical form (Sequence): {demo_seq[0]}")
print(f"Ready to go into LSTM (Padded): {demo_pad[0][:10]} ... (the rest are zeros)")
print("="*60 + "\n")

print("Building a Bi-LSTM network...")
model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    Bidirectional(LSTM(LSTM_UNITS)),
    Dropout(0.5),
    Dense(64, activation='  '),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print("\nStarting training (will take a few minutes)...")
history = model.fit(
    X_train_pad, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_pad, y_test),
    verbose=1
)

print("\nGenerating graphs in a folder 'plots/'...")
sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 5))
# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy during training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('plots/lstm_training_history.png')
plt.close()
print("-> plots/lstm_training_history.png")

y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion matrix (LSTM)')
plt.ylabel('Real')
plt.xlabel('Predicted')
plt.savefig('plots/lstm_confusion_matrix.png')
plt.close()
print("-> plots/lstm_confusion_matrix.png")

final_acc = accuracy_score(y_test, y_pred)
print(f"\nFINAL ACCURACY: {final_acc*100:.2f}%")

print("\n" + "="*60)
print("--- TEST WITH DIFFICULT SENTENCES ---")
print("="*60)

my_reviews = [
    "This is the best purchase I have ever made!",
    "Total waste of money, very disappointed.",
    "It arrived late but the quality is okay.",
    "Amazing product",
    "Late shipping and terrible quality",
    "Late delivery and terrible quality"
]

# 1. Clean
my_reviews_clean = [advanced_clean(r) for r in my_reviews]
# 2. Tokenize
my_seq = tokenizer.texts_to_sequences(my_reviews_clean)
# 3. Pad
my_pad = pad_sequences(my_seq, maxlen=MAX_LEN, padding='post', truncating='post')

predictions = model.predict(my_pad)

for i, review in enumerate(my_reviews):
    prob = predictions[i][0]
    sentiment = "POSITIVE" if prob > 0.5 else "NEGATIVE"
    # Confidence: how far it is from 0.5 (the limit)
    confidence = prob if prob > 0.5 else 1 - prob
    
    print(f"Review: '{review}'")
    print(f"   -> Forecast: {sentiment}")
    print(f"   -> Confidence: {confidence*100:.2f}% (Probability: {prob:.4f})")
    print("-" * 30)
