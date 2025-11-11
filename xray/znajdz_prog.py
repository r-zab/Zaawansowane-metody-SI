import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- 1. KONFIGURACJA ---
# Upewnij się, że ścieżka do modelu jest poprawna
MODEL_PATH = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\model_VGG16_transfer_v1.keras')
# Ścieżka do danych walidacyjnych
VAL_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray_split_500_processed\val')

IMG_SIZE = 512
BATCH_SIZE = 8  # Ustaw taki sam batch size jak przy treningu

# --- 2. Wczytanie modelu i danych ---
print(f"Wczytywanie modelu z: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Wczytywanie danych walidacyjnych z: {VAL_DIR}")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False  # Absolutnie kluczowe dla poprawnej kolejności
)

# Optymalizacja (bez .cache() żeby nie zjeść RAM-u przy wczytywaniu)
AUTOTUNE = tf.data.AUTOTUNE
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("Model i dane wczytane. Rozpoczynam predykcję...")

# --- 3. Pobranie Prawdziwych Etykiet i Predykcji ---

# y_true to prawdziwe etykiety (0 lub 1)
y_true = np.concatenate([y for x, y in val_ds], axis=0).astype(int)

# y_pred_probs to prawdopodobieństwa z modelu (np. 0.05, 0.49, 0.98)
y_pred_probs = model.predict(val_ds)

print("Predykcje wygenerowane. Analizowanie progów...")

# --- 4. Testowanie Różnych Progów ---
thresholds = np.arange(0.1, 0.91, 0.01)  # Testuj progi co 0.01
results = []  # Lista na wyniki

for t in thresholds:
    # Użyj progu 't' zamiast domyślnego 0.5
    y_pred = (y_pred_probs > t).astype(int).flatten()

    # Oblicz metryki
    # PNEUMONIA to klasa '1' (positive class)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # TO JEST CZUŁOŚĆ!
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Oblicz Falsy Negatywne (FN)
    # cm.ravel() -> [TN, FP, FN, TP]
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:  # Upewnij się, że mamy pełną macierz
        tn, fp, fn, tp = cm.ravel()
    else:  # Obsługa błędu, gdy model przewiduje tylko jedną klasę
        tn, fp, fn, tp = 0, 0, 0, 0
        if t > 0.98:  # Prawdopodobnie same zera
            tn = cm[0, 0]
            fn = cm[1, 0]
        if t < 0.02:  # Prawdopodobnie same jedynki
            tp = cm[1, 1]
            fp = cm[0, 1]

    results.append({
        'threshold': round(t, 2),
        'accuracy': accuracy,
        'precision': precision,
        'recall (czułość)': recall,
        'f1_score': f1,
        'False_Negatives (FN)': fn,
        'False_Positives (FP)': fp
    })

# --- 5. Wyświetlenie Wyników ---
# Konwertuj wyniki do Pandas DataFrame dla łatwej analizy
df_results = pd.DataFrame(results)

# Znajdź najlepszy próg na podstawie F1-Score
best_f1_threshold = df_results.loc[df_results['f1_score'].idxmax()]

# Znajdź najlepszy próg na podstawie Czułości (Recall)
best_recall_threshold = df_results.loc[df_results['recall (czułość)'].idxmax()]

print("\n--- Analiza Zakończona ---")
print("\nDomyślny próg (0.5) dla porównania:")
print(df_results[df_results['threshold'] == 0.50].to_string())

print("\n-----------------------------------------------------------")
print("🏆 Najlepszy próg (balans precyzji i czułości):")
print(best_f1_threshold.to_string())
print("-----------------------------------------------------------")

print("\n❤️ Najlepszy próg (maksymalne wykrycie PNEUMONIA):")
print(best_recall_threshold.to_string())
print("-----------------------------------------------------------")

print("\nUWAGA (Fałszywe Negatywy):")
print(f"Przy progu 0.5, liczba FN: {df_results[df_results['threshold'] == 0.50]['False_Negatives (FN)'].values[0]}")
print(f"Przy progu z max F1, liczba FN: {best_f1_threshold['False_Negatives (FN)']}")
print(f"Przy progu z max Czułości, liczba FN: {best_recall_threshold['False_Negatives (FN)']}")