import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- 1. KONFIGURACJA ---
# Ścieżka do Twojego najlepszego modelu
MODEL_PATH = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\model_VGG16_transfer_v1.keras')
# Ścieżka do danych walidacyjnych
VAL_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray_split_500_processed\val')
# Gdzie zapisać wykresy
PLOT_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\plots')

IMG_SIZE = 512
BATCH_SIZE = 8

# +++ NASZE TRZY PROGI ODKRYTE W OSTATNIM KROKU +++
thresholds_to_plot = {
    "Domyślny": 0.50,
    "Złoty_Środek (Max F1)": 0.39,
    "Bezpieczny (Max Czułość)": 0.10
}
# -------------------------

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Wczytanie modelu i danych ---
print(f"Wczytywanie modelu z: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Wczytywanie danych walidacyjnych z: {VAL_DIR}")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False  # Kluczowe dla poprawnej kolejności
)

# Zapisz nazwy klas ZANIM zrobisz .prefetch()
CLASS_NAMES = val_ds.class_names
print(f"Znalezione klasy: {CLASS_NAMES}")

# Optymalizacja
AUTOTUNE = tf.data.AUTOTUNE
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# --- 3. Pobranie Prawdziwych Etykiet i Predykcji (tylko raz) ---
print("Generowanie predykcji (tylko raz)...")
# y_true to prawdziwe etykiety (0 lub 1)
y_true = np.concatenate([y for x, y in val_ds], axis=0).astype(int)
# y_pred_probs to prawdopodobieństwa z modelu (np. 0.05, 0.49, 0.98)
y_pred_probs = model.predict(val_ds)

# --- 4. Generowanie i Zapisywanie Macierzy ---
print("Generowanie macierzy konfuzji dla każdego progu...")

for name, threshold in thresholds_to_plot.items():

    print(f"  Rysowanie macierzy dla: Próg = {threshold} ({name})...")

    # Krok 1: Oblicz predykcje dla TEGO progu
    y_pred = (y_pred_probs > threshold).astype(int).flatten()

    # Krok 2: Oblicz macierz
    cm = confusion_matrix(y_true, y_pred)

    # Krok 3: Narysuj wykres
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)

    # Oblicz FN i FP do tytułu
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:  # Awaryjnie
        tn, fp, fn, tp = 0, 0, 0, 0

    plt.title(f'Macierz Konfuzji (Próg = {threshold})\nFN: {fn} | FP: {fp}')
    plt.ylabel('Rzeczywista klasa')
    plt.xlabel('Przewidziana klasa')

    # Krok 4: Zapisz wykres do pliku
    plot_path = PLOT_DIR / f'confusion_matrix_threshold_{str(threshold).replace(".", "_")}.png'
    plt.savefig(plot_path)
    plt.close()  # Zamknij figurę, żeby się nie nadpisały

print("\n*** Gotowe! ***")
print(f"Trzy nowe macierze konfuzji zostały zapisane w folderze: {PLOT_DIR}")