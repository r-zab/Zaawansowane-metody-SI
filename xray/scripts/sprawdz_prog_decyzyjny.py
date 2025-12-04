import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. KONFIGURACJA ---
# Upewnij się, że ścieżki są poprawne

# Ścieżka do folderu ze zbiorem danych, na którym model był trenowany
DATA_DIR = Path(r'C:\Users\jakub\OneDrive\Pulpit\studia semestr 5\zaawansowane metody i techniki AI\laboratorium\Zaawansowane-metody-SI\chest_xray_processed_val1000_balanced')

# Ścieżka do ZAPISANEGO modelu .keras
MODEL_PATH = Path(r'C:\Users\jakub\OneDrive\Pulpit\studia semestr 5\zaawansowane metody i techniki AI\laboratorium\Zaawansowane-metody-SI\xray\plots_val_1000_dropout50\final_model_v4.keras')

# Ścieżka, gdzie zapisać nowe wykresy z progami
PLOT_DIR = Path(r'C:\Users\jakub\OneDrive\Pulpit\studia semestr 5\zaawansowane metody i techniki AI\laboratorium\Zaawansowane-metody-SI\xray\plots_val_1000')

IMG_SIZE = 512
BATCH_SIZE = 8
# -------------------------

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Wczytanie modelu ---
print(f"Wczytywanie modelu z: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("Model wczytany.")

# --- 3. Wczytanie danych walidacyjnych ---
# Używamy zbioru 'val' do znalezienia progu
# (Później użyjemy 'test' do ostatecznej oceny)
val_dir = DATA_DIR / 'val'

print(f"Wczytywanie danych walidacyjnych z: {val_dir}")
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False # Absolutnie kluczowe dla macierzy konfuzji
)
class_names = val_ds.class_names
print(f"Znalezione klasy: {class_names}")

# --- 4. Generowanie Predykcji (tylko RAZ) ---
print("Generowanie predykcji na zbiorze walidacyjnym (to chwilę potrwa)...")
# Pobieramy prawdziwe etykiety
y_true = np.concatenate([y for x, y in val_ds], axis=0).astype(int)
# Pobieramy surowe prawdopodobieństwa (np. 0.1, 0.4, 0.8)
y_pred_probs = model.predict(val_ds)
print("Predykcje wygenerowane.")


# --- 5. Testowanie Różnych Progów Decyzyjnych ---
print("\n" + "="*30)
print(" ROZPOCZYNAM TESTOWANIE PROGÓW ")
print("="*30)

# Lista progów, które chcemy sprawdzić
# Zaczynamy od domyślnego 0.5 i schodzimy w dół
thresholds_to_test = [0.5, 0.4, 0.3, 0.2]

for thresh in thresholds_to_test:
    print(f"\n--- Analiza dla progu: {thresh} ---")

    # TO JEST MIEJSCE, KTÓRE ZMIENIAMY:
    # Zamiast > 0.5, używamy naszej zmiennej 'thresh'
    y_pred = (y_pred_probs > thresh).astype(int).flatten()

    # 1. Obliczamy nową macierz konfuzji
    cm = confusion_matrix(y_true, y_pred)
    
    # 2. Drukujemy raport (on pokaże błędy FN)
    # 'recall' dla PNEUMONIA to najważniejsza metryka
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 3. Rysujemy i zapisujemy nową macierz konfuzji
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    title = f'Macierz Konfuzji (Próg = {thresh})'
    plt.title(title)
    plt.ylabel('Rzeczywista klasa')
    plt.xlabel('Przewidziana klasa')
    
    # Zapisujemy wykres z unikalną nazwą
    cm_plot_path = PLOT_DIR / f'confusion_matrix_threshold_{thresh}.png'
    plt.savefig(cm_plot_path)
    print(f"Macierz zapisana w: {cm_plot_path}")
    plt.close()

print("\n--- Zakończono testowanie progów ---")