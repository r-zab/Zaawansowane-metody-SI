import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input  # <--- KLUCZOWE DLA VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import sys

# --- 0. SPRAWDZENIE GPU ---
print("=" * 50)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"SUKCES! Wykryto GPU: {gpus}")
else:
    print("UWAGA! Nie wykryto GPU. Trening będzie wolny.")
print("=" * 50)

# --- 1. KONFIGURACJA ---
# ZMIEŃ TE ŚCIEŻKI NA SWOJE
PROCESSED_DIR = Path(
    r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\datasets\chest_xray_processed_val1000_balanced')
BASE_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray')

# Folder na wyniki (tworzy nowy folder z datą/opisem, żeby nie nadpisywać starych)
EXPERIMENT_NAME = 'VGG16_Transfer_v1'
RESULTS_DIR = BASE_DIR / 'wykresy' / EXPERIMENT_NAME
MODELS_DIR = BASE_DIR / 'models'

# PARAMETRY TRENINGU
IMG_SIZE = 512
BATCH_SIZE = 16  # VGG16 jest duży, 16 powinno wejść na 6GB VRAM. Jak wywali błąd OOM, zmniejsz do 8.
EPOCHS = 30
L2_STRENGTH = 1e-4
LEARNING_RATE = 1e-4

# Tworzenie folderów
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Wczytanie Danych ---
train_dir = PROCESSED_DIR / 'train'
val_dir = PROCESSED_DIR / 'val'

print(f"Wczytywanie danych treningowych z: {train_dir}")
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'  # 2 wyjścia (softmax)
)

print(f"Wczytywanie danych walidacyjnych z: {val_dir}")
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

class_names = train_ds.class_names
print(f"Znalezione klasy: {class_names}")

# --- 3. Obliczenie Wag Klas ---
n_normal = len(list(train_dir.glob('NORMAL/*.*')))
n_pneumonia = len(list(train_dir.glob('PNEUMONIA/*.*')))
total_train = n_normal + n_pneumonia

if n_normal > 0 and n_pneumonia > 0:
    weight_for_0 = (1 / n_normal) * (total_train / 2.0)
    weight_for_1 = (1 / n_pneumonia) * (total_train / 2.0)
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"Wagi klas: {class_weights}")
else:
    class_weights = None

# --- 4. Optymalizacja Wydajności ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# --- 5. Definicja Modelu (VGG16 TRANSFER LEARNING) ---
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="augmentacja"
)


def build_model_vgg16_pro():
    # 1. Baza: VGG16
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    # Zamrażamy bazę (nie trenujemy jej na początku)
    base_model.trainable = False

    # 2. Składanie modelu
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # WAŻNE: preprocess_input zamiast Rescaling(1./255)
    # VGG16 wymaga specyficznego formatowania kolorów
    x = layers.Lambda(preprocess_input)(inputs)

    x = data_augmentation(x)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    # Silna głowa klasyfikacyjna
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_STRENGTH))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_STRENGTH))(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(2, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="VGG16_Pro_Classifier")
    return model


model = build_model_vgg16_pro()

# --- 6. Kompilacja Modelu ---
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- 7. Trening z Callbacks ---
# Zapisz tylko NAJLEPSZY model (nie nadpisuj go gorszymi z późniejszych epok)
checkpoint_path = MODELS_DIR / f'{EXPERIMENT_NAME}_best.keras'

callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
]

print("\nRozpoczynam trening...")
if class_weights:
    history = model.fit(
        train_ds, epochs=EPOCHS, validation_data=val_ds,
        class_weight=class_weights, callbacks=callbacks_list
    )
else:
    history = None

# --- 8. Analiza Wyników ---
if history:
    # Wykresy
    print("Generowanie wykresów...")
    epochs_range = range(len(history.history['loss']))

    plt.figure(figsize=(14, 6))

    # Dokładność
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], label='Trening')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Walidacja')
    plt.title('Dokładność')
    plt.legend()

    # Strata
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label='Trening')
    plt.plot(epochs_range, history.history['val_loss'], label='Walidacja')
    plt.title('Strata (Loss)')
    plt.legend()

    plt.savefig(RESULTS_DIR / 'training_history.png')
    plt.close()

    # --- 9. Raport Medyczny ---
    print("\nGenerowanie raportu medycznego...")

    # Prawdziwe etykiety
    y_true_onehot = np.concatenate([y for x, y in val_ds], axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)

    # Predykcje
    y_pred_probs = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Macierz Konfuzji
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Macierz Konfuzji ({EXPERIMENT_NAME})')
    plt.ylabel('Prawda')
    plt.xlabel('Predykcja')
    plt.savefig(RESULTS_DIR / 'confusion_matrix.png')
    plt.close()

    # Obliczanie metryk
    # cm structure: [[TN, FP], [FN, TP]] dla układu [Normal, Pneumonia]
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    report_text = f"""
    RAPORT EWALUACJI MODELU: {EXPERIMENT_NAME}
    =========================================
    Model Base: VGG16 (ImageNet Weights)
    Rozmiar obrazu: {IMG_SIZE}x{IMG_SIZE}
    Batch Size: {BATCH_SIZE}
    -----------------------------------------
    MACIERZ KONFUZJI:
    TN (Prawdziwie Zdrowy): {tn}
    FP (Fałszywy Alarm):    {fp}
    FN (Niewykryty Chory):  {fn}  <-- KRYTYCZNE
    TP (Wykryty Chory):     {tp}
    -----------------------------------------
    METRYKI:
    Accuracy:    {accuracy:.4f}
    Czułość (Sensitivity): {sensitivity:.4f} (Zdolność wykrywania choroby)
    Swoistość (Specificity): {specificity:.4f} (Zdolność ignorowania zdrowych)
    Precyzja:    {precision:.4f}
    =========================================

    Pełny raport sklearn:
    {classification_report(y_true, y_pred, target_names=class_names)}
    """

    print(report_text)

    # Zapis do pliku
    with open(RESULTS_DIR / 'report_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✅ Wszystko gotowe! Wyniki w folderze: {RESULTS_DIR}")