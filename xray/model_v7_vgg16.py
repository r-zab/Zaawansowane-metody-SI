import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
# NOWE IMPORTY DLA VGG16
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

from keras.callbacks import EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- 1. KONFIGURACJA ---
PROCESSED_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray_split_500_processed')
PLOT_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\plots')

# PARAMETRY TRENINGU
IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 30
L2_STRENGTH = 1e-4
LEARNING_RATE = 1e-4
# -------------------------

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Wczytanie Danych ---
train_dir = PROCESSED_DIR / 'train'
val_dir = PROCESSED_DIR / 'val'

print(f"Wczytywanie danych treningowych z: {train_dir}")
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

print(f"Wczytywanie danych walidacyjnych z: {val_dir}")
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)
CLASS_NAMES = train_ds.class_names
print(f"Znalezione klasy: {CLASS_NAMES}")

# --- 3. Obliczenie Wag Klas ---
# ... (Ten kod jest identyczny, nie ruszamy) ...
print("Obliczanie wag klas...")
n_normal = len(list(train_dir.glob('NORMAL/*.*')))
n_pneumonia = len(list(train_dir.glob('PNEUMONIA/*.*')))
total_train = n_normal + n_pneumonia
if n_normal == 0 or n_pneumonia == 0:
    class_weights = None
else:
    weight_for_0 = (1 / n_normal) * (total_train / 2.0)
    weight_for_1 = (1 / n_pneumonia) * (total_train / 2.0)
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"Waga dla klasy 0 (NORMAL): {weight_for_0:.2f}")
    print(f"Waga dla klasy 1 (PNEUMONIA): {weight_for_1:.2f}\n")

# --- 4. Optymalizacja Wydajności Wczytywania ---
# Używamy .cache() do ładowania do RAM (tak jak w twoim udanym logu)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 5. Definicja Modelu (TRANSFER LEARNING VGG16) ---

# Augmentacja zostaje ta sama
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="augmentacja"
)


def build_model_vgg16():
    # Krok 1: Załaduj model bazowy VGG16
    base_model = VGG16(
        weights='imagenet',  # Zacznij od wag nauczonych na milionach zdjęć
        include_top=False,  # NIE dołączaj ostatniej warstwy (głowy)
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Krok 2: ZAMROŹ model bazowy
    # Mówimy mu: "Jesteś już mądry, nie zmieniaj swojej wiedzy"
    base_model.trainable = False

    # Krok 3: Zbuduj nowy model
    model = keras.Sequential(name="Transfer_Learning_VGG16")

    # Dodaj warstwę wejściową
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

    # Krok 3a: Użyj specjalnego preprocessingu VGG16
    # (Zastępuje naszą starą warstwę Rescaling)
    model.add(layers.Lambda(preprocess_input, name="vgg_preprocessing"))

    # Krok 3b: Augmentacja
    model.add(data_augmentation)

    # Krok 3c: Dodaj "zamrożony" model bazowy
    model.add(base_model)

    # Krok 3d: Doklej naszą starą, sprawdzoną "głowę"
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(L2_STRENGTH)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


model = build_model_vgg16()

# --- 6. Kompilacja Modelu ---
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Zobaczysz teraz MILIONY parametrów, ale prawie wszystkie
# będą w "Non-trainable params" - to jest POPRAWNE!
model.summary()

# --- 7. Definicja Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

# --- 8. Trening Modelu ---
print("\nRozpoczynam trening modelu (VGG16 Transfer Learning)...")
if class_weights:
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )
else:
    print("Trening zatrzymany z powodu błędu w obliczaniu wag klas.")
    history = None

# --- 9. Zapis Modelu i Wykresów ---
if history:
    MODEL_NAME = "model_VGG16_transfer_v1"  # Nowa nazwa
    print("Trening zakończony.")
    model_save_path = PROCESSED_DIR.parent / f'{MODEL_NAME}.keras'
    model.save(model_save_path)
    print(f"Model zapisany w: {model_save_path}")

    # === 9a. Generowanie wykresów ===
    print("Generowanie wykresów historii treningu...")
    epochs_ran = len(history.history['loss'])
    epochs_range = range(epochs_ran)
    # ... (reszta kodu wykresów jest identyczna) ...
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Dokładność Treningowa')
    plt.plot(epochs_range, val_acc, label='Dokładność Walidacyjna')
    plt.legend(loc='lower right')
    plt.title(f'Dokładność (Model {MODEL_NAME})')
    plt.xlabel('Epoki')
    plt.ylabel('Dokładność')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Strata Treningowa')
    plt.plot(epochs_range, val_loss, label='Strata Walidacyjna')
    plt.legend(loc='upper right')
    plt.title(f'Strata (Model {MODEL_NAME})')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')

    plot_path = PLOT_DIR / f'training_history_{MODEL_NAME}.png'
    plt.savefig(plot_path)
    print(f"Wykresy historii zapisane w: {plot_path}")
    plt.close()

    # === 9b. Generowanie macierzy konfuzji ===
    print("Generowanie macierzy konfuzji dla zbioru walidacyjnego...")
    y_true = np.concatenate([y for x, y in val_ds], axis=0).astype(int)
    y_pred_probs = model.predict(val_ds)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_true, y_pred)
    class_names = CLASS_NAMES

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Macierz Konfuzji (Val - {MODEL_NAME})')
    plt.ylabel('Rzeczywista klasa')
    plt.xlabel('Przewidziana klasa')

    cm_plot_path = PLOT_DIR / f'confusion_matrix_{MODEL_NAME}.png'
    plt.savefig(cm_plot_path)
    print(f"Macierz konfuzji zapisana w: {cm_plot_path}")
    plt.close()