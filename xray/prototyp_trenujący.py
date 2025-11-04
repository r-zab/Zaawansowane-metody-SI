import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from keras.callbacks import EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt
import os

# --- 1. KONFIGURACJA ---
# Upewnij się, że ta ścieżka prowadzi do folderu z obrazami 512x512
PROCESSED_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray_processed_padded')
PLOT_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\plots')

# PARAMETRY TRENINGU
IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 50
L2_STRENGTH = 1e-4
LEARNING_RATE = 1e-4
# -------------------------

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Wczytanie Danych ---
train_dir = PROCESSED_DIR / 'train'
val_dir = PROCESSED_DIR / 'test'  # Używamy 'test' jako walidacyjnego

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
    label_mode='binary'
)
print(f"Znalezione klasy: {train_ds.class_names}")

# --- 3. Obliczenie Wag Klas ---
print("Obliczanie wag klas...")
n_normal = len(list(train_dir.glob('NORMAL/*.*')))
n_pneumonia = len(list(train_dir.glob('PNEUMONIA/*.*')))
total_train = n_normal + n_pneumonia

if n_normal == 0 or n_pneumonia == 0:
    print("BŁĄD: Jeden z folderów (NORMAL lub PNEUMONIA) jest pusty.")
    class_weights = None
else:
    weight_for_0 = (1 / n_normal) * (total_train / 2.0)
    weight_for_1 = (1 / n_pneumonia) * (total_train / 2.0)
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"Waga dla klasy 0 (NORMAL): {weight_for_0:.2f}")
    print(f"Waga dla klasy 1 (PNEUMONIA): {weight_for_1:.2f}\n")

# --- 4. Optymalizacja Wydajności Wczytywania ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 5. Definicja Modelu ---

# ##################################################################
#  POPRAWKA: Dodajemy SILNIEJSZĄ augmentację, aby walczyć z overfittingiem
# ##################################################################
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),  # <-- DODANE
        layers.RandomContrast(0.1),  # <-- DODANE
    ],
    name="augmentacja"
)


def build_model_final():
    model = keras.Sequential(name="Efficient_CNN_v4_Strong_Aug")

    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(layers.Rescaling(1. / 255))
    model.add(data_augmentation)  # Używamy nowej, silniejszej augmentacji

    # Warstwy konwolucyjne z regularyzacją L2
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(L2_STRENGTH)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(L2_STRENGTH)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(L2_STRENGTH)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(L2_STRENGTH)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Głowa klasyfikacyjna
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(L2_STRENGTH)))
    model.add(layers.Dropout(0.5))  # Dropout też jest formą regularyzacji

    model.add(layers.Dense(1, activation='sigmoid'))

    return model


model = build_model_final()

# --- 6. Kompilacja Modelu ---
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 7. Definicja Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',  # Obserwuj stratę walidacyjną
    patience=5,  # Cierpliwość: 5 epok bez poprawy
    restore_best_weights=True,  # Przywróć model z najlepszej epoki
    verbose=1
)

# --- 8. Trening Modelu ---
print("\nRozpoczynam trening modelu (v4 ze wzmocnioną augmentacją)...")

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
    print("Trening zakończony.")
    model_save_path = PROCESSED_DIR.parent / 'final_model_v4.keras'
    model.save(model_save_path)
    print(f"Model zapisany w: {model_save_path}")

    print("Generowanie wykresów...")

    # Bierzemy tylko tyle epok, ile faktycznie trwało (przed zatrzymaniem)
    epochs_ran = len(history.history['loss'])
    epochs_range = range(epochs_ran)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Dokładność Treningowa')
    plt.plot(epochs_range, val_acc, label='Dokładność Walidacyjna')
    plt.legend(loc='lower right')
    plt.title('Dokładność (Model v4 - Strong Aug)')
    plt.xlabel('Epoki')
    plt.ylabel('Dokładność')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Strata Treningowa')
    plt.plot(epochs_range, val_loss, label='Strata Walidacyjna')
    plt.legend(loc='upper right')
    plt.title('Strata (Model v4 - Strong Aug)')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')

    plot_path = PLOT_DIR / 'training_history_v4.png'
    plt.savefig(plot_path)
    print(f"Wykresy zapisane w: {plot_path}")