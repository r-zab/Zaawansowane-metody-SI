import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
import os

# --- 1. KONFIGURACJA ---
PROCESSED_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray_processed_padded')
PLOT_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\plots')
IMG_SIZE = 512
BATCH_SIZE = 8  # Zacznij od 8. Przy nowym, lżejszym modelu możesz spróbować 16.
EPOCHS = 20
# -------------------------

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Wczytanie Danych ---
train_dir = PROCESSED_DIR / 'train'
val_dir = PROCESSED_DIR / 'test' # <--- UŻYWAMY ZBIORU 'test' JAKO WALIDACYJNEGO

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

# --- 3. Obliczenie Wag Klas (POPRAWIONA LOGIKA) ---
# Zlicz wszystkie pliki (*.*), aby poprawnie policzyć wagi
print("Obliczanie wag klas...")
n_normal = len(list(train_dir.glob('NORMAL/*.*')))
n_pneumonia = len(list(train_dir.glob('PNEUMONIA/*.*')))
total_train = n_normal + n_pneumonia

print(f"Obrazy 'NORMAL' w zbiorze treningowym: {n_normal}")
print(f"Obrazy 'PNEUMONIA' w zbiorze treningowym: {n_pneumonia}")

# Uniknięcie dzielenia przez zero, jeśli folder jest pusty
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

# --- 5. Definicja Modelu (NOWA, LEPSZA ARCHITEKTURA) ---

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ],
    name="augmentacja"
)


def build_model():
    model = keras.Sequential(name="Efficient_CNN_v1")

    # Warstwa wejściowa
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

    # 1. Normalizacja pikseli z [0, 255] do [0, 1]
    model.add(layers.Rescaling(1. / 255))

    # 2. Augmentacja
    model.add(data_augmentation)

    # --- Blok 1 ---
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Blok 2 ---
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Blok 3 ---
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Blok 4 (DODANY, ABY ZMNIEJSZYĆ WYMIAR) ---
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- GŁOWA KLASYFIKACYJNA (NOWA) ---

    # Zamiast Flatten() używamy GlobalAveragePooling2D
    # Redukuje to wymiar z (None, 32, 32, 128) do (None, 128)
    model.add(layers.GlobalAveragePooling2D())

    # Dodajemy gęstą warstwę, ale teraz ma ona znacznie mniej parametrów
    # (128 wejść * 64 wyjścia) + 64 = 8256 parametrów (zamiast 33.5M!)
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Warstwa wyjściowa
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


model = build_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 6. Trening Modelu ---
print("\nRozpoczynam trening modelu...")

# Upewnij się, że wagi zostały poprawnie obliczone
if class_weights:
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights
    )
else:
    print("Trening zatrzymany z powodu błędu w obliczaniu wag klas.")
    history = None  # Zatrzymujemy

# --- 7. Zapis Modelu i Wykresów (jeśli trening się odbył) ---
if history:
    print("Trening zakończony.")
    model_save_path = PROCESSED_DIR.parent / 'efficient_model_v1.keras'
    model.save(model_save_path)
    print(f"Model zapisany w: {model_save_path}")

    print("Generowanie wykresów...")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(EPOCHS)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Dokładność Treningowa')
    plt.plot(epochs_range, val_acc, label='Dokładność Walidacyjna')
    plt.legend(loc='lower right')
    plt.title('Dokładność Treningowa i Walidacyjna')
    plt.xlabel('Epoki')
    plt.ylabel('Dokładność')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Strata Treningowa')
    plt.plot(epochs_range, val_loss, label='Strata Walidacyjna')
    plt.legend(loc='upper right')
    plt.title('Strata Treningowa i Walidacyjna')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')

    plot_path = PLOT_DIR / 'training_history_v1.png'
    plt.savefig(plot_path)
    print(f"Wykresy zapisane w: {plot_path}")