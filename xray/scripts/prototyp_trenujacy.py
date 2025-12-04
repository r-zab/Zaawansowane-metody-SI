import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from keras.callbacks import EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np  
import seaborn as sns  
from sklearn.metrics import confusion_matrix, classification_report # <--- DODANO classification_report

# --- 1. KONFIGURACJA ---
#te ściezki musisz zmienic: processed dir i base dir
PROCESSED_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\datasets\chest_xray_processed_val1000_balanced')
BASE_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray')
RESULTS_DIR = BASE_DIR / 'wykresy' / 'wyniki_04_12_bs_16_img512' 
MODELS_DIR = BASE_DIR / 'models'
# PARAMETRY TRENINGU
IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 30
L2_STRENGTH = 1e-4
LEARNING_RATE = 1e-4
# -------------------------
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
    label_mode='categorical'
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
else:
    class_weights = None

# --- 4. Optymalizacja Wydajności ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE) 

# --- 5. Definicja Modelu ---
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="augmentacja"
)

def build_model_final():
    model = keras.Sequential(name="Efficient_CNN_v4_Two_Outputs")

    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(layers.Rescaling(1. / 255))
    model.add(data_augmentation)

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

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(L2_STRENGTH)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(2, activation='softmax')) # 2 wyjścia
    
    return model

model = build_model_final()

# --- 6. Kompilacja Modelu ---
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# --- 7. Trening ---
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
)

if class_weights:
    history = model.fit(
        train_ds, epochs=EPOCHS, validation_data=val_ds,
        class_weight=class_weights, callbacks=[early_stopping]
    )
else:
    history = None

# --- 9. Ewaluacja i Metryki ---
# === 9a. Generowanie wykresów historii treningu (TEGO BRAKOWAŁO) ===
    print("Generowanie wykresów historii treningu...")

    epochs_ran = len(history.history['loss'])
    epochs_range = range(epochs_ran)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 6))
    
    # Wykres Dokładności
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Dokładność Treningowa')
    plt.plot(epochs_range, val_acc, label='Dokładność Walidacyjna')
    plt.legend(loc='lower right')
    plt.title('Dokładność (2 Outputs)')
    plt.xlabel('Epoki')
    plt.ylabel('Accuracy')

    # Wykres Straty
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Strata Treningowa')
    plt.plot(epochs_range, val_loss, label='Strata Walidacyjna')
    plt.legend(loc='upper right')
    plt.title('Strata (Categorical Crossentropy)')
    plt.xlabel('Epoki')
    plt.ylabel('Loss')

    # Zapis do poprawnego folderu RESULTS_DIR
    plot_path = RESULTS_DIR / 'training_history_04_12_bs16_imgs_512.png'
    plt.savefig(plot_path)
    print(f"Wykresy historii zapisane w: {plot_path}")
    plt.close()
if history:
    # Zapis modelu
    model.save(MODELS_DIR / '04_12_dwieklasy_softmax_bs16_imgs512.keras')


    # === 9b. Macierz Konfuzji ===
    print("\nGenerowanie predykcji...")
    y_true_onehot = np.concatenate([y for x, y in val_ds], axis=0)
    y_true = np.argmax(y_true_onehot, axis=1) # [0, 1, 0...]
    
    y_pred_probs = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)  # [0, 1, 0...]

    cm = confusion_matrix(y_true, y_pred)
    
    # Rysowanie macierzy
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Macierz Konfuzji')
    plt.ylabel('Prawda')
    plt.xlabel('Predykcja')
    plt.savefig(RESULTS_DIR / 'confusion_matrix_04_12_bs16_imgs512.png')
    plt.close()

    # === 9c. ZAAWANSOWANE METRYKI MEDYCZNE (Nowość) ===
    print("\n" + "="*40)
    print("SZCZEGÓŁOWY RAPORT KLASYFIKACJI")
    print("="*40)
    
    # 1. Raport sklearn (Precyzja, Recall, F1 dla każdej klasy)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 2. Ręczne wyliczenie Czułości i Swoistości (dla pewności)
    # Zakładamy: 0 = NORMAL, 1 = PNEUMONIA
    # cm[0,0] -> TN (Prawdziwie Zdrowy)
    # cm[0,1] -> FP (Fałszywy Alarm - Zdrowy uznany za chorego)
    # cm[1,0] -> FN (Fałszywie Ujemny - Chory uznany za zdrowego) <--- NAJGROŹNIEJSZE!
    # cm[1,1] -> TP (Prawdziwie Chory)
    
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) # Czułość (Recall dla Pneumonia)
    specificity = tn / (tn + fp) # Swoistość (Recall dla Normal)
    precision = tp / (tp + fp)   # Precyzja
    
    print("-" * 30)
    print(f"METRYKI MEDYCZNE:")
    print(f"Czułość (Sensitivity/Recall): {sensitivity:.4f}  <-- Ważne: Ile chorych wykryliśmy?")
    print(f"Swoistość (Specificity):      {specificity:.4f}  <-- Ważne: Ile zdrowych nie dostało leków niepotrzebnie?")
    print(f"Precyzja (Precision):         {precision:.4f}")
    print("-" * 30)
    
    # Zapis wyników do pliku tekstowego
    with open(RESULTS_DIR/ 'wyniki_metryki_04_12_bs16_imgs512.txt', 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write(f"\nCzułość: {sensitivity:.4f}\nSwoistość: {specificity:.4f}\n")
        
    print(f"Pełny raport zapisano w: {RESULTS_DIR / 'wyniki_metryki_04_12_bs16_imgs512.txt'}")