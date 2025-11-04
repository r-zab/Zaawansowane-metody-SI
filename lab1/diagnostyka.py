import tensorflow as tf

print("--- DIAGNOSTYKA TENSORFLOW ---")
print("Wersja TensorFlow:", tf.__version__)

# Kluczowa komenda: Sprawdź, czy TF widzi jakiekolwiek GPU
gpus = tf.config.list_physical_devices('GPU')

print("Liczba znalezionych GPU:", len(gpus))

if gpus:
    print("\nGratulacje! Znaleziono następujące GPU:")
    try:
        # Spróbuj uzyskać szczegóły, aby potwierdzić, że CUDA działa
        for gpu in gpus:
            print(f"- {gpu.name} (Typ: {gpu.device_type})")
        print("\nWSZYSTKO WYGLĄDA DOBRZE!")
    except Exception as e:
        print(f"\n!!! BŁĄD: Znaleziono GPU, ale nie można uzyskać szczegółów: {e}")
        print("To może być problem ze sterownikiem lub cuDNN.")
else:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! KRYTYCZNY BŁĄD: TensorFlow NIE WIDZI TWOJEGO GPU !!!")
    print("!!! Trening będzie wykonywany na CPU (bardzo wolno). !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

print("--- Koniec diagnostyki ---")