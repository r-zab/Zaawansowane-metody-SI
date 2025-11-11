import os
from PIL import Image
from pathlib import Path
import sys

# --- 1. KONFIGURACJA ---
# Ustaw ścieżkę do GŁÓWNEGO folderu ze zbiorem danych (tam gdzie są 'train', 'test', 'val')
SOURCE_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray_split_500')

# Ustaw ścieżkę, gdzie skrypt ma zapisać NOWE, przetworzone obrazy
# Najlepiej, aby był to nowy, pusty folder
OUTPUT_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray_split_500_processed')

# Ustaw docelowy rozmiar obrazu (kwadrat).
# 512 to dobry kompromis.
# Jeśli dostaniesz błąd "CUDA out of memory" podczas treningu,
# zmień tę wartość na 256 i uruchom skrypt ponownie.
TARGET_SIZE = 512


# -------------------------


def standardize_image(img_path: Path, output_path: Path, size: int):
    """
    Wczytuje obraz, zachowuje proporcje przez dodanie czarnych pasów (padding)
    do kwadratu, a następnie skaluje go do docelowego rozmiaru.
    """
    try:
        # Otwórz obraz
        with Image.open(img_path) as img:

            # Konwertuj do 'RGB' (dla modeli CNN)
            img = img.convert('RGB')

            # --- Metoda: Padding do kwadratu + Skalowanie ---

            # 1. Zmieniamy rozmiar obrazu, ALE ZACHOWUJĄC PROPORCJE,
            #    tak aby zmieścił się w pudełku (size, size).
            #    Np. 1500x1000 -> 512x341 (jeśli size=512)
            #    Używamy LANCZOS dla najlepszej jakości.
            img.thumbnail((size, size), Image.Resampling.LANCZOS)

            # 2. Tworzymy nowe, kwadratowe, czarne tło
            padded_img = Image.new("RGB", (size, size), (0, 0, 0))

            # 3. Obliczamy, gdzie wkleić obraz, aby był na środku
            x_offset = (size - img.width) // 2
            y_offset = (size - img.height) // 2

            # 4. Wklejamy przeskalowany obraz na środek czarnego tła
            padded_img.paste(img, (x_offset, y_offset))

            # 5. Zapisujemy gotowy, kwadratowy obraz
            padded_img.save(output_path)

    except Exception as e:
        print(f"\n[BŁĄD] Nie można przetworzyć obrazu {img_path}: {e}", file=sys.stderr)


# --- Główna funkcja przetwarzająca ---

def process_dataset():
    if not SOURCE_DIR.exists():
        print(f"Błąd krytyczny: Ścieżka źródłowa nie istnieje: {SOURCE_DIR}")
        print("Popraw ścieżkę 'SOURCE_DIR' w skrypcie i spróbuj ponownie.")
        return

    # Stwórz folder wyjściowy, jeśli nie istnieje
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Rozpoczynam standaryzację obrazów ze źródła: {SOURCE_DIR}")
    print(f"Metoda: Padding (czarne pasy) + Skalowanie do {TARGET_SIZE}x{TARGET_SIZE} px")
    print(f"Przetworzone obrazy zostaną zapisane w: {OUTPUT_DIR}\n")

    sets_to_process = ['train', 'test', 'val']
    classes = ['NORMAL', 'PNEUMONIA']

    img_extensions = ('.jpeg', '.jpg', '.png')
    total_processed = 0

    # Przejdź przez wszystkie foldery (train/NORMAL, train/PNEUMONIA, itd.)
    for set_name in sets_to_process:
        for class_name in classes:

            # Ścieżka do folderu z oryginalnymi obrazami
            input_folder = SOURCE_DIR / set_name / class_name

            # Ścieżka do folderu, gdzie zapiszemy nowe obrazy
            output_folder = OUTPUT_DIR / set_name / class_name

            # Stwórz folder docelowy, jeśli nie istnieje
            output_folder.mkdir(parents=True, exist_ok=True)

            if not input_folder.exists():
                print(f"Pominięto, brak folderu: {input_folder}")
                continue

            print(f"--- Przetwarzanie: {input_folder.relative_to(SOURCE_DIR)} ---")

            # Znajdź wszystkie pliki obrazów w folderze
            image_files = []
            for ext in img_extensions:
                image_files.extend(input_folder.glob(f'*{ext}'))

            if not image_files:
                print("  Brak obrazów do przetworzenia.")
                continue

            # Przetwórz każdy obraz
            count_in_folder = 0
            for img_path in image_files:
                output_path = output_folder / img_path.name
                standardize_image(img_path, output_path, TARGET_SIZE)
                count_in_folder += 1
                total_processed += 1

                # Prosty wskaźnik postępu
                print(f"\r  Przetworzono {count_in_folder}/{len(image_files)} obrazów...", end="")
            print("\n  Gotowe.\n")

    print(f"--- Zakończono standaryzację! ---")
    print(f"Łącznie przetworzono {total_processed} obrazów.")
    print(f"Znajdziesz je w folderze: {OUTPUT_DIR}")


# Uruchomienie skryptu
if __name__ == "__main__":
    process_dataset()