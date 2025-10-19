import os
from PIL import Image
from pathlib import Path
import sys

# --- 1. KONFIGURACJA ---
# Ustaw ścieżkę do GŁÓWNEGO folderu ze zbiorem danych (tam gdzie są 'train', 'test', 'val')
# Użyj 'r' przed ścieżką, aby uniknąć problemów ze znakami '\'
SOURCE_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray')

# Ustaw ścieżkę, gdzie skrypt ma zapisać NOWE, przetworzone obrazy
# Najlepiej, aby był to nowy, pusty folder
OUTPUT_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray_processed')

# Ustaw docelowy rozmiar obrazu (kwadrat). 224x224 to popularny wybór.
TARGET_SIZE = 224


# -------------------------


def standardize_image(img_path: Path, output_path: Path, size: int):
    """
    Wczytuje obraz, wycina centralny kwadrat i skaluje go do docelowego rozmiaru.
    """
    try:
        # Otwórz obraz
        with Image.open(img_path) as img:

            # Konwertuj do 'RGB'. Większość modeli CNN oczekuje 3 kanałów.
            # Jeśli obraz jest w skali szarości (tryb 'L'),
            # to skopiuje jego wartości do wszystkich 3 kanałów (R, G, B).
            img = img.convert('RGB')

            # --- Metoda: Kadrowanie do kwadratu (sugerowane przez profesora) ---

            # Znajdź krótszy bok obrazu
            short_side = min(img.size)
            width, height = img.size

            # Oblicz koordynaty do wycięcia centralnego kwadratu
            left = (width - short_side) / 2
            top = (height - short_side) / 2
            right = (width + short_side) / 2
            bottom = (height + short_side) / 2

            # Wytnij centralny kwadrat
            img_cropped = img.crop((left, top, right, bottom))

            # Przeskaluj kwadrat do docelowego rozmiaru (size x size)
            # Używamy LANCZOS dla najlepszej jakości przeskalowania
            img_resized = img_cropped.resize((size, size), Image.Resampling.LANCZOS)

            # Zapisz przetworzony obraz w folderze docelowym
            img_resized.save(output_path)

    except Exception as e:
        print(f"\n[BŁĄD] Nie można przetworzyć obrazu {img_path}: {e}", file=sys.stderr)


# --- Główna funkcja przetwarzająca ---

def process_dataset():
    if not SOURCE_DIR.exists():
        print(f"Błąd krytyczny: Ścieżka źródłowa nie istnieje: {SOURCE_DIR}")
        print("Popraw ścieżkę 'SOURCE_DIR' w skrypcie i spróbuj ponownie.")
        return

    print(f"Rozpoczynam standaryzację obrazów ze źródła: {SOURCE_DIR}")
    print(f"Przetworzone obrazy zostaną zapisane w: {OUTPUT_DIR}\n")

    # Foldery w zbiorze danych (na podstawie Twojego zrzutu ekranu)
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