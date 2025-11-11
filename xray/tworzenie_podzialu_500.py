import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- 1. KONFIGURACJA ---

# Ścieżka do ORYGINALNEGO zbioru (tego z Kaggle)
# Upewnij się, że wskazuje na folder 'chest_xray'
ORIGINAL_DATA_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray')

# Gdzie zapisać NOWY podział (stworzy ten folder)
NEW_SPLIT_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray_split_500')

# Ile obrazów ma być w NOWYM zbiorze walidacyjnym
VAL_SIZE = 500


# -------------------------

def create_new_split():
    print("Rozpoczynam tworzenie nowego podziału danych...")

    original_train_dir = ORIGINAL_DATA_DIR / 'train'

    # Foldery docelowe
    new_train_dir = NEW_SPLIT_DIR / 'train'
    new_val_dir = NEW_SPLIT_DIR / 'val'
    new_test_dir = NEW_SPLIT_DIR / 'test'  # Test zostaje nietknięty

    # Stwórz całą strukturę folderów
    (new_train_dir / 'NORMAL').mkdir(parents=True, exist_ok=True)
    (new_train_dir / 'PNEUMONIA').mkdir(parents=True, exist_ok=True)
    (new_val_dir / 'NORMAL').mkdir(parents=True, exist_ok=True)
    (new_val_dir / 'PNEUMONIA').mkdir(parents=True, exist_ok=True)

    # --- 1. Znajdź wszystkie pliki ---
    normal_files = list((original_train_dir / 'NORMAL').glob('*.*'))
    pneumonia_files = list((original_train_dir / 'PNEUMONIA').glob('*.*'))

    all_files = normal_files + pneumonia_files
    # Etykiety: 0 dla NORMAL, 1 dla PNEUMONIA
    labels = [0] * len(normal_files) + [1] * len(pneumonia_files)

    print(f"Znaleziono {len(normal_files)} obrazów NORMAL i {len(pneumonia_files)} PNEUMONIA.")

    # --- 2. Podziel pliki na train i val (500) ---
    # Chcemy 500 obrazów w walidacji. Obliczmy, jaki to procent
    val_split_ratio = VAL_SIZE / len(all_files)

    # Użyj sklearn do inteligentnego podziału z zachowaniem proporcji klas
    train_files, val_files = train_test_split(
        all_files,
        test_size=val_split_ratio,  # Użyj proporcji, nie liczby stałej
        stratify=labels,  # Kluczowe: zachowaj proporcje klas
        random_state=42  # Ustal seed dla powtarzalności
    )

    print(f"\nNowy podział:")
    print(f"  Zbiór treningowy: {len(train_files)} plików")
    print(f"  Zbiór walidacyjny: {len(val_files)} plików (powinno być blisko {VAL_SIZE})")

    # --- 3. Kopiuj pliki do nowych lokalizacji ---

    def copy_files(file_list, destination_dir):
        count = 0
        for file_path in file_list:
            # Sprawdź, czy to NORMAL czy PNEUMONIA
            class_name = file_path.parent.name
            dest_folder = destination_dir / class_name

            # Kopiuj
            shutil.copy(file_path, dest_folder / file_path.name)
            count += 1
            print(f"\rKopiowanie do {destination_dir.name}: {count}/{len(file_list)}", end="")
        print("\nGotowe.")

    print("\n--- Kopiowanie plików treningowych ---")
    copy_files(train_files, new_train_dir)

    print("\n--- Kopiowanie plików walidacyjnych ---")
    copy_files(val_files, new_val_dir)

    # --- 4. Skopiuj oryginalny zbiór TEST (nietknięty) ---
    print("\n--- Kopiowanie zbioru testowego ---")
    original_test_dir = ORIGINAL_DATA_DIR / 'test'
    if new_test_dir.exists():
        shutil.rmtree(new_test_dir)  # Usuń, jeśli istnieje, by uniknąć błędów
    shutil.copytree(original_test_dir, new_test_dir)

    print(f"\n*** Sukces! ***")
    print(f"Nowy zbiór danych gotowy w: {NEW_SPLIT_DIR}")


if __name__ == "__main__":
    create_new_split()