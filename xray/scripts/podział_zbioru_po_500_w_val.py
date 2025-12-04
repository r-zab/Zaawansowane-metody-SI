import os
import shutil
import random
from pathlib import Path
import sys

# --- 1. KONFIGURACJA ---

# Ścieżka do folderu, w którym znajduje się 'chest_xray_processed_padding'
# (Folder nadrzędny)
SOURCE_BASE_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\datasets')

# Nazwa folderu źródłowego (tego, którego teraz używasz)
SOURCE_FOLDER_NAME = 'chest_xray_processed_padding'

# Nazwa NOWEGO folderu, który zostanie stworzony z idealnym podziałem
DEST_FOLDER_NAME = 'chest_xray_processed_val1000_balanced'

# Ile obrazów ma być w każdej klasie zbioru walidacyjnego
VAL_IMAGES_PER_CLASS = 500

# -------------------------

# Pełne ścieżki
SOURCE_DIR = SOURCE_BASE_DIR / SOURCE_FOLDER_NAME
DEST_DIR = SOURCE_BASE_DIR / DEST_FOLDER_NAME

def create_balanced_dataset():
    print(f"📖 Źródło danych: {SOURCE_DIR}")
    print(f"✨ Nowy folder docelowy: {DEST_DIR}\n")

    # --- Bezpieczeństwo ---
    if not SOURCE_DIR.exists():
        print(f"BŁĄD KRYTYCZNY: Folder źródłowy nie istnieje: {SOURCE_DIR}", file=sys.stderr)
        return

    if DEST_DIR.exists():
        print(f"BŁĄD: Folder docelowy '{DEST_DIR.name}' już istnieje!", file=sys.stderr)
        print("Usuń go ręcznie (lub zmień nazwę 'DEST_FOLDER_NAME') i spróbuj ponownie.", file=sys.stderr)
        return

    try:
        # --- Krok 1: Kopiowanie zbioru 'test' w całości ---
        # Ten zbiór pozostaje nietknięty
        print("Krok 1: Kopiowanie zbioru 'test'...")
        shutil.copytree(SOURCE_DIR / 'test', DEST_DIR / 'test')
        print(" -> Zbiór 'test' skopiowany.\n")

        # --- Krok 2: Tworzenie zrównoważonych zbiorów 'train' i 'val' ---
        print(f"Krok 2: Tworzenie zrównoważonego zbioru 'val' (po {VAL_IMAGES_PER_CLASS} obrazów)...")
        
        classes = ['NORMAL', 'PNEUMONIA']
        for class_name in classes:
            print(f"  Przetwarzanie klasy: {class_name}")
            
            # Ścieżki źródłowe
            source_train_dir = SOURCE_DIR / 'train' / class_name
            source_val_dir = SOURCE_DIR / 'val' / class_name
            
            # Ścieżki docelowe (tworzymy je)
            dest_train_dir = DEST_DIR / 'train' / class_name
            dest_val_dir = DEST_DIR / 'val' / class_name
            dest_train_dir.mkdir(parents=True, exist_ok=True)
            dest_val_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Stwórz jedną dużą pulę plików (z train i val)
            train_files = list(source_train_dir.glob('*.*'))
            val_files = list(source_val_dir.glob('*.*'))
            all_files = train_files + val_files
            random.shuffle(all_files) # Mieszamy całą pulę
            
            total_files = len(all_files)
            print(f"    -> Znaleziono łącznie {total_files} obrazów.")
            
            if total_files < VAL_IMAGES_PER_CLASS:
                print(f"    BŁĄD: Za mało obrazów ({total_files}), by stworzyć zbiór walidacyjny ({VAL_IMAGES_PER_CLASS})", file=sys.stderr)
                continue

            # 2. Podziel pulę na zbiór walidacyjny i treningowy
            files_for_val = all_files[:VAL_IMAGES_PER_CLASS]
            files_for_train = all_files[VAL_IMAGES_PER_CLASS:]
            
            print(f"    -> Przenoszenie {len(files_for_val)} obrazów do 'val'")
            print(f"    -> Przenoszenie {len(files_for_train)} obrazów do 'train'")

            # 3. Kopiuj pliki do nowych lokalizacji
            for f in files_for_val:
                shutil.copy(f, dest_val_dir / f.name)
            
            for f in files_for_train:
                shutil.copy(f, dest_train_dir / f.name)
            
            print("    -> Gotowe.\n")

        print("\n--- Zakończono! ---")
        print(f"🎉 Sukces! Nowy, zrównoważony zbiór danych jest gotowy w folderze:")
        print(f"{DEST_DIR}")

    except Exception as e:
        print(f"\nBŁĄD PODCZAS PRZETWARZANIA: {e}", file=sys.stderr)
        # W razie błędu, posprzątaj niekompletny folder
        if DEST_DIR.exists():
            print(f"Sprzątanie niekompletnego folderu: {DEST_DIR}", file=sys.stderr)
            shutil.rmtree(DEST_DIR)

# Uruchomienie skryptu
if __name__ == "__main__":
    create_balanced_dataset()