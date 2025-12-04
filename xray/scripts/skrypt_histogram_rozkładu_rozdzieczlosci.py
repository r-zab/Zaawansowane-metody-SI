import os
from PIL import Image
from pathlib import Path
import sys

# --- 1. KONFIGURACJA ---
# Ustaw ścieżkę do GŁÓWNEGO folderu ze zbiorem danych (tam gdzie są 'train', 'test', 'val')
# Użyj 'r' przed ścieżką, aby uniknąć problemów ze znakami '\'
SOURCE_DIR = Path(r'C:\Users\jakub\OneDrive\Pulpit\studia semestr 5\zaawansowane metody i techniki AI\laboratorium\Zaawansowane-metody-SI\chest_xray')


# -------------------------


def find_max_resolution():
    if not SOURCE_DIR.exists():
        print(f"Błąd krytyczny: Ścieżka źródłowa nie istnieje: {SOURCE_DIR}")
        print("Popraw ścieżkę 'SOURCE_DIR' w skrypcie i spróbuj ponownie.")
        return

    print(f"Rozpoczynam skanowanie folderu: {SOURCE_DIR}")
    print("Szukanie obrazu o największej rozdzielczości, to może potrwać chwilę...\n")

    max_pixels = 0
    max_dims = (0, 0)
    max_res_image_path = None

    # Lista rozszerzeń, których szukamy
    img_extensions = ('.jpeg', '.jpg', '.png')
    image_count = 0

    # Użyj rglob() do rekursywnego (głębokiego) przeszukania wszystkich podfolderów
    for file_path in SOURCE_DIR.rglob('*'):

        # Sprawdź, czy plik ma jedno z poszukiwanych rozszerzeń
        if file_path.suffix.lower() in img_extensions:
            image_count += 1

            try:
                # Otwórz obraz (tylko na tyle, by odczytać jego metadane, nie ładuje całego)
                with Image.open(file_path) as img:
                    width, height = img.size
                    current_pixels = width * height

                    # Sprawdź, czy ten obraz jest większy niż dotychczasowy rekordzista
                    if current_pixels > max_pixels:
                        max_pixels = current_pixels
                        max_dims = (width, height)
                        max_res_image_path = file_path

            except Exception as e:
                # Obsługa błędu, gdyby jakiś plik był uszkodzony
                print(f"\n[BŁĄD] Nie można odczytać obrazu {file_path.relative_to(SOURCE_DIR)}: {e}", file=sys.stderr)

            # Prosty wskaźnik postępu, żeby było widać, że coś się dzieje
            if image_count % 500 == 0:
                print(f"  ...przeskanowano {image_count} obrazów...")

    # --- Wyświetlenie wyników ---
    print(f"\n--- Skanowanie zakończone ---")
    print(f"Łącznie sprawdzono: {image_count} obrazów.\n")

    if max_res_image_path:
        print(f"🏆 Znaleziono obraz o największej rozdzielczości:")
        # Wyświetlamy ścieżkę względną, aby była bardziej czytelna
        print(f"  Ścieżka: {max_res_image_path.relative_to(SOURCE_DIR)}")
        print(f"  Rozdzielczość: {max_dims[0]} x {max_dims[1]} px")
        print(f"  Łączna liczba pikseli: {max_pixels:,}")  # formatuje liczbę z separatorami
    else:
        print("Nie znaleziono żadnych obrazów w podanym folderze.")


# Uruchomienie skryptu
if __name__ == "__main__":
    find_max_resolution()