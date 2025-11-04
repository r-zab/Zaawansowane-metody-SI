import os
from PIL import Image
from pathlib import Path
import sys

# --- 1. KONFIGURACJA ---
# Ustaw ścieżkę do GŁÓWNEGO folderu ze zbiorem danych (tam gdzie są 'train', 'test', 'val')
SOURCE_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray')


# -------------------------


def find_min_resolutions():
    if not SOURCE_DIR.exists():
        print(f"Błąd krytyczny: Ścieżka źródłowa nie istnieje: {SOURCE_DIR}")
        print("Popraw ścieżkę 'SOURCE_DIR' w skrypcie i spróbuj ponownie.")
        return

    print(f"Rozpoczynam skanowanie folderu: {SOURCE_DIR}")
    print("Szukanie obrazów o najmniejszych wymiarach, to może potrwać chwilę...\n")

    # Inicjujemy wartości 'nieskończonością',
    # aby każda rzeczywista wartość była mniejsza
    min_pixels = float('inf')
    min_width = float('inf')
    min_height = float('inf')

    min_pixels_dims = (0, 0)
    min_pixels_image_path = None
    min_width_image_path = None
    min_height_image_path = None

    img_extensions = ('.jpeg', '.jpg', '.png')
    image_count = 0

    # Użyj rglob() do rekursywnego (głębokiego) przeszukania wszystkich podfolderów
    for file_path in SOURCE_DIR.rglob('*'):

        if file_path.suffix.lower() in img_extensions:
            image_count += 1

            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    current_pixels = width * height

                    # 1. Sprawdź, czy ten obraz ma najmniejszą SZEROKOŚĆ
                    if width < min_width:
                        min_width = width
                        min_width_image_path = file_path

                    # 2. Sprawdź, czy ten obraz ma najmniejszą WYSOKOŚĆ
                    if height < min_height:
                        min_height = height
                        min_height_image_path = file_path

                    # 3. Sprawdź, czy ten obraz ma najmniejsze POLE (liczbę pikseli)
                    if current_pixels < min_pixels:
                        min_pixels = current_pixels
                        min_pixels_dims = (width, height)
                        min_pixels_image_path = file_path

            except Exception as e:
                print(f"\n[BŁĄD] Nie można odczytać obrazu {file_path.relative_to(SOURCE_DIR)}: {e}", file=sys.stderr)

            if image_count % 500 == 0:
                print(f"  ...przeskanowano {image_count} obrazów...")

    # --- Wyświetlenie wyników ---
    print(f"\n--- Skanowanie zakończone ---")
    print(f"Łącznie sprawdzono: {image_count} obrazów.\n")

    if min_width_image_path:
        print(f"📊 Najmniejsza znaleziona SZEROKOŚĆ:")
        print(f"  Wartość: {min_width} px")
        print(f"  Obraz:   {min_width_image_path.relative_to(SOURCE_DIR)}\n")

    if min_height_image_path:
        print(f"📊 Najmniejsza znaleziona WYSOKOŚĆ:")
        print(f"  Wartość: {min_height} px")
        print(f"  Obraz:   {min_height_image_path.relative_to(SOURCE_DIR)}\n")

    if min_pixels_image_path:
        print(f"📊 Obraz o najmniejszym POLU (najmniej pikseli):")
        print(f"  Rozdzielczość: {min_pixels_dims[0]} x {min_pixels_dims[1]} px")
        print(f"  Łączna liczba pikseli: {min_pixels:,}")
        print(f"  Obraz:   {min_pixels_image_path.relative_to(SOURCE_DIR)}\n")

    if not image_count:
        print("Nie znaleziono żadnych obrazów w podanym folderze.")


# Uruchomienie skryptu
if __name__ == "__main__":
    find_min_resolutions()