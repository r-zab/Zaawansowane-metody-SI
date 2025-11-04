import os
from PIL import Image
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. KONFIGURACJA ---
SOURCE_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\chest_xray')

# Gdzie zapisać wykresy
OUTPUT_PLOT_DIR = Path(r'C:\Users\rafal\PycharmProjects\Zaawansowane-metody-SI\xray\plots')


# -------------------------

def analyze_distribution():
    if not SOURCE_DIR.exists():
        print(f"Błąd krytyczny: Ścieżka źródłowa nie istnieje: {SOURCE_DIR}")
        return

    # Stwórz folder na wykresy
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Rozpoczynam skanowanie {SOURCE_DIR}...")

    image_data = []  # Lista na dane o każdym obrazku
    img_extensions = ('.jpeg', '.jpg', '.png')
    image_count = 0

    # Przejdź przez wszystkie pliki
    for file_path in SOURCE_DIR.rglob('*'):
        if file_path.suffix.lower() in img_extensions:
            image_count += 1
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    # Zapisz dane
                    image_data.append({
                        'path': file_path.relative_to(SOURCE_DIR),
                        'width': width,
                        'height': height,
                        'pixels': width * height,
                        'aspect_ratio': round(width / height, 2)
                    })

            except Exception as e:
                print(f"\n[BŁĄD] Nie można odczytać obrazu {file_path.relative_to(SOURCE_DIR)}: {e}", file=sys.stderr)

            if image_count % 500 == 0:
                print(f"  ...przeskanowano {image_count} obrazów...")

    print(f"\nSkanowanie zakończone. Przeanalizowano {image_count} obrazów.")

    if not image_data:
        print("Nie znaleziono obrazów.")
        return

    # --- Analiza z użyciem Pandas ---
    print("Rozpoczynam analizę statystyczną...")

    # Konwertuj listę słowników na DataFrame
    df = pd.DataFrame(image_data)

    # 1. Pokaż podstawowe statystyki (min, max, średnia, mediana (50%))
    print("\n--- Podstawowe statystyki (df.describe()) ---")
    # Używamy formatu .to_string(), żeby ładnie się wyświetliło
    print(df.describe().to_string())

    # --- Rysowanie Histogramów ---
    print(f"\nGenerowanie wykresów w folderze: {OUTPUT_PLOT_DIR}...")

    # Ustawiamy styl wykresów
    plt.style.use('ggplot')

    # 1. Histogram Szerokości
    plt.figure(figsize=(12, 7))
    df['width'].plot(kind='hist', bins=100, title='Dystrybucja Szerokości Obrazów (px)')
    plt.xlabel('Szerokość (px)')
    plt.ylabel('Liczba obrazów')
    plt.savefig(OUTPUT_PLOT_DIR / '1_width_distribution.png')
    plt.close()

    # 2. Histogram Wysokości
    plt.figure(figsize=(12, 7))
    df['height'].plot(kind='hist', bins=100, title='Dystrybucja Wysokości Obrazów (px)')
    plt.xlabel('Wysokość (px)')
    plt.ylabel('Liczba obrazów')
    plt.savefig(OUTPUT_PLOT_DIR / '2_height_distribution.png')
    plt.close()

    # 3. Histogram Łącznej Liczby Pikseli
    plt.figure(figsize=(12, 7))
    df['pixels'].plot(kind='hist', bins=100, title='Dystrybucja Łącznej Liczby Pikseli (w milionach)')
    plt.xlabel('Liczba Pikseli (x1,000,000)')
    plt.ylabel('Liczba obrazów')
    plt.savefig(OUTPUT_PLOT_DIR / '3_pixels_distribution.png')
    plt.close()

    # 4. Histogram Proporcji (Aspect Ratio)
    plt.figure(figsize=(12, 7))
    df['aspect_ratio'].plot(kind='hist', bins=50, title='Dystrybucja Proporcji (Szerokość / Wysokość)')
    plt.xlabel('Proporcje (np. 1.0 = kwadrat, >1 = poziomy, <1 = pionowy)')
    plt.ylabel('Liczba obrazów')
    plt.savefig(OUTPUT_PLOT_DIR / '4_aspect_ratio_distribution.png')
    plt.close()

    print("--- Gotowe! ---")
    print("Sprawdź pliki .png w folderze 'plots'. Pomogą wam zrozumieć dane.")


# Uruchomienie skryptu
if __name__ == "__main__":
    analyze_distribution()