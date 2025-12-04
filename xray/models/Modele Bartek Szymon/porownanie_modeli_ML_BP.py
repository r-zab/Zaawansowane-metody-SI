"""
UPROSZCZONY SKRYPT PORÓWNANIA MODELI - BEZ TENSORFLOW
Porównanie Regresji Logistycznej i Lasu Losowego
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, roc_curve)
import time
import warnings
warnings.filterwarnings('ignore')

# ======================== KONFIGURACJA ========================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'chest_xray_split_500_processed'
PLOT_DIR = BASE_DIR / 'plots'

IMG_SIZE = 128
BATCH_SIZE = 16

PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("PORÓWNANIE DWÓCH MODELI ML - KLASYFIKACJA CHEST X-RAY")
print("=" * 100)

# ======================== 1. WCZYTANIE DANYCH ========================
print("\n[DANE] Wczytywanie i przygotowywanie danych...")

def load_images_from_directory(directory, img_size=IMG_SIZE):
    """Wczytuje obrazy z katalogu i spłaszcza je"""
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
        class_dir = Path(directory) / class_name
        if not class_dir.exists():
            print(f"BŁĄD: Katalog {class_dir} nie istnieje!")
            continue
        
        # Wczytanie obrazów
        from PIL import Image
        for img_file in sorted(class_dir.glob('*.jpeg'))[:500]:  # Limit do 500 obrazów
            try:
                img = Image.open(img_file).convert('L')  # Konwersja na grayscale
                img_resized = img.resize((img_size, img_size))
                img_array = np.array(img_resized).flatten() / 255.0  # Normalizacja
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"Błąd przy wczytywaniu {img_file}: {e}")
                continue
    
    return np.array(images), np.array(labels)

print("[DANE] Wczytywanie danych treningowych...")
try:
    from PIL import Image
    X_train, y_train = load_images_from_directory(DATA_DIR / 'train')
    print(f"[DANE] ✓ Dane treningowe: {X_train.shape}")
except ImportError:
    print("[DANE] ✗ Pillow nie zainstalowany - używam danych losowych dla demo")
    X_train = np.random.rand(200, IMG_SIZE * IMG_SIZE)
    y_train = np.random.randint(0, 2, 200)

print("[DANE] Wczytywanie danych testowych...")
try:
    X_test, y_test = load_images_from_directory(DATA_DIR / 'test')
    print(f"[DANE] ✓ Dane testowe: {X_test.shape}")
except:
    print("[DANE] Brak danych testowych - generuję dane losowe")
    X_test = np.random.rand(100, IMG_SIZE * IMG_SIZE)
    y_test = np.random.randint(0, 2, 100)

# Normalizacja
print("\n[DANE] Normalizacja danych...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("[DANE] ✓ Normalizacja zakończona")

# ======================== 2. REGRESJA LOGISTYCZNA ========================
print("\n[LR] Trening Regresji Logistycznej...")
start_time = time.time()

lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, verbose=0)
lr_model.fit(X_train_scaled, y_train)

time_lr = time.time() - start_time

y_pred_lr = lr_model.predict(X_test_scaled)
y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
recall_lr = recall_score(y_test, y_pred_lr, zero_division=0)
f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)
auc_lr = roc_auc_score(y_test, y_proba_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)

print(f"[LR] ✓ Czas treningu: {time_lr:.2f}s")
print(f"[LR]   Dokładność: {acc_lr:.4f}")
print(f"[LR]   Precyzja: {prec_lr:.4f}")
print(f"[LR]   Recall: {recall_lr:.4f}")
print(f"[LR]   F1-Score: {f1_lr:.4f}")
print(f"[LR]   AUC-ROC: {auc_lr:.4f}")

# ======================== 3. LAS LOSOWY ========================
print("\n[RF] Trening Lasu Losowego...")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_model.fit(X_train_scaled, y_train)

time_rf = time.time() - start_time

y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
auc_rf = roc_auc_score(y_test, y_proba_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print(f"[RF] ✓ Czas treningu: {time_rf:.2f}s")
print(f"[RF]   Dokładność: {acc_rf:.4f}")
print(f"[RF]   Precyzja: {prec_rf:.4f}")
print(f"[RF]   Recall: {recall_rf:.4f}")
print(f"[RF]   F1-Score: {f1_rf:.4f}")
print(f"[RF]   AUC-ROC: {auc_rf:.4f}")

# ======================== 4. PORÓWNANIE WYNIKÓW ========================
print("\n[MAIN] PORÓWNANIE WYNIKÓW")
print("-" * 100)

results_df = pd.DataFrame({
    'Model': ['Regresja Logistyczna', 'Las Losowy'],
    'Dokładność': [acc_lr, acc_rf],
    'Precyzja': [prec_lr, prec_rf],
    'Recall': [recall_lr, recall_rf],
    'F1-Score': [f1_lr, f1_rf],
    'AUC-ROC': [auc_lr, auc_rf],
    'Czas treningu (s)': [time_lr, time_rf]
})

print("\n[MAIN] TABELA PORÓWNAWCZA:")
print(results_df.to_string(index=False))

results_df.to_csv(PLOT_DIR / 'porownanie_modeli_ML_BP.csv', index=False)
print(f"\n✓ Wyniki CSV: {PLOT_DIR / 'porownanie_modeli_ML_BP.csv'}")

# ======================== 5. WIZUALIZACJE ========================
print("\n[VIZ] Generowanie wizualizacji...")

fig = plt.figure(figsize=(14, 10))

models = results_df['Model']
colors = ['#1f77b4', '#ff7f0e']

# 1. Dokładność
ax1 = plt.subplot(2, 3, 1)
ax1.bar(models, results_df['Dokładność'], color=colors)
ax1.set_ylabel('Dokładność')
ax1.set_title('Porównanie Dokładności')
ax1.set_ylim([0, 1])
for i, v in enumerate(results_df['Dokładność']):
    ax1.text(i, v + 0.02, f'{v:.4f}', ha='center')

# 2. Precyzja
ax2 = plt.subplot(2, 3, 2)
ax2.bar(models, results_df['Precyzja'], color=colors)
ax2.set_ylabel('Precyzja')
ax2.set_title('Porównanie Precyzji')
ax2.set_ylim([0, 1])
for i, v in enumerate(results_df['Precyzja']):
    ax2.text(i, v + 0.02, f'{v:.4f}', ha='center')

# 3. Recall
ax3 = plt.subplot(2, 3, 3)
ax3.bar(models, results_df['Recall'], color=colors)
ax3.set_ylabel('Recall')
ax3.set_title('Porównanie Recall')
ax3.set_ylim([0, 1])
for i, v in enumerate(results_df['Recall']):
    ax3.text(i, v + 0.02, f'{v:.4f}', ha='center')

# 4. F1-Score
ax4 = plt.subplot(2, 3, 4)
ax4.bar(models, results_df['F1-Score'], color=colors)
ax4.set_ylabel('F1-Score')
ax4.set_title('Porównanie F1-Score')
ax4.set_ylim([0, 1])
for i, v in enumerate(results_df['F1-Score']):
    ax4.text(i, v + 0.02, f'{v:.4f}', ha='center')

# 5. AUC-ROC
ax5 = plt.subplot(2, 3, 5)
ax5.bar(models, results_df['AUC-ROC'], color=colors)
ax5.set_ylabel('AUC-ROC')
ax5.set_title('Porównanie AUC-ROC')
ax5.set_ylim([0, 1])
for i, v in enumerate(results_df['AUC-ROC']):
    ax5.text(i, v + 0.02, f'{v:.4f}', ha='center')

# 6. Czas treningu
ax6 = plt.subplot(2, 3, 6)
ax6.bar(models, results_df['Czas treningu (s)'], color=colors)
ax6.set_ylabel('Czas (sekundy)')
ax6.set_title('Porównanie Czasu Treningu')
for i, v in enumerate(results_df['Czas treningu (s)']):
    ax6.text(i, v + 0.1, f'{v:.2f}s', ha='center')

plt.tight_layout()
plt.savefig(PLOT_DIR / 'porownanie_metryki_ML_BP.png', dpi=300, bbox_inches='tight')
print(f"✓ Metryki: {PLOT_DIR / 'porownanie_metryki_ML_BP.png'}")

# ======================== MACIERZE BŁĘDÓW ========================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Macierz błędów - Regresja Logistyczna')
axes[0].set_ylabel('Rzeczywiste')
axes[0].set_xlabel('Predykcje')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False)
axes[1].set_title('Macierz błędów - Las Losowy')
axes[1].set_ylabel('Rzeczywiste')
axes[1].set_xlabel('Predykcje')

plt.tight_layout()
plt.savefig(PLOT_DIR / 'macierze_bledow_ML_BP.png', dpi=300, bbox_inches='tight')
print(f"✓ Macierze: {PLOT_DIR / 'macierze_bledow_ML_BP.png'}")

# ======================== KRZYWE ROC ========================
fig, ax = plt.subplots(figsize=(10, 8))

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

ax.plot(fpr_lr, tpr_lr, label=f'Regresja Logistyczna (AUC={auc_lr:.4f})', linewidth=2)
ax.plot(fpr_rf, tpr_rf, label=f'Las Losowy (AUC={auc_rf:.4f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Losowe (AUC=0.5)', linewidth=1)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Porównanie Krzywych ROC', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'krzywe_roc_ML_BP.png', dpi=300, bbox_inches='tight')
print(f"✓ Krzywe ROC: {PLOT_DIR / 'krzywe_roc_ML_BP.png'}")

# ======================== WNIOSKI ========================
print("\n" + "=" * 100)
print("WNIOSKI I REKOMENDACJE")
print("=" * 100)

best_model_idx = results_df['AUC-ROC'].idxmax()
best_model = results_df.loc[best_model_idx, 'Model']
best_auc = results_df.loc[best_model_idx, 'AUC-ROC']

print(f"\n✓ NAJLEPSZY MODEL: {best_model}")
print(f"  AUC-ROC: {best_auc:.4f}")

print("\n📊 SZCZEGÓŁOWA ANALIZA:")

for idx, row in results_df.iterrows():
    print(f"\n  {idx + 1}. {row['Model']}")
    print(f"     Dokładność: {row['Dokładność']:.4f}")
    print(f"     Precyzja:   {row['Precyzja']:.4f}")
    print(f"     Recall:     {row['Recall']:.4f}")
    print(f"     F1-Score:   {row['F1-Score']:.4f}")
    print(f"     AUC-ROC:    {row['AUC-ROC']:.4f}")
    print(f"     Czas:       {row['Czas treningu (s)']:.2f}s")

# Analiza porównawcza
improvement_pct = ((auc_rf - auc_lr) / auc_lr) * 100

print(f"\n🔍 ANALIZA PORÓWNAWCZA:")
print(f"  Poprawa AUC (Las Losowy vs Regresja): {improvement_pct:+.2f}%")

if auc_rf > auc_lr:
    if improvement_pct > 5:
        print(f"\n  ✅ LAS LOSOWY ISTOTNIE LEPSZY")
        print(f"     Poprawa AUC wynosi {improvement_pct:.2f}% (>5%)")
    else:
        print(f"\n  ⚠️  LAS LOSOWY MARGINALNIE LEPSZY")
        print(f"     Poprawa AUC wynosi {improvement_pct:.2f}% (<5%)")
else:
    print(f"\n  ℹ️  REGRESJA LOGISTYCZNA LEPSZA")
    print(f"     Przewaga: {abs(improvement_pct):.2f}%")

print("\n" + "=" * 100)
print("✓ ANALIZA MODELI ML ZAKOŃCZONA!")
print("=" * 100)
print("\nPliki wynikowe w folderze 'plots':")
print("  - porownanie_modeli_ML_BP.csv")
print("  - porownanie_metryki_ML_BP.png")
print("  - macierze_bledow_ML_BP.png")
print("  - krzywe_roc_ML_BP.png")
print("=" * 100)
