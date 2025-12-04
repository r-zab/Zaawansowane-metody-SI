"""
GŁÓWNY SKRYPT PORÓWNANIA MODELI
Orchestracja wszystkich modułów
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve

# Importy własnych modułów
from przygotowanie_danych_BP import DataPreparer
from modele_ml_BP import LogisticRegressionModel, RandomForestModel
from siec_neuronowa_BP import NeuralNetworkModel
from wizualizacja_BP import ResultsVisualizer

# ======================== KONFIGURACJA ========================
BASE_DIR = Path(__file__).parent
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS_NN = 20

print("=" * 100)
print("PORÓWNANIE TRZECH MODELI - KLASYFIKACJA CHEST X-RAY")
print("=" * 100)

# ======================== ETAP 1: PRZYGOTOWANIE DANYCH ========================
print("\n[MAIN] ETAP 1: PRZYGOTOWANIE DANYCH")
print("-" * 100)

data_preparer = DataPreparer(BASE_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
data = data_preparer.get_all_data()

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
X_train_scaled = data['X_train_scaled']
X_test_scaled = data['X_test_scaled']
train_ds = data['train_ds']
val_ds = data['val_ds']
test_ds = data['test_ds']

# ======================== ETAP 2: REGRESJA LOGISTYCZNA ========================
print("\n[MAIN] ETAP 2: REGRESJA LOGISTYCZNA")
print("-" * 100)

lr_model = LogisticRegressionModel()
lr_model.train(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_proba_lr = lr_model.predict_proba(X_test_scaled)
results_lr = lr_model.evaluate(X_test_scaled, y_test, y_pred_lr, y_proba_lr)

# ======================== ETAP 3: LAS LOSOWY ========================
print("\n[MAIN] ETAP 3: LAS LOSOWY")
print("-" * 100)

rf_model = RandomForestModel(n_estimators=100, max_depth=20)
rf_model.train(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)
results_rf = rf_model.evaluate(X_test_scaled, y_test, y_pred_rf, y_proba_rf)

# ======================== ETAP 4: SIEĆ NEURONOWA ========================
print("\n[MAIN] ETAP 4: SIEĆ NEURONOWA")
print("-" * 100)

nn_model = NeuralNetworkModel(img_size=IMG_SIZE, learning_rate=1e-4)
nn_model.train(train_ds, val_ds, epochs=EPOCHS_NN)
y_proba_nn = nn_model.predict(test_ds).flatten()
y_pred_nn = (y_proba_nn > 0.5).astype(int)
results_nn = nn_model.evaluate(test_ds, y_test, y_pred_nn, y_proba_nn)

# ======================== ETAP 5: PORÓWNANIE WYNIKÓW ========================
print("\n[MAIN] ETAP 5: PORÓWNANIE WYNIKÓW")
print("-" * 100)

results_dict = {
    'models': ['Regresja Logistyczna', 'Las Losowy', 'Sieć Neuronowa'],
    'accuracy': [results_lr['accuracy'], results_rf['accuracy'], results_nn['accuracy']],
    'precision': [results_lr['precision'], results_rf['precision'], results_nn['precision']],
    'recall': [results_lr['recall'], results_rf['recall'], results_nn['recall']],
    'f1': [results_lr['f1'], results_rf['f1'], results_nn['f1']],
    'auc': [results_lr['auc'], results_rf['auc'], results_nn['auc']],
    'time': [results_lr['training_time'], results_rf['training_time'], results_nn['training_time']]
}

visualizer = ResultsVisualizer(BASE_DIR / 'plots')

# Tworzy tabelę
results_df = visualizer.create_comparison_table(results_dict)

print("\n[MAIN] TABELA WYNIKÓW:")
print(results_df.to_string(index=False))

# ======================== ETAP 6: WIZUALIZACJE ========================
print("\n[MAIN] ETAP 6: GENEROWANIE WIZUALIZACJI")
print("-" * 100)

# Wykresy metryk
visualizer.plot_metrics_comparison(results_df)

# Macierze błędów
cm_dict = {
    'Regresja Logistyczna': results_lr['confusion_matrix'],
    'Las Losowy': results_rf['confusion_matrix'],
    'Sieć Neuronowa': results_nn['confusion_matrix']
}
visualizer.plot_confusion_matrices(cm_dict)

# Krzywe ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_proba_nn)

roc_dict = {
    'Regresja Logistyczna': (fpr_lr, tpr_lr, results_lr['auc']),
    'Las Losowy': (fpr_rf, tpr_rf, results_rf['auc']),
    'Sieć Neuronowa': (fpr_nn, tpr_nn, results_nn['auc'])
}
visualizer.plot_roc_curves(roc_dict)

# Historia treningu NN
visualizer.plot_training_history(results_nn['history'])

# Raport tekstowy
visualizer.generate_summary_report(results_df)

# Zapisanie wyników do CSV
results_df.to_csv(BASE_DIR / 'plots' / 'wyniki_porownania_BP.csv', index=False)
print(f"[VIZ] ✓ Wyniki CSV zapisane: {BASE_DIR / 'plots' / 'wyniki_porownania_BP.csv'}")

# ======================== ETAP 7: WNIOSKI ========================
print("\n[MAIN] ETAP 7: ANALIZA I WNIOSKI")
print("-" * 100)

best_model_idx = results_df['AUC-ROC'].idxmax()
best_model = results_df.loc[best_model_idx, 'Model']
best_auc = results_df.loc[best_model_idx, 'AUC-ROC']

print(f"\n✓ NAJLEPSZY MODEL: {best_model}")
print(f"  AUC-ROC: {best_auc:.4f}")

print("\n📊 PODSUMOWANIE WYDAJNOŚCI:")

for idx, row in results_df.iterrows():
    print(f"\n  {idx + 1}. {row['Model']}")
    print(f"     Dokładność: {row['Dokładność']:.4f}")
    print(f"     Precyzja:   {row['Precyzja']:.4f}")
    print(f"     Recall:     {row['Recall']:.4f}")
    print(f"     F1-Score:   {row['F1-Score']:.4f}")
    print(f"     AUC-ROC:    {row['AUC-ROC']:.4f}")
    print(f"     Czas:       {row['Czas treningu (s)']:.2f}s")

# Analiza sensu budowy sieci neuronowej
print("\n🔍 OCENA SENSU BUDOWY SIECI NEURONOWEJ:")

improvement_rf_vs_lr = ((results_df.loc[1, 'AUC-ROC'] - results_df.loc[0, 'AUC-ROC']) / results_df.loc[0, 'AUC-ROC']) * 100
improvement_nn_vs_rf = ((results_df.loc[2, 'AUC-ROC'] - results_df.loc[1, 'AUC-ROC']) / results_df.loc[1, 'AUC-ROC']) * 100
improvement_nn_vs_lr = ((results_df.loc[2, 'AUC-ROC'] - results_df.loc[0, 'AUC-ROC']) / results_df.loc[0, 'AUC-ROC']) * 100

print(f"\n  Poprawa Las Losowy vs Regresja: {improvement_rf_vs_lr:+.2f}%")
print(f"  Poprawa Sieć vs Las Losowy:    {improvement_nn_vs_rf:+.2f}%")
print(f"  Poprawa Sieć vs Regresja:      {improvement_nn_vs_lr:+.2f}%")

if results_df.loc[2, 'AUC-ROC'] > results_df.loc[1, 'AUC-ROC'] and improvement_nn_vs_rf > 5:
    print(f"\n  ✅ SIEĆ NEURONOWA UZASADNIONA")
    print(f"     Poprawa AUC wynosi {improvement_nn_vs_rf:.2f}% (>5%)")
    print(f"     Warto inwestować w tę architekturę")
elif results_df.loc[2, 'AUC-ROC'] > results_df.loc[1, 'AUC-ROC']:
    print(f"\n  ⚠️  SIEĆ NEURONOWA MARGINALNIE LEPSZA")
    print(f"     Poprawa AUC wynosi tylko {improvement_nn_vs_rf:.2f}% (<5%)")
    print(f"     Można rozważyć użycie prostszego modelu Las Losowy")
else:
    print(f"\n  ❌ SIEĆ NEURONOWA GORSZA OD LASU LOSOWEGO")
    print(f"     Rekomendacja: Użyć Las Losowy")

print("\n" + "=" * 100)
print("✓ ANALIZA ZAKOŃCZONA!")
print("=" * 100)
print("\nPliki wynikowe znajdują się w folderze 'plots':")
print("  - wyniki_porownania_BP.csv")
print("  - porownanie_metryki_BP.png")
print("  - macierze_bledow_BP.png")
print("  - krzywe_roc_BP.png")
print("  - historia_treningu_BP.png")
print("  - raport_porownania_BP.txt")
print("=" * 100)
