"""
MODUŁY MODELI - REGRESJA I LAS LOSOWY
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42, 
                                       n_jobs=-1, verbose=0)
        self.results = {}
        
    def train(self, X_train, y_train):
        """Trening modelu"""
        print("\n[LR] Trening Regresji Logistycznej...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        print(f"[LR] ✓ Czas treningu: {self.training_time:.2f}s")
        
    def predict(self, X_test):
        """Predykcja"""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Predykcja probabilistyczna"""
        return self.model.predict_proba(X_test)[:, 1]
    
    def evaluate(self, X_test, y_test, y_pred, y_proba):
        """Ewaluacja modelu"""
        self.results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'training_time': self.training_time
        }
        
        print(f"[LR]   Dokładność: {self.results['accuracy']:.4f}")
        print(f"[LR]   Precyzja: {self.results['precision']:.4f}")
        print(f"[LR]   Recall: {self.results['recall']:.4f}")
        print(f"[LR]   F1-Score: {self.results['f1']:.4f}")
        print(f"[LR]   AUC-ROC: {self.results['auc']:.4f}")
        
        return self.results


class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=20):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        self.results = {}
        
    def train(self, X_train, y_train):
        """Trening modelu"""
        print("\n[RF] Trening Lasu Losowego...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        print(f"[RF] ✓ Czas treningu: {self.training_time:.2f}s")
        
    def predict(self, X_test):
        """Predykcja"""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Predykcja probabilistyczna"""
        return self.model.predict_proba(X_test)[:, 1]
    
    def evaluate(self, X_test, y_test, y_pred, y_proba):
        """Ewaluacja modelu"""
        self.results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'training_time': self.training_time,
            'feature_importance': self.model.feature_importances_
        }
        
        print(f"[RF]   Dokładność: {self.results['accuracy']:.4f}")
        print(f"[RF]   Precyzja: {self.results['precision']:.4f}")
        print(f"[RF]   Recall: {self.results['recall']:.4f}")
        print(f"[RF]   F1-Score: {self.results['f1']:.4f}")
        print(f"[RF]   AUC-ROC: {self.results['auc']:.4f}")
        
        return self.results
    
    def get_feature_importance(self):
        """Zwraca ważność cech"""
        return self.model.feature_importances_
