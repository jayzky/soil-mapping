import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SoilPredictor:
    def __init__(self):
        self.models = {
            'soil_type': RandomForestClassifier(n_estimators=100, random_state=42),
            'soil_texture': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.label_encoders = {
            'soil_type': LabelEncoder(),
            'soil_texture': LabelEncoder()
        }
        self.feature_importances = None
        self.feature_columns = ['elevation', 'slope', 'aspect', 'ndvi', 'precipitation', 'temperature']
        self.target_columns = ['soil_type', 'soil_texture']
        self.scaler = StandardScaler()
        self._trained = False
        
    def is_trained(self):
        return self._trained
        
    def train(self, X, y):
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        # 对标签进行编码并训练模型
        y_encoded = pd.DataFrame()
        for target in self.target_columns:
            y_encoded[target] = self.label_encoders[target].fit_transform(y[target])
            self.models[target].fit(X_scaled, y_encoded[target])
        
        # 计算平均特征重要性
        self.feature_importances = np.mean([
            self.models[target].feature_importances_ 
            for target in self.target_columns
        ], axis=0)
        
        self._trained = True
    
    def predict(self, X):
        if not self._trained:
            raise ValueError("模型未训练，请先训练模型")
            
        # 特征标准化
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        # 预测
        predictions = {}
        for target in self.target_columns:
            pred = self.models[target].predict(X_scaled)
            predictions[target] = self.label_encoders[target].inverse_transform(pred)
            
        return pd.DataFrame(predictions)
    
    def get_feature_importances(self):
        if not self._trained:
            raise ValueError("模型未训练，请先训练模型")
        return list(zip(self.feature_columns, self.feature_importances.tolist()))
    
    def evaluate(self, X, y):
        if not self._trained:
            raise ValueError("模型未训练，请先训练模型")
            
        # 特征标准化
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        # 预测并计算指标
        y_pred = self.predict(X)
        metrics = {}
        
        for target in self.target_columns:
            y_true = y[target]
            y_pred_target = y_pred[target]
            
            # 计算各项指标
            metrics[target] = {
                'accuracy': float(accuracy_score(y_true, y_pred_target)),
                'precision': float(precision_score(y_true, y_pred_target, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred_target, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred_target, average='weighted', zero_division=0))
            }
            
        return metrics
    
    def save(self, path):
        if not self._trained:
            raise ValueError("模型未训练，请先训练模型")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'models': self.models,
            'label_encoders': self.label_encoders,
            'feature_importances': self.feature_importances,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        joblib.dump(model_data, path)
    
    def load(self, path):
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.label_encoders = model_data['label_encoders']
        self.feature_importances = model_data['feature_importances']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.target_columns = model_data['target_columns']
        self._trained = True 