import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
import logging

class MLSlotClassifier:
    """Classificador de machine learning para verificação de slots baseado em treinamento OK/NG."""
    
    def __init__(self, slot_id: str = None):
        self.slot_id = slot_id
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.model_path = None
        
        # Configurações do classificador
        self.classifier_type = 'random_forest'  # 'random_forest' ou 'svm'
        self.min_samples_for_training = 4  # Mínimo de amostras para treinar (2 OK + 2 NG)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def extract_features(self, roi_image: np.ndarray) -> np.ndarray:
        """Extrai características da imagem ROI para classificação.
        
        Args:
            roi_image: Imagem da região de interesse (ROI)
            
        Returns:
            Array numpy com as características extraídas
        """
        try:
            if roi_image is None or roi_image.size == 0:
                return np.array([])
                
            # Converte para escala de cinza se necessário
            if len(roi_image.shape) == 3:
                gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi_image.copy()
                
            features = []
            
            # 1. Características estatísticas básicas
            features.extend([
                np.mean(gray),           # Média da intensidade
                np.std(gray),            # Desvio padrão
                np.min(gray),            # Valor mínimo
                np.max(gray),            # Valor máximo
                np.median(gray),         # Mediana
                np.percentile(gray, 25), # Primeiro quartil
                np.percentile(gray, 75), # Terceiro quartil
            ])
            
            # 2. Características de histograma
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normaliza
            features.extend(hist.tolist())
            
            # 3. Características de textura (LBP simplificado)
            lbp_features = self._calculate_lbp_features(gray)
            features.extend(lbp_features)
            
            # 4. Características de contorno
            contour_features = self._calculate_contour_features(gray)
            features.extend(contour_features)
            
            # 5. Características de gradiente
            gradient_features = self._calculate_gradient_features(gray)
            features.extend(gradient_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair características: {e}")
            return np.array([])
    
    def _calculate_lbp_features(self, gray: np.ndarray) -> List[float]:
        """Calcula características de Local Binary Pattern simplificado."""
        try:
            # LBP simplificado - compara pixel central com vizinhos
            h, w = gray.shape
            lbp = np.zeros_like(gray)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray[i, j]
                    code = 0
                    code |= (gray[i-1, j-1] >= center) << 7
                    code |= (gray[i-1, j] >= center) << 6
                    code |= (gray[i-1, j+1] >= center) << 5
                    code |= (gray[i, j+1] >= center) << 4
                    code |= (gray[i+1, j+1] >= center) << 3
                    code |= (gray[i+1, j] >= center) << 2
                    code |= (gray[i+1, j-1] >= center) << 1
                    code |= (gray[i, j-1] >= center) << 0
                    lbp[i, j] = code
            
            # Histograma do LBP
            hist_lbp = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [16], [0, 256])
            hist_lbp = hist_lbp.flatten() / (hist_lbp.sum() + 1e-7)
            
            return hist_lbp.tolist()
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular LBP: {e}")
            return [0.0] * 16
    
    def _calculate_contour_features(self, gray: np.ndarray) -> List[float]:
        """Calcula características baseadas em contornos."""
        try:
            # Detecção de bordas
            edges = cv2.Canny(gray, 50, 150)
            
            # Encontra contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return [0.0] * 6
            
            # Características do maior contorno
            largest_contour = max(contours, key=cv2.contourArea)
            
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Evita divisão por zero
            if perimeter == 0:
                compactness = 0
            else:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            extent = float(area) / (w * h) if (w * h) != 0 else 0
            
            # Convex hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area != 0 else 0
            
            return [
                area / (gray.shape[0] * gray.shape[1]),  # Área normalizada
                perimeter / (2 * (gray.shape[0] + gray.shape[1])),  # Perímetro normalizado
                compactness,
                aspect_ratio,
                extent,
                solidity
            ]
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular características de contorno: {e}")
            return [0.0] * 6
    
    def _calculate_gradient_features(self, gray: np.ndarray) -> List[float]:
        """Calcula características baseadas em gradientes."""
        try:
            # Gradientes Sobel
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Magnitude e direção do gradiente
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)
            
            return [
                np.mean(magnitude),
                np.std(magnitude),
                np.mean(np.abs(grad_x)),
                np.mean(np.abs(grad_y)),
                np.std(direction),
            ]
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular características de gradiente: {e}")
            return [0.0] * 5
    
    def train(self, training_samples: List[Dict]) -> Dict:
        """Treina o classificador com as amostras fornecidas.
        
        Args:
            training_samples: Lista de dicionários com 'roi' e 'label' ('OK' ou 'NG')
            
        Returns:
            Dicionário com métricas de treinamento
        """
        try:
            if len(training_samples) < self.min_samples_for_training:
                raise ValueError(f"Mínimo de {self.min_samples_for_training} amostras necessárias para treinamento")
            
            # Extrai características e labels
            X = []
            y = []
            
            for sample in training_samples:
                roi = sample['roi']
                label = sample['label']
                
                features = self.extract_features(roi)
                if features.size > 0:
                    X.append(features)
                    y.append(1 if label == 'OK' else 0)  # 1 para OK, 0 para NG
            
            if len(X) == 0:
                raise ValueError("Nenhuma característica válida foi extraída")
            
            X = np.array(X)
            y = np.array(y)
            
            # Normaliza as características
            X_scaled = self.scaler.fit_transform(X)
            
            # Divide em treino e teste se há amostras suficientes
            if len(X) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_test = X_scaled, X_scaled
                y_train, y_test = y, y
            
            # Cria e treina o classificador
            if self.classifier_type == 'random_forest':
                self.classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
            else:  # SVM
                self.classifier = SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42,
                    class_weight='balanced'
                )
            
            self.classifier.fit(X_train, y_train)
            
            # Avalia o modelo
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation se há amostras suficientes
            if len(X) >= 8:
                cv_scores = cross_val_score(self.classifier, X_scaled, y, cv=min(5, len(X)//2))
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
            else:
                cv_mean = accuracy
                cv_std = 0.0
            
            self.is_trained = True
            
            # Salva nomes das características para referência
            self.feature_names = self._get_feature_names()
            
            metrics = {
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'n_samples': len(training_samples),
                'n_ok': sum(1 for s in training_samples if s['label'] == 'OK'),
                'n_ng': sum(1 for s in training_samples if s['label'] == 'NG'),
                'classifier_type': self.classifier_type
            }
            
            self.logger.info(f"Modelo treinado com sucesso. Acurácia: {accuracy:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro durante treinamento: {e}")
            raise
    
    def predict(self, roi_image: np.ndarray) -> Tuple[str, float]:
        """Classifica uma imagem ROI.
        
        Args:
            roi_image: Imagem da região de interesse
            
        Returns:
            Tupla com (classificação, confiança)
        """
        try:
            if not self.is_trained:
                raise ValueError("Modelo não foi treinado")
            
            features = self.extract_features(roi_image)
            if features.size == 0:
                return 'NG', 0.0
            
            # Normaliza as características
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predição
            prediction = self.classifier.predict(features_scaled)[0]
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            
            # Confiança é a probabilidade da classe predita
            confidence = probabilities[prediction]
            
            label = 'OK' if prediction == 1 else 'NG'
            
            return label, confidence
            
        except Exception as e:
            self.logger.error(f"Erro durante predição: {e}")
            return 'NG', 0.0
    
    def save_model(self, filepath: str) -> bool:
        """Salva o modelo treinado em arquivo.
        
        Args:
            filepath: Caminho para salvar o modelo
            
        Returns:
            True se salvou com sucesso
        """
        try:
            if not self.is_trained:
                raise ValueError("Modelo não foi treinado")
            
            model_data = {
                'classifier': self.classifier,
                'scaler': self.scaler,
                'classifier_type': self.classifier_type,
                'feature_names': self.feature_names,
                'slot_id': self.slot_id
            }
            
            # Cria diretório se não existir
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(model_data, filepath)
            self.model_path = filepath
            
            self.logger.info(f"Modelo salvo em: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Carrega um modelo salvo.
        
        Args:
            filepath: Caminho do modelo salvo
            
        Returns:
            True se carregou com sucesso
        """
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
            
            model_data = joblib.load(filepath)
            
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.classifier_type = model_data.get('classifier_type', 'random_forest')
            self.feature_names = model_data.get('feature_names', [])
            self.slot_id = model_data.get('slot_id', self.slot_id)
            self.is_trained = True
            self.model_path = filepath
            
            self.logger.info(f"Modelo carregado de: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def _get_feature_names(self) -> List[str]:
        """Retorna os nomes das características extraídas."""
        names = [
            'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
            'median_intensity', 'q1_intensity', 'q3_intensity'
        ]
        
        # Histograma (32 bins)
        names.extend([f'hist_bin_{i}' for i in range(32)])
        
        # LBP (16 bins)
        names.extend([f'lbp_bin_{i}' for i in range(16)])
        
        # Contornos
        names.extend([
            'normalized_area', 'normalized_perimeter', 'compactness',
            'aspect_ratio', 'extent', 'solidity'
        ])
        
        # Gradientes
        names.extend([
            'mean_magnitude', 'std_magnitude', 'mean_grad_x',
            'mean_grad_y', 'std_direction'
        ])
        
        return names
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna a importância das características (apenas para Random Forest).
        
        Returns:
            Dicionário com nome da característica e sua importância
        """
        if not self.is_trained or self.classifier_type != 'random_forest':
            return {}
        
        try:
            importances = self.classifier.feature_importances_
            feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importances))]
            
            return dict(zip(feature_names, importances))
            
        except Exception as e:
            self.logger.error(f"Erro ao obter importância das características: {e}")
            return {}
    
    def evaluate_model(self, test_samples: List[Dict]) -> Dict:
        """Avalia o modelo com amostras de teste.
        
        Args:
            test_samples: Lista de amostras para teste
            
        Returns:
            Dicionário com métricas de avaliação
        """
        try:
            if not self.is_trained:
                raise ValueError("Modelo não foi treinado")
            
            predictions = []
            true_labels = []
            confidences = []
            
            for sample in test_samples:
                roi = sample['roi']
                true_label = sample['label']
                
                pred_label, confidence = self.predict(roi)
                
                predictions.append(pred_label)
                true_labels.append(true_label)
                confidences.append(confidence)
            
            # Calcula métricas
            correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
            accuracy = correct / len(predictions) if predictions else 0
            
            # Matriz de confusão
            tp = sum(1 for p, t in zip(predictions, true_labels) if p == 'OK' and t == 'OK')
            tn = sum(1 for p, t in zip(predictions, true_labels) if p == 'NG' and t == 'NG')
            fp = sum(1 for p, t in zip(predictions, true_labels) if p == 'OK' and t == 'NG')
            fn = sum(1 for p, t in zip(predictions, true_labels) if p == 'NG' and t == 'OK')
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'mean_confidence': np.mean(confidences),
                'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
                'n_samples': len(test_samples)
            }
            
        except Exception as e:
            self.logger.error(f"Erro durante avaliação: {e}")
            return {}