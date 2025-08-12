#!/usr/bin/env python3
"""
Script de demonstração para treinamento de modelo de Machine Learning.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Adicionar diretório modulos ao path
modulos_dir = Path(__file__).parent / "modulos"
sys.path.insert(0, str(modulos_dir))

def create_training_samples():
    """Cria amostras sintéticas para demonstração."""
    print("🎨 Criando amostras de treinamento sintéticas...")
    
    samples = []
    
    # Amostras OK - padrões regulares
    for i in range(5):
        # Cria imagem com padrão regular (OK)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Adiciona formas geométricas regulares
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
        cv2.circle(img, (50, 50), 15, (0, 0, 255), -1)
        
        # Adiciona ruído sutil
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        img = np.clip(img, 0, 255)
        
        samples.append({
            'roi': img,
            'label': 'OK'
        })
        print(f"✅ Amostra OK {i+1} criada")
    
    # Amostras NG - padrões irregulares
    for i in range(5):
        # Cria imagem com padrão irregular (NG)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Adiciona formas irregulares
        points = np.array([
            [20, 20], [80, 30], [70, 80], [30, 70], [20, 20]
        ], np.int32)
        cv2.fillPoly(img, [points], (255, 255, 255))
        
        # Adiciona ruído mais intenso
        noise = np.random.normal(0, 30, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        img = np.clip(img, 0, 255)
        
        samples.append({
            'roi': img,
            'label': 'NG'
        })
        print(f"❌ Amostra NG {i+1} criada")
    
    return samples

def demo_training():
    """Demonstra o treinamento de um modelo ML."""
    try:
        print("🚀 Demonstração de Treinamento ML")
        print("=" * 50)
        
        # Importa o classificador ML
        from modulos.ml_classifier import MLSlotClassifier
        
        # Cria classificador
        clf = MLSlotClassifier(slot_id="demo_slot")
        print(f"✅ Classificador criado - Tipo: {clf.classifier_type}")
        
        # Cria amostras de treinamento
        training_samples = create_training_samples()
        print(f"\n📊 Total de amostras criadas: {len(training_samples)}")
        print(f"   • OK: {sum(1 for s in training_samples if s['label'] == 'OK')}")
        print(f"   • NG: {sum(1 for s in training_samples if s['label'] == 'NG')}")
        
        # Treina o modelo
        print(f"\n🧠 Treinando modelo...")
        metrics = clf.train(training_samples)
        
        print(f"✅ Modelo treinado com sucesso!")
        print(f"📈 Acurácia: {metrics['accuracy']:.1%}")
        print(f"🔄 Validação Cruzada: {metrics['cv_mean']:.1%} (±{metrics['cv_std']:.1%})")
        print(f"📊 Amostras utilizadas: {metrics['n_samples']}")
        
        # Testa o modelo treinado
        print(f"\n🧪 Testando modelo treinado...")
        
        # Testa com amostras de treinamento
        correct = 0
        total = 0
        
        for i, sample in enumerate(training_samples):
            prediction, confidence = clf.predict(sample['roi'])
            expected = sample['label']
            
            is_correct = prediction == expected
            if is_correct:
                correct += 1
            total += 1
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} Amostra {i+1}: Esperado={expected}, Predito={prediction}, Confiança={confidence:.3f}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n📊 Acurácia no conjunto de treinamento: {accuracy:.1%}")
        
        # Testa com novas imagens
        print(f"\n🆕 Testando com novas imagens...")
        
        # Nova imagem OK
        new_ok = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(new_ok, (25, 25), (75, 75), (255, 255, 255), -1)
        cv2.circle(new_ok, (50, 50), 12, (0, 0, 255), -1)
        
        pred_ok, conf_ok = clf.predict(new_ok)
        print(f"   🆕 Nova OK: Predito={pred_ok}, Confiança={conf_ok:.3f}")
        
        # Nova imagem NG
        new_ng = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.ellipse(new_ng, (50, 50), (30, 20), 45, 0, 360, (255, 255, 255), -1)
        
        pred_ng, conf_ng = clf.predict(new_ng)
        print(f"   🆕 Nova NG: Predito={pred_ng}, Confiança={conf_ng:.3f}")
        
        # Mostra importância das características (se Random Forest)
        if clf.classifier_type == 'random_forest':
            print(f"\n🌳 Importância das características (top 10):")
            importances = clf.get_feature_importance()
            
            # Ordena por importância
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                print(f"   {i+1:2d}. {feature}: {importance:.4f}")
        
        # Salva o modelo
        print(f"\n💾 Salvando modelo...")
        model_path = "demo_model.joblib"
        if clf.save_model(model_path):
            print(f"✅ Modelo salvo em: {model_path}")
        else:
            print("❌ Falha ao salvar modelo")
        
        print("\n" + "=" * 50)
        print("✅ Demonstração de treinamento concluída!")
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("💡 Verifique se todas as dependências estão instaladas")
    except Exception as e:
        print(f"❌ Erro durante demonstração: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_training()
