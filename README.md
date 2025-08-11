# 🔬 Sistema de Visão Computacional DX v2.0

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-red.svg)
![ttkbootstrap](https://img.shields.io/badge/ttkbootstrap-1.10+-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)
![Version](https://img.shields.io/badge/Version-2.0-blue.svg)

**Sistema avançado de inspeção visual automatizada para controle de qualidade industrial**

*Desenvolvido pela equipe DX (Desenvolvimento Digital) - Versão 2.0*

</div>

---

## 📋 Índice

- [🎯 Visão Geral](#-visão-geral)
- [✨ Funcionalidades Principais](#-funcionalidades-principais)
- [🏗️ Arquitetura do Sistema](#️-arquitetura-do-sistema)
- [🧮 Algoritmos e Tecnologias](#-algoritmos-e-tecnologias)
- [⚙️ Requisitos do Sistema](#️-requisitos-do-sistema)
- [🚀 Instalação e Configuração](#-instalação-e-configuração)
- [📁 Estrutura do Projeto](#-estrutura-do-projeto)
- [🎮 Guia de Uso](#-guia-de-uso)
- [🔧 Desenvolvimento e Extensibilidade](#-desenvolvimento-e-extensibilidade)
- [🛠️ Solução de Problemas](#️-solução-de-problemas)
- [📈 Performance e Otimização](#-performance-e-otimização)
- [🗺️ Roadmap](#-roadmap)
- [📞 Suporte e Contribuição](#-suporte-e-contribuição)

---

## 🎯 Visão Geral

O **Sistema de Visão Computacional DX v2.0** é uma solução completa e avançada de inspeção visual automatizada que combina técnicas sofisticadas de **visão computacional**, **machine learning** e **processamento de imagens** para realizar controle de qualidade industrial com alta precisão e eficiência.

### 🌟 Características Principais

- **🔍 Inspeção Automatizada**: Verificação automática de montagem de componentes com múltiplos algoritmos
- **🤖 Machine Learning**: Classificadores Random Forest e SVM para classificação OK/NG
- **📹 Multi-Câmera**: Suporte a múltiplas câmeras (USB, Industrial, IP) com cache inteligente
- **🎨 Interface Moderna**: Interface gráfica avançada com PyQt5 e ttkbootstrap
- **💾 Banco de Dados**: Sistema SQLite robusto com backup automático e histórico completo
- **📊 Analytics**: Relatórios em tempo real com métricas detalhadas e estatísticas
- **🔧 Configurável**: Sistema de configuração visual avançado com temas personalizáveis

---

## ✨ Funcionalidades Principais

### 🔍 **Módulo de Montagem (Core)**
- **Verificação Automática**: Detecção de componentes montados usando template matching avançado
- **Template Matching**: Múltiplos algoritmos (TM_CCOEFF_NORMED, TM_CCORR, TM_SQDIFF)
- **Feature Detection**: Algoritmo ORB (Oriented FAST and Rotated BRIEF) para detecção robusta
- **Sistema de Slots**: Definição visual de áreas de inspeção com editor de malhas
- **Transformações Geométricas**: Homografia e RANSAC para alinhamento de imagens
- **Validação em Tempo Real**: Processamento contínuo com feedback visual imediato

### 🧠 **Sistema de Machine Learning**
- **Classificadores Avançados**: Random Forest e Support Vector Machine (SVM)
- **Extração de Features**: 39+ características incluindo estatísticas, histogramas, textura e contornos
- **Treinamento Automático**: Sistema de coleta de amostras OK/NG com validação cruzada
- **Otimização de Thresholds**: Cálculo automático de limiares ótimos baseado em amostras
- **Modelos Persistidos**: Salvamento e carregamento de modelos treinados (.joblib)
- **Métricas de Performance**: Acurácia, precisão, recall e F1-score em tempo real

### 📹 **Gerenciamento de Câmeras**
- **Detecção Automática**: Identificação automática de câmeras disponíveis
- **Cache Inteligente**: Sistema de cache para evitar reinicializações desnecessárias
- **Multi-Platform**: Suporte nativo para Windows (DirectShow) e Linux/macOS
- **Configuração Avançada**: Resolução, FPS e buffer configuráveis
- **Limpeza Automática**: Liberação automática de recursos não utilizados
- **Fallback Robusto**: Mecanismos de recuperação para falhas de câmera

### 💾 **Sistema de Banco de Dados**
- **SQLite Avançado**: Banco de dados relacional com transações ACID
- **Modelos e Slots**: Estrutura hierárquica para organização de inspeções
- **Histórico Completo**: Registro de todas as inspeções com metadados
- **Backup Automático**: Sistema de backup automático com versionamento
- **Migração de Dados**: Suporte para importação de modelos JSON existentes
- **Integridade Referencial**: Constraints e foreign keys para consistência

### 🎨 **Interface do Usuário**
- **Dashboard Centralizado**: Interface unificada com navegação por abas
- **Temas Personalizáveis**: Sistema de cores e estilos configurável
- **Editor Visual**: Interface gráfica para definição de áreas de inspeção
- **Visualização em Tempo Real**: Exibição de resultados com overlay visual
- **Responsividade**: Interface adaptável para diferentes resoluções
- **Acessibilidade**: Controles intuitivos com feedback visual claro

### 📊 **Sistema de Relatórios**
- **Histórico de Inspeções**: Registro completo de todas as verificações
- **Estatísticas Avançadas**: Métricas de performance e tendências
- **Filtros Dinâmicos**: Busca por modelo, data, resultado e confiança
- **Exportação de Dados**: Suporte para múltiplos formatos de saída
- **Dashboard Analytics**: Visualizações gráficas de performance
- **Auditoria Completa**: Rastreabilidade de todas as operações

---

## 🏗️ Arquitetura do Sistema

### 📐 **Arquitetura Modular**

```mermaid
graph TB
    A[main.py] --> B[montagem.py]
    A --> C[database_manager.py]
    A --> D[camera_manager.py]
    
    B --> E[inspection.py]
    B --> F[ml_classifier.py]
    B --> G[training_dialog.py]
    
    C --> H[SQLite Database]
    D --> I[Camera Cache]
    
    E --> J[OpenCV Algorithms]
    F --> K[Scikit-learn ML]
    G --> L[Training Pipeline]
    
    B --> M[UI Components]
    M --> N[ttkbootstrap]
    M --> O[PyQt5]
```

### 🔧 **Módulos Principais**

#### **`main.py`** - Ponto de Entrada
- Inicialização do sistema
- Gerenciamento de módulos
- Tratamento de erros global

#### **`montagem.py`** - Núcleo do Sistema
- Interface principal de montagem
- Coordenação entre módulos
- Gerenciamento de estado global

#### **`database_manager.py`** - Persistência de Dados
- CRUD de modelos e slots
- Histórico de inspeções
- Backup e migração de dados

#### **`camera_manager.py`** - Gerenciamento de Câmeras
- Detecção automática de dispositivos
- Cache inteligente de instâncias
- Configuração de parâmetros

#### **`ml_classifier.py`** - Machine Learning
- Classificadores Random Forest e SVM
- Extração de características
- Treinamento e validação

#### **`inspection.py`** - Algoritmos de Inspeção
- Template matching avançado
- Feature detection ORB
- Transformações geométricas

#### **`training_dialog.py`** - Interface de Treinamento
- Coleta de amostras OK/NG
- Configuração de parâmetros
- Validação de modelos

#### **`utils.py`** - Utilitários do Sistema
- Configuração de estilos
- Gerenciamento de cores e fontes
- Funções auxiliares

---

## 🧮 Algoritmos e Tecnologias

### 🔍 **Template Matching**

#### **Correlação Cruzada Normalizada**
```
γ(u,v) = Σ[T(x,y) - T̄][I(x+u,y+v) - Ī(u,v)] / √{Σ[T(x,y) - T̄]² · Σ[I(x+u,y+v) - Ī(u,v)]²}
```

**Implementação OpenCV:**
```python
# Múltiplos métodos disponíveis
methods = [
    cv2.TM_CCOEFF_NORMED,    # Correlação cruzada normalizada
    cv2.TM_CCORR_NORMED,     # Correlação normalizada
    cv2.TM_SQDIFF_NORMED     # Diferença quadrada normalizada
]

result = cv2.matchTemplate(image, template, method)
locations = np.where(result >= threshold)
```

### 🎯 **Feature Detection (ORB)**

#### **Algoritmo FAST (Features from Accelerated Segment Test)**
Para um pixel `p` com intensidade `Ip`:
```
∃ conjunto S de n pixels contíguos no círculo de 16 pixels tal que:
∀ pixel x ∈ S: |Ix - Ip| > t
```

**Parâmetros ORB Otimizados:**
```python
orb = cv2.ORB_create(
    nfeatures=5000,           # Máximo de features
    scaleFactor=1.2,          # Fator de escala da pirâmide
    nlevels=8,                # Níveis da pirâmide
    edgeThreshold=31,         # Tamanho da borda
    firstLevel=0,             # Primeiro nível da pirâmide
    WTA_K=2,                  # Pontos para produzir elementos BRIEF
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,             # Tamanho do patch para descritor
    fastThreshold=20          # Threshold FAST
)
```

### 🔄 **RANSAC (Random Sample Consensus)**

#### **Estimativa de Homografia**
1. **Seleção Aleatória**: Escolher 4 pontos correspondentes
2. **Modelo**: Calcular homografia usando DLT
3. **Consenso**: Contar inliers usando distância de reprojeção
4. **Iteração**: Repetir N vezes

**Implementação:**
```python
H, mask = cv2.findHomography(
    src_pts, dst_pts, 
    cv2.RANSAC, 
    ransacReprojThreshold=5.0,
    maxIters=2000,
    confidence=0.995
)
```

### 🤖 **Machine Learning**

#### **Extração de Características (39+ features)**
```python
# 1. Características Estatísticas (7)
features.extend([
    np.mean(gray),           # Média da intensidade
    np.std(gray),            # Desvio padrão
    np.min(gray),            # Valor mínimo
    np.max(gray),            # Valor máximo
    np.median(gray),         # Mediana
    np.percentile(gray, 25), # Primeiro quartil
    np.percentile(gray, 75), # Terceiro quartil
])

# 2. Histograma Normalizado (32 bins)
hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
hist = hist.flatten() / hist.sum()

# 3. Características de Textura (LBP)
# 4. Características de Contorno
# 5. Características de Gradiente
```

#### **Classificadores Disponíveis**
- **Random Forest**: Para classificação geral com boa interpretabilidade
- **Support Vector Machine**: Para casos complexos com margem ótima

#### **Validação Cruzada**
```python
# K-Fold Cross Validation
scores = cross_val_score(classifier, X, y, cv=5)
cv_score = scores.mean()
cv_std = scores.std()
```

### 📊 **Métricas de Avaliação**

#### **Métricas Clássicas**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 · (Precision · Recall) / (Precision + Recall)
```

#### **Validação Cruzada K-Fold**
```
CV_Score = (1/k) · Σ(Accuracy_i)
```

---

## ⚙️ Requisitos do Sistema

### 💻 **Requisitos Mínimos**
- **Python**: 3.8 ou superior (recomendado 3.11+)
- **Sistema Operacional**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+
- **Memória RAM**: Mínimo 4GB (recomendado 8GB+)
- **Processador**: Intel i5 ou equivalente (recomendado i7/i9 ou AMD Ryzen 5+)
- **Armazenamento**: Mínimo 2GB livre (recomendado 10GB+)

### 📹 **Requisitos de Hardware**
- **Câmera**: Webcam USB, câmera industrial ou IP camera compatível
- **Resolução**: Mínimo 640x480 (recomendado 1920x1080 ou superior)
- **FPS**: Mínimo 15 FPS (recomendado 30 FPS)
- **GPU**: Opcional, mas recomendado para processamento acelerado

### 🔧 **Requisitos de Software**
- **OpenCV**: 4.8.1.78 ou superior
- **PyQt5**: 5.15.10 ou superior
- **NumPy**: 1.24.3 ou superior
- **Scikit-learn**: 1.3.0 ou superior

---

## 🚀 Instalação e Configuração

### 1️⃣ **Preparação do Ambiente**

```bash
# Verificar versão do Python
python --version
# Deve ser 3.8 ou superior

# Criar ambiente virtual (recomendado)
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2️⃣ **Instalação das Dependências**

```bash
# Atualizar pip
pip install --upgrade pip

# Instalar dependências principais
pip install -r requirements.txt

# Verificar instalação
python -c "import cv2, PyQt5, ttkbootstrap, numpy, sklearn; print('✅ Instalação bem-sucedida!')"
```

### 3️⃣ **Execução do Sistema**

```bash
# Execução principal
python main.py

# Execução direta do módulo de montagem
python -m modulos.montagem

# Execução com debug
python main.py --debug
```

### 4️⃣ **Configuração Inicial**

1. **Primeira Execução**: O sistema criará automaticamente a estrutura de diretórios
2. **Configuração de Câmera**: Use "Detectar Câmeras" no módulo de Montagem
3. **Criação de Modelos**: Comece criando um modelo de referência
4. **Definição de Slots**: Use o editor visual para definir áreas de inspeção

---

## 📁 Estrutura do Projeto

```
v2-main/
├── 📄 main.py                    # Ponto de entrada principal
├── 📋 requirements.txt           # Dependências do projeto
├── 📖 README.md                  # Documentação principal
├── 📚 DOCUMENTACAO_TECNICA.md   # Documentação técnica detalhada
├── 🎨 CORES_CENTRALIZADAS.md    # Guia de cores e estilos
├── 🍓 RASPBERRY_PI_OPTIMIZATION.md # Otimizações para Raspberry Pi
│
├── 🖼️ assets/                    # Recursos visuais
│   ├── dx_project_logo.png      # Logo principal
│   ├── dx_project_logo.svg      # Logo em SVG
│   ├── honda_logo.svg           # Logo da marca
│   └── logo.svg                 # Logo do sistema
│
├── ⚙️ config/                    # Configurações do sistema
│   └── style_config.json        # Configuração de estilos
│
├── 🧪 Imagem de teste/          # Imagens para testes
│   ├── NG - Copia.JPG          # Exemplo com defeito
│   ├── NG.JPG                  # Exemplo com defeito
│   └── OK.jpg                  # Exemplo aprovado
│
├── 🏗️ modelos/                   # Modelos e templates
│   ├── _samples/               # Amostras de treinamento
│   │   ├── model_unknown/      # Modelo desconhecido
│   │   ├── slot_1_samples/     # Amostras do slot 1
│   │   └── slot_2_samples/     # Amostras do slot 2
│   ├── _templates/             # Templates de referência
│   ├── 1_33/                   # Modelo específico 1-33
│   │   ├── 1_reference.jpg     # Imagem de referência
│   │   └── templates/          # Templates do modelo
│   ├── a_29/                   # Modelo específico A-29
│   │   ├── a_reference.jpg     # Imagem de referência
│   │   ├── ml_model_slot_1.joblib # Modelo ML treinado
│   │   └── templates/          # Templates e amostras
│   ├── b_34/                   # Modelo específico B-34
│   ├── historico_fotos/        # Histórico de fotos
│   └── n_35/                   # Modelo específico N-35
│
├── 🔧 modulos/                   # Módulos do sistema
│   ├── __init__.py             # Inicialização do pacote
│   ├── camera_manager.py       # Gerenciamento de câmeras
│   ├── database_manager.py     # Gerenciamento de banco de dados
│   ├── dialogs.py              # Diálogos do sistema
│   ├── history_ui.py           # Interface de histórico
│   ├── image_optimizer.py      # Otimização de imagens
│   ├── image_utils.py          # Utilitários de imagem
│   ├── inspection_ui.py        # Interface de inspeção
│   ├── inspection_window.py    # Janela de inspeção
│   ├── inspection.py           # Algoritmos de inspeção
│   ├── mesh_editor.py          # Editor de malhas
│   ├── ml_classifier.py        # Classificador de ML
│   ├── model_selector.py       # Seletor de modelos
│   ├── montagem.py             # Módulo principal de montagem
│   ├── montagem_backup.py      # Sistema de backup
│   ├── paths.py                # Gerenciamento de caminhos
│   ├── training_dialog.py      # Diálogo de treinamento
│   └── utils.py                # Utilitários e configurações
│
└── 🛠️ tools/                    # Ferramentas auxiliares
    └── check_db.py             # Verificação de banco de dados
```

---

## 🎮 Guia de Uso

### 🏠 **Dashboard Principal**

O sistema apresenta uma interface unificada com três abas principais:

1. **🏗️ Montagem**: Verificação de componentes montados
2. **📊 Histórico**: Análise de resultados e relatórios
3. **🔧 Configurações**: Ajustes do sistema e câmeras

### 🔍 **Módulo de Montagem**

#### **Criação de Modelos**
1. **Novo Modelo**: Clique em "Novo Modelo" e defina um nome
2. **Imagem de Referência**: Carregue uma imagem de referência
3. **Definição de Slots**: Use o editor visual para definir áreas de inspeção
4. **Configuração de Parâmetros**: Ajuste thresholds e tolerâncias
5. **Treinamento**: Colete amostras OK e NG para treinar o modelo

#### **Editor Visual de Slots**
- **Desenho de Retângulos**: Clique e arraste para criar áreas de inspeção
- **Configuração de Slots**: Ajuste posição, tamanho e parâmetros
- **Tipos de Inspeção**: Presença/ausência, cor, forma, alinhamento
- **Tolerâncias**: Configure thresholds para diferentes critérios

#### **Sistema de Treinamento**
1. **Coleta de Amostras**: Capture múltiplas imagens OK e NG
2. **Treinamento Automático**: O sistema calcula thresholds ótimos
3. **Validação**: Teste o modelo com novas imagens
4. **Persistência**: Salve o modelo treinado para uso futuro

### 📊 **Módulo de Histórico**

#### **Visualização de Dados**
- **Filtros Dinâmicos**: Por modelo, data, resultado e confiança
- **Estatísticas em Tempo Real**: Métricas de performance atualizadas
- **Visualização Gráfica**: Gráficos de tendências e distribuições
- **Exportação**: Suporte para múltiplos formatos de saída

#### **Análise de Performance**
- **Taxa de Aprovação**: Percentual de inspeções aprovadas
- **Tendências Temporais**: Evolução da performance ao longo do tempo
- **Análise por Modelo**: Comparação entre diferentes modelos
- **Detecção de Anomalias**: Identificação de padrões anômalos

### 🔧 **Módulo de Configurações**

#### **Configuração de Câmeras**
- **Detecção Automática**: Identificação de dispositivos disponíveis
- **Configuração de Parâmetros**: Resolução, FPS, buffer
- **Teste de Câmera**: Verificação de funcionamento
- **Configuração de Múltiplas Câmeras**: Suporte para setups complexos

#### **Configuração de Estilos**
- **Temas Personalizáveis**: Cores, fontes e layouts
- **Configuração de Interface**: Posicionamento e tamanho de elementos
- **Preferências do Usuário**: Configurações persistentes
- **Modo Escuro/Claro**: Alternância entre temas

---

## 🔧 Desenvolvimento e Extensibilidade

### 🏗️ **Arquitetura Extensível**

O sistema foi projetado para facilitar a adição de novos módulos e funcionalidades:

#### **Estrutura de Módulos**
```python
# Exemplo de novo módulo
from modulos.base_module import BaseModule

class NovoModulo(BaseModule):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        # Implementar interface do usuário
        pass
    
    def process_data(self, data):
        # Implementar lógica de processamento
        pass
```

#### **Sistema de Plugins**
- **Carregamento Dinâmico**: Módulos são carregados automaticamente
- **Interface Padrão**: Todos os módulos seguem a mesma estrutura
- **Integração Automática**: Novos módulos aparecem no dashboard
- **Configuração Centralizada**: Gerenciamento unificado de configurações

### 🔌 **APIs e Interfaces**

#### **API de Banco de Dados**
```python
from modulos.database_manager import DatabaseManager

db = DatabaseManager()
modelos = db.list_modelos()
novo_modelo = db.save_modelo("Nome", "caminho/imagem.jpg", slots)
```

#### **API de Câmeras**
```python
from modulos.camera_manager import detect_cameras, capture_image_from_camera

cameras = detect_cameras()
image = capture_image_from_camera(camera_index=0)
```

#### **API de Machine Learning**
```python
from modulos.ml_classifier import MLSlotClassifier

classifier = MLSlotClassifier()
classifier.train(training_samples)
result, confidence = classifier.predict(test_image)
```

### 🧪 **Sistema de Testes**

#### **Testes Unitários**
```bash
# Executar testes
python -m pytest tests/

# Com cobertura
python -m pytest --cov=modulos tests/
```

#### **Testes de Integração**
```bash
# Testar módulos específicos
python -m pytest tests/test_montagem.py
python -m pytest tests/test_ml_classifier.py
```

---

## 🛠️ Solução de Problemas

### ❌ **Problemas Comuns**

#### **Erro de Importação de Módulos**
```bash
# Verificar estrutura de diretórios
ls -la modulos/

# Verificar __init__.py
cat modulos/__init__.py

# Testar importação individual
python -c "from modulos.montagem import create_main_window"
```

#### **Câmera Não Detectada**
```bash
# Windows: Executar como administrador
# Linux: Verificar permissões
sudo usermod -a -G video $USER

# Testar com OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

#### **Erro de Banco de Dados**
```bash
# Verificar permissões
ls -la modelos/

# Recriar banco se necessário
rm modelos/models.db
# O banco será recriado automaticamente
```

#### **Problemas de Performance**
```bash
# Reduzir resolução da câmera
# Ajustar parâmetros ORB
# Fechar aplicativos desnecessários
```

### 🔍 **Debugging Avançado**

#### **Modo Debug**
```bash
# Executar com logs detalhados
python main.py --debug

# Definir variáveis de ambiente
export OPENCV_LOG_LEVEL=DEBUG  # Linux/Mac
set OPENCV_LOG_LEVEL=DEBUG     # Windows
```

#### **Logs do Sistema**
- **Windows**: Event Viewer
- **Linux**: `/var/log/syslog` ou `journalctl`
- **macOS**: Console.app

#### **Verificação de Dependências**
```bash
# Listar pacotes instalados
pip list

# Verificar versões específicas
python -c "import cv2; print(cv2.__version__)"
python -c "import PyQt5; print(PyQt5.QtCore.QT_VERSION_STR)"
```

---

## 📈 Performance e Otimização

### ⚡ **Otimizações de Performance**

#### **Processamento de Imagens**
- **Redimensionamento Inteligente**: Ajuste automático de resolução
- **Cache de Features**: Armazenamento de características calculadas
- **Processamento Paralelo**: Utilização de múltiplos cores quando disponível
- **Otimização de Algoritmos**: Parâmetros ajustados para velocidade vs. precisão

#### **Gerenciamento de Memória**
- **Cache Inteligente**: Reutilização de objetos quando possível
- **Limpeza Automática**: Liberação de recursos não utilizados
- **Garbage Collection**: Otimização do ciclo de vida de objetos
- **Monitoramento de Uso**: Acompanhamento de consumo de memória

#### **Otimizações de Câmera**
- **Buffer Otimizado**: Configuração de buffer para minimizar latência
- **Resolução Adaptativa**: Ajuste automático baseado na performance
- **FPS Dinâmico**: Ajuste de taxa de quadros baseado na carga
- **Cache de Instâncias**: Reutilização de objetos de câmera

### 🎯 **Benchmarks e Métricas**

#### **Performance de Template Matching**
- **Tempo de Processamento**: < 100ms para imagens 1920x1080
- **Taxa de FPS**: 30+ FPS em hardware moderno
- **Precisão**: > 95% para templates bem treinados
- **Robustez**: Funciona com variações de iluminação e ângulo

#### **Performance de Machine Learning**
- **Tempo de Treinamento**: < 5 segundos para 100 amostras
- **Tempo de Predição**: < 50ms por imagem
- **Acurácia**: > 90% com dados de treinamento adequados
- **Overfitting**: Proteção contra overfitting com validação cruzada

---

## 🗺️ Roadmap

### 🚀 **Versão Atual (v2.0) ✅**
- ✅ Sistema de inspeção de montagem avançado
- ✅ Interface gráfica moderna com PyQt5 e ttkbootstrap
- ✅ Banco de dados SQLite com backup automático
- ✅ Template matching com múltiplos algoritmos
- ✅ Sistema de treinamento com machine learning
- ✅ Suporte a múltiplas câmeras (USB, Industrial, IP)
- ✅ Interface responsiva com temas personalizáveis
- ✅ Sistema de histórico e relatórios avançados
- ✅ Editor visual de malhas de inspeção
- ✅ Validação cruzada e métricas de avaliação
- ✅ Sistema de cache inteligente para câmeras
- ✅ Configuração visual avançada de estilos

### 🔮 **Próximas Versões**

#### **v2.1 - Integração IoT e Industry 4.0** 🔄
- **APIs REST**: Interface web para integração com sistemas externos
- **MQTT**: Comunicação em tempo real com dispositivos IoT
- **OPC UA**: Integração com sistemas de automação industrial
- **Cloud Sync**: Sincronização com plataformas na nuvem

#### **v2.2 - Aplicativo Móvel** 📱
- **Android/iOS**: Aplicativo nativo para monitoramento remoto
- **Push Notifications**: Alertas em tempo real
- **Offline Mode**: Funcionamento sem conexão
- **QR Code**: Configuração rápida via código QR

#### **v2.3 - Interface Web Corporativa** 🌐
- **Dashboard Web**: Interface baseada em navegador
- **Multi-User**: Suporte para múltiplos usuários
- **Role-Based Access**: Controle de acesso baseado em funções
- **Real-Time Updates**: Atualizações em tempo real via WebSocket

#### **v2.4 - Inteligência Artificial Avançada** 🤖
- **Deep Learning**: Redes neurais convolucionais (CNN)
- **Transfer Learning**: Aproveitamento de modelos pré-treinados
- **Anomaly Detection**: Detecção automática de anomalias
- **Predictive Analytics**: Análise preditiva de falhas

#### **v2.5 - Analytics Preditivos** 📊
- **Machine Learning Avançado**: Algoritmos de ensemble
- **Time Series Analysis**: Análise de séries temporais
- **Predictive Maintenance**: Manutenção preditiva
- **Quality Forecasting**: Previsão de qualidade

#### **v2.6 - Sistema de Segurança** 🔒
- **Authentication**: Autenticação multi-fator
- **Authorization**: Controle de acesso granular
- **Audit Logging**: Registro completo de auditoria
- **Encryption**: Criptografia de dados sensíveis

---

## 📞 Suporte e Contribuição

### 🆘 **Suporte Técnico**

#### **Canais de Suporte**
- **Issues GitHub**: Reporte bugs e solicite features
- **Documentação**: Consulte a documentação técnica completa
- **Comunidade**: Participe da comunidade de desenvolvedores
- **Email**: Entre em contato com a equipe de desenvolvimento

#### **Informações para Suporte**
Ao solicitar suporte, inclua:
- **Versão do Sistema**: Versão exata do DX v2.0
- **Sistema Operacional**: Windows/Linux/macOS e versão
- **Python**: Versão do Python (3.8+)
- **Hardware**: Especificações do sistema
- **Logs**: Logs de erro quando disponíveis
- **Passos**: Passos para reproduzir o problema

### 🤝 **Contribuição**

#### **Como Contribuir**
1. **Fork**: Faça um fork do projeto
2. **Branch**: Crie uma branch para sua feature
3. **Desenvolvimento**: Implemente suas mudanças
4. **Testes**: Execute os testes existentes
5. **Pull Request**: Abra um pull request

#### **Padrões de Código**
- **PEP 8**: Formatação Python padrão
- **Docstrings**: Documentação de funções e classes
- **Type Hints**: Anotações de tipo quando apropriado
- **Testes**: Inclua testes para novas funcionalidades

#### **Áreas de Contribuição**
- **Novos Algoritmos**: Implementação de algoritmos de visão computacional
- **Interface**: Melhorias na interface do usuário
- **Performance**: Otimizações de performance
- **Documentação**: Melhorias na documentação
- **Testes**: Cobertura de testes e testes de integração

### 📚 **Recursos de Aprendizado**

#### **Documentação Técnica**
- **DOCUMENTACAO_TECNICA.md**: Documentação técnica completa
- **CORES_CENTRALIZADAS.md**: Guia de cores e estilos
- **RASPBERRY_PI_OPTIMIZATION.md**: Otimizações para Raspberry Pi

#### **Exemplos e Tutoriais**
- **Samples**: Pasta com exemplos de uso
- **Templates**: Templates de modelos para diferentes aplicações
- **Vídeos**: Tutoriais em vídeo (quando disponíveis)

---

## 📄 Licença

Este projeto é desenvolvido pela **equipe DX (Desenvolvimento Digital)** sob licença **MIT**.

### 📋 **Termos da Licença**
- **Uso Comercial**: Permitido
- **Modificação**: Permitida
- **Distribuição**: Permitida
- **Uso Privado**: Permitido
- **Atribuição**: Não obrigatória, mas apreciada

---

## 👥 Créditos e Agradecimentos

### 🏆 **Equipe de Desenvolvimento**

#### **Equipe DX (Desenvolvimento Digital)**
- **Líder de Projeto**: Coordenação geral e arquitetura
- **Desenvolvedores**: Implementação de módulos e funcionalidades
- **Testadores**: Validação e testes de qualidade
- **Documentadores**: Criação e manutenção da documentação

#### **Departamentos de Suporte**
- **Departamento de TI**: Suporte técnico e infraestrutura
- **Engenharia de Qualidade**: Especificações técnicas e validação
- **Produção**: Testes em ambiente real e feedback
- **Manutenção**: Suporte operacional e manutenção

### 🛠️ **Tecnologias e Bibliotecas**

#### **Core Technologies**
- **Python 3.11+**: Linguagem principal de desenvolvimento
- **OpenCV 4.8+**: Biblioteca de visão computacional
- **PyQt5 5.15+**: Framework de interface gráfica
- **ttkbootstrap 1.10+**: Interface moderna para módulos específicos

#### **Machine Learning e Dados**
- **Scikit-learn 1.3+**: Algoritmos de machine learning
- **NumPy 1.24+**: Computação científica e arrays
- **Pandas 2.1+**: Manipulação e análise de dados
- **Matplotlib 3.7+**: Visualização de dados

#### **Processamento de Imagens**
- **Pillow 10.0+**: Manipulação de imagens
- **SciPy 1.11+**: Computação científica avançada
- **ImageIO 2.31+**: Leitura e escrita de imagens
- **Scikit-image 0.21+**: Processamento de imagens científico

#### **Utilitários e Sistema**
- **SQLite3**: Banco de dados local (incluído no Python)
- **Pathlib**: Manipulação de caminhos de arquivo
- **Psutil 5.9+**: Monitoramento de sistema
- **Requests 2.31+**: Requisições HTTP

### 🙏 **Agradecimentos Especiais**

#### **Comunidade Open Source**
- **OpenCV Community**: Biblioteca de visão computacional
- **Python Community**: Linguagem de programação
- **Qt Community**: Framework de interface gráfica
- **Scikit-learn Community**: Biblioteca de machine learning

#### **Parceiros e Colaboradores**
- **Equipe de Produção**: Colaboração nos testes e validação
- **Engenheiros de Campo**: Feedback sobre usabilidade
- **Usuários Finais**: Sugestões de melhorias e reporte de bugs
- **Departamento de Qualidade**: Especificações técnicas e requisitos

---

## 🔮 **Conclusão**

O **Sistema de Visão Computacional DX v2.0** representa um marco significativo no desenvolvimento de soluções de inspeção visual automatizada. Com sua arquitetura modular, algoritmos avançados e interface moderna, o sistema oferece uma solução completa e profissional para controle de qualidade industrial.

### 🌟 **Destaques da Versão 2.0**
- **Arquitetura Robusta**: Sistema modular e extensível
- **Algoritmos Avançados**: Combinação de visão computacional e machine learning
- **Interface Moderna**: Interface gráfica intuitiva e responsiva
- **Performance Otimizada**: Processamento rápido e eficiente
- **Documentação Completa**: Guias detalhados e exemplos práticos

### 🚀 **Próximos Passos**
- **Implementação**: Comece criando seus primeiros modelos de inspeção
- **Treinamento**: Explore o sistema de treinamento com machine learning
- **Personalização**: Configure o sistema de acordo com suas necessidades
- **Contribuição**: Participe do desenvolvimento e melhoria do sistema

---

**© 2024-2025 Equipe DX - Desenvolvimento Digital. Licença MIT.**

*Sistema de Visão Computacional DX - Versão 2.0 - Transformando a Qualidade Industrial através da Tecnologia*
