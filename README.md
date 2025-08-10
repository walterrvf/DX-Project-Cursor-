# 🔬 Sistema de Visão Computacional DX

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-red.svg)
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
- [🧮 Algoritmos Matemáticos](#-algoritmos-matemáticos)
- [⚙️ Requisitos do Sistema](#️-requisitos-do-sistema)
- [🚀 Instalação](#-instalação)
- [📊 Arquitetura do Sistema](#-arquitetura-do-sistema)
- [🔧 Configuração e Uso](#-configuração-e-uso)
- [📈 Performance e Otimização](#-performance-e-otimização)
- [🛠️ Desenvolvimento](#️-desenvolvimento)
- [📞 Suporte](#-suporte)

---

## 🎯 Visão Geral

O **Sistema de Visão Computacional DX v2.0** é uma solução completa de inspeção visual automatizada que combina técnicas avançadas de **visão computacional**, **machine learning** e **processamento de imagens** para realizar controle de qualidade industrial com alta precisão e eficiência. 

Esta versão inclui melhorias significativas em performance, interface do usuário e algoritmos de detecção, além de suporte completo para múltiplas câmeras e modelos de inspeção avançados.

### 🏗️ Arquitetura Modular

```mermaid
graph TB
    A[Dashboard Principal] --> B[Módulo de Montagem]
    A --> C[Gerenciador de BD]
    A --> D[Seletor de Modelos]
    B --> E[Template Matching]
    B --> F[Feature Detection]
    B --> G[Machine Learning]
    B --> H[Análise de Histogramas]
```

## ✨ Funcionalidades Principais

### 🔍 **Módulo de Montagem Avançado**
- ✅ **Verificação automática** de montagem de componentes
- 🎯 **Template matching** com múltiplos algoritmos
- 🤖 **Sistema de treinamento** com amostras OK/NG
- 📐 **Detecção de alinhamento** e posicionamento
- 📹 **Suporte a múltiplas câmeras** (USB, Industrial, IP)
- ⚙️ **Interface de configuração** avançada
- 📊 **Relatórios em tempo real** com métricas detalhadas
- 🔄 **Sistema de backup** automático de modelos
- 📱 **Interface responsiva** com temas personalizáveis

### 🧠 **Inteligência Artificial Integrada**
- 🌲 **Random Forest Classifier** para classificação OK/NG
- 🎯 **Support Vector Machine (SVM)** para casos complexos
- 📈 **Validação cruzada** automática
- 🔄 **Retreinamento** de slots específicos
- 📊 **Métricas de performance** em tempo real

### 🎨 **Interface Moderna e Intuitiva**
- 🖥️ **Dashboard centralizado** com PyQt5
- 🎨 **Interface moderna** com ttkbootstrap
- 📱 **Design responsivo** e adaptável
- 🔧 **Configuração visual** de parâmetros
- 📊 **Visualização em tempo real** dos resultados
- 🌙 **Modo escuro/claro** com temas personalizáveis
- 📱 **Interface adaptativa** para diferentes resoluções
- 🎯 **Navegação intuitiva** com breadcrumbs e menus contextuais

## 🧮 Algoritmos Matemáticos

### 📐 **Template Matching**

O sistema utiliza correlação cruzada normalizada para detectar componentes:

**Fórmula da Correlação Cruzada Normalizada:**

```
γ(u,v) = Σ[T(x,y) - T̄][I(x+u,y+v) - Ī(u,v)] / √{Σ[T(x,y) - T̄]² · Σ[I(x+u,y+v) - Ī(u,v)]²}
```

Onde:
- `T(x,y)` = Template de referência
- `I(x,y)` = Imagem de entrada
- `T̄` = Média do template
- `Ī(u,v)` = Média da região da imagem
- `γ(u,v)` = Coeficiente de correlação (-1 ≤ γ ≤ 1)

**Implementação OpenCV:**
```python
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
locations = np.where(result >= threshold)  # threshold ∈ [0.7, 0.95]
```

### 🎯 **Feature Detection (ORB)**

**Algoritmo FAST (Features from Accelerated Segment Test):**

Para um pixel `p` com intensidade `Ip`, um ponto é considerado corner se:

```
∃ conjunto S de n pixels contíguos no círculo de 16 pixels tal que:
∀ pixel x ∈ S: |Ix - Ip| > t
```

Onde `t` é o threshold de intensidade e `n ≥ 12` para FAST-12.

**Descritor BRIEF:**

Para um patch de imagem suavizada `S`, o descritor binário é:

```
τ(S; x, y) = { 1 se S(x) < S(y)
             { 0 caso contrário
```

**Parâmetros ORB Otimizados:**
```python
orb = cv2.ORB_create(
    nfeatures=500,        # Máximo de features
    scaleFactor=1.2,      # Fator de escala da pirâmide
    nlevels=8,            # Níveis da pirâmide
    edgeThreshold=31,     # Tamanho da borda
    firstLevel=0,         # Primeiro nível da pirâmide
    WTA_K=2,              # Pontos para produzir elementos BRIEF
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,         # Tamanho do patch para descritor
    fastThreshold=20      # Threshold FAST
)
```

### 🔄 **RANSAC (Random Sample Consensus)**

**Algoritmo para Estimativa de Homografia:**

1. **Seleção Aleatória:** Escolher 4 pontos correspondentes aleatoriamente
2. **Modelo:** Calcular homografia `H` usando DLT (Direct Linear Transform)
3. **Consenso:** Contar inliers usando distância de reprojeção:

```
d = ||x'i - H·xi|| < threshold
```

4. **Iteração:** Repetir N vezes onde:

```
N = log(1-p) / log(1-(1-ε)^s)
```

Onde:
- `p` = probabilidade de sucesso (0.99)
- `ε` = proporção de outliers
- `s` = número mínimo de pontos (4)

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

### 📊 **Análise de Histogramas**

**Comparação de Histogramas HSV:**

**Correlação de Histogramas:**
```
ρ(H1,H2) = Σ[H1(i) - H̄1][H2(i) - H̄2] / √{Σ[H1(i) - H̄1]² · Σ[H2(i) - H̄2]²}
```

**Chi-Square Distance:**
```
χ²(H1,H2) = Σ[(H1(i) - H2(i))² / (H1(i) + H2(i))]
```

**Bhattacharyya Distance:**
```
dB(H1,H2) = √{1 - (1/√(H̄1·H̄2·N²)) · Σ√(H1(i)·H2(i))}
```

### 🤖 **Machine Learning**

**Random Forest Classifier:**

**Entropia para Divisão de Nós:**
```
H(S) = -Σ(pi · log2(pi))
```

**Information Gain:**
```
IG(S,A) = H(S) - Σ(|Sv|/|S| · H(Sv))
```

**Support Vector Machine:**

**Função de Decisão:**
```
f(x) = sign(Σ(αi·yi·K(xi,x)) + b)
```

**Kernel RBF:**
```
K(xi,xj) = exp(-γ||xi - xj||²)
```

### 📈 **Métricas de Avaliação**

**Acurácia:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precisão:**
```
Precision = TP / (TP + FP)
```

**Recall (Sensibilidade):**
```
Recall = TP / (TP + FN)
```

**F1-Score:**
```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

**Validação Cruzada K-Fold:**
```
CV_Score = (1/k) · Σ(Accuracy_i)
```

---

## ⚙️ Requisitos do Sistema

- **Python**: 3.8 ou superior (recomendado 3.11+)
- **Sistema Operacional**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+
- **Memória RAM**: Mínimo 4GB (recomendado 8GB+)
- **Câmera**: Webcam USB, câmera industrial ou IP camera compatível
- **Processador**: Intel i5 ou equivalente (recomendado i7/i9 ou AMD Ryzen 5+)
- **Armazenamento**: Mínimo 2GB livre (recomendado 10GB+)
- **GPU**: Opcional, mas recomendado para processamento acelerado

## 🚀 Instalação

### 1. Preparação do Ambiente

Certifique-se de ter o Python 3.8 ou superior instalado:
```bash
python --version
# ou
python3 --version
```

### 2. Clone ou Baixe o Projeto
```bash
git clone https://github.com/walterrvf/DX-Project-Cursor-.git
cd DX-Project-Cursor-
```

### 3. Crie um Ambiente Virtual (Recomendado)
```bash
python -m venv venv
```

### 4. Ative o Ambiente Virtual

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 5. Instale as Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Verificação da Instalação
```bash
python -c "import cv2, PyQt5, ttkbootstrap; print('✅ Instalação bem-sucedida!')"
```

### 7. Execução do Sistema
```bash
python main.py
```

## 🎮 Executando o Sistema

### Execução Padrão
1. Certifique-se de que o ambiente virtual está ativado
2. Execute o programa principal:
```bash
python main.py
```

### Execução do Módulo de Montagem
O módulo de montagem pode ser executado independentemente para testes:
```bash
# Módulo de Montagem
python -m modulos.montagem

# Ou diretamente
python modulos/montagem.py
```

## 📁 Estrutura do Projeto

```
DX-Project-Cursor-/
├── main.py                     # Aplicação principal do sistema
├── requirements.txt            # Dependências do projeto
├── README.md                   # Documentação do projeto
├── DOCUMENTACAO_TECNICA.md     # Documentação técnica completa
│
├── assets/                     # Recursos visuais
│   └── logo.svg               # Logo do sistema
│
├── modelos/                    # Modelos e templates
│   ├── _templates/            # Templates de referência
│   ├── _samples/              # Amostras de treinamento
│   ├── a_29/                  # Modelo específico A-29
│   ├── b_34/                  # Modelo específico B-34
│   ├── n_35/                  # Modelo específico N-35
│   ├── 1_33/                  # Modelo específico 1-33
│   └── models.db              # Banco de dados SQLite
│
├── modulos/                    # Módulos do sistema
│   ├── __init__.py            # Inicialização do pacote
│   ├── database_manager.py    # Gerenciador de banco de dados
│   ├── model_selector.py      # Seletor de modelos
│   ├── montagem.py            # Módulo principal de verificação de montagem
│   ├── camera_manager.py      # Gerenciador de câmeras
│   ├── image_utils.py         # Utilitários de imagem
│   ├── ml_classifier.py       # Classificador de machine learning
│   ├── inspection_ui.py       # Interface de inspeção
│   ├── history_ui.py          # Interface de histórico
│   ├── dialogs.py             # Diálogos do sistema
│   ├── mesh_editor.py         # Editor de malhas
│   ├── paths.py               # Gerenciamento de caminhos
│   ├── training_dialog.py     # Diálogo de treinamento
│   └── utils.py               # Utilitários e configurações
│
└── Imagem de teste/           # Imagens para testes
    ├── NG.JPG                # Exemplo de imagem com defeito
    └── OK.jpg                # Exemplo de imagem aprovada
```

## ⚙️ Configuração Inicial

### Configuração de Câmera
1. Conecte sua câmera USB, webcam ou IP camera
2. Execute o sistema e acesse o módulo de Montagem
3. Use a função "Detectar Câmeras" para identificar dispositivos disponíveis
4. Selecione a câmera desejada nas configurações
5. Configure parâmetros de resolução e FPS para otimizar performance

### Criação de Modelos
1. Acesse o módulo de Montagem
2. Clique em "Novo Modelo" e defina um nome
3. Carregue uma imagem de referência
4. Defina as áreas de inspeção (slots) com o editor visual
5. Configure parâmetros de cada slot (threshold, tipo de inspeção)
6. Treine o modelo com amostras OK e NG
7. Valide o modelo com imagens de teste
8. Salve o modelo no banco de dados

## 🎮 Uso do Sistema

### 🏠 Dashboard Principal
O dashboard oferece acesso centralizado a todos os módulos do sistema:
- **Montagem**: Verificação de componentes montados
- **Histórico**: Análise de resultados e relatórios
- **Configurações**: Ajustes do sistema e câmeras
- **Treinamento**: Criação e gestão de modelos

### 🔧 Módulo de Montagem - Funcionalidades Avançadas

#### Criação de Slots de Inspeção
1. Carregue uma imagem de referência
2. Use o mouse para desenhar retângulos nas áreas a serem inspecionadas
3. Configure parâmetros específicos para cada slot:
   - Limiar de correlação
   - Tipo de inspeção (presença/ausência, cor, forma)
   - Tolerâncias

#### Sistema de Treinamento
1. Capture múltiplas amostras OK (aprovadas)
2. Capture amostras NG (rejeitadas)
3. O sistema calculará automaticamente os limiares ótimos
4. Teste o modelo com novas imagens

#### Inspeção em Tempo Real
1. Selecione um modelo treinado
2. Ative a captura ao vivo
3. O sistema processará automaticamente cada frame
4. Resultados são exibidos em tempo real

## Dependências Detalhadas

### Principais Bibliotecas
- **PyQt5**: Interface gráfica principal
- **ttkbootstrap**: Interface moderna para módulos específicos
- **OpenCV**: Processamento de imagem e visão computacional
- **NumPy**: Operações matemáticas e arrays
- **Pillow**: Manipulação de imagens
- **SQLite3**: Banco de dados (incluído no Python)

### Algoritmos Utilizados
- **Template Matching**: Detecção de componentes
- **ORB (Oriented FAST and Rotated BRIEF)**: Detecção de features
- **RANSAC**: Estimativa robusta de transformações
- **Correlação Cruzada**: Análise de similaridade

## Adicionando Novos Módulos

### Estrutura Básica
Para adicionar um novo módulo:

1. Crie um arquivo `.py` na pasta `modulos/`
2. Implemente uma classe que herde de `QMainWindow`
3. Adicione uma função `main()` para execução independente
4. O módulo será automaticamente detectado pelo dashboard

### Exemplo de Módulo
```python
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

class NovoModuloWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Novo Módulo')
        self.setGeometry(150, 150, 600, 400)
        self.setStyleSheet('background-color: white;')
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        title = QLabel('Novo Módulo')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('font-size: 24px; color: #212529; margin: 20px;')
        layout.addWidget(title)

def main():
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = NovoModuloWindow()
    window.show()
    return window

if __name__ == "__main__":
    main()
```

## Solução de Problemas

### Problemas de Instalação

#### Erro ao instalar OpenCV
```bash
# Se houver erro com opencv-python, tente:
pip install opencv-python-headless==4.8.1.78

# Ou instale as dependências do sistema (Linux):
sudo apt-get install python3-opencv
```

#### Erro ao instalar PyQt5
```bash
# Windows - instale Visual C++ Redistributable
# Linux:
sudo apt-get install python3-pyqt5

# macOS:
brew install pyqt5
```

#### Problemas com ttkbootstrap
```bash
# Se houver conflitos, instale versão específica:
pip install ttkbootstrap==1.10.1 --force-reinstall
```

### Problemas de Execução

#### Programa não inicia
1. **Verifique as dependências:**
   ```bash
   pip list | grep -E "PyQt5|opencv|ttkbootstrap"
   ```

2. **Teste a importação:**
   ```bash
   python -c "import PyQt5, cv2, ttkbootstrap; print('OK')"
   ```

3. **Verifique o ambiente virtual:**
   ```bash
   which python  # Linux/Mac
   where python  # Windows
   ```

#### Módulo não aparece no dashboard
1. Verifique se o arquivo está em `modulos/`
2. Confirme se há uma função `main()` no módulo
3. Verifique erros de sintaxe:
   ```bash
   python -m py_compile modulos/nome_do_modulo.py
   ```

#### Problemas com câmera
1. **Câmera não detectada:**
   - Verifique se a câmera está conectada
   - Teste com outros aplicativos
   - Execute como administrador (Windows)

2. **Erro de permissão (Linux):**
   ```bash
   sudo usermod -a -G video $USER
   # Reinicie a sessão
   ```

3. **Múltiplas câmeras:**
   - Use a função "Detectar Câmeras" no módulo
   - Teste diferentes índices (0, 1, 2...)

#### Problemas de performance
1. **Sistema lento:**
   - Reduza a resolução da câmera
   - Ajuste os parâmetros ORB
   - Feche outros aplicativos

2. **Alto uso de CPU:**
   - Aumente o intervalo entre frames
   - Reduza o número de features ORB
   - Use modo de inspeção por demanda

### Problemas com Banco de Dados

#### Erro ao salvar modelo
```bash
# Verifique permissões da pasta
ls -la modelos/

# Recrie o banco se necessário
rm modelos/models.db
# O banco será recriado automaticamente
```

#### Modelos não carregam
1. Verifique a integridade do banco:
   ```python
   import sqlite3
   conn = sqlite3.connect('modelos/models.db')
   cursor = conn.cursor()
   cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
   print(cursor.fetchall())
   ```

### Problemas com Imagens

#### Erro ao carregar logo
1. Verifique se `assets/logo.svg` existe
2. Teste com formato alternativo (PNG/JPG)
3. Verifique permissões do arquivo

#### Imagens não processam corretamente
1. **Formatos suportados:** JPG, PNG, BMP, TIFF
2. **Tamanho máximo:** Recomendado até 4K (3840x2160)
3. **Verificar codificação:**
   ```python
   import cv2
   img = cv2.imread('caminho/para/imagem.jpg')
   print(f"Imagem carregada: {img is not None}")
   ```

### Logs e Debugging

#### Ativar modo debug
```bash
# Execute com logs detalhados
python app.py --debug

# Ou defina variável de ambiente
export OPENCV_LOG_LEVEL=DEBUG  # Linux/Mac
set OPENCV_LOG_LEVEL=DEBUG     # Windows
```

#### Verificar logs do sistema
- **Windows:** Event Viewer
- **Linux:** `/var/log/syslog` ou `journalctl`
- **macOS:** Console.app

### Contato e Suporte

Se os problemas persistirem:
1. Colete informações do sistema:
   ```bash
   python --version
   pip list
   # Inclua essas informações ao reportar problemas
   ```
2. Documente os passos para reproduzir o erro
3. Inclua screenshots ou logs de erro quando possível

## Contribuição

### Como Contribuir
1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

### Padrões de Código
- Use PEP 8 para formatação Python
- Adicione docstrings para funções e classes
- Inclua testes para novas funcionalidades
- Mantenha compatibilidade com Python 3.8+

### Reportando Bugs
Ao reportar bugs, inclua:
- Versão do Python e sistema operacional
- Lista de dependências (`pip list`)
- Passos para reproduzir o problema
- Screenshots ou logs de erro
- Comportamento esperado vs. atual

## 🗺️ Roadmap

### Versão Atual (v2.0) ✅
- ✅ Sistema de inspeção de montagem avançado
- ✅ Interface gráfica moderna com PyQt5
- ✅ Banco de dados SQLite com backup automático
- ✅ Template matching com múltiplos algoritmos
- ✅ Sistema de treinamento com machine learning
- ✅ Suporte a múltiplas câmeras (USB, Industrial, IP)
- ✅ Interface responsiva com temas personalizáveis
- ✅ Sistema de histórico e relatórios avançados
- ✅ Editor visual de malhas de inspeção
- ✅ Validação cruzada e métricas de avaliação

### Próximas Versões 🚀
- 🔄 **v2.1**: Integração com APIs de IoT e Industry 4.0
- 📱 **v2.2**: Aplicativo móvel para monitoramento remoto
- 🌐 **v2.3**: Interface web para dashboards corporativos
- 🤖 **v2.4**: Inteligência artificial avançada com deep learning
- 📊 **v2.5**: Analytics preditivos e manutenção preventiva
- 🔒 **v2.6**: Sistema de segurança e auditoria avançados

## 📄 Licença

Este projeto é desenvolvido pela equipe DX (Desenvolvimento Digital) sob licença MIT.

## 👥 Créditos

### Desenvolvido por
- **Equipe DX (Desenvolvimento Digital)**
- **Departamento de Visão Computacional**
- **Centro de Inovação Tecnológica**

### 🛠️ Tecnologias Utilizadas
- **Python 3.11+**: Linguagem principal
- **OpenCV 4.8+**: Biblioteca de visão computacional
- **PyQt5 5.15+**: Framework de interface gráfica
- **NumPy 1.24+**: Computação científica
- **SQLite3**: Banco de dados local
- **Scikit-learn**: Machine learning e validação
- **Pillow**: Processamento de imagens
- **ttkbootstrap**: Interface moderna para módulos específicos

### 🙏 Agradecimentos
- Equipe de Produção pela colaboração nos testes
- Departamento de TI pelo suporte técnico
- Engenheiros de Qualidade pelas especificações técnicas
- Comunidade open source pelas bibliotecas utilizadas

---

**© 2024-2025 Equipe DX - Desenvolvimento Digital. Licença MIT.**

*Sistema de Visão Computacional DX - Versão 2.0*
