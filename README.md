# Sistema de Visão Computacional Honda

Sistema avançado de inspeção visual automatizada para controle de qualidade na linha de produção Honda. O sistema utiliza técnicas de visão computacional e machine learning para detectar defeitos, verificar montagem de componentes, contar peças e medir dimensões com alta precisão.

## Funcionalidades Principais

### 🔍 Módulo de Montagem
- Verificação automática de montagem de componentes
- Template matching para detecção de peças
- Sistema de treinamento com amostras OK/NG
- Detecção de alinhamento e posicionamento
- Suporte a múltiplas câmeras
- Interface de configuração avançada

### 📊 Módulo de Contagem
- Contagem automática de peças em linha de produção
- Algoritmos de detecção de objetos
- Relatórios de produtividade

### 📏 Módulo de Dimensões
- Medição precisa de dimensões de componentes
- Calibração automática de câmera
- Tolerâncias configuráveis

### 🔄 Módulo de Rotação
- Medição de ângulos e rotação de peças
- Detecção de orientação incorreta

## Requisitos do Sistema

- **Python**: 3.8 ou superior
- **Sistema Operacional**: Windows 10/11, Linux, macOS
- **Memória RAM**: Mínimo 4GB (recomendado 8GB)
- **Câmera**: Webcam USB ou câmera industrial compatível
- **Processador**: Intel i5 ou equivalente (recomendado i7)

## Instalação

### 1. Preparação do Ambiente

Certifique-se de ter o Python 3.8 ou superior instalado:
```bash
python --version
```

### 2. Clone ou Baixe o Projeto
```bash
git clone [URL_DO_REPOSITORIO]
cd vis-o-computacional
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
python -c "import cv2, PyQt5, ttkbootstrap; print('Instalação bem-sucedida!')"
```

## Executando o Sistema

### Execução Padrão
1. Certifique-se de que o ambiente virtual está ativado
2. Execute o programa principal:
```bash
python app.py
```

### Execução de Módulos Individuais
Cada módulo pode ser executado independentemente para testes:
```bash
# Módulo de Montagem
python -m modulos.montagem

# Módulo de Contagem
python -m modulos.contagem

# Módulo de Dimensões
python -m modulos.dimensoes

# Módulo de Rotação
python -m modulos.rotacao
```

## Estrutura do Projeto

```
vis-o-computacional/
├── app.py                      # Dashboard principal do sistema
├── requirements.txt            # Dependências do projeto
├── README.md                   # Documentação do projeto
├── RELATORIO_ANALISE_MONTAGEM.md # Relatório técnico detalhado
│
├── assets/                     # Recursos visuais
│   └── honda_logo.svg         # Logo oficial da Honda
│
├── modelos/                    # Modelos e templates
│   ├── _templates/            # Templates de referência
│   │   ├── slot_1_template.png
│   │   ├── slot_2_template.png
│   │   └── slot_3_template.png
│   ├── HRV_17/               # Modelos específicos do HRV 2017
│   ├── walter ramos_18/      # Outros modelos específicos
│   └── models.db             # Banco de dados SQLite
│
├── modulos/                    # Módulos do sistema
│   ├── __pycache__/           # Cache Python (gerado automaticamente)
│   ├── contagem.py            # Módulo de contagem de peças
│   ├── database_manager.py    # Gerenciador de banco de dados
│   ├── dimensoes.py           # Módulo de medição de dimensões
│   ├── model_selector.py      # Seletor de modelos
│   ├── montagem.py            # Módulo principal de verificação de montagem
│   └── rotacao.py             # Módulo de medição de rotação
│
└── Imagem de teste/           # Imagens para testes
    ├── NG.JPG                # Exemplo de imagem com defeito
    └── OK.jpg                # Exemplo de imagem aprovada
```

## Configuração Inicial

### Configuração de Câmera
1. Conecte sua câmera USB ou webcam
2. Execute o sistema e acesse o módulo de Montagem
3. Use a função "Detectar Câmeras" para identificar dispositivos disponíveis
4. Selecione a câmera desejada nas configurações

### Criação de Modelos
1. Acesse o módulo de Montagem
2. Clique em "Novo Modelo" e defina um nome
3. Carregue uma imagem de referência
4. Defina as áreas de inspeção (slots)
5. Treine o modelo com amostras OK e NG
6. Salve o modelo no banco de dados

## Uso do Sistema

### Dashboard Principal
O dashboard oferece acesso rápido a todos os módulos:
- **Montagem**: Verificação de componentes montados
- **Contagem**: Contagem automática de peças
- **Dimensões**: Medição de dimensões
- **Rotação**: Análise de orientação

### Módulo de Montagem - Funcionalidades Avançadas

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
1. Verifique se `assets/honda_logo.svg` existe
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
1. Verifique a documentação técnica em `RELATORIO_ANALISE_MONTAGEM.md`
2. Colete informações do sistema:
   ```bash
   python --version
   pip list
   # Inclua essas informações ao reportar problemas
   ```
3. Documente os passos para reproduzir o erro
4. Inclua screenshots ou logs de erro quando possível

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

## Roadmap

### Versão Atual (v1.0)
- ✅ Sistema de inspeção de montagem
- ✅ Interface gráfica com PyQt5
- ✅ Banco de dados SQLite
- ✅ Template matching
- ✅ Sistema de treinamento

### Próximas Versões
- 🔄 **v1.1**: Melhorias na interface do usuário
- 📋 **v1.2**: Relatórios avançados e exportação
- 🤖 **v2.0**: Integração com machine learning
- 🌐 **v2.1**: Interface web para monitoramento remoto
- 📊 **v2.2**: Dashboard de analytics em tempo real

## Licença

Este projeto é desenvolvido para uso interno da Honda. Todos os direitos reservados.

**Uso Restrito**: Este software é propriedade da Honda e destina-se exclusivamente ao uso em suas operações de controle de qualidade. A distribuição, modificação ou uso não autorizado é estritamente proibido.

## Créditos

### Desenvolvido por
- **Equipe de Engenharia Honda**
- **Departamento de Visão Computacional**

### Tecnologias Utilizadas
- **Python**: Linguagem principal
- **OpenCV**: Biblioteca de visão computacional
- **PyQt5**: Framework de interface gráfica
- **NumPy**: Computação científica
- **SQLite**: Banco de dados

### Agradecimentos
- Equipe de Produção Honda pela colaboração nos testes
- Departamento de TI pelo suporte técnico
- Engenheiros de Qualidade pelas especificações técnicas

---

**© 2024 Honda Motor Co., Ltd. Todos os direitos reservados.**

*Sistema de Visão Computacional Honda - Versão 1.0*
