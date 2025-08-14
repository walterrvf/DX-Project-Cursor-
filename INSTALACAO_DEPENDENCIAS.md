# 📦 Instalação de Dependências - Sistema DX

## 🎯 Visão Geral

Este projeto possui três arquivos de dependências para diferentes cenários de uso:

- **`requirements-minimal.txt`** - Instalação mínima para produção
- **`requirements.txt`** - Instalação completa recomendada
- **`requirements-dev.txt`** - Instalação para desenvolvimento

## 🚀 Instalação Rápida

### Para Produção (Recomendado)
```bash
pip install -r requirements.txt
```

### Para Produção Mínima
```bash
pip install -r requirements-minimal.txt
```

### Para Desenvolvimento
```bash
pip install -r requirements-dev.txt
```

## 📋 Dependências por Categoria

### 🔧 Essenciais (Sempre Instaladas)
- **ttkbootstrap** - Interface gráfica moderna
- **opencv-python** - Visão computacional e câmeras
- **Pillow** - Manipulação de imagens
- **numpy** - Arrays e operações matemáticas
- **scikit-learn** - Machine learning
- **joblib** - Serialização de modelos

### 📊 Completas (requirements.txt)
- **scikit-image** - Processamento avançado de imagem
- **openpyxl** - Leitura/escrita de arquivos Excel

### 🛠️ Desenvolvimento (requirements-dev.txt)
- **pytest** - Framework de testes
- **flake8** - Linting de código
- **black** - Formatação de código
- **mypy** - Análise estática
- **sphinx** - Geração de documentação

## 💻 Requisitos do Sistema

- **Python**: 3.8 ou superior
- **Sistema Operacional**: Windows 10/11, Linux, macOS
- **RAM**: 4GB+ recomendado
- **Espaço**: ~500MB para dependências

## 🔍 Verificação da Instalação

Após instalar as dependências, execute:

```bash
python -c "import cv2, numpy, ttkbootstrap, sklearn; print('✅ Todas as dependências instaladas com sucesso!')"
```

## 🚨 Solução de Problemas

### Erro de Compilação OpenCV
```bash
# Windows
pip install opencv-python-headless

# Linux
sudo apt-get install python3-opencv
```

### Erro de Tkinter
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# CentOS/RHEL
sudo yum install tkinter
```

### Erro de Permissão
```bash
pip install --user -r requirements.txt
```

## 📊 Comparação de Tamanhos

| Arquivo | Dependências | Tamanho Estimado | Uso Recomendado |
|---------|--------------|------------------|-----------------|
| `requirements-minimal.txt` | 6 | ~200MB | Produção básica |
| `requirements.txt` | 8 | ~300MB | Produção completa |
| `requirements-dev.txt` | 15+ | ~500MB | Desenvolvimento |

## 🔄 Atualização de Dependências

Para atualizar todas as dependências:

```bash
pip install --upgrade -r requirements.txt
```

Para atualizar uma dependência específica:

```bash
pip install --upgrade opencv-python
```

## 📝 Notas Importantes

1. **Dependências Nativas**: Muitas funcionalidades usam bibliotecas nativas do Python (não precisam ser instaladas)
2. **Compatibilidade**: Todas as versões são testadas para compatibilidade
3. **Performance**: As versões foram escolhidas para otimizar performance e estabilidade
4. **Segurança**: Todas as dependências são de fontes confiáveis e verificadas

## 🤝 Contribuição

Se encontrar problemas com dependências:

1. Verifique a versão do Python
2. Tente instalar com `--user` flag
3. Verifique se há conflitos com outros pacotes
4. Abra uma issue no repositório

---

**💡 Dica**: Para desenvolvimento, sempre use `requirements-dev.txt` para ter todas as ferramentas necessárias!
