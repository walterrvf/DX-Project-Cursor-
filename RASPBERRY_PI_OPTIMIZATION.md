# Otimizações para Raspberry Pi 4 - Sistema de Visão Computacional DX

## 🚀 **Guia Completo de Otimização para Raspberry Pi 4**

Este documento detalha as otimizações necessárias para executar o Sistema de Visão Computacional DX no Raspberry Pi 4 com performance otimizada.

---

## 📋 **Requisitos de Hardware**

### **Raspberry Pi 4 - Configuração Recomendada:**
- **Modelo**: Raspberry Pi 4B com 4GB ou 8GB RAM
- **Armazenamento**: MicroSD Classe 10 (32GB+) ou SSD USB 3.0
- **Câmera**: Raspberry Pi Camera Module v2 ou webcam USB compatível
- **Alimentação**: Fonte oficial 5V/3A
- **Refrigeração**: Dissipador + ventilador (recomendado)

### **Sistema Operacional:**
- **Raspberry Pi OS 64-bit** (Bullseye ou superior)
- **Ubuntu 22.04 LTS ARM64** (alternativa)

---

## ⚙️ **Configurações do Sistema**

### **1. Configuração de GPU e Memória**

Edite `/boot/config.txt`:
```bash
# Aumentar memória da GPU para processamento de imagem
gpu_mem=128

# Habilitar câmera
camera_auto_detect=1
start_x=1

# Otimizações de performance
arm_freq=2000
gpu_freq=750
over_voltage=6

# Habilitar hardware de vídeo
dtoverlay=vc4-kms-v3d
max_framebuffers=2
```

### **2. Configurações de Sistema**

Edite `/etc/dphys-swapfile`:
```bash
# Aumentar swap para processamento de imagem
CONF_SWAPSIZE=2048
```

Reinicie o sistema:
```bash
sudo reboot
```

---

## 📦 **Instalação Otimizada**

### **1. Preparação do Ambiente**

```bash
# Atualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependências do sistema
sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libqt5gui5 libqt5core5a libqt5dbus5 qttools5-dev-tools
sudo apt install -y libjpeg-dev libtiff5-dev libpng-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgtk2.0-dev libcanberra-gtk-module
sudo apt install -y libxvidcore-dev libx264-dev
```

### **2. Ambiente Virtual Python**

```bash
# Criar ambiente virtual
python3 -m venv venv_rpi
source venv_rpi/bin/activate

# Atualizar pip
pip install --upgrade pip setuptools wheel
```

### **3. Dependências Otimizadas**

Crie `requirements_rpi.txt`:
```txt
# Interface gráfica - versões otimizadas para ARM
PyQt5==5.15.9
ttkbootstrap==1.10.1

# Processamento de imagem - usar versão pré-compilada
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.1

# Utilitários
pathlib2==2.3.7

# Dependências adicionais para Raspberry Pi
raspberry-gpio==0.7.0
picamera2==0.3.12
```

Instale as dependências:
```bash
pip install -r requirements_rpi.txt
```

---

## 🔧 **Otimizações de Código**

### **1. Arquivo de Configuração Otimizada**

Crie `config/rpi_config.py`:
```python
"""
Configuração otimizada para Raspberry Pi 4
"""

# Parâmetros de Canvas e Preview - REDUZIDOS
PREVIEW_W = 800   # Reduzido de 1200
PREVIEW_H = 600   # Reduzido de 900

# Parâmetros ORB - OTIMIZADOS
ORB_FEATURES = 300        # Reduzido de 5000
ORB_SCALE_FACTOR = 1.3    # Aumentado para menos níveis
ORB_N_LEVELS = 6          # Reduzido de 8

# Parâmetros de Câmera - OTIMIZADOS
CAMERA_WIDTH = 640        # Resolução reduzida
CAMERA_HEIGHT = 480
CAMERA_FPS = 15           # FPS reduzido

# Parâmetros de Processamento
PROCESSING_INTERVAL = 200  # ms - Aumentado para reduzir carga
TEMPLATE_RESIZE_FACTOR = 0.5  # Redimensionar templates

# Threading e Multiprocessing
USE_THREADING = True
MAX_WORKERS = 2           # Limitado para Pi 4

# Otimizações de Memória
IMAGE_CACHE_SIZE = 5      # Reduzido
GARBAGE_COLLECT_INTERVAL = 10  # Frames

# Configurações específicas do Pi
USE_GPU_ACCELERATION = True
USE_NEON_OPTIMIZATION = True
ENABLE_FAST_MATH = True
```

### **2. Otimizações no Código Principal**

Crie `modulos/rpi_optimizations.py`:
```python
"""
Otimizações específicas para Raspberry Pi 4
"""

import cv2
import numpy as np
import gc
from threading import Thread
from queue import Queue
import time

class RPiOptimizer:
    """Classe para otimizações específicas do Raspberry Pi"""
    
    def __init__(self):
        self.frame_queue = Queue(maxsize=2)
        self.processing_queue = Queue(maxsize=1)
        self.last_gc = time.time()
        
    def optimize_opencv(self):
        """Otimiza configurações do OpenCV para Pi"""
        # Habilitar otimizações NEON se disponível
        cv2.setUseOptimized(True)
        
        # Configurar número de threads
        cv2.setNumThreads(2)
        
        # Usar aceleração de hardware quando possível
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
    
    def resize_for_processing(self, image, max_width=640):
        """Redimensiona imagem para processamento otimizado"""
        height, width = image.shape[:2]
        if width > max_width:
            ratio = max_width / width
            new_height = int(height * ratio)
            return cv2.resize(image, (max_width, new_height), 
                            interpolation=cv2.INTER_LINEAR)
        return image
    
    def optimize_orb_params(self):
        """Retorna parâmetros ORB otimizados para Pi"""
        return {
            'nfeatures': 300,
            'scaleFactor': 1.3,
            'nlevels': 6,
            'edgeThreshold': 31,
            'firstLevel': 0,
            'WTA_K': 2,
            'scoreType': cv2.ORB_HARRIS_SCORE,
            'patchSize': 31,
            'fastThreshold': 20
        }
    
    def memory_cleanup(self):
        """Limpeza periódica de memória"""
        current_time = time.time()
        if current_time - self.last_gc > 5:  # A cada 5 segundos
            gc.collect()
            self.last_gc = current_time
    
    def create_optimized_camera(self, camera_index=0):
        """Cria captura de câmera otimizada"""
        cap = cv2.VideoCapture(camera_index)
        
        # Configurações otimizadas
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Configurações específicas para Pi Camera
        if camera_index == 0:  # Pi Camera
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        return cap

class ThreadedCamera:
    """Captura de câmera em thread separada"""
    
    def __init__(self, camera_index=0):
        self.cap = RPiOptimizer().create_optimized_camera(camera_index)
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.thread = None
    
    def start(self):
        """Inicia captura em thread"""
        self.running = True
        self.thread = Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()
    
    def _capture_frames(self):
        """Loop de captura de frames"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # Remove frame antigo
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except:
                        pass
            time.sleep(0.033)  # ~30 FPS max
    
    def get_frame(self):
        """Obtém último frame disponível"""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
    
    def stop(self):
        """Para captura"""
        self.running = False
        if self.thread:
            self.thread.join()
        self.cap.release()

def apply_rpi_optimizations():
    """Aplica todas as otimizações para Raspberry Pi"""
    optimizer = RPiOptimizer()
    optimizer.optimize_opencv()
    
    # Configurações globais
    import os
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    
    return optimizer
```

### **3. Modificações no Arquivo Principal**

Adicione no início de `modulos/montagem.py`:
```python
# Importar otimizações para Raspberry Pi
try:
    import platform
    if 'arm' in platform.machine().lower() or 'aarch64' in platform.machine().lower():
        from .rpi_optimizations import apply_rpi_optimizations, ThreadedCamera, RPiOptimizer
        RPI_MODE = True
        # Aplicar otimizações
        rpi_optimizer = apply_rpi_optimizations()
    else:
        RPI_MODE = False
except ImportError:
    RPI_MODE = False

# Parâmetros otimizados para Raspberry Pi
if RPI_MODE:
    # Reduzir parâmetros para melhor performance
    PREVIEW_W = 800
    PREVIEW_H = 600
    ORB_FEATURES = 300
    ORB_SCALE_FACTOR = 1.3
    ORB_N_LEVELS = 6
```

---

## 🎯 **Otimizações Específicas**

### **1. Processamento de Imagem**

```python
def optimized_template_matching(image, template, threshold=0.7):
    """Template matching otimizado para Pi"""
    # Redimensionar se necessário
    if image.shape[1] > 640:
        scale = 640 / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale)
        template = cv2.resize(template, None, fx=scale, fy=scale)
    
    # Usar método mais rápido
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    
    return len(locations[0]) > 0

def optimized_orb_matching(img1, img2):
    """ORB matching otimizado para Pi"""
    # Parâmetros otimizados
    orb = cv2.ORB_create(
        nfeatures=300,
        scaleFactor=1.3,
        nlevels=6,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20
    )
    
    # Detectar features
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return 0
    
    # Matching otimizado
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Filtrar matches
    good_matches = [m for m in matches if m.distance < 50]
    
    return len(good_matches)
```

### **2. Gerenciamento de Memória**

```python
class MemoryManager:
    """Gerenciador de memória para Raspberry Pi"""
    
    def __init__(self, max_cache_size=5):
        self.image_cache = {}
        self.max_cache_size = max_cache_size
        self.frame_count = 0
    
    def cache_image(self, key, image):
        """Cache inteligente de imagens"""
        if len(self.image_cache) >= self.max_cache_size:
            # Remove imagem mais antiga
            oldest_key = next(iter(self.image_cache))
            del self.image_cache[oldest_key]
        
        self.image_cache[key] = image.copy()
    
    def get_cached_image(self, key):
        """Recupera imagem do cache"""
        return self.image_cache.get(key)
    
    def periodic_cleanup(self):
        """Limpeza periódica"""
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            gc.collect()
            
        if self.frame_count % 50 == 0:
            # Limpeza mais agressiva
            self.image_cache.clear()
            gc.collect()
```

---

## 🚀 **Script de Inicialização**

Crie `start_rpi.py`:
```python
#!/usr/bin/env python3
"""
Script de inicialização otimizado para Raspberry Pi 4
"""

import os
import sys
import platform
import subprocess

def check_rpi_environment():
    """Verifica se está rodando em Raspberry Pi"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            return 'Raspberry Pi' in cpuinfo
    except:
        return False

def optimize_system():
    """Aplica otimizações do sistema"""
    print("🔧 Aplicando otimizações do sistema...")
    
    # Configurações de ambiente
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    
    # Prioridade do processo
    try:
        os.nice(-5)  # Aumenta prioridade
    except:
        pass

def check_dependencies():
    """Verifica dependências essenciais"""
    print("📦 Verificando dependências...")
    
    required_modules = ['cv2', 'numpy', 'PyQt5', 'ttkbootstrap']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing.append(module)
            print(f"❌ {module}")
    
    if missing:
        print(f"\n⚠️  Módulos faltando: {', '.join(missing)}")
        print("Execute: pip install -r requirements_rpi.txt")
        return False
    
    return True

def main():
    """Função principal"""
    print("🍓 Sistema de Visão Computacional DX - Raspberry Pi 4")
    print("=" * 50)
    
    # Verificar ambiente
    if check_rpi_environment():
        print("✅ Raspberry Pi detectado")
    else:
        print("⚠️  Não foi possível confirmar Raspberry Pi")
    
    # Verificar dependências
    if not check_dependencies():
        sys.exit(1)
    
    # Aplicar otimizações
    optimize_system()
    
    # Iniciar aplicação
    print("\n🚀 Iniciando aplicação...")
    try:
        from app import main as app_main
        app_main()
    except Exception as e:
        print(f"❌ Erro ao iniciar aplicação: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## 📊 **Monitoramento de Performance**

Crie `utils/rpi_monitor.py`:
```python
"""
Monitoramento de performance para Raspberry Pi
"""

import psutil
import time
from threading import Thread

class RPiMonitor:
    """Monitor de performance do Raspberry Pi"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'temperature': 0,
            'fps': 0
        }
    
    def get_cpu_temperature(self):
        """Obtém temperatura da CPU"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read()) / 1000.0
                return temp
        except:
            return 0
    
    def start_monitoring(self):
        """Inicia monitoramento"""
        self.monitoring = True
        thread = Thread(target=self._monitor_loop)
        thread.daemon = True
        thread.start()
    
    def _monitor_loop(self):
        """Loop de monitoramento"""
        while self.monitoring:
            self.stats['cpu_percent'] = psutil.cpu_percent()
            self.stats['memory_percent'] = psutil.virtual_memory().percent
            self.stats['temperature'] = self.get_cpu_temperature()
            time.sleep(1)
    
    def get_stats(self):
        """Retorna estatísticas atuais"""
        return self.stats.copy()
    
    def stop_monitoring(self):
        """Para monitoramento"""
        self.monitoring = False
```

---

## 🔧 **Comandos de Instalação Completa**

```bash
#!/bin/bash
# Script completo de instalação para Raspberry Pi 4

echo "🍓 Instalação Sistema de Visão Computacional DX - Raspberry Pi 4"
echo "================================================================"

# Atualizar sistema
echo "📦 Atualizando sistema..."
sudo apt update && sudo apt upgrade -y

# Instalar dependências do sistema
echo "🔧 Instalando dependências do sistema..."
sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libqt5gui5 libqt5core5a libqt5dbus5 qttools5-dev-tools
sudo apt install -y libjpeg-dev libtiff5-dev libpng-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgtk2.0-dev libcanberra-gtk-module
sudo apt install -y libxvidcore-dev libx264-dev
sudo apt install -y git htop

# Configurar GPU e câmera
echo "🎥 Configurando GPU e câmera..."
sudo raspi-config nonint do_camera 0
sudo raspi-config nonint do_memory_split 128

# Criar ambiente virtual
echo "🐍 Criando ambiente virtual..."
python3 -m venv venv_rpi
source venv_rpi/bin/activate

# Instalar dependências Python
echo "📚 Instalando dependências Python..."
pip install --upgrade pip setuptools wheel
pip install -r requirements_rpi.txt

# Configurar inicialização automática (opcional)
echo "🚀 Configuração concluída!"
echo "Para iniciar o sistema: python3 start_rpi.py"
```

---

## 📈 **Resultados Esperados**

### **Performance Otimizada:**
- **FPS**: 10-15 FPS (vs 5-8 FPS sem otimização)
- **Uso de CPU**: 60-80% (vs 90-100% sem otimização)
- **Uso de RAM**: 1.5-2GB (vs 2.5-3GB sem otimização)
- **Temperatura**: 65-75°C (com refrigeração adequada)

### **Funcionalidades Mantidas:**
- ✅ Template matching otimizado
- ✅ ORB feature detection (reduzido)
- ✅ Sistema de treinamento OK/NG
- ✅ Interface gráfica responsiva
- ✅ Detecção de múltiplas câmeras
- ✅ Banco de dados SQLite

---

## 🎯 **Próximos Passos**

### **Melhorias Futuras:**
1. **Aceleração por Hardware**: Usar GPU do Pi 4 para OpenCV
2. **Otimização de Algoritmos**: Implementar versões ARM-específicas
3. **Interface Web**: Dashboard web para monitoramento remoto
4. **Edge AI**: Integração com modelos TensorFlow Lite
5. **Clustering**: Múltiplos Pi 4 trabalhando em conjunto

### **Monitoramento Contínuo:**
- Implementar logs de performance
- Alertas de temperatura
- Backup automático de modelos
- Atualizações OTA (Over-The-Air)

---

## 📞 **Suporte e Troubleshooting**

### **Problemas Comuns:**

1. **Sistema lento demais**
   - Verificar temperatura da CPU
   - Reduzir resolução da câmera
   - Aumentar swap do sistema

2. **Erro de memória**
   - Reduzir ORB_FEATURES para 200
   - Limpar cache mais frequentemente
   - Usar imagens menores

3. **Câmera não funciona**
   - Verificar cabo da Pi Camera
   - Habilitar câmera: `sudo raspi-config`
   - Testar: `raspistill -o test.jpg`

### **Comandos de Debug:**
```bash
# Verificar temperatura
vcgencmd measure_temp

# Verificar memória
free -h

# Verificar CPU
htop

# Testar câmera
raspistill -o test.jpg

# Logs do sistema
journalctl -f
```

---

**🍓 Desenvolvido e otimizado para Raspberry Pi 4**

*Equipe DX (Desenvolvimento Digital)*
*Versão: 1.0 - Otimizada para ARM64*
*Data: Janeiro 2025*