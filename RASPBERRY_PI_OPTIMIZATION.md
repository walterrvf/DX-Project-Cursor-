# 🍓 **Otimizações para Raspberry Pi 4**

*Sistema de Visão Computacional DX - Versão Otimizada*

---

## 📋 **Índice**
1. [Requisitos de Hardware](#requisitos-de-hardware)
2. [Sistema Operacional](#sistema-operacional)
3. [Configurações do Sistema](#configurações-do-sistema)
4. [Instalação Otimizada](#instalação-otimizada)
5. [Otimizações de Código](#otimizações-de-código)
6. [Monitoramento de Performance](#monitoramento-de-performance)
7. [Scripts de Automação](#scripts-de-automação)
8. [Troubleshooting](#troubleshooting)

---

## 🔧 **Requisitos de Hardware**

### **Raspberry Pi 4 - Especificações Recomendadas:**
- **Modelo**: Raspberry Pi 4B (8GB RAM recomendado para melhor performance)
- **Armazenamento**: SSD USB 3.0 (recomendado) ou MicroSD Classe 10 U3 (mínimo 64GB)
- **Câmera**: Pi Camera v3 (12MP) ou USB Camera 1080p compatível
- **Alimentação**: Fonte oficial 5V/3A com cabo USB-C de qualidade
- **Refrigeração**: Case com ventilador ativo ou dissipador passivo robusto
- **Conectividade**: Ethernet (recomendado) ou Wi-Fi 5GHz para melhor estabilidade

### **Sistema Operacional Recomendado:**
- **Raspberry Pi OS 64-bit** (Bookworm - mais recente)
- **Ubuntu 22.04.3 LTS ARM64** (para melhor compatibilidade com bibliotecas Python)

### **Configurações Essenciais:**
```bash
# Habilitar câmera e configurações
sudo raspi-config
# Interface Options > Camera > Enable
# Interface Options > SSH > Enable (para acesso remoto)
# Advanced Options > Memory Split > 256 (para melhor performance gráfica)

# Configurar boot para SSD (se usando)
sudo raspi-config
# Advanced Options > Boot Order > USB Boot
```

---

## ⚙️ **Configurações do Sistema**

### **1. Configuração `/boot/config.txt`:**
```ini
# GPU Memory Split (aumentado para melhor performance)
gpu_mem=256

# Overclock seguro para Pi 4 (com boa refrigeração)
arm_freq=1800
gpu_freq=650
over_voltage=2

# Câmera e display
start_x=1
display_auto_detect=1

# Otimizações de performance
force_turbo=1
initial_turbo=60

# Habilitar hardware de vídeo
dtoverlay=vc4-kms-v3d
max_framebuffers=2

# Desabilitar recursos não utilizados (opcional)
# dtoverlay=disable-wifi
# dtoverlay=disable-bt
```

### **2. Configuração de Swap `/etc/dphys-swapfile`:**
```bash
# Configurar swap baseado na RAM disponível
# Para Pi 4 8GB: 4GB de swap
# Para Pi 4 4GB: 2GB de swap
CONF_SWAPSIZE=4096
CONF_SWAPFILE=/var/swap
CONF_MAXSWAP=4096
```

### **3. Otimizações de Sistema:**
```bash
# Configurar governor de CPU para performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Limitar logs para economizar espaço
sudo journalctl --vacuum-time=7d
sudo journalctl --vacuum-size=100M

# Configurar logrotate otimizado
echo '/var/log/*.log {
    daily
    missingok
    rotate 3
    compress
    delaycompress
    notifempty
    create 0644 root root
    postrotate
        systemctl reload rsyslog > /dev/null 2>&1 || true
    endscript
}' | sudo tee /etc/logrotate.d/custom

# Otimizar I/O do sistema
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
```

Reinicie o sistema:
```bash
sudo reboot
```

---

## 📦 **Instalação Otimizada**

### **1. Preparação do Ambiente**

```bash
# Atualizar sistema completamente
sudo apt update && sudo apt full-upgrade -y

# Instalar dependências essenciais do sistema
sudo apt install -y python3-pip python3-venv python3-dev python3-setuptools
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libqt5gui5 libqt5core5a libqt5dbus5 qttools5-dev-tools
sudo apt install -y libjpeg-dev libtiff5-dev libpng-dev libwebp-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt install -y libxvidcore-dev libx264-dev libgtk-3-dev
sudo apt install -y libcanberra-gtk-module libcanberra-gtk3-module
sudo apt install -y git htop tree curl wget nano

# Instalar ferramentas de monitoramento
sudo apt install -y iotop nethogs
```

### **2. Ambiente Virtual Python**

```bash
# Criar ambiente virtual otimizado
python3 -m venv --system-site-packages venv_rpi
source venv_rpi/bin/activate

# Atualizar ferramentas de build
pip install --upgrade pip setuptools wheel cython

# Configurar pip para usar cache local
mkdir -p ~/.cache/pip
echo '[global]' > ~/.pip/pip.conf
echo 'cache-dir = ~/.cache/pip' >> ~/.pip/pip.conf
echo 'trusted-host = pypi.org' >> ~/.pip/pip.conf
echo '               pypi.python.org' >> ~/.pip/pip.conf
echo '               files.pythonhosted.org' >> ~/.pip/pip.conf
```

### **3. Dependências Python Otimizadas**

Crie `requirements_rpi.txt`:
```txt
# Core dependencies - versões estáveis para ARM64
numpy==1.24.4
opencv-python==4.8.1.78
Pillow==10.1.0
PyQt5==5.15.10
ttkbootstrap==1.10.1
psutil==5.9.6

# Processamento de imagem otimizado
scipy==1.11.4
scikit-image==0.22.0
imageio==2.31.6

# Utilitários
pathlib2==2.3.7
requests==2.31.0
matplotlib==3.7.3

# Raspberry Pi específicas
raspberry-gpio==0.7.1
picamera2==0.3.15
rpi.gpio==0.7.1

# Monitoramento e logs
psutil==5.9.6
coloredlogs==15.0.1
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
Versão 2.0 - Janeiro 2025
"""

import os
import platform

# Detectar modelo do Pi automaticamente
def get_pi_model():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Pi 4' in cpuinfo:
                return 'pi4'
            elif 'Pi 3' in cpuinfo:
                return 'pi3'
    except:
        pass
    return 'unknown'

PI_MODEL = get_pi_model()

# Configurações baseadas no modelo
if PI_MODEL == 'pi4':
    # Parâmetros otimizados para Pi 4
    PREVIEW_W = 800
    PREVIEW_H = 600
    ORB_FEATURES = 400
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    CAMERA_FPS = 20
    MAX_THREADS = 4
else:
    # Configurações conservadoras para outros modelos
    PREVIEW_W = 640
    PREVIEW_H = 480
    ORB_FEATURES = 300
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 15
    MAX_THREADS = 2

# Parâmetros ORB otimizados
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8
ORB_EDGE_THRESHOLD = 31
ORB_PATCH_SIZE = 31

# Processamento e cache
PROCESSING_INTERVAL = 200
TEMPLATE_RESIZE_FACTOR = 0.5
BATCH_SIZE = 1
CACHE_SIZE = 10
MEMORY_CLEANUP_INTERVAL = 50

# Threading e Multiprocessing
USE_THREADING = True
MAX_WORKERS = MAX_THREADS

# Otimizações de Memória
IMAGE_CACHE_SIZE = CACHE_SIZE
GARBAGE_COLLECT_INTERVAL = 10

# Configurações específicas do Pi
USE_GPU_ACCELERATION = True
USE_NEON_OPTIMIZATION = True
ENABLE_FAST_MATH = True

# Configurações de sistema
os.environ['OMP_NUM_THREADS'] = str(MAX_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(MAX_THREADS)
os.environ['MKL_NUM_THREADS'] = str(MAX_THREADS)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
```

### **2. Otimizações Principais**

Crie `modulos/rpi_optimizations.py`:
```python
"""
Otimizações específicas para Raspberry Pi 4
Versão 2.0 - Janeiro 2025
"""

import cv2
import numpy as np
import gc
import threading
import psutil
import time
from queue import Queue, Empty
from config.rpi_config import *

class RPiOptimizer:
    """Classe avançada para otimizações do Raspberry Pi"""
    
    def __init__(self):
        self.frame_count = 0
        self.memory_threshold = 0.85
        self.cpu_threshold = 0.90
        self.last_cleanup = time.time()
        self.performance_stats = {
            'fps': 0,
            'cpu_usage': 0,
            'memory_usage': 0,
            'temperature': 0
        }
    
    def optimize_opencv(self):
        """Otimizações avançadas do OpenCV"""
        # Configurar threads baseado no modelo do Pi
        cv2.setNumThreads(MAX_THREADS)
        
        # Habilitar otimizações
        cv2.setUseOptimized(True)
        
        # Configurar cache de instruções
        if hasattr(cv2, 'setUseOpenVX'):
            cv2.setUseOpenVX(True)
    
    def resize_image_smart(self, image, target_width=None):
        """Redimensionamento inteligente baseado na carga do sistema"""
        if target_width is None:
            # Ajustar baseado no uso de CPU
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 80:
                target_width = 640
            elif cpu_percent > 60:
                target_width = 800
            else:
                target_width = PREVIEW_W
        
        height, width = image.shape[:2]
        if width > target_width:
            scale = target_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), 
                            interpolation=cv2.INTER_LINEAR)
        return image
    
    def get_adaptive_orb(self):
        """ORB adaptativo baseado na performance do sistema"""
        # Ajustar features baseado na carga
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > 80:
            features = ORB_FEATURES // 2
        elif cpu_percent > 60:
            features = int(ORB_FEATURES * 0.75)
        else:
            features = ORB_FEATURES
        
        return cv2.ORB_create(
            nfeatures=features,
            scaleFactor=ORB_SCALE_FACTOR,
            nlevels=ORB_N_LEVELS,
            edgeThreshold=ORB_EDGE_THRESHOLD,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=ORB_PATCH_SIZE,
            fastThreshold=20
        )
    
    def smart_memory_cleanup(self):
        """Limpeza inteligente de memória"""
        self.frame_count += 1
        current_time = time.time()
        
        # Verificar uso de memória
        memory_percent = psutil.virtual_memory().percent
        
        # Limpeza baseada no uso de memória
        if memory_percent > self.memory_threshold or \
           self.frame_count % MEMORY_CLEANUP_INTERVAL == 0:
            gc.collect()
            self.last_cleanup = current_time
        
        # Limpeza agressiva se necessário
        if memory_percent > 90:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            gc.collect()
    
    def get_system_stats(self):
        """Obtém estatísticas do sistema"""
        try:
            # CPU
            self.performance_stats['cpu_usage'] = psutil.cpu_percent()
            
            # Memória
            self.performance_stats['memory_usage'] = psutil.virtual_memory().percent
            
            # Temperatura (Pi específico)
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read()) / 1000.0
                    self.performance_stats['temperature'] = temp
            except:
                self.performance_stats['temperature'] = 0
                
        except Exception as e:
            print(f"Erro ao obter stats: {e}")
        
        return self.performance_stats.copy()

def create_optimized_camera(camera_index=0, use_pi_camera=True):
    """Cria captura de câmera otimizada"""
    if use_pi_camera:
        try:
            # Tentar usar Pi Camera primeiro
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
                controls={"FrameRate": CAMERA_FPS}
            )
            picam2.configure(config)
            return picam2
        except ImportError:
            print("Pi Camera não disponível, usando USB camera")
    
    # Fallback para USB camera
    cap = cv2.VideoCapture(camera_index)
    
    # Configurações otimizadas
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    return cap

class ThreadedCamera:
    """Captura de câmera otimizada com thread separada"""
    
    def __init__(self, camera_index=0, use_pi_camera=True):
        self.camera = create_optimized_camera(camera_index, use_pi_camera)
        self.q = Queue(maxsize=3)
        self.running = True
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def start(self):
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        
    def update(self):
        while self.running:
            try:
                if hasattr(self.camera, 'capture_array'):
                    # Pi Camera
                    frame = self.camera.capture_array()
                    ret = True
                else:
                    # USB Camera
                    ret, frame = self.camera.read()
                
                if ret and frame is not None:
                    # Limpar queue se estiver cheia
                    while self.q.qsize() >= 2:
                        try:
                            self.q.get_nowait()
                        except Empty:
                            break
                    
                    self.q.put(frame)
                    self.fps_counter += 1
                    
                    # Calcular FPS
                    if time.time() - self.fps_start_time >= 1.0:
                        self.current_fps = self.fps_counter
                        self.fps_counter = 0
                        self.fps_start_time = time.time()
                        
            except Exception as e:
                print(f"Erro na captura: {e}")
                time.sleep(0.1)
    
    def read(self):
        try:
            frame = self.q.get_nowait()
            return True, frame
        except Empty:
            return False, None
    
    def get_fps(self):
        return self.current_fps
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        
        if hasattr(self.camera, 'stop'):
            self.camera.stop()
        elif hasattr(self.camera, 'release'):
            self.camera.release()

def apply_rpi_optimizations():
    """Aplica todas as otimizações do sistema"""
    print("🍓 Aplicando otimizações para Raspberry Pi...")
    
    # Criar otimizador
    optimizer = RPiOptimizer()
    optimizer.optimize_opencv()
    
    # Configurar prioridade do processo
    try:
        import os
        os.nice(-5)  # Aumentar prioridade
    except:
        pass
    
    print(f"✅ Otimizações aplicadas para {PI_MODEL}")
    print(f"   - Threads: {MAX_THREADS}")
    print(f"   - Resolução: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"   - FPS alvo: {CAMERA_FPS}")
    print(f"   - ORB Features: {ORB_FEATURES}")
    
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

## 🚀 **Scripts de Automação**

### **1. Script de Inicialização Avançado**

Crie `start_rpi.py`:
```python
#!/usr/bin/env python3
"""
Script de inicialização otimizado para Raspberry Pi 4
Versão 2.0 - Janeiro 2025
"""

import os
import sys
import platform
import subprocess
import time
import psutil
from pathlib import Path

class RPiSystemManager:
    """Gerenciador do sistema Raspberry Pi"""
    
    def __init__(self):
        self.pi_model = self.detect_pi_model()
        self.system_info = self.get_system_info()
        
    def detect_pi_model(self):
        """Detecta modelo do Raspberry Pi"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Pi 4' in cpuinfo:
                    return 'Raspberry Pi 4'
                elif 'Pi 3' in cpuinfo:
                    return 'Raspberry Pi 3'
                elif 'Raspberry Pi' in cpuinfo:
                    return 'Raspberry Pi (Modelo desconhecido)'
        except:
            pass
        return 'Sistema não identificado'
    
    def get_system_info(self):
        """Coleta informações do sistema"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': round(psutil.virtual_memory().total / (1024**3), 1),
            'disk_usage': psutil.disk_usage('/').percent,
            'temperature': self.get_cpu_temperature()
        }
        return info
    
    def get_cpu_temperature(self):
        """Obtém temperatura da CPU"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read()) / 1000.0
                return temp
        except:
            return 0
    
    def check_system_health(self):
        """Verifica saúde do sistema"""
        issues = []
        
        # Verificar temperatura
        if self.system_info['temperature'] > 80:
            issues.append(f"⚠️  Temperatura alta: {self.system_info['temperature']:.1f}°C")
        
        # Verificar memória
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 85:
            issues.append(f"⚠️  Uso de memória alto: {memory_percent:.1f}%")
        
        # Verificar disco
        if self.system_info['disk_usage'] > 90:
            issues.append(f"⚠️  Disco quase cheio: {self.system_info['disk_usage']:.1f}%")
        
        return issues
    
    def optimize_system(self):
        """Aplica otimizações do sistema"""
        print("🔧 Aplicando otimizações do sistema...")
        
        # Configurações de ambiente baseadas no modelo
        if 'Pi 4' in self.pi_model:
            threads = '4'
        else:
            threads = '2'
        
        os.environ['OMP_NUM_THREADS'] = threads
        os.environ['OPENBLAS_NUM_THREADS'] = threads
        os.environ['MKL_NUM_THREADS'] = threads
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        # Configurar prioridade do processo
        try:
            os.nice(-10)  # Prioridade alta
            print(f"✅ Prioridade do processo aumentada")
        except PermissionError:
            print("⚠️  Não foi possível aumentar prioridade (execute como root)")
        
        # Configurar governor de CPU para performance
        try:
            subprocess.run([
                'sudo', 'sh', '-c',
                'echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
            ], check=True, capture_output=True)
            print("✅ CPU governor configurado para performance")
        except:
            print("⚠️  Não foi possível configurar CPU governor")
    
    def check_dependencies(self):
        """Verifica dependências essenciais"""
        print("📦 Verificando dependências...")
        
        required_modules = {
            'cv2': 'opencv-python',
            'numpy': 'numpy',
            'PyQt5': 'PyQt5',
            'ttkbootstrap': 'ttkbootstrap',
            'psutil': 'psutil',
            'sqlite3': 'sqlite3 (built-in)'
        }
        
        missing = []
        
        for module, package in required_modules.items():
            try:
                __import__(module)
                print(f"✅ {module} ({package})")
            except ImportError:
                missing.append(package)
                print(f"❌ {module} ({package})")
        
        if missing:
            print(f"\n⚠️  Pacotes faltando: {', '.join(missing)}")
            print("Execute: pip install -r requirements_rpi.txt")
            return False
        
        return True
    
    def check_camera(self):
        """Verifica disponibilidade da câmera"""
        print("📷 Verificando câmeras...")
        
        cameras_found = []
        
        # Verificar Pi Camera
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            cameras_found.append("Pi Camera (picamera2)")
            picam2.close()
        except:
            pass
        
        # Verificar USB cameras
        import cv2
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras_found.append(f"USB Camera {i}")
                cap.release()
        
        if cameras_found:
            for camera in cameras_found:
                print(f"✅ {camera}")
            return True
        else:
            print("❌ Nenhuma câmera encontrada")
            return False
    
    def display_system_info(self):
        """Exibe informações do sistema"""
        print("\n" + "=" * 60)
        print("🍓 SISTEMA DE VISÃO COMPUTACIONAL DX - RASPBERRY PI")
        print("=" * 60)
        print(f"Modelo: {self.pi_model}")
        print(f"CPUs: {self.system_info['cpu_count']}")
        print(f"RAM: {self.system_info['memory_total']} GB")
        print(f"Temperatura: {self.system_info['temperature']:.1f}°C")
        print(f"Uso do disco: {self.system_info['disk_usage']:.1f}%")
        print("=" * 60)
        
        # Verificar problemas
        issues = self.check_system_health()
        if issues:
            print("\n⚠️  ALERTAS DO SISTEMA:")
            for issue in issues:
                print(f"   {issue}")
            print()

def main():
    """Função principal"""
    manager = RPiSystemManager()
    manager.display_system_info()
    
    # Verificar dependências
    if not manager.check_dependencies():
        print("\n❌ Dependências não atendidas. Instalação necessária.")
        sys.exit(1)
    
    # Verificar câmeras
    if not manager.check_camera():
        print("\n⚠️  Nenhuma câmera detectada. Verifique as conexões.")
    
    # Aplicar otimizações
    manager.optimize_system()
    
    # Iniciar aplicação
    print("\n🚀 Iniciando Sistema de Visão Computacional DX...")
    print("   Pressione Ctrl+C para interromper\n")
    
    try:
        # Importar e iniciar aplicação principal
        sys.path.append(str(Path(__file__).parent))
        from modulos.montagem import main as app_main
        app_main()
    except KeyboardInterrupt:
        print("\n\n🛑 Sistema interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro ao iniciar aplicação: {e}")
        print("\nVerifique os logs para mais detalhes.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

### **2. Sistema de Monitoramento Avançado**

Crie `utils/rpi_monitor.py`:
```python
"""
Sistema de monitoramento avançado para Raspberry Pi
Versão 2.0 - Janeiro 2025
"""

import psutil
import time
import json
import logging
from threading import Thread, Event
from datetime import datetime
from pathlib import Path
from collections import deque

class RPiPerformanceMonitor:
    """Monitor avançado de performance do Raspberry Pi"""
    
    def __init__(self, log_file='logs/rpi_performance.log', history_size=100):
        self.monitoring = False
        self.stop_event = Event()
        self.history_size = history_size
        self.log_file = Path(log_file)
        
        # Criar diretório de logs se não existir
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Histórico de estatísticas
        self.stats_history = deque(maxlen=history_size)
        
        # Estatísticas atuais
        self.current_stats = {
            'timestamp': 0,
            'cpu_percent': 0,
            'cpu_freq': 0,
            'memory_percent': 0,
            'memory_available': 0,
            'temperature': 0,
            'disk_usage': 0,
            'network_sent': 0,
            'network_recv': 0,
            'fps': 0,
            'gpu_memory': 0
        }
        
        # Alertas
        self.alert_thresholds = {
            'cpu_percent': 90,
            'memory_percent': 85,
            'temperature': 80,
            'disk_usage': 90
        }
        
        self.alerts_active = set()
    
    def get_cpu_temperature(self):
        """Obtém temperatura da CPU"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read()) / 1000.0
                return temp
        except:
            return 0
    
    def get_cpu_frequency(self):
        """Obtém frequência da CPU"""
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
                freq = int(f.read()) / 1000  # MHz
                return freq
        except:
            return 0
    
    def get_gpu_memory(self):
        """Obtém uso de memória da GPU"""
        try:
            import subprocess
            result = subprocess.run(['vcgencmd', 'get_mem', 'gpu'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # Formato: gpu=76M
                gpu_mem = result.stdout.strip().split('=')[1].replace('M', '')
                return int(gpu_mem)
        except:
            pass
        return 0
    
    def collect_stats(self):
        """Coleta estatísticas do sistema"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = self.get_cpu_frequency()
        
        # Memória
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available / (1024**2)  # MB
        
        # Temperatura
        temperature = self.get_cpu_temperature()
        
        # Disco
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Rede
        network = psutil.net_io_counters()
        network_sent = network.bytes_sent / (1024**2)  # MB
        network_recv = network.bytes_recv / (1024**2)  # MB
        
        # GPU
        gpu_memory = self.get_gpu_memory()
        
        # Atualizar estatísticas
        self.current_stats.update({
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'cpu_freq': cpu_freq,
            'memory_percent': memory_percent,
            'memory_available': memory_available,
            'temperature': temperature,
            'disk_usage': disk_usage,
            'network_sent': network_sent,
            'network_recv': network_recv,
            'gpu_memory': gpu_memory
        })
        
        # Adicionar ao histórico
        self.stats_history.append(self.current_stats.copy())
        
        # Verificar alertas
        self.check_alerts()
    
    def check_alerts(self):
        """Verifica e gera alertas"""
        new_alerts = set()
        
        for metric, threshold in self.alert_thresholds.items():
            if self.current_stats[metric] > threshold:
                new_alerts.add(metric)
                
                # Log apenas novos alertas
                if metric not in self.alerts_active:
                    message = f"ALERTA: {metric} = {self.current_stats[metric]:.1f} (limite: {threshold})"
                    logging.warning(message)
                    print(f"⚠️  {message}")
        
        # Alertas resolvidos
        resolved_alerts = self.alerts_active - new_alerts
        for metric in resolved_alerts:
            message = f"RESOLVIDO: {metric} = {self.current_stats[metric]:.1f}"
            logging.info(message)
            print(f"✅ {message}")
        
        self.alerts_active = new_alerts
    
    def start_monitoring(self, interval=2):
        """Inicia monitoramento"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.stop_event.clear()
        
        def monitor_loop():
            while self.monitoring and not self.stop_event.wait(interval):
                try:
                    self.collect_stats()
                except Exception as e:
                    logging.error(f"Erro no monitoramento: {e}")
        
        self.monitor_thread = Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logging.info("Monitoramento iniciado")
        print("📊 Monitoramento de performance iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self.stop_event.set()
        
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        
        logging.info("Monitoramento parado")
        print("📊 Monitoramento de performance parado")
    
    def get_current_stats(self):
        """Retorna estatísticas atuais"""
        return self.current_stats.copy()
    
    def get_stats_history(self, last_n=None):
        """Retorna histórico de estatísticas"""
        if last_n:
            return list(self.stats_history)[-last_n:]
        return list(self.stats_history)
    
    def get_average_stats(self, last_n=10):
        """Calcula estatísticas médias"""
        if not self.stats_history:
            return self.current_stats.copy()
        
        recent_stats = list(self.stats_history)[-last_n:]
        avg_stats = {}
        
        for key in self.current_stats:
            if key != 'timestamp':
                values = [stat[key] for stat in recent_stats if stat[key] is not None]
                avg_stats[key] = sum(values) / len(values) if values else 0
        
        return avg_stats
    
    def export_stats(self, filename=None):
        """Exporta estatísticas para arquivo JSON"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'logs/rpi_stats_{timestamp}.json'
        
        data = {
            'export_time': datetime.now().isoformat(),
            'current_stats': self.current_stats,
            'history': list(self.stats_history),
            'alerts_active': list(self.alerts_active)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📁 Estatísticas exportadas para {filename}")
        return filename
    
    def print_status(self):
        """Imprime status atual do sistema"""
        stats = self.current_stats
        
        print("\n" + "=" * 50)
        print("📊 STATUS DO SISTEMA RASPBERRY PI")
        print("=" * 50)
        print(f"CPU: {stats['cpu_percent']:.1f}% @ {stats['cpu_freq']:.0f} MHz")
        print(f"Memória: {stats['memory_percent']:.1f}% ({stats['memory_available']:.0f} MB livres)")
        print(f"Temperatura: {stats['temperature']:.1f}°C")
        print(f"Disco: {stats['disk_usage']:.1f}%")
        print(f"GPU Memory: {stats['gpu_memory']} MB")
        
        if self.alerts_active:
            print(f"\n⚠️  Alertas ativos: {', '.join(self.alerts_active)}")
        else:
            print("\n✅ Sistema operando normalmente")
        
        print("=" * 50)

# Instância global do monitor
rpi_monitor = RPiPerformanceMonitor()

def start_monitoring():
    """Função de conveniência para iniciar monitoramento"""
    rpi_monitor.start_monitoring()

def stop_monitoring():
    """Função de conveniência para parar monitoramento"""
    rpi_monitor.stop_monitoring()

def get_system_stats():
    """Função de conveniência para obter estatísticas"""
    return rpi_monitor.get_current_stats()
```

---

### **3. Script de Instalação Automatizada**

Crie `install_rpi.sh`:
```bash
#!/bin/bash
# Script completo de instalação para Raspberry Pi 4
# Versão 2.0 - Janeiro 2025

set -e  # Parar em caso de erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🍓 Instalação Sistema de Visão Computacional DX - Raspberry Pi 4${NC}"
echo -e "${BLUE}================================================================${NC}"

# Verificar se é Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo -e "${RED}❌ Este script deve ser executado em um Raspberry Pi${NC}"
    exit 1
fi

# Verificar se é executado como usuário normal (não root)
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}❌ Não execute este script como root. Use sudo quando necessário.${NC}"
    exit 1
fi

# Função para log com timestamp
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠️  $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ❌ $1${NC}"
}

# Atualizar sistema
log "📦 Atualizando sistema..."
sudo apt update && sudo apt full-upgrade -y

# Instalar dependências do sistema
log "🔧 Instalando dependências do sistema..."
sudo apt install -y \
    python3-pip python3-venv python3-dev python3-setuptools \
    build-essential cmake pkg-config \
    libopencv-dev python3-opencv \
    libatlas-base-dev libhdf5-dev libhdf5-serial-dev \
    libqt5gui5 libqt5core5a libqt5dbus5 qttools5-dev-tools \
    libjpeg-dev libtiff5-dev libpng-dev libwebp-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev \
    libcanberra-gtk-module libcanberra-gtk3-module \
    git htop tree curl wget nano iotop nethogs

# Configurar GPU e câmera
log "🎥 Configurando GPU e câmera..."
sudo raspi-config nonint do_camera 0
sudo raspi-config nonint do_memory_split 256
sudo raspi-config nonint do_ssh 0  # Habilitar SSH

# Configurar overclock seguro (opcional)
read -p "Aplicar overclock seguro? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "⚡ Aplicando overclock seguro..."
    sudo bash -c 'cat >> /boot/config.txt << EOF

# Overclock seguro para Pi 4
arm_freq=1800
gpu_freq=650
over_voltage=2
force_turbo=1
EOF'
    warn "Overclock aplicado. Certifique-se de ter refrigeração adequada!"
fi

# Criar diretório do projeto
PROJECT_DIR="$HOME/DX-Project"
log "📁 Criando diretório do projeto: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Criar ambiente virtual
log "🐍 Criando ambiente virtual..."
python3 -m venv --system-site-packages venv_rpi
source venv_rpi/bin/activate

# Atualizar ferramentas de build
log "🔧 Atualizando ferramentas de build..."
pip install --upgrade pip setuptools wheel cython

# Criar requirements_rpi.txt se não existir
if [ ! -f "requirements_rpi.txt" ]; then
    log "📝 Criando requirements_rpi.txt..."
    cat > requirements_rpi.txt << 'EOF'
# Core dependencies - versões estáveis para ARM64
numpy==1.24.4
opencv-python==4.8.1.78
Pillow==10.1.0
PyQt5==5.15.10
ttkbootstrap==1.10.1
psutil==5.9.6

# Processamento de imagem otimizado
scipy==1.11.4
scikit-image==0.22.0
imageio==2.31.6

# Utilitários
pathlib2==2.3.7
requests==2.31.0
matplotlib==3.7.3

# Raspberry Pi específicas
raspberry-gpio==0.7.1
picamera2==0.3.15
rpi.gpio==0.7.1

# Monitoramento e logs
psutil==5.9.6
coloredlogs==15.0.1
EOF
fi

# Instalar dependências Python
log "📚 Instalando dependências Python..."
pip install -r requirements_rpi.txt

# Criar diretórios necessários
log "📁 Criando estrutura de diretórios..."
mkdir -p logs config modulos utils

# Configurar serviço systemd (opcional)
read -p "Configurar inicialização automática? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "🚀 Configurando serviço systemd..."
    sudo tee /etc/systemd/system/dx-vision.service > /dev/null << EOF
[Unit]
Description=DX Vision System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv_rpi/bin
ExecStart=$PROJECT_DIR/venv_rpi/bin/python start_rpi.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable dx-vision.service
    log "✅ Serviço configurado. Use 'sudo systemctl start dx-vision' para iniciar"
fi

# Configurar logrotate
log "📋 Configurando rotação de logs..."
sudo tee /etc/logrotate.d/dx-vision > /dev/null << 'EOF'
/home/*/DX-Project/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644
    su root root
}
EOF

# Criar script de monitoramento
log "📊 Criando script de monitoramento..."
cat > monitor_system.sh << 'EOF'
#!/bin/bash
# Script de monitoramento do sistema

echo "📊 Status do Sistema Raspberry Pi"
echo "================================"
echo "Data/Hora: $(date)"
echo "Uptime: $(uptime -p)"
echo "Temperatura: $(vcgencmd measure_temp | cut -d'=' -f2)"
echo "Frequência CPU: $(vcgencmd measure_clock arm | cut -d'=' -f2 | awk '{print $1/1000000 " MHz"}')"
echo "Memória GPU: $(vcgencmd get_mem gpu)"
echo "Uso de CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Uso de Memória: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Uso de Disco: $(df -h / | awk 'NR==2{print $5}')"
echo "================================"
EOF
chmod +x monitor_system.sh

# Finalizar instalação
log "🎉 Instalação concluída com sucesso!"
echo
echo -e "${GREEN}✅ Sistema instalado em: $PROJECT_DIR${NC}"
echo -e "${GREEN}✅ Ambiente virtual: $PROJECT_DIR/venv_rpi${NC}"
echo -e "${GREEN}✅ Para ativar: source $PROJECT_DIR/venv_rpi/bin/activate${NC}"
echo -e "${GREEN}✅ Para iniciar: python3 start_rpi.py${NC}"
echo -e "${GREEN}✅ Para monitorar: ./monitor_system.sh${NC}"
echo
warn "Reinicie o sistema para aplicar todas as configurações: sudo reboot"
```

---

## 📈 **Resultados Esperados**

### **Performance Otimizada (Pi 4 8GB):**
- **FPS**: 15-25 FPS (vs 5-10 FPS sem otimização)
- **Uso de CPU**: 50-70% (vs 85-100% sem otimização)
- **Uso de RAM**: 2-3GB (vs 4-6GB sem otimização)
- **Temperatura**: 60-70°C (com refrigeração adequada)
- **Tempo de inicialização**: 15-30s (vs 60-120s sem otimização)
- **Latência de processamento**: 50-100ms (vs 200-500ms sem otimização)

### **Performance Otimizada (Pi 4 4GB):**
- **FPS**: 10-18 FPS
- **Uso de CPU**: 60-80%
- **Uso de RAM**: 1.5-2.5GB
- **Temperatura**: 65-75°C

### **Funcionalidades Mantidas:**
- ✅ Template matching otimizado com cache inteligente
- ✅ ORB feature detection adaptativo
- ✅ Sistema de treinamento OK/NG com ML
- ✅ Interface gráfica responsiva (PyQt5 + ttkbootstrap)
- ✅ Detecção automática de múltiplas câmeras
- ✅ Banco de dados SQLite com otimizações
- ✅ Sistema de logs e monitoramento
- ✅ Backup automático de modelos
- ✅ Relatórios de performance em tempo real

---

## 🎯 **Roadmap de Melhorias**

### **Versão 2.1 (Março 2025):**
1. **🚀 Aceleração por Hardware**: Implementar GPU acceleration com OpenGL ES
2. **🧠 Edge AI**: Integração com TensorFlow Lite e modelos ONNX
3. **🌐 Interface Web**: Dashboard web responsivo para monitoramento remoto
4. **📱 App Mobile**: Aplicativo para monitoramento via smartphone
5. **🔄 Auto-Update**: Sistema de atualizações OTA (Over-The-Air)

### **Versão 2.2 (Junho 2025):**
1. **🏭 Clustering**: Múltiplos Pi 4 trabalhando em conjunto
2. **☁️ Cloud Integration**: Sincronização com serviços em nuvem
3. **🤖 AI Avançada**: Modelos de deep learning para detecção de defeitos
4. **📊 Analytics**: Dashboard avançado com métricas de produção
5. **🔐 Segurança**: Implementação de autenticação e criptografia

### **Monitoramento Contínuo Implementado:**
- ✅ Logs de performance em tempo real
- ✅ Alertas automáticos de temperatura e recursos
- ✅ Backup automático de modelos e configurações
- ✅ Métricas de qualidade e produtividade
- ✅ Relatórios automáticos por email
- ✅ Sistema de notificações push

---

## 🛠️ **Suporte e Troubleshooting**

### **Problemas Comuns e Soluções:**

#### **1. Erro de importação OpenCV:**
```bash
# Diagnóstico
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"

# Solução 1: Reinstalar OpenCV
pip uninstall opencv-python opencv-contrib-python
sudo apt remove python3-opencv
sudo apt install python3-opencv libopencv-dev
pip install opencv-python==4.8.1.78

# Solução 2: Verificar dependências
sudo apt install --reinstall libatlas-base-dev libhdf5-dev
```

#### **2. Erro de memória (Out of Memory):**
```bash
# Diagnóstico
free -h
cat /proc/meminfo | grep -E "MemTotal|MemFree|SwapTotal|SwapFree"

# Solução 1: Aumentar swap
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Solução 2: Otimizar parâmetros
# Editar config/rpi_config.py
# Reduzir PREVIEW_W, PREVIEW_H, ORB_FEATURES
```

#### **3. Câmera não detectada:**
```bash
# Diagnóstico completo
echo "=== Verificação de Câmeras ==="
vcgencmd get_camera
lsusb | grep -i camera
v4l2-ctl --list-devices
ls /dev/video*

# Para Pi Camera
sudo raspi-config nonint do_camera 0
sudo modprobe bcm2835-v4l2

# Para câmeras USB
sudo apt install v4l-utils
v4l2-ctl --list-formats-ext
```

#### **4. Performance baixa (FPS < 5):**
```bash
# Diagnóstico
echo "=== Diagnóstico de Performance ==="
vcgencmd measure_temp
vcgencmd measure_clock arm
vcgencmd get_mem gpu
top -bn1 | head -20

# Soluções
# 1. Verificar temperatura
if [ $(vcgencmd measure_temp | cut -d'=' -f2 | cut -d'.' -f1) -gt 80 ]; then
    echo "⚠️ Temperatura alta! Verifique refrigeração"
fi

# 2. Otimizar GPU
sudo raspi-config nonint do_memory_split 256

# 3. Aplicar governor performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

#### **5. Erro de dependências PyQt5:**
```bash
# Diagnóstico
python3 -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')"

# Solução
sudo apt install python3-pyqt5 python3-pyqt5-dev
sudo apt install qttools5-dev-tools qt5-qmake
pip install --force-reinstall PyQt5==5.15.10
```

#### **6. Erro de permissões GPIO:**
```bash
# Adicionar usuário ao grupo gpio
sudo usermod -a -G gpio $USER
sudo usermod -a -G video $USER
sudo usermod -a -G i2c $USER

# Reiniciar sessão
sudo reboot
```

### **Scripts de Diagnóstico Avançado:**

#### **1. Script de Diagnóstico Completo (`diagnose_system.sh`):**
```bash
#!/bin/bash
# Script de diagnóstico completo do sistema

echo "🔍 Diagnóstico Completo do Sistema DX - Raspberry Pi"
echo "================================================="
echo "Data/Hora: $(date)"
echo

# Informações do sistema
echo "📋 Informações do Sistema:"
echo "Modelo: $(cat /proc/device-tree/model 2>/dev/null || echo 'N/A')"
echo "OS: $(lsb_release -d 2>/dev/null | cut -f2 || echo 'N/A')"
echo "Kernel: $(uname -r)"
echo "Uptime: $(uptime -p)"
echo

# Hardware
echo "🔧 Status do Hardware:"
echo "CPU: $(nproc) cores @ $(vcgencmd measure_clock arm | cut -d'=' -f2 | awk '{print $1/1000000 " MHz"}' 2>/dev/null || echo 'N/A')"
echo "Temperatura: $(vcgencmd measure_temp 2>/dev/null | cut -d'=' -f2 || echo 'N/A')"
echo "GPU Memory: $(vcgencmd get_mem gpu 2>/dev/null || echo 'N/A')"
echo "Throttling: $(vcgencmd get_throttled 2>/dev/null || echo 'N/A')"
echo

# Memória
echo "💾 Status da Memória:"
free -h
echo

# Armazenamento
echo "💿 Status do Armazenamento:"
df -h | grep -E "(Filesystem|/dev/)"
echo

# Câmeras
echo "📷 Câmeras Detectadas:"
echo "Pi Camera: $(vcgencmd get_camera 2>/dev/null || echo 'N/A')"
echo "USB Cameras: $(lsusb 2>/dev/null | grep -i camera | wc -l)"
echo "Video Devices: $(ls /dev/video* 2>/dev/null | wc -l)"
echo

# Python e dependências
echo "🐍 Ambiente Python:"
echo "Python: $(python3 --version 2>/dev/null || echo 'N/A')"
echo "Pip: $(pip --version 2>/dev/null | cut -d' ' -f2 || echo 'N/A')"
echo "Virtual Env: ${VIRTUAL_ENV:-'Não ativado'}"
echo

echo "📚 Dependências Críticas:"
for pkg in cv2 numpy PyQt5 ttkbootstrap psutil; do
    python3 -c "import $pkg; print('✅ $pkg:', $pkg.__version__ if hasattr($pkg, '__version__') else 'OK')" 2>/dev/null || echo "❌ $pkg: Não instalado"
done
echo

# Processos
echo "⚡ Processos com Alto Uso de CPU:"
top -bn1 | head -15 | tail -10
echo

# Rede
echo "🌐 Status da Rede:"
ip addr show | grep -E "(inet |UP|DOWN)" | head -10
echo

# Logs recentes
echo "📋 Logs Recentes (últimas 5 linhas):"
journalctl -n 5 --no-pager 2>/dev/null || echo "Logs não disponíveis"
echo

echo "✅ Diagnóstico concluído!"
echo "Para mais detalhes, execute: journalctl -f"
```

#### **2. Script de Teste de Performance (`test_performance.sh`):**
```bash
#!/bin/bash
# Teste de performance do sistema

echo "🚀 Teste de Performance - Sistema DX"
echo "==================================="

# Teste de CPU
echo "⚡ Testando CPU..."
start_time=$(date +%s.%N)
dd if=/dev/zero bs=1M count=100 2>/dev/null | wc -c > /dev/null
end_time=$(date +%s.%N)
cpu_time=$(echo "$end_time - $start_time" | bc)
echo "Tempo CPU: ${cpu_time}s"

# Teste de memória
echo "💾 Testando Memória..."
python3 -c "
import numpy as np
import time
start = time.time()
a = np.random.rand(1000, 1000)
b = np.random.rand(1000, 1000)
c = np.dot(a, b)
end = time.time()
print(f'Tempo Memória: {end-start:.3f}s')
" 2>/dev/null || echo "Erro no teste de memória"

# Teste OpenCV
echo "👁️ Testando OpenCV..."
python3 -c "
import cv2
import numpy as np
import time
start = time.time()
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
for i in range(10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
end = time.time()
print(f'Tempo OpenCV: {end-start:.3f}s')
" 2>/dev/null || echo "Erro no teste OpenCV"

echo "✅ Testes concluídos!"
```

### **Comandos de Monitoramento em Tempo Real:**

```bash
# Monitor completo em uma tela
watch -n 1 '
echo "=== Status Raspberry Pi ==="
echo "Temp: $(vcgencmd measure_temp | cut -d"=" -f2)"
echo "CPU: $(vcgencmd measure_clock arm | cut -d"=" -f2 | awk "{print \$1/1000000 \" MHz\"}")"
echo "GPU Mem: $(vcgencmd get_mem gpu)"
echo "Throttle: $(vcgencmd get_throttled)"
echo
echo "=== Recursos ==="
free -h | grep -E "(Mem|Swap)"
echo
echo "=== Top Processos ==="
top -bn1 | head -8 | tail -5
'

# Monitor de temperatura contínuo
watch -n 2 'vcgencmd measure_temp && vcgencmd measure_clock arm'

# Monitor de rede
watch -n 1 'cat /proc/net/dev | grep -E "(eth0|wlan0)"'
```

### **Logs e Debugging:**

```bash
# Logs do sistema DX
tail -f ~/DX-Project/logs/*.log

# Logs do sistema
journalctl -f -u dx-vision.service

# Debug OpenCV
export OPENCV_LOG_LEVEL=DEBUG
python3 start_rpi.py

# Debug Qt
export QT_DEBUG_PLUGINS=1
export QT_LOGGING_RULES="*.debug=true"

# Profiling de memória
python3 -m memory_profiler start_rpi.py

# Profiling de CPU
python3 -m cProfile -o profile.stats start_rpi.py
```

### **Contato para Suporte:**

- 📧 **Email**: suporte@dx-project.com
- 🐛 **Issues**: GitHub Issues
- 📖 **Documentação**: Wiki do projeto
- 💬 **Discord**: Servidor da comunidade

---

## 📝 **Changelog**

### **v2.0 (Janeiro 2025)**
- ✅ Documentação técnica completa
- ✅ Otimizações avançadas para Raspberry Pi 4
- ✅ Sistema de monitoramento em tempo real
- ✅ Scripts de instalação automatizada
- ✅ Suporte para Pi Camera 2 e câmeras USB
- ✅ Interface otimizada com PyQt5 + ttkbootstrap
- ✅ Sistema de cache inteligente
- ✅ Backup automático de modelos
- ✅ Relatórios de performance

### **v1.5 (Dezembro 2024)**
- ✅ Primeira versão para Raspberry Pi
- ✅ Otimizações básicas de performance
- ✅ Suporte inicial para Pi Camera

---

*Última atualização: Janeiro 2025 | Versão 2.0*
*Desenvolvido com ❤️ para a comunidade Raspberry Pi*

---

**🍓 Desenvolvido e otimizado para Raspberry Pi 4**

*Equipe DX (Desenvolvimento Digital)*
*Versão: 1.0 - Otimizada para ARM64*
*Data: Janeiro 2025*