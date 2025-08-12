"""
Módulo para a janela de inspeção (InspecaoWindow).
"""

import cv2
import numpy as np
from pathlib import Path
import ttkbootstrap as ttk
from ttkbootstrap.constants import (LEFT, BOTH, DISABLED, NORMAL, X, Y, BOTTOM, RIGHT, HORIZONTAL, VERTICAL, NW, CENTER)
from tkinter import (Canvas, filedialog, messagebox, simpledialog, Toplevel, StringVar, Text,
                     colorchooser, DoubleVar)
from tkinter.ttk import Combobox
from PIL import Image, ImageTk
from datetime import datetime
import os
import time

try:
    from database_manager import DatabaseManager
    from model_selector import ModelSelectorDialog, SaveModelDialog
    from dialogs import EditSlotDialog, SystemConfigDialog
    from training_dialog import SlotTrainingDialog
    from utils import load_style_config, save_style_config, apply_style_config, get_style_config_path, get_color, get_colors_group, get_font
    from ml_classifier import MLSlotClassifier
    from camera_manager import (
        detect_cameras,
        get_cached_camera,
        release_cached_camera,
        cleanup_unused_cameras,
        release_all_cached_cameras,
        capture_image_from_camera,
            initialize_persistent_pool,
            start_frame_pump,
            stop_frame_pump,
            capture_image_from_persistent_pool,
    )
    from dual_camera_driver import (
         get_dual_camera_manager,
         initialize_dual_cameras,
         get_camera_frame,
         get_all_camera_frames,
         stop_dual_cameras
     )
    from image_utils import cv2_to_tk
    from paths import get_model_dir, get_template_dir, get_model_template_dir
    from inspection import find_image_transform, check_slot
except ImportError:
    from database_manager import DatabaseManager
    from model_selector import ModelSelectorDialog, SaveModelDialog
    from dialogs import EditSlotDialog, SystemConfigDialog
    from training_dialog import SlotTrainingDialog
    from utils import load_style_config, save_style_config, apply_style_config, get_style_config_path, get_color, get_colors_group, get_font
    from ml_classifier import MLSlotClassifier
    from camera_manager import (
        detect_cameras,
        get_cached_camera,
        release_cached_camera,
        cleanup_unused_cameras,
        release_all_cached_cameras,
        capture_image_from_camera,
    )
    from image_utils import cv2_to_tk
    from paths import get_model_dir, get_template_dir, get_model_template_dir
    from inspection import find_image_transform, check_slot

# Variáveis globais
MODEL_DIR = get_model_dir()

class InspecaoWindow(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        from inspection_ui import InspecaoWindow as _InspecaoWindow
        # delega para módulo dedicado de UI de inspeção
        self._delegate = _InspecaoWindow(master, montagem_instance=None)
        # Atributos mínimos esperados por outros métodos
        self.slots = []
        self.img_reference = None
        self.img_test = None
        self.img_display = None
        self.scale_factor = 1.0
        self.current_model_id = None
        self.inspection_results = []
        self.camera = None
        self.live_capture = False
        # Garante atributo usado por callbacks diversos
        self.live_view = False
        # Gerenciador de banco
        try:
            db_path = MODEL_DIR / "models.db"
            self.db_manager = DatabaseManager(str(db_path))
        except Exception:
            self.db_manager = None
        self.latest_frame = None
        
        # Controle de webcam - Sistema de pool persistente
        self.available_cameras = detect_cameras()
        self.selected_camera = 0
        self.current_camera_index = self.available_cameras[0] if self.available_cameras else 0
        
        # Gerenciador de múltiplas câmeras ativas
        self.active_cameras = {}  # Dicionário para manter câmeras ativas
        self.camera_frames = {}   # Frames mais recentes de cada câmera
        
        # Importa funções do pool persistente
        from camera_manager import (
            initialize_persistent_pool, 
            get_persistent_camera, 
            capture_image_from_persistent_pool,
            get_pool_status,
            shutdown_persistent_pool,
            start_frame_pump,
            stop_frame_pump,
        )
        
        # Referências para funções do pool
        self.initialize_persistent_pool = initialize_persistent_pool
        self.get_persistent_camera = get_persistent_camera
        self.capture_image_from_persistent_pool = capture_image_from_persistent_pool
        self.get_pool_status = get_pool_status
        self.shutdown_persistent_pool = shutdown_persistent_pool
        
        self.setup_ui()
        self.update_button_states()
        
        # Registra método de limpeza para encerramento
        import atexit
        atexit.register(self.cleanup_on_exit)
        
        # Estado para inspeção com múltiplos programas
        self.selected_program_ids = []  # Lista de IDs dos programas selecionados
        
        # Inicia pool persistente de câmeras após inicialização completa
        if self.available_cameras:
            self.after(500, self.initialize_persistent_camera_pool)
            # Sistema dual de câmeras para modo 2 modelos
            self.after(1000, self.initialize_multiple_cameras)
            # Monitora status do pool periodicamente
            self.after(15000, self.monitor_pool_status)  # Verifica a cada 15 segundos
    
    def initialize_persistent_camera_pool(self):
        """Inicializa o pool persistente de câmeras para acesso rápido e sem limpeza de cache."""
        try:
            print("Inicializando pool persistente de câmeras...")
            
            # Inicializa o pool com as câmeras disponíveis
            success = self.initialize_persistent_pool(self.available_cameras)
            
            if success:
                print(f"Pool persistente inicializado com sucesso para câmeras: {self.available_cameras}")
                
                # Define a câmera principal ativa
                if self.available_cameras:
                    self.current_camera_index = self.available_cameras[0]
                    self.camera = self.get_persistent_camera(self.current_camera_index)
                    
                    if self.camera:
                        print(f"Câmera principal {self.current_camera_index} ativa no pool")
                    else:
                        print(f"Erro: Câmera principal {self.current_camera_index} não disponível no pool")
                        
                # Inicia frame pump para capturar frames contínuos sem exibir
                try:
                    try:
                        from utils import load_style_config
                        fps_cfg = int(load_style_config().get('system', {}).get('frame_pump_fps', 30))
                    except Exception:
                        fps_cfg = 30
                    start_frame_pump(self.available_cameras, fps=fps_cfg)
                    print(f"Frame pump iniciado para captura em segundo plano (FPS={fps_cfg})")
                except Exception as e:
                    print(f"Falha ao iniciar frame pump: {e}")

                # Exibe status do pool
                pool_status = self.get_pool_status()
                print(f"Status do pool: {pool_status['active_cameras']} câmeras ativas de {len(pool_status['cameras'])} disponíveis")
                
            else:
                print("Falha ao inicializar pool persistente, usando método tradicional")
                # Fallback para método tradicional se necessário
                self.initialize_traditional_cameras()
                
        except Exception as e:
            print(f"Erro ao inicializar pool persistente: {e}")
            # Fallback para método tradicional
            self.initialize_traditional_cameras()
    
    def initialize_traditional_cameras(self):
        """Método de fallback para inicialização tradicional de câmeras."""
        try:
            print("Inicializando câmeras usando método tradicional...")
            
            if self.available_cameras:
                self.current_camera_index = self.available_cameras[0]
                
                # Usa o cache tradicional do camera_manager
                from camera_manager import get_cached_camera
                self.camera = get_cached_camera(self.current_camera_index)
                
                if self.camera:
                    print(f"Câmera {self.current_camera_index} inicializada tradicionalmente")
                else:
                    print(f"Erro ao inicializar câmera {self.current_camera_index} tradicionalmente")
                    
        except Exception as e:
            print(f"Erro na inicialização tradicional: {e}")
    
    def initialize_multiple_cameras(self):
        """Inicializa sistema dual de câmeras com drivers específicos"""
        try:
            print("Inicializando sistema dual de câmeras...")
            
            # Determina índices das câmeras interna e externa
            internal_index = 0  # Câmera interna (webcam)
            external_index = 1 if len(self.available_cameras) > 1 else None
            
            # Inicializa o gerenciador dual
            if external_index is not None:
                success = initialize_dual_cameras(internal_index, external_index)
                if success:
                    print(f"Sistema dual inicializado: Interna({internal_index}) + Externa({external_index})")
                    # Atualiza referências para compatibilidade
                    self.dual_manager = get_dual_camera_manager()
                    self.active_cameras = {internal_index: True, external_index: True}
                    self.camera_frames = {internal_index: None, external_index: None}
                else:
                    print("Falha ao inicializar sistema dual, usando método tradicional")
                    self.initialize_traditional_cameras()
            else:
                print("Apenas uma câmera detectada, usando driver interno")
                # Inicializa apenas a câmera interna, sem tentar usar o mesmo índice como externo
                success = initialize_dual_cameras(internal_index, external_index or internal_index)
                if success:
                    self.dual_manager = get_dual_camera_manager()
                    self.active_cameras = {internal_index: True}
                    self.camera_frames = {internal_index: None}
                else:
                    self.initialize_traditional_cameras()
                    
        except Exception as e:
            print(f"Erro ao inicializar sistema dual: {e}")
            self.initialize_traditional_cameras()
    
    def initialize_traditional_cameras(self):
        """Método tradicional de inicialização como fallback"""
        try:
            print("Usando inicialização tradicional de câmeras...")
            
            # Inicializa câmeras ativas
            for camera_id in self.available_cameras:
                try:
                    cap = cv2.VideoCapture(camera_id)
                    
                    if cap.isOpened():
                        # Configurações centralizadas (inclui exposição/ganho/WB)
                        try:
                            from camera_manager import configure_video_capture
                            configure_video_capture(cap, camera_id)
                        except Exception:
                            pass
                        
                        self.active_cameras[camera_id] = cap
                        self.camera_frames[camera_id] = None
                        print(f"Câmera {camera_id} inicializada")
                    else:
                        cap.release()
                        
                except Exception as e:
                    print(f"Erro ao inicializar câmera {camera_id}: {e}")
            
            # Inicia captura contínua para todas as câmeras ativas
            if self.active_cameras:
                print(f"Captura iniciada para {len(self.active_cameras)} câmeras")
                self.start_multi_camera_capture()
                
        except Exception as e:
            print(f"Erro ao inicializar câmeras tradicionais: {e}")
    
    def monitor_pool_status(self):
        """Monitora o status do pool persistente periodicamente."""
        try:
            pool_status = self.get_pool_status()
            
            if pool_status['initialized']:
                active_count = pool_status['active_cameras']
                total_count = len(pool_status['cameras'])
                
                if active_count < total_count:
                    print(f"AVISO: Apenas {active_count} de {total_count} câmeras estão saudáveis no pool")
                    
                # Reagenda próxima verificação
                self.after(15000, self.monitor_pool_status)
            else:
                print("Pool persistente não está inicializado")
                
        except Exception as e:
            print(f"Erro ao monitorar status do pool: {e}")
            # Reagenda mesmo com erro
            self.after(15000, self.monitor_pool_status)

    def monitor_camera_health(self):
        """Monitora a saúde das câmeras no pool sem fechá-las desnecessariamente."""
        try:
            healthy_cameras = []
            for camera_index in list(self.active_cameras.keys()):
                camera = self.active_cameras.get(camera_index)
                if camera and camera.isOpened():
                    # Verifica se a câmera está respondendo
                    try:
                        # Tenta uma leitura rápida sem bloquear
                        ret, frame = camera.read()
                        if ret and frame is not None:
                            healthy_cameras.append(camera_index)
                        else:
                            print(f"Câmera {camera_index}: Sem resposta, mas mantendo conexão")
                            healthy_cameras.append(camera_index)  # Mantém mesmo sem resposta
                    except Exception as e:
                        print(f"Câmera {camera_index}: Erro na verificação de saúde: {e}")
                        healthy_cameras.append(camera_index)  # Mantém mesmo com erro
                else:
                    print(f"Câmera {camera_index}: Conexão perdida, mas mantendo no pool")
            
            if healthy_cameras:
                print(f"Pool de câmeras ativo: {healthy_cameras}")
            
            # Reagenda próxima verificação
            self.after(30000, self.monitor_camera_health)  # A cada 30 segundos
            
        except Exception as e:
            print(f"Erro no monitoramento de câmeras: {e}")
            # Reagenda mesmo com erro
            self.after(30000, self.monitor_camera_health)
    
    def start_camera_connection(self, camera_index):
        """Inicia conexão com uma câmera específica usando pool compartilhado."""
        try:
            # Verifica se a câmera já está no pool ativo
            if camera_index in self.active_cameras:
                camera = self.active_cameras[camera_index]
                if camera and camera.isOpened():
                    print(f"Câmera {camera_index} já está ativa no pool, reutilizando conexão")
                    return
                else:
                    # Remove câmera inválida do pool
                    del self.active_cameras[camera_index]
            
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            print(f"Inicializando nova conexão para câmera {camera_index}...")
            
            # Configurações otimizadas para inicialização mais rápida
            if is_windows:
                camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                camera = cv2.VideoCapture(camera_index)
            
            if not camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações centralizadas (inclui exposição/ganho/WB)
            try:
                from camera_manager import configure_video_capture
                configure_video_capture(camera, camera_index)
            except Exception:
                pass
            
            # Armazena a câmera no pool ativo
            self.active_cameras[camera_index] = camera
            self.camera_frames[camera_index] = None
            
            print(f"Câmera {camera_index} adicionada ao pool ativo com sucesso")
            
        except Exception as e:
            print(f"Erro ao conectar câmera {camera_index}: {e}")
            raise
    
    def restart_camera_connection(self, camera_index):
        """Tenta recuperar a conexão com uma câmera específica sem fechá-la desnecessariamente."""
        try:
            import time
            
            # Sistema de debounce para evitar tentativas muito frequentes
            if not hasattr(self, 'camera_restart_cooldown'):
                self.camera_restart_cooldown = {}
            
            current_time = time.time()
            cooldown_period = 15.0  # 15 segundos de cooldown aumentado
            
            # Verifica se ainda está no período de cooldown
            if camera_index in self.camera_restart_cooldown:
                time_since_last_restart = current_time - self.camera_restart_cooldown[camera_index]
                if time_since_last_restart < cooldown_period:
                    print(f"Câmera {camera_index} em cooldown, aguardando {cooldown_period - time_since_last_restart:.1f}s")
                    return
            
            print(f"Tentando recuperar câmera {camera_index} sem reinicialização...")
            self.camera_restart_cooldown[camera_index] = current_time
            
            # Primeiro, tenta apenas limpar o buffer da câmera existente
            if camera_index in self.active_cameras:
                camera = self.active_cameras[camera_index]
                if camera and camera.isOpened():
                    try:
                        # Limpa buffer antigo
                        for _ in range(5):
                            ret, frame = camera.read()
                            if ret and frame is not None:
                                print(f"Câmera {camera_index} recuperada sem reinicialização")
                                return
                        print(f"Câmera {camera_index} não está respondendo, mantendo conexão")
                        return
                    except Exception as e:
                        print(f"Erro ao tentar recuperar câmera {camera_index}: {e}")
            
            # Só reinicializa se realmente necessário e após múltiplas falhas
            print(f"Câmera {camera_index} será mantida ativa para próxima tentativa")
                
        except Exception as e:
            print(f"Erro ao tentar recuperar câmera {camera_index}: {e}")
    
    def start_multi_camera_capture(self):
        """Inicia captura de frames para todas as câmeras ativas."""
        try:
            import threading
            
            for camera_index in self.active_cameras.keys():
                # Cria thread separada para cada câmera
                thread = threading.Thread(
                    target=self.capture_camera_frames, 
                    args=(camera_index,), 
                    daemon=True
                )
                thread.start()
                
        except Exception as e:
            print(f"Erro ao iniciar captura multi-câmera: {e}")
    
    def capture_camera_frames(self, camera_index):
        """Captura frames continuamente de uma câmera específica com alta tolerância a falhas."""
        try:
            import threading
            import time
            camera = self.active_cameras.get(camera_index)
            if not camera:
                return
            
            # Cria lock específico para esta câmera se não existir
            if not hasattr(self, 'camera_locks'):
                self.camera_locks = {}
            if camera_index not in self.camera_locks:
                self.camera_locks[camera_index] = threading.Lock()
                
            consecutive_failures = 0
            max_failures = 50  # Muito mais tolerante - 50 falhas antes de tentar recuperação
            recovery_attempts = 0
            max_recovery_attempts = 3  # Máximo 3 tentativas de recuperação por sessão
                
            while camera_index in self.active_cameras and camera.isOpened():
                try:
                    ret, frame = camera.read()
                    if ret and frame is not None:
                        # Usa lock para evitar condições de corrida
                        with self.camera_locks[camera_index]:
                            self.camera_frames[camera_index] = frame.copy()
                            
                            # Atualiza latest_frame se esta é a câmera ativa
                            if camera_index == self.current_camera_index:
                                self.latest_frame = frame.copy()
                        
                        consecutive_failures = 0  # Reset contador de falhas
                        recovery_attempts = 0  # Reset tentativas de recuperação
                    else:
                        consecutive_failures += 1
                        # Só tenta recuperação após muitas falhas e se ainda não tentou muito
                        if consecutive_failures >= max_failures and recovery_attempts < max_recovery_attempts:
                            print(f"Câmera {camera_index}: {consecutive_failures} falhas consecutivas, tentativa de recuperação {recovery_attempts + 1}/{max_recovery_attempts}")
                            self.restart_camera_connection(camera_index)
                            consecutive_failures = 0
                            recovery_attempts += 1
                            time.sleep(2.0)  # Pausa maior após tentativa de recuperação
                        elif recovery_attempts >= max_recovery_attempts:
                            print(f"Câmera {camera_index}: Máximo de tentativas de recuperação atingido, mantendo conexão")
                            consecutive_failures = 0  # Reset para evitar spam de logs
                            
                    time.sleep(0.05)  # ~20 FPS - Reduz carga do sistema
                except Exception as e:
                    consecutive_failures += 1
                    # Só loga erros ocasionalmente para evitar spam
                    if consecutive_failures % 10 == 1:  # Loga a cada 10 erros
                        print(f"Erro na captura da câmera {camera_index}: {e} (erro #{consecutive_failures})")
                    
                    # Só tenta recuperação após muitas falhas
                    if consecutive_failures >= max_failures and recovery_attempts < max_recovery_attempts:
                        print(f"Câmera {camera_index}: Tentativa de recuperação após {consecutive_failures} erros")
                        self.restart_camera_connection(camera_index)
                        consecutive_failures = 0
                        recovery_attempts += 1
                        time.sleep(2.0)  # Pausa maior após tentativa de recuperação
                    else:
                        time.sleep(0.2)  # Pausa menor em caso de erro
                    
        except Exception as e:
            print(f"Erro geral na captura da câmera {camera_index}: {e}")

    def acquire_camera_exclusive(self, camera_index, timeout=5.0):
        """Adquire acesso exclusivo a uma câmera com timeout."""
        import threading
        import time
        
        # Inicializa mutex se não existir
        if not hasattr(self, 'camera_mutex'):
            self.camera_mutex = {}
        if camera_index not in self.camera_mutex:
            self.camera_mutex[camera_index] = threading.Lock()
            
        # Inicializa controle de uso se não existir
        if not hasattr(self, 'camera_in_use'):
            self.camera_in_use = {}
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.camera_mutex[camera_index].acquire(blocking=False):
                self.camera_in_use[camera_index] = True
                return True
            time.sleep(0.1)
        return False
    
    def release_camera_exclusive(self, camera_index):
        """Libera acesso exclusivo a uma câmera."""
        if hasattr(self, 'camera_mutex') and camera_index in self.camera_mutex:
            self.camera_in_use[camera_index] = False
            self.camera_mutex[camera_index].release()
    
    def capture_all_cameras_and_run_multi_inspection(self, program_ids):
        """Captura imagens simultâneas de duas câmeras e executa inspeção de múltiplos programas (dual)."""
        try:
            import time
            from dual_camera_driver import get_all_camera_frames
            
            # Dicionário para armazenar imagens capturadas por câmera
            self.captured_camera_images = {}
            
            self.status_var.set("Capturando frames das câmeras...")
            
            captured_count = 0
            
            # 1) Captura simultânea via dual_camera_driver (se estiver inicializado)
            try:
                if hasattr(self, 'dual_manager') and self.dual_manager and getattr(self.dual_manager, 'is_initialized', False):
                    frames = get_all_camera_frames()
                    for cam_idx, frame in frames.items():
                        if frame is not None:
                            self.captured_camera_images[cam_idx] = frame.copy()
                    captured_count = len(self.captured_camera_images)
                    if captured_count:
                        print(f"Capturadas {captured_count} imagens via sistema dual: {list(self.captured_camera_images.keys())}")
            except Exception as e:
                print(f"Erro ao capturar via sistema dual: {e}")
            
            # 2) Se não, tenta via frames recentes do pool tradicional
            if captured_count == 0 and hasattr(self, 'camera_frames') and self.camera_frames:
                for cam_idx, frame in list(self.camera_frames.items()):
                    if frame is not None:
                        self.captured_camera_images[cam_idx] = frame.copy()
                captured_count = len(self.captured_camera_images)
                if captured_count:
                    print(f"Capturadas {captured_count} imagens via frames recentes: {list(self.captured_camera_images.keys())}")
            
            # 3) Se ainda não, captura direta das câmeras ativas
            if captured_count == 0 and hasattr(self, 'active_cameras') and self.active_cameras:
                for cam_idx, cap in list(self.active_cameras.items()):
                    try:
                        if cap and hasattr(cap, 'read'):
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                self.captured_camera_images[cam_idx] = frame.copy()
                    except Exception:
                        pass
                captured_count = len(self.captured_camera_images)
                if captured_count:
                    print(f"Capturadas {captured_count} imagens via captura direta: {list(self.captured_camera_images.keys())}")
            
            # 4) Último fallback: camera_manager por índice
            if captured_count == 0 and hasattr(self, 'available_cameras'):
                from camera_manager import capture_image_from_camera
                for cam_idx in self.available_cameras:
                    try:
                        img = capture_image_from_camera(cam_idx)
                        if img is not None:
                            self.captured_camera_images[cam_idx] = img
                    except Exception:
                        pass
                captured_count = len(self.captured_camera_images)
                if captured_count:
                    print(f"Capturadas {captured_count} imagens via camera_manager: {list(self.captured_camera_images.keys())}")
            
            # Mantém no máximo 2 câmeras
            if captured_count > 2:
                ordered = sorted(self.captured_camera_images.items(), key=lambda kv: kv[0])
                self.captured_camera_images = dict(ordered[:2])
                captured_count = 2
            
            if captured_count == 0:
                print("AVISO: Nenhuma imagem capturada! Usando método tradicional...")
                self.run_multi_program_inspection(program_ids)
                return
            
            # Executa inspeção mapeando cada programa para uma câmera capturada
            self.run_multi_program_inspection_with_captured_images(program_ids)
            
        except Exception as e:
            print(f"Erro ao capturar de duas câmeras: {e}")
            self.status_var.set(f"Erro na captura: {str(e)}")
            self.run_multi_program_inspection(program_ids)

    def run_multi_program_inspection_with_captured_images(self, program_ids):
        """Executa inspeção mapeando programas para imagens capturadas de múltiplas câmeras."""
        if not program_ids:
            return
        try:
            # Armazena o modelo e câmera originais para restaurar ao final
            original_model_id = getattr(self, 'current_model_id', None)
            original_camera_index = self.current_camera_index
            original_img_reference = getattr(self, 'img_reference', None)
            overall_all_ok = True
            per_program_results = []
            program_images = []  # Lista para armazenar imagens com anotações por programa

            # Obtém imagens capturadas por câmera (se disponíveis)
            camera_images = getattr(self, 'captured_camera_images', {}) or {}
            cam_keys = sorted(list(camera_images.keys()))

            # Fallback: se não houver imagens capturadas, tenta usar a câmera atual como antes
            if not cam_keys:
                current_camera = self.current_camera_index
                if current_camera not in camera_images:
                    self.status_var.set("Sem imagem capturada para inspecionar")
                    return
                cam_keys = [current_camera]

            total_programs = len(program_ids)
            for idx, mid in enumerate(program_ids, start=1):
                try:
                    model_data = self.db_manager.load_modelo(mid)
                    
                    # Mapeamento inteligente: usa camera_index do modelo se disponível
                    model_camera_index = model_data.get('camera_index', 0)
                    if model_camera_index in camera_images:
                        cam_idx = model_camera_index
                    else:
                        # Fallback: distribuição round-robin baseada no índice
                        cam_idx = cam_keys[(idx - 1) % len(cam_keys)]
                    
                    captured_image = camera_images.get(cam_idx)
                    if captured_image is None:
                        per_program_results.append({
                            'id': mid,
                            'name': model_data.get('nome', str(mid)),
                            'success': False,
                            'details': f"Sem imagem para a câmera {cam_idx}",
                            'camera': cam_idx
                        })
                        overall_all_ok = False
                        continue

                    # Carrega o modelo e configurações (incluindo img_reference)
                    self.status_var.set(f"CARREGANDO PROGRAMA {idx}/{total_programs}...")
                    self.load_model_from_db(mid)
                    time.sleep(0.1)  # Reduzido de 0.2 para 0.1

                    # Validação crítica: img_reference deve estar carregado
                    if not hasattr(self, 'img_reference') or self.img_reference is None:
                        per_program_results.append({
                            'id': mid,
                            'name': model_data.get('nome', str(mid)),
                            'success': False,
                            'details': f"Imagem de referência não carregada",
                            'camera': cam_idx
                        })
                        overall_all_ok = False
                        continue

                    # Define a imagem de teste (da câmera mapeada)
                    self.img_test = captured_image.copy()

                    # Executa inspeção
                    self.run_inspection()

                    # Coleta resultado
                    if hasattr(self, 'inspection_results') and self.inspection_results:
                        passed = sum(1 for r in self.inspection_results if r.get('passou', False))
                        total_slots = len(self.inspection_results)
                        success = passed == total_slots

                        # Cria imagem anotada para este programa
                        program_image = self._create_annotated_image(
                            model_data.get('nome', str(mid)),
                            success,
                            f"{passed}/{total_slots} OK (Câm:{cam_idx})",
                            source_image=captured_image
                        )
                        if program_image is not None:
                            program_images.append(program_image)

                        per_program_results.append({
                            'id': mid,
                            'name': model_data.get('nome', str(mid)),
                            'success': success,
                            'details': f"{passed}/{total_slots} slots OK (Câmera {cam_idx})",
                            'camera': cam_idx
                        })
                        overall_all_ok = overall_all_ok and success
                    else:
                        # Sem resultados de inspeção
                        per_program_results.append({
                            'id': mid,
                            'name': model_data.get('nome', str(mid)),
                            'success': False,
                            'details': "Falha no alinhamento ou processamento",
                            'camera': cam_idx
                        })
                        overall_all_ok = False

                except Exception as e:
                    print(f"Erro ao processar programa {mid}: {e}")
                    per_program_results.append({
                        'id': mid,
                        'name': str(mid),
                        'success': False,
                        'details': f"Erro: {str(e)[:50]}...",
                        'camera': cam_keys[0] if cam_keys else -1
                    })
                    overall_all_ok = False

            # Restauração cuidadosa do estado original
            try:
                if original_model_id is not None and original_model_id != self.current_model_id:
                    self.load_model_from_db(original_model_id)
                elif original_img_reference is not None:
                    self.img_reference = original_img_reference
                    
                if original_camera_index != self.current_camera_index:
                    if hasattr(self, 'camera_combo'):
                        self.camera_combo.set(str(original_camera_index))
                    self.on_camera_changed()
            except Exception as restore_error:
                print(f"Erro ao restaurar estado original: {restore_error}")

            # Atualiza status final
            self.inspection_status_var.set("FINALIZADO")
            success_count = sum(1 for r in per_program_results if r.get('success', False))
            if overall_all_ok:
                self.status_var.set(f"TODOS OS PROGRAMAS: APROVADO ({success_count}/{total_programs})")
            else:
                self.status_var.set(f"ALGUNS PROGRAMAS: REPROVADO ({success_count}/{total_programs})")

            # Exibe resumo visual
            if program_images:
                self._show_multi_program_visual_summary(program_images, per_program_results, overall_all_ok)

            # Limpa as imagens capturadas da memória
            if hasattr(self, 'captured_camera_images'):
                del self.captured_camera_images

        except Exception as e:
            print(f"Erro na inspeção de múltiplos programas: {e}")
            self.status_var.set(f"Erro na inspeção: {str(e)[:100]}...")

    def start_background_camera_direct(self, camera_index):
        """Inicia a câmera diretamente em segundo plano com índice específico."""
        try:
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            # Preferir pool persistente
            try:
                from camera_manager import get_persistent_camera
                self.camera = get_persistent_camera(camera_index)
            except Exception:
                self.camera = None
            if not self.camera:
                if is_windows:
                    self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                else:
                    self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações centralizadas (inclui exposição/ganho/WB)
            try:
                from camera_manager import configure_video_capture
                configure_video_capture(self.camera, camera_index)
            except Exception:
                pass
            
            self.live_capture = True
            print(f"Webcam {camera_index} inicializada com sucesso em segundo plano")
            
            # Inicia captura de frames em thread separada
            self.start_background_frame_capture()
            
        except Exception as e:
            print(f"Erro ao inicializar webcam {camera_index}: {e}")
            self.camera = None
            self.live_capture = False
    
    def on_camera_changed(self, event=None):
        """Callback chamado quando o usuário muda a seleção da câmera - Troca instantânea."""
        try:
            if not hasattr(self, 'camera_combo') or not self.camera_combo.get():
                return
                
            new_camera_index = int(self.camera_combo.get())
            current_camera_index = getattr(self, 'current_camera_index', 0)
            
            # Só muda se for uma câmera diferente
            if new_camera_index != current_camera_index:
                print(f"Trocando instantaneamente da câmera {current_camera_index} para câmera {new_camera_index}")

                # Se o sistema dual estiver ativo, atualiza somente o índice e o frame mais recente
                if hasattr(self, 'dual_manager') and self.dual_manager and getattr(self.dual_manager, 'is_initialized', False):
                    self.current_camera_index = new_camera_index
                    frame = None
                    try:
                        # Tentativa de obter frame diretamente do driver dual
                        from dual_camera_driver import get_camera_frame
                        frame = get_camera_frame(new_camera_index)
                    except Exception:
                        frame = None
                    # Fallback para último frame armazenado do pool
                    if frame is None and hasattr(self, 'camera_frames') and new_camera_index in self.camera_frames and self.camera_frames[new_camera_index] is not None:
                        frame = self.camera_frames[new_camera_index]
                    if frame is not None:
                        try:
                            self.latest_frame = frame.copy()
                        except Exception:
                            self.latest_frame = frame
                    print(f"Câmera {new_camera_index} ativada via sistema dual")
                    return
                
                # Verifica se a nova câmera está disponível no sistema de múltiplas câmeras
                if hasattr(self, 'active_cameras') and new_camera_index in self.active_cameras and hasattr(self.active_cameras[new_camera_index], 'read'):
                    # Troca instantânea - apenas atualiza referências
                    self.current_camera_index = new_camera_index
                    self.camera = self.active_cameras[new_camera_index]
                    
                    # Atualiza latest_frame com o frame mais recente da nova câmera
                    if hasattr(self, 'camera_frames') and new_camera_index in self.camera_frames and self.camera_frames[new_camera_index] is not None:
                        try:
                            self.latest_frame = self.camera_frames[new_camera_index].copy()
                        except Exception:
                            self.latest_frame = self.camera_frames[new_camera_index]
                    
                    print(f"Câmera {new_camera_index} ativada instantaneamente")
                else:
                    # Fallback para o método tradicional se a câmera não estiver no sistema múltiplo
                    print(f"Câmera {new_camera_index} não encontrada no sistema múltiplo, usando método tradicional")
                    
                    # Para todas as capturas ativas
                    if hasattr(self, 'live_capture') and self.live_capture:
                        try:
                            self.stop_live_capture_inspection()
                            self.stop_live_capture_manual_inspection()
                        except Exception as stop_error:
                            print(f"Erro ao parar captura ao trocar câmera: {stop_error}")
                    
                    # Para visualização ao vivo se estiver ativa
                    if hasattr(self, 'live_view') and self.live_view:
                        try:
                            self.stop_live_view()
                        except Exception as stop_view_error:
                            print(f"Erro ao parar visualização ao trocar câmera: {stop_view_error}")
                    
                    # Libera câmera atual
                    if hasattr(self, 'camera') and self.camera:
                        try:
                            self.camera.release()
                            self.camera = None
                        except Exception:
                            pass
                    
                    # Atualiza índice atual
                    self.current_camera_index = new_camera_index
                    
                    # Usa o pool persistente para troca rápida
                    try:
                        self.camera = self.get_persistent_camera(new_camera_index)
                        if self.camera:
                            print(f"Câmera {new_camera_index} obtida do pool persistente")
                        else:
                            print(f"Câmera {new_camera_index} não disponível no pool persistente")
                            # Fallback para cache tradicional
                            from camera_manager import get_cached_camera
                            self.camera = get_cached_camera(new_camera_index)
                            if self.camera:
                                print(f"Câmera {new_camera_index} obtida do cache tradicional")
                    except Exception as pool_error:
                        print(f"Erro ao obter câmera do pool persistente: {pool_error}")
                        # Fallback para cache tradicional
                        try:
                            from camera_manager import get_cached_camera
                            self.camera = get_cached_camera(new_camera_index)
                            if self.camera:
                                print(f"Câmera {new_camera_index} obtida do cache tradicional (fallback)")
                        except Exception as cache_error:
                            print(f"Erro ao obter câmera do cache tradicional: {cache_error}")
                
        except Exception as e:
            print(f"Erro ao trocar câmera: {e}")
    
    def setup_ui(self):
        # Configuração de estilo industrial Keyence
        self.style = ttk.Style()
        
        # Carrega as configurações de estilo personalizadas
        style_config = load_style_config()
        
        # Cores industriais Keyence com personalização
        self.bg_color = get_color('colors.background_color', style_config)  # Fundo escuro mais profundo
        self.panel_color = get_color('colors.canvas_colors.panel_bg', style_config)  # Cor dos painéis
        self.accent_color = get_color('colors.button_color', style_config)  # Cor de destaque
        self.success_color = get_color('colors.ok_color', style_config)  # Verde brilhante industrial
        self.warning_color = get_color('colors.status_colors.warning_bg', style_config)  # Amarelo industrial
        self.danger_color = get_color('colors.ng_color', style_config)  # Vermelho industrial
        self.text_color = get_color('colors.text_color', style_config)  # Texto branco
        self.button_bg = get_color('colors.canvas_colors.button_bg')  # Cor de fundo dos botões
        self.button_active = get_color('colors.canvas_colors.button_active')  # Cor quando botão ativo
        
        # Configurar estilos
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.text_color)
        self.style.configure('TLabelframe', background=self.panel_color, borderwidth=2, relief='groove')
        self.style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.accent_color, 
                             font=style_config["ok_font"])
        
        # Botões com estilo industrial
        self.style.configure('TButton', background=self.button_bg, foreground=self.text_color, 
                             font=style_config["ok_font"], borderwidth=2, relief='raised')
        self.style.map('TButton', 
                       background=[('active', self.button_active), ('pressed', self.accent_color)],
                       foreground=[('pressed', 'white')])
        
        # Estilo para botão de inspeção (destaque)
        self.style.configure('Inspect.TButton', font=style_config["ok_font"], background=self.accent_color)
        self.style.map('Inspect.TButton',
                       background=[('active', get_color('colors.button_colors.inspect_active')), ('pressed', get_color('colors.button_colors.inspect_pressed'))])
        
        # Estilos para resultados
        self.style.configure('Success.TFrame', background=get_color('colors.inspection_colors.pass_bg'))
        self.style.configure('Danger.TFrame', background=get_color('colors.inspection_colors.fail_bg'))
        
        # Estilos para Entry e Combobox
        self.style.configure('TEntry', fieldbackground=get_color('colors.dialog_colors.entry_bg'), foreground=self.text_color)
        self.style.map('TEntry',
                       fieldbackground=[('readonly', get_color('colors.dialog_colors.entry_readonly_bg'))],
                       foreground=[('readonly', self.text_color)])
        
        self.style.configure('TCombobox', fieldbackground=get_color('colors.dialog_colors.entry_bg'), foreground=self.text_color, selectbackground=get_color('colors.dialog_colors.combobox_select_bg'))
        self.style.map('TCombobox',
                       fieldbackground=[('readonly', get_color('colors.dialog_colors.entry_readonly_bg'))],
                       foreground=[('readonly', self.text_color)])
        
        # Configurar cores para a interface - usando style em vez de configure diretamente
        # Nota: widgets ttk não suportam configuração direta de background
        # self.configure(background=self.bg_color) # Esta linha causava erro
        
        # Frame principal com layout horizontal de três painéis
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Painel esquerdo - Controles
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 10))
        
        # Painel central - Apenas imagem
        center_panel = ttk.Frame(main_frame)
        center_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        
        # Painel direito - Resultados e status
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=RIGHT, fill=Y, padx=(0, 0), pady=0, ipadx=0)
        
        # === PAINEL ESQUERDO ===
        
        # Cabeçalho com título estilo Keyence
        header_frame = ttk.Frame(left_panel, style='Header.TFrame')
        header_frame.pack(fill=X, pady=(0, 15))
        
        # Estilo para o cabeçalho
        self.style.configure('Header.TFrame', background=self.accent_color)
        
        # Logo DX Project
        try:
            from tkinter import PhotoImage
            from PIL import Image, ImageTk
            import os
            
            logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "dx_project_logo.png")

            
            if os.path.exists(logo_path):
                # Carregar e redimensionar a imagem
                pil_image = Image.open(logo_path)
                # Redimensionar mantendo proporção - altura de aproximadamente 100px
                original_width, original_height = pil_image.size
                new_height = 100
                new_width = int((new_height * original_width) / original_height)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Converter para PhotoImage
                logo_image = ImageTk.PhotoImage(pil_image)
                
                # Frame para a logo - sem estilo para evitar fundo verde
                logo_frame = ttk.Frame(header_frame)
                logo_frame.pack(pady=15, fill=X)
                
                # Label com a imagem da logo - sem background para ficar transparente
                logo_label = ttk.Label(logo_frame, image=logo_image)
                logo_label.image = logo_image  # Manter referência para evitar garbage collection
                logo_label.pack(side="left", padx=(20, 20))
            else:
                # Fallback para texto se a imagem não existir
                logo_frame = ttk.Frame(header_frame, style='Header.TFrame')
                logo_frame.pack(pady=10, fill=X)
                
                # Texto DX em estilo grande
                dx_label = ttk.Label(logo_frame, text="DX", 
                                    font=get_font('title_font'), foreground=get_color('colors.special_colors.green_text'),
                                    background=self.accent_color)
                dx_label.pack(side="left", padx=(20, 5))
                
                # Ícone de olho simulado
                eye_label = ttk.Label(logo_frame, text="👁", 
                                    font=get_font('subtitle_font'), foreground=get_color('colors.special_colors.green_text'),
                                    background=self.accent_color)
                eye_label.pack(side="left", padx=5)
                
                # Texto PROJECT
                project_label = ttk.Label(logo_frame, text="PROJECT", 
                                        font=get_font('header_font'), foreground=get_color('colors.special_colors.green_text'),
                                        background=self.accent_color)
                project_label.pack(side="left", padx=(5, 20))
            
        except Exception as e:
            # Fallback para texto simples se houver erro
            header_label = ttk.Label(header_frame, text="DX PROJECT - VISUAL INSPECTION", 
                                    font=get_font('ok_font'), foreground=get_color('colors.special_colors.green_text'),
                                    background=self.accent_color)
            header_label.pack(pady=10, fill=X)
        
        # Versão do sistema
        version_label = ttk.Label(header_frame, text="V1.0.0 - INDUSTRIAL INSPECTION", 
                                font=style_config["ok_font"].replace("12", "8"), foreground="gray")
        version_label.pack(pady=(0, 10))
        
        # Seção de Modelo - Estilo industrial Keyence
        model_frame = ttk.LabelFrame(left_panel, text="Modelo de Inspeção")
        model_frame.pack(fill=X, pady=(0, 10))
        
        # Indicador de modelo carregado
        model_indicator_frame = ttk.Frame(model_frame)
        model_indicator_frame.pack(fill=X, padx=5, pady=2)
        
        ttk.Label(model_indicator_frame, text="Status:", font=get_font('tiny_font')).pack(side=LEFT, padx=(0, 5))
        
        self.model_status_var = StringVar(value="Não carregado")
        model_status = ttk.Label(model_indicator_frame, textvariable=self.model_status_var, 
                                foreground=self.danger_color, font=get_font('tiny_font'))
        model_status.pack(side=LEFT)
        
        # Botão com ícone industrial
        self.btn_load_model = ttk.Button(model_frame, text="Carregar Modelo", 
                                       command=self.load_model_dialog, )
        self.btn_load_model.pack(fill=X, padx=5, pady=5)
        
        # Seção de Imagem de Teste - Estilo industrial
        test_frame = ttk.LabelFrame(left_panel, text="Imagem de Teste")
        test_frame.pack(fill=X, pady=(0, 10))
        
        self.btn_load_test = ttk.Button(test_frame, text="Carregar Imagem", 
                                       command=self.load_test_image)
        self.btn_load_test.pack(fill=X, padx=5, pady=2)
        
        # Seção de Webcam - Estilo industrial
        webcam_frame = ttk.LabelFrame(left_panel, text="Câmera")
        webcam_frame.pack(fill=X, pady=(0, 10))
        
        # Combobox para seleção de câmera
        camera_selection_frame = ttk.Frame(webcam_frame)
        camera_selection_frame.pack(fill=X, padx=5, pady=2)
        
        ttk.Label(camera_selection_frame, text="Câmera:").pack(side=LEFT)
        self.camera_combo = Combobox(camera_selection_frame, 
                                   values=[str(i) for i in self.available_cameras],
                                   state="readonly", width=5)
        self.camera_combo.pack(side=RIGHT)
        if self.available_cameras:
            self.camera_combo.set(str(self.available_cameras[0]))
        
        # Adiciona evento para detectar mudança de câmera
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_changed)
        
        # Nota informativa sobre o ajuste automático
        info_frame = ttk.Frame(webcam_frame)
        info_frame.pack(fill=X, padx=5, pady=2)
        
        ttk.Label(info_frame, text="A imagem será ajustada automaticamente", 
                 font=get_font('small_font'), foreground=get_color('colors.special_colors.gray_text'))\
            .pack(side=LEFT, padx=(0, 5))
        
        # Botão para iniciar/parar captura contínua
        self.btn_capture_test = ttk.Button(webcam_frame, text="CAPTURAR IMAGEM", 
                                          command=self.capture_test_from_webcam)
        self.btn_capture_test.pack(fill=X, padx=5, pady=2)
        
        # Seção de Inspeção - Estilo industrial Keyence com destaque
        inspection_frame = ttk.LabelFrame(left_panel, text="INSPEÇÃO AUTOMÁTICA")
        inspection_frame.pack(fill=X, pady=(0, 10))
        
        # Indicador de status de inspeção
        inspection_status_frame = ttk.Frame(inspection_frame)
        inspection_status_frame.pack(fill=X, padx=5, pady=2)
        
        ttk.Label(inspection_status_frame, text="SISTEMA:", font=("Arial", 8, "bold")).pack(side=LEFT, padx=(0, 5))
        
        self.inspection_status_var = StringVar(value="PRONTO")
        self.inspection_status_label = ttk.Label(inspection_status_frame, textvariable=self.inspection_status_var, 
                                     foreground=self.success_color, font=("Arial", 8, "bold"))
        self.inspection_status_label.pack(side=LEFT)
        
        # Botões de inspeção contínua removidos conforme solicitado pelo usuário
        
        # Botão para inspecionar sem tirar foto
        self.btn_inspect_only = ttk.Button(inspection_frame, text="INSPECIONAR SEM CAPTURAR", 
                                        command=self.inspect_without_capture,
                                        )
        self.btn_inspect_only.pack(fill=X, padx=5, pady=5)
        
        # Botão para inspeção com múltiplos programas
        self.btn_dual_inspect = ttk.Button(inspection_frame, text="INSPECIONAR COM PROGRAMAS...", 
                                        command=self.open_multi_program_dialog,
                                        style='Inspect.TButton')
        self.btn_dual_inspect.pack(fill=X, padx=5, pady=5)
        
        # Label grande para resultado NG/OK
        self.result_display_label = ttk.Label(inspection_frame, text="--", 
                                            font=("Arial", 36, "bold"), 
                                            foreground=get_color('colors.status_colors.muted_text'), 
                                            background=get_color('colors.status_colors.muted_bg'),
                                            anchor="center",
                                            relief="raised",
                                            borderwidth=4,
                                            padding=(20, 15))
        self.result_display_label.pack(fill=X, padx=5, pady=(10, 5), ipady=20)
        
        # === PAINEL CENTRAL - CANVAS DE INSPEÇÃO ===
        
        # Canvas de inspeção com estilo industrial - Ocupando toda a área central
        canvas_frame = ttk.LabelFrame(center_panel, text="VISUALIZAÇÃO DE INSPEÇÃO")
        canvas_frame.pack(fill=BOTH, expand=True, pady=(0, 5))
        
        # Frame para canvas e scrollbars
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=VERTICAL)
        v_scrollbar.pack(side=RIGHT, fill=Y)
        
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=HORIZONTAL)
        h_scrollbar.pack(side=BOTTOM, fill=X)
        
        # Canvas com fundo escuro estilo industrial - Ampliado para ocupar toda a área
        self.canvas = Canvas(canvas_container, bg=get_color('colors.canvas_colors.canvas_dark_bg'),
                           yscrollcommand=v_scrollbar.set,
                           xscrollcommand=h_scrollbar.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Configurar scrollbars
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        
        # Adicionar evento de redimensionamento para ajustar a imagem
        def on_canvas_configure(event):
            # Atualiza a exibição quando o canvas é redimensionado
            if hasattr(self, 'img_test') and self.img_test is not None:
                self.update_display()
        
        # Vincular evento de configuração (redimensionamento) ao canvas
        self.canvas.bind('<Configure>', on_canvas_configure)
        
        # === PAINEL DIREITO - STATUS E RESULTADOS ===
        
        # Reorganização: Painel de status expandido no topo
        status_summary_frame = ttk.LabelFrame(right_panel, text="PAINEL DE STATUS")
        status_summary_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # Frame interno para o grid de status
        self.status_grid_frame = ttk.Frame(status_summary_frame)
        self.status_grid_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Resultados - Estilo industrial Keyence (reduzido) - Movido para parte inferior
        results_frame = ttk.LabelFrame(right_panel, text="RESULTADOS DE INSPEÇÃO")
        results_frame.pack(fill=X, expand=False, side=BOTTOM, pady=(0, 10))
        
        # Painel de resumo de resultados
        summary_frame = ttk.Frame(results_frame)
        summary_frame.pack(fill=X, padx=5, pady=5)
        
        # Criar painel de resumo de status
        self.create_status_summary_panel(summary_frame)
        
        # Lista de resultados com estilo industrial (altura reduzida)
        list_container = ttk.Frame(results_frame)
        list_container.pack(fill=X, expand=False, padx=5, pady=5)
        
        scrollbar_results = ttk.Scrollbar(list_container)
        scrollbar_results.pack(side=RIGHT, fill=Y)
        
        # Configurar estilo da Treeview para parecer com sistemas Keyence
        self.style.configure("Treeview", 
                           foreground=self.text_color, 
                           borderwidth=1,
                           relief="solid")
        self.style.configure("Treeview.Heading", 
                           font=style_config["ok_font"], 
                           foreground=get_color('colors.special_colors.white_text'))
        self.style.map("Treeview", 
                      background=[("selected", get_color('colors.selection_color', style_config))],
                      foreground=[("selected", get_color('colors.special_colors.black_bg'))])
        
        # Altura reduzida para 4 linhas em vez de 8
        self.results_listbox = ttk.Treeview(list_container, yscrollcommand=scrollbar_results.set, height=4)
        self.results_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar_results.config(command=self.results_listbox.yview)
        
        # Configurar colunas da lista de resultados
        self.results_listbox["columns"] = ("status", "score", "detalhes")
        self.results_listbox.column("#0", width=40, minwidth=40)
        self.results_listbox.column("status", width=60, minwidth=60, anchor="center")
        self.results_listbox.column("score", width=60, minwidth=60, anchor="center")
        self.results_listbox.column("detalhes", width=120, minwidth=120)
        
        self.results_listbox.heading("#0", text="SLOT")
        self.results_listbox.heading("status", text="STATUS")
        self.results_listbox.heading("score", text="SCORE")
        self.results_listbox.heading("detalhes", text="DETALHES")
        
        # Configurar tags para resultados
        self.results_listbox.tag_configure("pass", background=get_color('colors.inspection_colors.pass_bg'), foreground=get_color('colors.special_colors.white_text'))
        self.results_listbox.tag_configure("fail", background=get_color('colors.inspection_colors.fail_bg'), foreground=get_color('colors.special_colors.white_text'))
        
        # Dicionário para armazenar widgets de status
        self.status_widgets = {}
        
        # Status bar estilo industrial
        status_bar_frame = ttk.Frame(self)
        status_bar_frame.pack(side=BOTTOM, fill=X)
        
        self.status_var = StringVar()
        self.status_var.set("SISTEMA PRONTO - CARREGUE UM MODELO PARA COMEÇAR")
        
        # Armazenar referência ao status_bar para poder modificar suas propriedades
        self.status_bar = ttk.Label(status_bar_frame, textvariable=self.status_var, 
                                  relief="sunken", font=style_config["ok_font"].replace("12", "9"))
        self.status_bar.pack(side=LEFT, fill=X, expand=True, padx=2, pady=2)
    
    def load_model_dialog(self):
        """Abre diálogo para selecionar modelo do banco de dados."""
        dialog = ModelSelectorDialog(self, self.db_manager)
        result = dialog.show()
        
        if result:
            if result['action'] == 'load':
                self.load_model_from_db(result['model_id'])
    
    def load_model_from_db(self, model_id):
        """Carrega um modelo do banco de dados."""
        try:
            # Carrega dados do modelo
            model_data = self.db_manager.load_modelo(model_id)
            
            # Carrega imagem de referência
            image_path = model_data['image_path']
            
            # Tenta caminho absoluto primeiro
            if not Path(image_path).exists():
                # Tenta caminho relativo ao diretório de modelos
                relative_path = MODEL_DIR / Path(image_path).name
                if relative_path.exists():
                    image_path = str(relative_path)
                else:
                    raise FileNotFoundError(f"Imagem de referência não encontrada: {image_path}")
            
            self.img_reference = cv2.imread(str(image_path))
            if self.img_reference is None:
                raise ValueError(f"Não foi possível carregar a imagem de referência: {image_path}")
            
            # Carrega slots
            self.slots = model_data['slots']
            self.current_model_id = model_id
            # Define o modelo atual para uso em outras funções
            self.current_model = model_data
            
            # Configurar câmera padrão associada ao modelo, se disponível
            camera_index = model_data.get('camera_index', 0)
            current_camera_index = int(self.camera_combo.get()) if hasattr(self, 'camera_combo') and self.camera_combo.get() else 0
            
            if hasattr(self, 'camera_combo') and str(camera_index) in [self.camera_combo['values'][i] for i in range(len(self.camera_combo['values']))]:
                self.camera_combo.set(str(camera_index))
                
                # Reinicializa a câmera se o índice mudou
                if camera_index != current_camera_index:
                    # Para captura atual se estiver ativa
                    if hasattr(self, 'live_capture') and self.live_capture:
                        try:
                            self.stop_live_capture_inspection()
                            self.stop_live_capture_manual_inspection()
                        except Exception as stop_error:
                            print(f"Erro ao parar captura ao trocar câmera: {stop_error}")
                    
                    # Para visualização ao vivo se estiver ativa
                    if hasattr(self, 'live_view') and self.live_view:
                        try:
                            self.stop_live_view()
                        except Exception as stop_view_error:
                            print(f"Erro ao parar visualização ao trocar câmera: {stop_view_error}")
                    
                    # Usa o pool de câmeras em vez de inicializar em segundo plano
                    self.current_camera_index = camera_index
                    if hasattr(self, 'camera_pool') and camera_index in self.camera_pool:
                        self.camera = self.camera_pool[camera_index]
                        print(f"Câmera {camera_index} obtida do pool para modelo")
                    elif hasattr(self, 'active_cameras') and camera_index in self.active_cameras:
                        self.camera = self.active_cameras[camera_index]
                        print(f"Câmera {camera_index} obtida do sistema ativo para modelo")
                    else:
                        # Tenta obter do cache do camera_manager
                        try:
                            cached_camera = get_cached_camera(camera_index)
                            if cached_camera:
                                self.camera = cached_camera
                                print(f"Câmera {camera_index} obtida do cache para modelo")
                            else:
                                print(f"Câmera {camera_index} não disponível para modelo")
                        except Exception as cache_error:
                            print(f"Erro ao obter câmera do cache para modelo: {cache_error}")
            
            # Limpa resultados de inspeção anteriores
            self.inspection_results = []
            
            # Limpa a lista de resultados na interface
            children = self.results_listbox.get_children()
            if children:
                self.results_listbox.delete(*children)
            
            # Resetar o label grande de resultado
            if hasattr(self, 'result_display_label'):
                self.result_display_label.config(
                    text="--",
                    foreground=get_color('colors.status_colors.muted_text'),
                    background=get_color('colors.status_colors.muted_bg')
                )
            
            # Criar painel de resumo de status
            self.create_status_summary_panel()
            
            self.status_var.set(f"Modelo carregado: {model_data['nome']} ({len(self.slots)} slots)")
            self.update_button_states()
            
            print(f"Modelo de inspeção '{model_data['nome']}' carregado com sucesso: {len(self.slots)} slots")
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.status_var.set(f"Erro ao carregar modelo: {str(e)}")
    
    def load_test_image(self):
        """Carrega imagem de teste."""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem de Teste",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.img_test = cv2.imread(str(file_path))
                if self.img_test is None:
                    raise ValueError(f"Não foi possível carregar a imagem: {file_path}")
                

                # Limpa resultados de inspeção anteriores
                self.inspection_results = []
                
                # Resetar o label grande de resultado
                if hasattr(self, 'result_display_label'):
                    self.result_display_label.config(
                        text="--",
                        foreground=get_color('colors.status_colors.muted_text'),
                        background=get_color('colors.status_colors.muted_bg')
                    )
                
                self.update_display()
                self.status_var.set(f"Imagem de teste carregada: {Path(file_path).name}")
                self.update_button_states()
                
            except Exception as e:
                print(f"Erro ao carregar imagem de teste: {e}")
                self.status_var.set(f"Erro ao carregar imagem: {str(e)}")
    
    def start_live_capture_inspection(self):
        """Inicia captura contínua da câmera em segundo plano para inspeção automática."""
        # Verifica se o atributo live_capture existe
        if not hasattr(self, 'live_capture'):
            self.live_capture = False
            
        if self.live_capture:
            return
            
        try:
            # Desativa o modo de inspeção manual se estiver ativo
            if hasattr(self, 'manual_inspection_mode') and self.manual_inspection_mode:
                try:
                    self.stop_live_capture_manual_inspection()
                except Exception as stop_error:
                    print(f"Erro ao parar inspeção manual: {stop_error}")
                
            # Verifica se o atributo camera_combo existe
            if not hasattr(self, 'camera_combo'):
                raise ValueError("Seletor de câmera não encontrado")
                
            camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
            
            # Verifica se o atributo live_view existe
            if not hasattr(self, 'live_view'):
                self.live_view = False
                
            # Para live view se estiver ativo
            if self.live_view:
                try:
                    self.stop_live_view()
                except Exception as stop_view_error:
                    print(f"Erro ao parar visualização ao vivo: {stop_view_error}")
            
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            # Preferir pool persistente
            try:
                from camera_manager import get_persistent_camera
                self.camera = get_persistent_camera(camera_index)
            except Exception:
                self.camera = None
            if not self.camera:
                if is_windows:
                    self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                else:
                    self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações centralizadas (inclui exposição/ganho/WB)
            try:
                from camera_manager import configure_video_capture
                configure_video_capture(self.camera, camera_index)
            except Exception:
                pass
            
            # Inicializa contador de frames para inspeção automática
            self._inspection_frame_count = 0
            
            self.live_capture = True
            self.manual_inspection_mode = False  # Garante que o modo de inspeção manual está desativado
            
            # Inicia o processamento de frames
            try:
                self.process_live_frame_inspection()
            except Exception as process_error:
                print(f"Erro ao iniciar processamento de frames: {process_error}")
                
            # Atualiza o status
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Inspeção com Enter iniciada - Câmera {camera_index} ativa - Pressione ENTER para inspecionar")
            
            # Limpa resultados anteriores
            self.inspection_results = []
            
            # Resetar o label grande de resultado
            if hasattr(self, 'result_display_label'):
                self.result_display_label.config(
                    text="--",
                    foreground=get_color('colors.status_colors.muted_text'),
                    background=get_color('colors.status_colors.muted_bg')
                )
            self.update_results_list()
            
            # Configura o bind da tecla Enter para inspeção
            if hasattr(self, 'master'):
                try:
                    self.master.bind('<Return>', self.on_enter_key_continuous_inspection)
                except Exception as bind_error:
                    print(f"Erro ao configurar tecla Enter para inspeção contínua: {bind_error}")
            
        except Exception as e:
            print(f"Erro ao iniciar câmera para inspeção contínua: {e}")
            messagebox.showerror("Erro", f"Erro ao iniciar câmera para inspeção contínua: {str(e)}")
    
    def stop_live_capture_inspection(self):
        """Para a captura contínua da câmera para inspeção."""
        try:
            # Verifica se os atributos existem antes de acessá-los
            if hasattr(self, 'live_capture'):
                self.live_capture = False
            
            # Verifica se o atributo live_view existe
            if not hasattr(self, 'live_view'):
                self.live_view = False
                
            # Libera a câmera se existir e não estiver sendo usada pelo live_view
            if hasattr(self, 'camera') and self.camera is not None and not self.live_view:
                try:
                    self.camera.release()
                    self.camera = None
                except Exception as release_error:
                    print(f"Erro ao liberar câmera: {release_error}")
            
            # Remove o bind da tecla Enter
            if hasattr(self, 'master'):
                try:
                    self.master.unbind('<Return>')
                except Exception as unbind_error:
                    print(f"Erro ao remover bind da tecla Enter: {unbind_error}")
            
            # Limpa o frame mais recente
            if hasattr(self, 'latest_frame'):
                self.latest_frame = None
                
            # Reseta o contador de frames de inspeção
            if hasattr(self, '_inspection_frame_count'):
                self._inspection_frame_count = 0
                
            # Atualiza o status
            if hasattr(self, 'live_view') and not self.live_view and hasattr(self, 'status_var'):
                self.status_var.set("Câmera desconectada")
                
        except Exception as e:
            print(f"Erro ao parar captura para inspeção contínua: {e}")
            # Não exibe messagebox para evitar interrupção da interface
            
    def start_live_capture_manual_inspection(self):
        """Inicia captura contínua da câmera em segundo plano para inspeção manual com Enter."""
        # Verifica se o atributo live_capture existe
        if not hasattr(self, 'live_capture'):
            self.live_capture = False
            
        if self.live_capture:
            return
        
        try:
            # Verifica se há um modelo carregado para inspeção
            if not hasattr(self, 'slots') or not self.slots or not hasattr(self, 'img_reference') or self.img_reference is None:
                # Apenas atualiza o status
                if hasattr(self, 'status_var'):
                    self.status_var.set("É necessário carregar um modelo de inspeção para iniciar a captura")
                # Referência ao botão removido - btn_continuous_inspect
                return
            
            # Verifica se o atributo camera_combo existe
            if not hasattr(self, 'camera_combo'):
                raise ValueError("Seletor de câmera não encontrado")
                
            camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
            
            # Verifica se o atributo live_view existe
            if not hasattr(self, 'live_view'):
                self.live_view = False
                
            # Para live view se estiver ativo
            if self.live_view:
                try:
                    self.stop_live_view()
                except Exception as stop_view_error:
                    print(f"Erro ao parar visualização ao vivo: {stop_view_error}")
            
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            # Configurações otimizadas para inicialização mais rápida
            # Usa DirectShow no Windows para melhor compatibilidade
            if is_windows:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações centralizadas (inclui exposição/ganho/WB)
            try:
                from camera_manager import configure_video_capture
                configure_video_capture(self.camera, camera_index)
            except Exception:
                pass
            
            self.live_capture = True
            self.manual_inspection_mode = True  # Modo de inspeção manual
            
            # Inicia o processamento de frames
            try:
                self.process_live_frame_manual_inspection()
            except Exception as process_error:
                print(f"Erro ao iniciar processamento de frames para inspeção manual: {process_error}")
                
            # Atualiza o status
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Câmera {camera_index} ativa - Pressione ENTER para capturar e inspecionar")
            
            # Referência ao botão removido - btn_continuous_inspect
            
            # Limpa resultados anteriores
            if hasattr(self, 'inspection_results'):
                self.inspection_results = []
                
                # Resetar o label grande de resultado
                if hasattr(self, 'result_display_label'):
                    self.result_display_label.config(
                        text="--",
                        foreground=get_color('colors.status_colors.muted_text'),
                        background=get_color('colors.status_colors.muted_bg')
                    )
                
            # Atualiza a lista de resultados
            if hasattr(self, 'update_results_list'):
                try:
                    self.update_results_list()
                except Exception as update_error:
                    print(f"Erro ao atualizar lista de resultados: {update_error}")
            
            # Configura o bind da tecla Enter para inspeção
            if hasattr(self, 'master'):
                try:
                    self.master.bind('<Return>', self.on_enter_key_inspection)
                except Exception as bind_error:
                    print(f"Erro ao configurar tecla Enter para inspeção: {bind_error}")
            
        except Exception as e:
            print(f"Erro ao iniciar câmera para inspeção manual: {e}")
            # Não exibe messagebox quando chamado automaticamente ao entrar na aba
            if hasattr(self, 'status_var'):
                self.status_var.set(f"Erro ao iniciar câmera: {str(e)}")
            # messagebox.showerror("Erro", f"Erro ao iniciar câmera para inspeção manual: {str(e)}")
    
    def stop_live_capture_manual_inspection(self):
        """Para a captura contínua da câmera para inspeção manual."""
        try:
            # Verifica se os atributos existem antes de acessá-los
            if hasattr(self, 'live_capture'):
                self.live_capture = False
            
            if hasattr(self, 'manual_inspection_mode'):
                self.manual_inspection_mode = False
            
            # Verifica se o atributo live_view existe
            if not hasattr(self, 'live_view'):
                self.live_view = False
                
            # Libera a câmera se existir e não estiver sendo usada pelo live_view
            if hasattr(self, 'camera') and self.camera is not None and not self.live_view:
                try:
                    self.camera.release()
                    self.camera = None
                except Exception as release_error:
                    print(f"Erro ao liberar câmera: {release_error}")
            
            # Limpa o frame mais recente
            if hasattr(self, 'latest_frame'):
                self.latest_frame = None
            
            # Remove o bind da tecla Enter
            if hasattr(self, 'master'):
                try:
                    self.master.unbind('<Return>')
                except Exception as unbind_error:
                    print(f"Erro ao remover bind da tecla Enter: {unbind_error}")
            
            # Referência ao botão removido - btn_continuous_inspect
            
            # Atualiza o status
            if hasattr(self, 'status_var'):
                self.status_var.set("Câmera desconectada")
                
        except Exception as e:
            print(f"Erro ao parar captura para inspeção manual: {e}")
            # Não exibe messagebox para evitar interrupção da interface
    
    def toggle_live_capture_manual_inspection(self):
        """Alterna entre iniciar e parar a captura contínua para inspeção manual com Enter."""
        try:
            if not hasattr(self, 'live_capture'):
                self.live_capture = False
                
            if not self.live_capture:
                # Verifica se há um modelo carregado para inspeção
                if not hasattr(self, 'slots') or not self.slots or not hasattr(self, 'img_reference') or self.img_reference is None:
                    self.status_var.set("É necessário carregar um modelo de inspeção antes de iniciar a captura")
                    return
                    
                self.start_live_capture_manual_inspection()
                # O texto do botão e status são atualizados na função start_live_capture_manual_inspection
            else:
                self.stop_live_capture_manual_inspection()
                # O texto do botão e status são atualizados na função stop_live_capture_manual_inspection
        except Exception as e:
            print(f"Erro ao alternar modo de inspeção manual: {e}")
            self.status_var.set(f"Erro ao alternar modo de inspeção: {str(e)}")
            # Referência ao botão removido - btn_continuous_inspect
            if hasattr(self, 'status_var'):
                self.status_var.set("Erro ao iniciar captura")
    
    def process_live_frame_manual_inspection(self):
        """Processa frames da câmera em segundo plano para inspeção manual (apenas captura, sem exibição ao vivo)."""
        # Verifica se todos os atributos necessários existem
        if not hasattr(self, 'live_capture') or not self.live_capture:
            return
            
        if not hasattr(self, 'camera') or not self.camera:
            return
            
        if not hasattr(self, 'manual_inspection_mode') or not self.manual_inspection_mode:
            return
        
        try:
            ret, frame = self.camera.read()
            if ret:
                self.latest_frame = frame.copy()
                
                # NÃO atualiza a exibição automaticamente - apenas mantém o frame mais recente
                # A exibição será atualizada apenas quando Enter for pressionado
        except Exception as e:
            print(f"Erro ao capturar frame: {e}")
            # Para a captura em caso de erro
            try:
                self.stop_live_capture_manual_inspection()
            except Exception as stop_error:
                print(f"Erro ao parar captura após falha: {stop_error}")
            return
        
        # Agenda próximo frame (100ms para melhor estabilidade)
        if hasattr(self, 'live_capture') and self.live_capture and hasattr(self, 'manual_inspection_mode') and self.manual_inspection_mode:
            self.master.after(100, self.process_live_frame_manual_inspection)
    
    def on_enter_key_inspection(self, event=None):
        """Manipulador de evento para a tecla Enter durante a inspeção manual."""
        # Verifica se todos os atributos necessários existem
        if not hasattr(self, 'manual_inspection_mode') or not self.manual_inspection_mode:
            return
            
        if not hasattr(self, 'live_capture') or not self.live_capture:
            return
        
        try:
            if not hasattr(self, 'latest_frame') or self.latest_frame is None:
                self.status_var.set("Nenhum frame disponível para inspeção")
                return
                
            # Usa o frame mais recente para inspeção
            self.img_test = self.latest_frame.copy()
            
            # Salva a imagem no histórico de fotos
            try:
                self.save_to_photo_history(self.img_test)
            except Exception as save_error:
                print(f"Erro ao salvar no histórico: {save_error}")
            
            # Exibe a imagem em tela cheia
            self.show_fullscreen_image()
            
            # Executa inspeção
            try:
                self.run_inspection()
            except Exception as inspect_error:
                print(f"Erro durante inspeção: {inspect_error}")
                self.status_var.set(f"Erro durante inspeção: {str(inspect_error)}")
            
            # Atualiza status
            self.status_var.set("Inspeção realizada - Pressione ENTER para nova inspeção")
        except Exception as e:
            print(f"Erro ao realizar inspeção manual: {e}")
            self.status_var.set(f"Erro ao realizar inspeção manual: {str(e)}")
            
    def on_enter_key_continuous_inspection(self, event=None):
        """Manipulador de evento para a tecla Enter durante a inspeção contínua."""
        # Verifica se está no modo de inspeção contínua
        if not hasattr(self, 'live_capture') or not self.live_capture:
            return
            
        # Verifica se não está no modo de inspeção manual
        if hasattr(self, 'manual_inspection_mode') and self.manual_inspection_mode:
            return
        
        try:
            if not hasattr(self, 'latest_frame') or self.latest_frame is None:
                self.status_var.set("Nenhum frame disponível para inspeção")
                return
                
            # Usa o frame mais recente para inspeção
            self.img_test = self.latest_frame.copy()
            
            # Salva a imagem no histórico de fotos
            try:
                self.save_to_photo_history(self.img_test)
            except Exception as save_error:
                print(f"Erro ao salvar no histórico: {save_error}")
            
            # Exibe a imagem em tela cheia
            self.show_fullscreen_image()
            
            # Executa inspeção
            try:
                self.run_inspection()
            except Exception as inspect_error:
                print(f"Erro durante inspeção: {inspect_error}")
                self.status_var.set(f"Erro durante inspeção: {str(inspect_error)}")
            
            # Atualiza status
            self.status_var.set("Inspeção realizada - Pressione ENTER para nova inspeção")
        except Exception as e:
            print(f"Erro ao realizar inspeção contínua: {e}")
            self.status_var.set(f"Erro ao realizar inspeção contínua: {str(e)}")
    
    def inspect_without_capture(self):
        """Executa inspeção na imagem atual sem capturar uma nova foto."""
        try:
            # Verifica se há uma imagem carregada
            if not hasattr(self, 'img_test') or self.img_test is None:
                self.status_var.set("Nenhuma imagem disponível para inspeção")
                return
                
            # Verifica se há um modelo carregado
            if not hasattr(self, 'slots') or not self.slots or not hasattr(self, 'img_reference') or self.img_reference is None:
                self.status_var.set("É necessário carregar um modelo de inspeção")
                return
            
            # Exibe a imagem em tela cheia
            self.show_fullscreen_image()
            
            # Executa inspeção
            self.run_inspection()
            
            # Atualiza status
            self.status_var.set("Inspeção realizada com sucesso")
            
        except Exception as e:
            print(f"Erro ao inspecionar sem capturar: {e}")
            self.status_var.set(f"Erro ao inspecionar: {str(e)}")
    
    def show_fullscreen_image(self):
        """Exibe a imagem atual em tela cheia temporariamente."""
        try:
            if not hasattr(self, 'img_test') or self.img_test is None:
                return
                
            # Obtém dimensões do canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Se o canvas ainda não foi renderizado, use valores padrão
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 640
                canvas_height = 480
            
            # Usa a função cv2_to_tk para manter consistência com o resto do código
            self.img_display, self.scale_factor = cv2_to_tk(self.img_test, max_w=canvas_width, max_h=canvas_height)
            
            if self.img_display is None:
                return
            
            # Calcula dimensões da imagem redimensionada
            img_height, img_width = self.img_test.shape[:2]
            new_width = int(img_width * self.scale_factor)
            new_height = int(img_height * self.scale_factor)
            
            # Remove apenas overlays, não toda a imagem
            self.canvas.delete("result_overlay")
            self.canvas.delete("inspection")
            
            # Calcula offsets para centralização (consistente com update_display)
            self.x_offset = max(0, (canvas_width - new_width) // 2)
            self.y_offset = max(0, (canvas_height - new_height) // 2)
            
            # Cria ou atualiza imagem (consistente com update_display)
            if not hasattr(self, '_canvas_image_id') or self._canvas_image_id is None:
                self._canvas_image_id = self.canvas.create_image(self.x_offset, self.y_offset, anchor=NW, image=self.img_display)
            else:
                try:
                    self.canvas.itemconfig(self._canvas_image_id, image=self.img_display)
                    self.canvas.coords(self._canvas_image_id, self.x_offset, self.y_offset)
                except Exception:
                    # Se falhar, cria nova imagem
                    self._canvas_image_id = self.canvas.create_image(self.x_offset, self.y_offset, anchor=NW, image=self.img_display)
            
            # Atualiza o canvas
            self.canvas.update()
            
            # Aguarda um momento para que o usuário veja a imagem
            self.master.update()
            
        except Exception as e:
            print(f"Erro ao exibir imagem em tela cheia: {e}")
    
    def open_dual_inspection_dialog(self):
        """Abre diálogo para configurar inspeção dual com dois modelos."""
        # Redireciona para o diálogo multi-programas (compatibilidade)
        self.open_multi_program_dialog()
    
    def open_multi_program_dialog(self):
        """Abre diálogo para seleção dinâmica de múltiplos programas (com botão +)."""
        try:
            if not self.db_manager:
                messagebox.showerror("Erro", "Banco de dados não disponível")
                return
            dialog = MultiProgramDialog(self, self.db_manager, preselected_ids=self.selected_program_ids)
            result = dialog.show()
            if result and result.get('program_ids'):
                # Salva seleção para uso nas capturas
                self.selected_program_ids = result['program_ids']
                # Feedback na UI
                try:
                    names = []
                    for mid in self.selected_program_ids:
                        info = self.db_manager.get_model_by_id(mid)
                        names.append(info.get('nome', str(mid)) if info else str(mid))
                    self.status_var.set(f"Programas selecionados: {', '.join(names)}")
                except Exception:
                    self.status_var.set(f"Programas selecionados: {self.selected_program_ids}")
        except Exception as e:
            print(f"Erro ao abrir diálogo multi-programas: {e}")
            messagebox.showerror("Erro", f"Erro ao abrir diálogo: {str(e)}")
    
    def _create_annotated_image(self, program_name, success, details, source_image=None):
        """Cria uma imagem anotada com o resultado da inspeção do programa."""
        try:
            # Usa a imagem fornecida ou fallback para img_test
            if source_image is not None:
                base_image = source_image
            elif hasattr(self, 'img_test') and self.img_test is not None:
                base_image = self.img_test
            else:
                return None
                
            # Cria uma cópia da imagem base
            img_copy = base_image.copy()
            h, w = img_copy.shape[:2]
            
            # Escala dinâmica para melhor qualidade de texto
            font_scale_base = max(0.5, min(1.5, w / 800))  # Escala baseada na largura da imagem
            
            # Carrega as configurações de estilo
            style_config = load_style_config()
            
            # Função para converter cor hex para BGR
            def hex_to_bgr(hex_color):
                if hex_color.startswith('#'):
                    hex_color = hex_color[1:]
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return (b, g, r)  # OpenCV usa BGR
            
            # Faixa de status sobre a imagem inspecionada (overlay)
            try:
                status_color = hex_to_bgr(get_color('colors.ok_color', style_config)) if success else hex_to_bgr(get_color('colors.ng_color', style_config))
                overlay = img_copy.copy()
                h, w = img_copy.shape[:2]
                faixa_h = max(40, int(0.08 * h))  # 8% da altura ou 40px mínimo
                cv2.rectangle(overlay, (0, 0), (w, faixa_h), status_color, -1)
                img_copy = cv2.addWeighted(overlay, 0.45, img_copy, 0.55, 0)
                status_text = f"{program_name} - {'APROVADO' if success else 'REPROVADO'}"
                font_scale = font_scale_base * 0.8
                thickness = max(1, int(font_scale * 2))
                (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                tx = max(10, (w - tw) // 2)
                ty = min(faixa_h - 10, faixa_h - (faixa_h - th)//2)
                # Contorno escuro
                cv2.putText(img_copy, status_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                # Texto principal
                cv2.putText(img_copy, status_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            except Exception:
                pass
            
            # Desenha os slots se existem resultados de inspeção
            if hasattr(self, 'inspection_results') and self.inspection_results:
                for result in self.inspection_results:
                    slot = result['slot_data']
                    
                    # Coordenadas do slot
                    x1, y1 = int(slot['x']), int(slot['y'])
                    x2, y2 = int(slot['x'] + slot['w']), int(slot['y'] + slot['h'])
                    
                    # Cores baseadas no resultado
                    if result['passou']:
                        outline_color = hex_to_bgr(get_color('colors.ok_color', style_config))
                        text_bg_color = hex_to_bgr(get_color('colors.ok_color', style_config))
                    else:
                        outline_color = hex_to_bgr(get_color('colors.ng_color', style_config))
                        text_bg_color = hex_to_bgr(get_color('colors.ng_color', style_config))
                    
                    # Desenha retângulo do slot
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), outline_color, 3)
                    
                    # Cria fundo para o texto
                    text_bg_width, text_bg_height = 100, 25
                    cv2.rectangle(img_copy, (x1, y1), (x1 + text_bg_width, y1 + text_bg_height), text_bg_color, -1)
                    cv2.rectangle(img_copy, (x1, y1), (x1 + text_bg_width, y1 + text_bg_height), outline_color, 2)
                    
                    # Adiciona texto com resultado usando escala dinâmica
                    status_text = "OK" if result['passou'] else "NG"
                    text = f"S{slot['id']}: {status_text}"
                    slot_font_scale = font_scale_base * 0.5
                    slot_thickness = max(1, int(slot_font_scale * 2))
                    cv2.putText(img_copy, text, (x1 + 5, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, slot_font_scale, (255, 255, 255), slot_thickness, cv2.LINE_AA)
                    
                    # Adiciona score com anti-aliasing
                    score_text = f"{result['score']:.2f}"
                    score_font_scale = font_scale_base * 0.4
                    score_thickness = max(1, int(score_font_scale * 2))
                    cv2.putText(img_copy, score_text, (x2 - 40, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, outline_color, score_thickness, cv2.LINE_AA)
            
            # Adiciona título do programa na parte superior
            title_height = 40
            title_img = np.zeros((title_height, img_copy.shape[1], 3), dtype=np.uint8)
            
            # Cor de fundo do título baseada no resultado
            bg_color = hex_to_bgr(get_color('colors.ok_color', style_config)) if success else hex_to_bgr(get_color('colors.ng_color', style_config))
            title_img[:] = bg_color
            
            # Adiciona texto do título com escala dinâmica
            title_text = f"{program_name} - {'OK' if success else 'NG'}"
            title_font_scale = font_scale_base * 0.7
            title_thickness = max(1, int(title_font_scale * 2))
            text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_thickness)[0]
            text_x = (title_img.shape[1] - text_size[0]) // 2
            cv2.putText(title_img, title_text, (text_x, 25), cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)
            
            # Adiciona detalhes na segunda linha com anti-aliasing
            if details:
                details_font_scale = font_scale_base * 0.4
                details_thickness = max(1, int(details_font_scale * 2))
                cv2.putText(title_img, details, (text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, details_font_scale, (255, 255, 255), details_thickness, cv2.LINE_AA)
            
            # Combina título com imagem
            final_img = np.vstack([title_img, img_copy])
            
            return final_img
            
        except Exception as e:
            print(f"Erro ao criar imagem anotada: {e}")
            return None

    def _show_multi_program_visual_summary(self, program_images, results, overall_success):
        """Exibe resumo visual com uma única imagem em tamanho completo no canvas."""
        try:
            if not program_images:
                return
                
            # Usa apenas a primeira imagem em tamanho completo
            # ao invés de criar thumbnails ou grid
            composite_img = program_images[0] if program_images else None
            
            if composite_img is not None:
                self._display_composite_image(composite_img, overall_success)
                
        except Exception as e:
            print(f"Erro ao mostrar resumo visual: {e}")

    def _create_side_by_side_layout(self, images):
        """Cria layout lado a lado para 2 programas."""
        try:
            if len(images) != 2:
                return None
                
            img1, img2 = images
            
            # Redimensiona para altura uniforme
            target_height = min(img1.shape[0], img2.shape[0], 600)
            
            # Calcula novas larguras mantendo proporção
            ratio1 = target_height / img1.shape[0]
            ratio2 = target_height / img2.shape[0]
            new_w1 = int(img1.shape[1] * ratio1)
            new_w2 = int(img2.shape[1] * ratio2)
            
            # Redimensiona imagens com interpolação adequada
            interp1 = cv2.INTER_AREA if ratio1 < 1.0 else cv2.INTER_LANCZOS4
            interp2 = cv2.INTER_AREA if ratio2 < 1.0 else cv2.INTER_LANCZOS4
            img1_resized = cv2.resize(img1, (new_w1, target_height), interpolation=interp1)
            img2_resized = cv2.resize(img2, (new_w2, target_height), interpolation=interp2)
            
            # Combina lado a lado
            composite = np.hstack([img1_resized, img2_resized])
            
            return composite
            
        except Exception as e:
            print(f"Erro ao criar layout lado a lado: {e}")
            return None

    def _create_three_program_layout(self, images):
        """Cria layout com 2 em cima e 1 embaixo para 3 programas."""
        try:
            if len(images) != 3:
                return None
                
            img1, img2, img3 = images
            
            # Define altura alvo
            target_height = 400
            
            # Redimensiona as duas primeiras imagens
            ratio1 = target_height / img1.shape[0]
            ratio2 = target_height / img2.shape[0]
            new_w1 = int(img1.shape[1] * ratio1)
            new_w2 = int(img2.shape[1] * ratio2)
            
            interp1 = cv2.INTER_AREA if ratio1 < 1.0 else cv2.INTER_LANCZOS4
            interp2 = cv2.INTER_AREA if ratio2 < 1.0 else cv2.INTER_LANCZOS4
            img1_resized = cv2.resize(img1, (new_w1, target_height), interpolation=interp1)
            img2_resized = cv2.resize(img2, (new_w2, target_height), interpolation=interp2)
            
            # Combina as duas primeiras lado a lado
            top_row = np.hstack([img1_resized, img2_resized])
            
            # Redimensiona a terceira imagem para ter a mesma largura do topo
            target_width = top_row.shape[1]
            ratio3 = target_width / img3.shape[1]
            new_h3 = int(img3.shape[0] * ratio3)
            
            interp3 = cv2.INTER_AREA if ratio3 < 1.0 else cv2.INTER_LANCZOS4
            img3_resized = cv2.resize(img3, (target_width, new_h3), interpolation=interp3)
            
            # Combina verticalmente
            composite = np.vstack([top_row, img3_resized])
            
            return composite
            
        except Exception as e:
            print(f"Erro ao criar layout de três programas: {e}")
            return None

    def _create_grid_layout(self, images):
        """Cria layout em grid para mais de 3 programas."""
        try:
            num_images = len(images)
            if num_images == 0:
                return None
                
            # Calcula dimensões do grid
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
            
            # Altura e largura alvo para cada imagem
            target_height = 360
            target_width = 480
            
            # Redimensiona todas as imagens mantendo a proporção dentro do alvo
            resized_images = []
            for img in images:
                h, w = img.shape[:2]
                scale = min(target_width / w, target_height / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
                # Coloca em um canvas do tamanho alvo para alinhamento uniforme
                canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                y_off = (target_height - new_h) // 2
                x_off = (target_width - new_w) // 2
                canvas[y_off:y_off+new_h, x_off:x_off+new_w] = img_resized
                resized_images.append(canvas)
            
            # Preenche com imagens em branco se necessário
            while len(resized_images) < rows * cols:
                blank_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                resized_images.append(blank_img)
            
            # Cria o grid
            grid_rows = []
            for r in range(rows):
                row_images = resized_images[r * cols:(r + 1) * cols]
                if row_images:
                    row = np.hstack(row_images)
                    grid_rows.append(row)
            
            if grid_rows:
                composite = np.vstack(grid_rows)
                return composite
                
            return None
            
        except Exception as e:
            print(f"Erro ao criar layout de grid: {e}")
            return None

    def _display_composite_image(self, composite_img, overall_success):
        """Exibe a imagem composta no canvas temporariamente."""
        try:
            if not hasattr(self, 'canvas') or composite_img is None:
                return
            
            # Flag para bloquear update_display enquanto composta está ativa
            self._composite_active = True
                
            # Salva estado atual da imagem antes de substituir
            self._save_current_display_state()
                
            # Converte para formato Tk
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 800, 600
                
            composite_tk, scale = cv2_to_tk(composite_img, max_w=canvas_width, max_h=canvas_height)
            
            if composite_tk is None:
                self._composite_active = False
                return
                
            # Remove apenas overlays, preservando a imagem base se possível
            self.canvas.delete("result_overlay")
            self.canvas.delete("composite_display")
            self.canvas.delete("composite_text")
            
            # Centraliza a imagem composta e calcula offsets
            img_height, img_width = composite_img.shape[:2]
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            x_offset = max(0, (canvas_width - new_width) // 2)
            y_offset = max(0, (canvas_height - new_height) // 2)
            
            # Cria imagem composta com tag específica
            self._composite_image_id = self.canvas.create_image(
                x_offset, y_offset, anchor=NW, image=composite_tk, tags=("composite_display", "result_overlay")
            )
            
            # Armazena referência para evitar garbage collection
            self._composite_image_ref = composite_tk
            
            # Adiciona texto de alta qualidade diretamente no canvas usando Tkinter
            self._add_canvas_text_overlay(x_offset, y_offset, new_width, new_height, overall_success)
            
            # Agenda restauração da imagem original após alguns segundos (sem popup)
            if hasattr(self, '_composite_restore_timer'):
                try:
                    self.master.after_cancel(self._composite_restore_timer)
                except Exception:
                    pass
            self._composite_restore_timer = self.master.after(5000, self._restore_original_display)
            
        except Exception as e:
            print(f"Erro ao exibir imagem composta: {e}")
            self._composite_active = False
    
    def _add_canvas_text_overlay(self, img_x, img_y, img_width, img_height, overall_success):
        """Adiciona texto de alta qualidade diretamente no canvas usando Tkinter."""
        try:
            # Carrega configurações de estilo
            style_config = load_style_config()
            
            # Cores baseadas no resultado
            if overall_success:
                bg_color = get_color('colors.ok_color', style_config)
                text_color = "white"
            else:
                bg_color = get_color('colors.ng_color', style_config)
                text_color = "white"
            
            # Calcula tamanho da fonte baseado na largura da imagem
            base_font_size = max(12, min(24, img_width // 40))
            
            # Cria faixa de status no topo da imagem
            status_height = max(30, img_height // 15)
            status_rect = self.canvas.create_rectangle(
                img_x, img_y, img_x + img_width, img_y + status_height,
                fill=bg_color, outline=bg_color, tags="composite_text"
            )
            
            # Texto principal do status
            status_text = f"RESULTADO GERAL: {'APROVADO' if overall_success else 'REPROVADO'}"
            status_font = ("Arial", base_font_size, "bold")
            
            status_text_id = self.canvas.create_text(
                img_x + img_width // 2, img_y + status_height // 2,
                text=status_text, fill=text_color, font=status_font,
                anchor="center", tags="composite_text"
            )
            
            # Se há resultados de inspeção, adiciona informações dos slots
            if hasattr(self, 'inspection_results') and self.inspection_results:
                y_pos = img_y + status_height + 10
                slot_font = ("Arial", max(10, base_font_size - 4), "normal")
                
                for i, result in enumerate(self.inspection_results[:5]):  # Máximo 5 slots para não sobrecarregar
                    slot = result['slot_data']
                    slot_status = "OK" if result['passou'] else "NG"
                    slot_text = f"Slot {slot['id']}: {slot_status} ({result['score']:.2f})"
                    
                    # Cor do texto baseada no resultado do slot
                    slot_color = "#00AA00" if result['passou'] else "#AA0000"
                    
                    self.canvas.create_text(
                        img_x + 10, y_pos,
                        text=slot_text, fill=slot_color, font=slot_font,
                        anchor="nw", tags="composite_text"
                    )
                    y_pos += max(15, base_font_size)
            
            # Adiciona timestamp no canto inferior direito
            timestamp = datetime.now().strftime("%H:%M:%S")
            timestamp_font = ("Arial", max(8, base_font_size - 6), "normal")
            
            self.canvas.create_text(
                img_x + img_width - 10, img_y + img_height - 10,
                text=timestamp, fill="#CCCCCC", font=timestamp_font,
                anchor="se", tags="composite_text"
            )
            
        except Exception as e:
            print(f"Erro ao adicionar texto no canvas: {e}")

    def _save_current_display_state(self):
        """Salva o estado atual do display para restauração posterior"""
        try:
            # Salva referências da imagem atual
            if hasattr(self, 'img_display'):
                self._saved_img_display = self.img_display
            if hasattr(self, '_canvas_image_id'):
                self._saved_canvas_image_id = self._canvas_image_id
            if hasattr(self, 'x_offset'):
                self._saved_x_offset = self.x_offset
            if hasattr(self, 'y_offset'):
                self._saved_y_offset = self.y_offset
                
            # Salva o estado atual dos labels de resultado
            if hasattr(self, 'result_display_label'):
                self._saved_result_text = self.result_display_label.cget('text')
                self._saved_result_fg = self.result_display_label.cget('foreground')
                self._saved_result_bg = self.result_display_label.cget('background')
                
        except Exception as e:
            print(f"Erro ao salvar estado do display: {e}")

    def _restore_original_display(self):
        """Restaura a exibição original no canvas mantendo a imagem visível."""
        try:
            # Remove flag de bloqueio
            self._composite_active = False
            
            # Cancela timer se existir
            if hasattr(self, '_composite_restore_timer'):
                self.master.after_cancel(self._composite_restore_timer)
                delattr(self, '_composite_restore_timer')
                
            # Remove apenas overlays, mantém a imagem base
            if hasattr(self, 'canvas'):
                self.canvas.delete("composite_display")
                self.canvas.delete("composite_text")
                self.canvas.delete("result_overlay")
                # Reset do ID da imagem para forçar recriação limpa
                if hasattr(self, '_canvas_image_id'):
                    self._canvas_image_id = None
            
            # Limpa referência da imagem composta
            if hasattr(self, '_composite_image_ref'):
                delattr(self, '_composite_image_ref')
            if hasattr(self, '_composite_image_id'):
                delattr(self, '_composite_image_id')
            
            # Garante que a imagem atual permaneça visível
            if hasattr(self, 'img_test') and self.img_test is not None:
                try:
                    # Sempre usa update_display para garantir consistência
                    self.update_display()
                    
                    # Restaura o estado dos labels de resultado se foram salvos
                    if hasattr(self, '_saved_result_text') and hasattr(self, 'result_display_label'):
                        try:
                            self.result_display_label.config(
                                text=self._saved_result_text,
                                foreground=self._saved_result_fg,
                                background=self._saved_result_bg
                            )
                        except Exception as label_error:
                            print(f"Erro ao restaurar label de resultado: {label_error}")
                            
                    # Redesenha resultados de inspeção se existirem
                    if hasattr(self, 'inspection_results') and self.inspection_results:
                        self.draw_inspection_results()
                        
                except Exception as update_error:
                    print(f"Erro ao atualizar display na restauração: {update_error}")
                    # Fallback final: tenta recriar a imagem do zero
                    try:
                        if hasattr(self, 'img_test') and self.img_test is not None:
                            self.update_display()
                            print("Fallback: imagem restaurada via update_display")
                    except Exception as final_error:
                        print(f"Erro no fallback final: {final_error}")
            else:
                print("Aviso: img_test não disponível para restauração")
            
            # Limpa estados salvos após uso
            for attr in ['_saved_img_display', '_saved_canvas_image_id', '_saved_x_offset', '_saved_y_offset',
                        '_saved_result_text', '_saved_result_fg', '_saved_result_bg']:
                if hasattr(self, attr):
                    delattr(self, attr)
                
        except Exception as e:
            print(f"Erro ao restaurar display original: {e}")
            # Fallback final: força atualização se possível
            try:
                if hasattr(self, 'img_test') and self.img_test is not None:
                    self.update_display()
            except:
                pass

    def run_dual_inspection(self, model1_id, model2_id):
        """Executa inspeção sequencial com dois modelos."""
        # Redireciona para execução multi, preservando compatibilidade
        try:
            self.run_multi_program_inspection([model1_id, model2_id])
        except Exception as e:
            print(f"Erro geral na inspeção dual: {e}")
            self.status_var.set(f"ERRO NA INSPEÇÃO DUAL: {str(e)}")
            messagebox.showerror("Erro", f"Erro na inspeção dual: {str(e)}")
    
    def show_dual_inspection_result(self, results):
        """Exibe resultado da inspeção dual no canvas (sem popups)."""
        try:
            # Carrega informações dos modelos
            model1_data = self.db_manager.load_modelo(results['model1']['id'])
            model2_data = self.db_manager.load_modelo(results['model2']['id'])
            
            # Cria resumo visual no canvas
            try:
                # Dimensões do canvas
                canvas_w = self.canvas.winfo_width() if hasattr(self, 'canvas') else 800
                canvas_h = self.canvas.winfo_height() if hasattr(self, 'canvas') else 600
                if canvas_w <= 1 or canvas_h <= 1:
                    canvas_w, canvas_h = 800, 600

                # Imagem de fundo
                bg_color = (30, 30, 30)  # BGR
                img = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

                # Cabeçalho
                header_h = int(0.14 * canvas_h)
                header_color = (0, 140, 0) if results['overall_success'] else (0, 0, 160)  # verde/vermelho
                cv2.rectangle(img, (0, 0), (canvas_w, header_h), header_color, thickness=-1)

                final_status = "APROVADO" if results['overall_success'] else "REPROVADO"
                title_text = f"RESULTADO DUAL: {final_status}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                title_scale = 1.0
                title_thickness = 2
                (tw, th), _ = cv2.getTextSize(title_text, font, title_scale, title_thickness)
                cv2.putText(img, title_text, ((canvas_w - tw) // 2, header_h // 2 + th // 2), font, title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)

                # Corpo com detalhes dos programas
                y = header_h + int(0.06 * canvas_h)
                line_scale = 0.7
                line_thickness = 2
                line_height = int(28 * (canvas_h / 600))

                # Programa 1
                prog1_line = f"PROGRAMA 1: {model1_data['nome']} - {'APROVADO' if results['model1']['success'] else 'REPROVADO'}"
                cv2.putText(img, prog1_line, (30, y), font, line_scale, (255, 255, 255), line_thickness, cv2.LINE_AA)
                y += line_height
                if results['model1']['details']:
                    cv2.putText(img, f"Detalhes: {results['model1']['details']}", (50, y), font, line_scale, (200, 200, 200), 1, cv2.LINE_AA)
                    y += line_height
                y += line_height // 2

                # Programa 2
                prog2_line = f"PROGRAMA 2: {model2_data['nome']} - {'APROVADO' if results['model2']['success'] else 'REPROVADO'}"
                cv2.putText(img, prog2_line, (30, y), font, line_scale, (255, 255, 255), line_thickness, cv2.LINE_AA)
                y += line_height
                if results['model2']['details']:
                    cv2.putText(img, f"Detalhes: {results['model2']['details']}", (50, y), font, line_scale, (200, 200, 200), 1, cv2.LINE_AA)
                    y += line_height
                y += line_height

                # Status final
                final_msg = "✅ Ambos os programas passaram!" if results['overall_success'] else "❌ Um ou ambos falharam!"
                cv2.putText(img, final_msg, (30, y), font, line_scale, (255, 255, 255), line_thickness, cv2.LINE_AA)

                # Exibir no canvas
                self._display_composite_image(img, results['overall_success'])
                
            except Exception as canvas_error:
                print(f"Erro ao criar resumo visual dual: {canvas_error}")
                
        except Exception as e:
            print(f"Erro ao exibir resultado dual: {e}")

    def run_multi_program_inspection(self, program_ids):
        """Executa inspeção sequencial para todos os programas fornecidos."""
        if not program_ids:
            return
        try:
            # Armazena o modelo original para restaurar ao final
            original_model_id = getattr(self, 'current_model_id', None)
            overall_all_ok = True
            per_program_results = []
            program_images = []  # Lista para armazenar imagens com anotações
            
            for idx, mid in enumerate(program_ids, start=1):
                try:
                    model_data = self.db_manager.load_modelo(mid)
                    # Troca de modelo (também ajusta câmera se necessário)
                    self.status_var.set(f"CARREGANDO PROGRAMA {idx}/{len(program_ids)}...")
                    self.load_model_from_db(mid)
                    time.sleep(0.2)
                    
                    # Executa inspeção na imagem atual (sem recapturar)
                    if self.img_test is None:
                        # se não tiver imagem, não há o que inspecionar
                        self.status_var.set("Sem imagem capturada para inspecionar")
                        break
                    else:
                        self.run_inspection()
                    
                    # Coletar resumo
                    if hasattr(self, 'inspection_results') and self.inspection_results:
                        passed = sum(1 for r in self.inspection_results if r.get('passou', False))
                        total = len(self.inspection_results)
                        success = passed == total
                        
                        # Criar imagem com anotações para este programa
                        program_image = self._create_annotated_image(
                            model_data.get('nome', str(mid)), 
                            success, 
                            f"{passed}/{total} OK"
                        )
                        if program_image is not None:
                            program_images.append(program_image)
                        
                        per_program_results.append({
                            'id': mid,
                            'name': model_data.get('nome', str(mid)),
                            'success': success,
                            'details': f"{passed}/{total} slots OK",
                        })
                        overall_all_ok = overall_all_ok and success
                    else:
                        per_program_results.append({
                            'id': mid,
                            'name': model_data.get('nome', str(mid)),
                            'success': False,
                            'details': 'Erro na inspeção',
                        })
                        overall_all_ok = False
                        
                except Exception as e:
                    print(f"Erro ao inspecionar programa {mid}: {e}")
                    per_program_results.append({
                        'id': mid,
                        'name': str(mid),
                        'success': False,
                        'details': f"Erro: {e}",
                    })
                    overall_all_ok = False
                    
            # Restaurar modelo original ao final
            if original_model_id is not None and (original_model_id not in program_ids or original_model_id != self.current_model_id):
                try:
                    self.load_model_from_db(original_model_id)
                except Exception:
                    pass
                    
            # Atualizar barra e mostrar resumo
            self.inspection_status_var.set("FINALIZADO")
            if overall_all_ok:
                self.status_var.set("TODOS OS PROGRAMAS: APROVADO")
            else:
                self.status_var.set("ALGUNS PROGRAMAS: REPROVADO")
                
            # Exibir resumo visual se há imagens coletadas
            if program_images:
                self._show_multi_program_visual_summary(program_images, per_program_results, overall_all_ok)
            else:
                # Fallback visual no canvas (sem popups)
                try:
                    # Dimensões do canvas
                    canvas_w = self.canvas.winfo_width() if hasattr(self, 'canvas') else 800
                    canvas_h = self.canvas.winfo_height() if hasattr(self, 'canvas') else 600
                    if canvas_w <= 1 or canvas_h <= 1:
                        canvas_w, canvas_h = 800, 600

                    # Imagem de fundo
                    bg_color = (30, 30, 30)  # BGR
                    img = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

                    # Cabeçalho
                    header_h = int(0.14 * canvas_h)
                    header_color = (0, 140, 0) if overall_all_ok else (0, 0, 160)  # verde/azulado para contraste
                    cv2.rectangle(img, (0, 0), (canvas_w, header_h), header_color, thickness=-1)

                    title_text = f"RESULTADO FINAL: {'APROVADO' if overall_all_ok else 'REPROVADO'}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    title_scale = 1.0
                    title_thickness = 2
                    (tw, th), _ = cv2.getTextSize(title_text, font, title_scale, title_thickness)
                    cv2.putText(img, title_text, ((canvas_w - tw) // 2, header_h // 2 + th // 2), font, title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)

                    # Corpo com detalhes por programa
                    y = header_h + int(0.06 * canvas_h)
                    line_scale = 0.7
                    line_thickness = 2
                    line_height = int(28 * (canvas_h / 600))

                    for r in per_program_results:
                        program_line = f"PROGRAMA: {r['name']} (ID {r['id']}) - {'APROVADO' if r['success'] else 'REPROVADO'}"
                        cv2.putText(img, program_line, (30, y), font, line_scale, (255, 255, 255), line_thickness, cv2.LINE_AA)
                        y += line_height
                        details = str(r.get('details', ''))
                        if details:
                            # Quebra detalhes em linhas menores se necessário
                            max_chars = 70
                            for i in range(0, len(details), max_chars):
                                cv2.putText(img, details[i:i+max_chars], (50, y), font, line_scale, (200, 200, 200), 1, cv2.LINE_AA)
                                y += line_height
                        y += line_height // 2

                    # Exibir no canvas
                    self._display_composite_image(img, overall_all_ok)
                except Exception as _e:
                    print(f"Falha ao exibir fallback visual: {_e}")
                    pass
                    
        except Exception as e:
            print(f"Erro na inspeção multi-programas: {e}")
            # Exibe erro apenas no status, sem pop-up
            self.status_var.set(f"ERRO: {str(e)}")
            self.inspection_status_var.set("ERRO")
    
    def toggle_live_capture_inspection(self):
        """Alterna entre iniciar e parar a captura contínua para inspeção automática."""
        # Redireciona para a função de inspeção manual com Enter, que agora é a única forma de inspeção
        self.toggle_live_capture_manual_inspection()
    
    def process_live_frame_inspection(self):
        """Processa frames da câmera em segundo plano para inspeção (apenas captura, sem exibição ao vivo)."""
        # Verifica se todos os atributos necessários existem
        if not hasattr(self, 'live_capture') or not self.live_capture:
            return
            
        if not hasattr(self, 'camera') or not self.camera:
            return
        
        try:
            ret, frame = self.camera.read()
            if ret:
                self.latest_frame = frame.copy()
                
                # NÃO atualiza a exibição automaticamente - apenas mantém o frame mais recente
                # A exibição e inspeção serão executadas apenas quando Enter for pressionado
        except Exception as e:
            print(f"Erro ao capturar frame: {e}")
            # Para a captura em caso de erro
            try:
                self.stop_live_capture_inspection()
            except Exception as stop_error:
                print(f"Erro ao parar captura após falha: {stop_error}")
            return
        
        # Agenda próximo frame (100ms para melhor estabilidade)
        if hasattr(self, 'live_capture') and self.live_capture:
            self.master.after(100, self.process_live_frame_inspection)
    
    def capture_test_from_webcam(self):
        """Captura instantânea da imagem mais recente da câmera para inspeção."""
        try:
            if not self.live_capture or self.latest_frame is None:
                # Fallback para captura única se não há captura contínua
                camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
                captured_image = capture_image_from_camera(camera_index)
            else:
                # Usa o frame mais recente da captura contínua
                captured_image = self.latest_frame.copy()
            
            if captured_image is not None:
                # Para de captura ao vivo se estiver ativa
                if hasattr(self, 'live_view') and self.live_view:
                    try:
                        self.stop_live_view()
                    except Exception:
                        pass
                
                # Carrega a imagem capturada
                self.img_test = captured_image
                
                # Limpa resultados de inspeção anteriores
                self.inspection_results = []
                
                # Atualiza estado dos botões
                self.update_button_states()
                
                camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
                self.status_var.set(f"Imagem capturada da câmera {camera_index}")
                
                # Salva a imagem no histórico de fotos
                self.save_to_photo_history(captured_image)
                
                # Exibe a imagem em tela cheia
                self.show_fullscreen_image()
                
                # Se houver múltiplos programas selecionados, capturar de todas as câmeras e executar sequencialmente
                if hasattr(self, 'selected_program_ids') and self.selected_program_ids:
                    self.capture_all_cameras_and_run_multi_inspection(self.selected_program_ids)
                else:
                    # Executa inspeção automática se modelo carregado
                    if hasattr(self, 'slots') and self.slots and hasattr(self, 'img_reference') and self.img_reference is not None:
                        self.run_inspection()
            else:
                self.status_var.set("Nenhuma imagem disponível para captura")
                
        except Exception as e:
            print(f"Erro ao capturar da webcam: {e}")
            self.status_var.set(f"Erro ao capturar da webcam: {str(e)}")
    
    def save_to_photo_history(self, image):
        """Salva a imagem capturada no histórico de fotos com otimização."""
        try:
            # Importar otimizador de imagens
            from modulos.image_optimizer import optimize_image_for_history
            
            # Cria o diretório de histórico se não existir
            historico_dir = MODEL_DIR / "historico_fotos"
            historico_dir.mkdir(exist_ok=True)
            
            # Cria diretório para capturas manuais
            capturas_dir = historico_dir / "Capturas"
            capturas_dir.mkdir(exist_ok=True)
            
            # Gera nome de arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Obtém o modelo atual se disponível para incluir no nome do arquivo
            model_name = "sem_modelo"
            model_id = getattr(self, 'current_model_id', '--')
            if model_id != '--' and hasattr(self, 'db_manager'):
                try:
                    model_info = self.db_manager.get_model_by_id(model_id)
                    if model_info and 'nome' in model_info:
                        model_name = model_info['nome']
                        # Substitui caracteres inválidos para nome de arquivo
                        model_name = model_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                except Exception as e:
                    print(f"Erro ao obter informações do modelo: {e}")
            
            file_name = f"foto_{model_name}_{timestamp}.png"
            file_path = capturas_dir / file_name
            
            # Salva a imagem com otimização (original + histórico + thumbnail)
            result = optimize_image_for_history(image, str(file_path))
            
            if result['success']:
                print(f"Foto salva no histórico com otimização:")
                print(f"  Original: {result['original']}")
                print(f"  Histórico: {result['history']}")
                print(f"  Thumbnail: {result['thumbnail']}")
            else:
                # Fallback para salvamento simples se a otimização falhar
                cv2.imwrite(str(file_path), image)
                print(f"Foto salva no histórico (fallback): {file_path}")
                
        except Exception as e:
            print(f"Erro ao salvar foto no histórico: {e}")
            # Fallback para salvamento simples em caso de erro
            try:
                cv2.imwrite(str(file_path), image)
                print(f"Foto salva no histórico (fallback após erro): {file_path}")
            except Exception as fallback_error:
                print(f"Erro no fallback: {fallback_error}")
    
    def save_inspection_result_to_history(self, status, passed, total):
        """Salva a imagem com os resultados da inspeção no histórico de fotos."""
        try:
            if self.img_test is None:
                return
                
            # Cria o diretório de histórico se não existir
            historico_dir = MODEL_DIR / "historico_fotos"
            historico_dir.mkdir(exist_ok=True)
            
            # Cria diretórios separados para OK e NG
            ok_dir = historico_dir / "OK"
            ng_dir = historico_dir / "NG"
            ok_dir.mkdir(exist_ok=True)
            ng_dir.mkdir(exist_ok=True)
            
            # Cria uma cópia da imagem para adicionar anotações
            img_result = self.img_test.copy()
            
            # Adiciona informações da inspeção na imagem
            # Obtém o modelo atual se disponível
            model_id = getattr(self, 'current_model_id', '--')
            model_name = "--"
            if hasattr(self, 'db_manager') and model_id != '--':
                try:
                    model_info = self.db_manager.get_model_by_id(model_id)
                    if model_info:
                        model_name = model_info['nome']
                except:
                    pass
            
            # Adiciona texto com informações da inspeção
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            cv2.putText(img_result, f"Data: {timestamp}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_result, f"Modelo: {model_name}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Cor baseada no resultado
            result_color = (0, 255, 0) if status == "APROVADO" else (0, 0, 255)
            cv2.putText(img_result, f"Resultado: {status}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
            cv2.putText(img_result, f"Slots OK: {passed}/{total}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Desenha os resultados dos slots na imagem
            for result in self.inspection_results:
                is_ok = result['passou']
                corners = result['corners']
                bbox = result['bbox']
                slot_id = result['slot_id']
                
                # Cores baseadas no resultado
                color = (0, 255, 0) if is_ok else (0, 0, 255)
                
                if corners is not None:
                    # Desenha polígono transformado
                    corners_array = np.array(corners, dtype=np.int32)
                    cv2.polylines(img_result, [corners_array], True, color, 2)
                    
                    # Adiciona texto com ID do slot e resultado
                    x, y = corners[0]
                    status_text = "OK" if is_ok else "NG"
                    cv2.putText(img_result, f"S{slot_id}: {status_text}", 
                               (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, color, 2)
                elif bbox != [0,0,0,0]:  # Fallback para bbox
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(img_result, (x, y), (x+w, y+h), color, 2)
                    # Texto de erro removido - apenas retângulo é exibido
            
            # Gera nome de arquivo com timestamp e resultado
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_tag = "OK" if status == "APROVADO" else "NG"
            file_name = f"inspecao_{timestamp}.png"
            
            # Seleciona o diretório correto com base no resultado
            target_dir = ok_dir if status == "APROVADO" else ng_dir
            
            # Obtém o modelo atual se disponível para incluir no nome do arquivo
            model_id = getattr(self, 'current_model_id', '--')
            if model_id != '--' and hasattr(self, 'db_manager'):
                try:
                    model_info = self.db_manager.get_model_by_id(model_id)
                    if model_info and 'nome' in model_info:
                        model_name = model_info['nome']
                        # Substitui caracteres inválidos para nome de arquivo
                        model_name = model_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                        file_name = f"inspecao_{model_name}_{timestamp}.png"
                except Exception as e:
                    print(f"Erro ao obter informações do modelo: {e}")
            
            file_path = target_dir / file_name
            
            # Salva a imagem com otimização
            try:
                from modulos.image_optimizer import optimize_image_for_history
                result = optimize_image_for_history(img_result, str(file_path))
                
                if result['success']:
                    print(f"Resultado de inspeção salvo no histórico com otimização:")
                    print(f"  Original: {result['original']}")
                    print(f"  Histórico: {result['history']}")
                    print(f"  Thumbnail: {result['thumbnail']}")
                else:
                    # Fallback para salvamento simples
                    cv2.imwrite(str(file_path), img_result)
                    print(f"Resultado de inspeção salvo no histórico (fallback): {file_path}")
            except Exception as opt_error:
                print(f"Erro na otimização, usando fallback: {opt_error}")
                cv2.imwrite(str(file_path), img_result)
                print(f"Resultado de inspeção salvo no histórico (fallback após erro): {file_path}")
        except Exception as e:
            print(f"Erro ao salvar resultado de inspeção no histórico: {e}")
    

    
    def stop_live_view(self):
        """Para a captura ao vivo."""
        try:
            # Verifica se o atributo live_view existe
            if hasattr(self, 'live_view'):
                self.live_view = False
            
            # Libera a câmera se existir
            if hasattr(self, 'camera') and self.camera is not None:
                try:
                    self.camera.release()
                    self.camera = None
                except Exception as release_error:
                    print(f"Erro ao liberar câmera no stop_live_view: {release_error}")
                    
        except Exception as e:
            print(f"Erro ao parar visualização ao vivo: {e}")
            # Não exibe messagebox para evitar interrupção da interface
    
    def process_live_frame(self):
        """Processa frame da câmera de forma otimizada"""
        try:
            # Verifica se os atributos necessários existem
            if not hasattr(self, 'live_view') or not hasattr(self, 'camera'):
                return
                
            if not self.live_view or not self.camera:
                return
            
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Atualiza a imagem de teste
                    if hasattr(self, 'img_test'):
                        self.img_test = frame
                    
                    # Atualiza o display
                    try:
                        self.update_display()
                    except Exception as display_error:
                        print(f"Erro ao atualizar display: {display_error}")
                    
                    # Inspeção automática otimizada (menos frequente)
                    # Pausa inspeções automáticas se uma imagem composta está sendo exibida
                    if not getattr(self, '_composite_active', False):
                        if hasattr(self, 'slots') and self.slots and hasattr(self, '_frame_count'):
                            self._frame_count += 1
                            # Executa inspeção a cada 5 frames para melhor performance
                            if self._frame_count % 5 == 0:
                                try:
                                    self.run_inspection(show_message=False)
                                except Exception as inspection_error:
                                    print(f"Erro durante inspeção automática: {inspection_error}")
                        elif hasattr(self, 'slots') and self.slots:
                            self._frame_count = 0
            except Exception as camera_error:
                print(f"Erro ao ler frame da câmera: {camera_error}")
            
            # Agenda próximo frame
            if hasattr(self, 'live_view') and self.live_view and hasattr(self, 'master'):
                try:
                    self.master.after(100, self.process_live_frame)
                except Exception as schedule_error:
                    print(f"Erro ao agendar próximo frame: {schedule_error}")
                    
        except Exception as e:
            print(f"Erro geral no processamento de frame: {e}")
            # Tenta agendar o próximo frame mesmo com erro para manter a continuidade
            if hasattr(self, 'master') and hasattr(self, 'live_view') and self.live_view:
                try:
                    self.master.after(100, self.process_live_frame)
                except Exception:
                    pass  # Ignora erro no agendamento de recuperação
    
    def update_display(self):
        """Atualiza exibição no canvas de forma otimizada"""
        try:
            # Se uma imagem composta está ativa, não atualiza para evitar sobreposição da base
            if getattr(self, '_composite_active', False):
                return
            
            # Verifica se os atributos necessários existem
            if not hasattr(self, 'img_test') or self.img_test is None:
                return
                
            if not hasattr(self, 'canvas'):
                print("Erro: Canvas não encontrado")
                return
            
            # === AJUSTE AUTOMÁTICO AO CANVAS ===
            try:
                # Obtém o tamanho atual do canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                # Se o canvas ainda não foi renderizado, use valores padrão
                if canvas_width <= 1 or canvas_height <= 1:
                    canvas_width = 640
                    canvas_height = 480
            except Exception as canvas_error:
                print(f"Erro ao obter dimensões do canvas: {canvas_error}")
                canvas_width = 640
                canvas_height = 480
            
            # Converte a imagem para o tamanho do canvas
            try:
                self.img_display, self.scale_factor = cv2_to_tk(self.img_test, max_w=canvas_width, max_h=canvas_height)
            except Exception as convert_error:
                print(f"Erro ao converter imagem para exibição: {convert_error}")
                return
            
            if self.img_display is None:
                return
            
            # === ATUALIZAÇÃO EFICIENTE DO CANVAS ===
            try:
                # Remove apenas overlays, mantém imagem base quando possível
                self.canvas.delete("result_overlay")
                self.canvas.delete("inspection")
                
                # Calcula dimensões da imagem redimensionada e offsets para centralização
                img_height, img_width = self.img_test.shape[:2]
                new_width = int(img_width * self.scale_factor)
                new_height = int(img_height * self.scale_factor)
                self.x_offset = max(0, (self.canvas.winfo_width() - new_width) // 2)
                self.y_offset = max(0, (self.canvas.winfo_height() - new_height) // 2)
                
                # Cria ou atualiza imagem
                if not hasattr(self, '_canvas_image_id') or self._canvas_image_id is None:
                    self._canvas_image_id = self.canvas.create_image(self.x_offset, self.y_offset, anchor=NW, image=self.img_display)
                else:
                    try:
                        self.canvas.itemconfig(self._canvas_image_id, image=self.img_display)
                        self.canvas.coords(self._canvas_image_id, self.x_offset, self.y_offset)
                    except Exception:
                        # Se falhar, cria nova imagem
                        self._canvas_image_id = self.canvas.create_image(self.x_offset, self.y_offset, anchor=NW, image=self.img_display)
            except Exception as canvas_update_error:
                print(f"Erro ao atualizar canvas: {canvas_update_error}")
                return
            
            # Desenha resultados se disponíveis
            if hasattr(self, 'inspection_results') and self.inspection_results:
                try:
                    self.draw_inspection_results()
                except Exception as draw_error:
                    print(f"Erro ao desenhar resultados de inspeção: {draw_error}")
            
            # Atualiza scroll region apenas se necessário
            try:
                bbox = self.canvas.bbox("all")
            except Exception as bbox_error:
                print(f"Erro ao obter bbox do canvas: {bbox_error}")
                return
                if bbox != self.canvas.cget("scrollregion"):
                    try:
                        self.canvas.configure(scrollregion=bbox)
                    except Exception as scroll_error:
                        print(f"Erro ao configurar região de scroll: {scroll_error}")
        except Exception as e:
            print(f"Erro geral ao atualizar display: {e}")
    
    def run_inspection(self, show_message=False):
        """Executa inspeção otimizada com estilo industrial Keyence"""
        try:
            # === ATUALIZAÇÃO DE STATUS ===
            try:
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("PROCESSANDO...")
                    if hasattr(self, 'update_idletasks'):
                        self.update_idletasks()  # Força atualização da UI
            except Exception as e:
                print(f"Erro ao atualizar status: {e}")
            
            # === VALIDAÇÃO INICIAL ===
            if not hasattr(self, 'slots') or not self.slots or \
               not hasattr(self, 'img_reference') or self.img_reference is None or \
               not hasattr(self, 'img_test') or self.img_test is None:
                if hasattr(self, 'status_var'):
                    self.status_var.set("Carregue o modelo de referência E a imagem de teste antes de inspecionar")
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("ERRO")
                return
            
            print("--- Iniciando Inspeção Keyence ---")
            
            # Limpa resultados anteriores
            if hasattr(self, 'canvas'):
                try:
                    self.canvas.delete("result_overlay")
                except Exception as canvas_error:
                    print(f"Erro ao limpar canvas: {canvas_error}")
            
            # === 1. ALINHAMENTO DE IMAGEM ===
            try:
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("ALINHANDO...")
                if hasattr(self, 'update_idletasks'):
                    self.update_idletasks()  # Força atualização da UI
                M, _, align_error = find_image_transform(self.img_reference, self.img_test)
            except Exception as e:
                print(f"Erro durante alinhamento: {e}")
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("ERRO")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Erro durante alinhamento: {e}")
                return
            
            if M is None:
                print(f"FALHA no Alinhamento: {align_error}")
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("FALHA DE ALINHAMENTO")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Falha no Alinhamento: Não foi possível alinhar as imagens. Erro: {align_error}")
                
                # Desenha slots de referência em cor de erro (estilo Keyence)
                if hasattr(self, 'canvas') and hasattr(self, 'scale_factor'):
                    try:
                        for slot in self.slots:
                            xr, yr, wr, hr = slot['x'], slot['y'], slot['w'], slot['h']
                            xa, ya = xr * self.scale_factor + self.x_offset, yr * self.scale_factor + self.y_offset
                            wa, ha = wr * self.scale_factor, hr * self.scale_factor
                            self.canvas.create_rectangle(xa, ya, xa+wa, ya+ha, outline=get_color('colors.inspection_colors.align_fail_color'), width=2, tags="result_overlay")
                            # Carrega as configurações de estilo
                            try:
                                style_config = load_style_config()
                                self.canvas.create_text(xa + wa/2, ya + ha/2, text=f"S{slot['id']}\nFAIL", fill=get_color('colors.inspection_colors.align_fail_color'), font=style_config["ng_font"], tags="result_overlay", justify="center")
                            except Exception as style_error:
                                print(f"Erro ao carregar configurações de estilo: {style_error}")
                                # Fallback para fonte padrão
                                self.canvas.create_text(xa + wa/2, ya + ha/2, text=f"S{slot['id']}\nFAIL", fill=get_color('colors.inspection_colors.align_fail_color'), tags="result_overlay", justify="center")
                    except Exception as draw_error:
                        print(f"Erro ao desenhar slots de referência: {draw_error}")
                return
            
            # === 2. VERIFICAÇÃO DOS SLOTS (ESTILO KEYENCE) ===
            try:
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("INSPECIONANDO...")
                if hasattr(self, 'update_idletasks'):
                    self.update_idletasks()  # Força atualização da UI
                
                overall_ok = True
                self.inspection_results = []
                failed_slots = []  # Para log otimizado
                
                # Resetar o label grande de resultado
                if hasattr(self, 'result_display_label'):
                    self.result_display_label.config(
                        text="--",
                        foreground=get_color('colors.status_colors.muted_text'),
                        background=get_color('colors.status_colors.muted_bg')
                    )
                
                # Adicionar modelo_id aos resultados se disponível
                model_id = getattr(self, 'current_model_id', '--')
                
                for i, slot in enumerate(self.slots):
                    # Atualizar status com progresso
                    progress = f"SLOT {i+1}/{len(self.slots)}"
                    if hasattr(self, 'inspection_status_var'):
                        self.inspection_status_var.set(progress)
                    if hasattr(self, 'update_idletasks'):
                        self.update_idletasks()  # Força atualização da UI
                    
                    try:
                        # Processamento otimizado sem logs excessivos
                        is_ok, correlation, pixels, corners, bbox, log_msgs = check_slot(self.img_test, slot, M)
                        
                        # Log apenas para falhas (reduz overhead)
                        if not is_ok:
                            failed_slots.append(f"S{slot['id']}({slot['tipo']})")
                            for msg in log_msgs:
                                print(f"  -> {msg}")
                        
                        # Armazena resultado otimizado com estilo Keyence
                        result = {
                            'slot_id': slot['id'],
                            'passou': is_ok,
                            'score': correlation,
                            'detalhes': f"Score: {correlation:.3f}, Pixels: {pixels}",
                            'slot_data': slot,
                            'corners': corners,
                            'bbox': bbox,
                            'model_id': model_id
                        }
                        self.inspection_results.append(result)
                        
                        if not is_ok:
                            overall_ok = False
                    except Exception as slot_error:
                        print(f"Erro ao processar slot {slot['id']}: {slot_error}")
                        # Continua com o próximo slot em caso de erro
            except Exception as e:
                print(f"Erro durante inspeção: {e}")
                if hasattr(self, 'inspection_status_var'):
                    self.inspection_status_var.set("ERRO")
                if hasattr(self, 'status_var'):
                    self.status_var.set(f"Erro durante inspeção: {e}")
                return
            
            # === 3. DESENHO OTIMIZADO NO CANVAS COM ESTILO KEYENCE ===
            try:
                if not hasattr(self, 'canvas') or not hasattr(self, 'inspection_results') or not hasattr(self, 'scale_factor'):
                    print("Atributos necessários para desenho não estão disponíveis")
                    return
                    
                for result in self.inspection_results:
                    try:
                        is_ok = result.get('passou', False)
                        corners = result.get('corners', None)
                        bbox = result.get('bbox', [0,0,0,0])
                        slot_id = result.get('slot_id', '?')
                        
                        # Cores no estilo Keyence
                        fill_color = get_color('colors.inspection_colors.pass_color') if is_ok else get_color('colors.inspection_colors.fail_color')
                        
                        if corners is not None:
                            try:
                                # Conversão otimizada de coordenadas
                                canvas_corners = [(int(pt[0] * self.scale_factor) + self.x_offset, int(pt[1] * self.scale_factor) + self.y_offset) for pt in corners]
                                
                                # Desenha polígono transformado estilo Keyence
                                self.canvas.create_polygon(canvas_corners, outline=fill_color, fill="", width=2, tags="result_overlay")
                                
                                # Adiciona um pequeno retângulo de status no canto estilo Keyence
                                status_x, status_y = canvas_corners[0][0], canvas_corners[0][1] - 20
                                self.canvas.create_rectangle(status_x, status_y, status_x + 40, status_y + 16, 
                                                           fill=fill_color, outline="", tags="result_overlay")
                                
                                # Label otimizado estilo Keyence
                                try:
                                    # Carrega as configurações de estilo
                                    style_config = load_style_config()
                                    self.canvas.create_text(status_x + 20, status_y + 8,
                                                          text=f"S{slot_id}", fill=get_color('colors.special_colors.white_text'), anchor="center", tags="result_overlay",
                                                          font=style_config["ok_font"])
                                    
                                    # Adiciona indicador de status
                                    status_text = "OK" if is_ok else "NG"
                                    # Escolhe a fonte baseada no resultado
                                    font_str = style_config["ok_font"] if is_ok else style_config["ng_font"]
                                    self.canvas.create_text(canvas_corners[0][0] + 60, canvas_corners[0][1] - 12,
                                                          text=status_text, fill=fill_color, anchor="nw", tags="result_overlay",
                                                          font=font_str)
                                except Exception as style_error:
                                    print(f"Erro ao carregar estilo ou criar texto: {style_error}")
                                    # Fallback para texto simples sem estilo
                                    self.canvas.create_text(status_x + 20, status_y + 8,
                                                          text=f"S{slot_id}", fill=get_color('colors.special_colors.white_text'), anchor="center", tags="result_overlay")
                                    self.canvas.create_text(canvas_corners[0][0] + 60, canvas_corners[0][1] - 12,
                                                          text="OK" if is_ok else "NG", fill=fill_color, anchor="nw", tags="result_overlay")
                            except Exception as corner_error:
                                print(f"Erro ao processar corners para slot {slot_id}: {corner_error}")
                        
                        if bbox != [0,0,0,0]:  # Fallback para bbox com estilo Keyence
                                try:
                                    xa, ya = bbox[0] * self.scale_factor, bbox[1] * self.scale_factor
                                    wa, ha = bbox[2] * self.scale_factor, bbox[3] * self.scale_factor
                                    # Linha pontilhada de erro removida conforme solicitado pelo usuário
                                    # Indicador de erro estilo Keyence removido conforme solicitado pelo usuário.
                                except Exception as bbox_error:
                                    print(f"Erro ao processar bbox para slot {slot_id}: {bbox_error}")
                    except Exception as result_error:
                        print(f"Erro ao processar resultado de inspeção: {result_error}")
                        continue  # Continua com o próximo resultado
            except Exception as draw_error:
                print(f"Erro ao desenhar resultados no canvas: {draw_error}")
            # Continua com o processamento para atualizar o status
            
            # === 4. RESULTADO FINAL ESTILO KEYENCE ===
            try:
                if not hasattr(self, 'inspection_results'):
                    print("Resultados de inspeção não disponíveis")
                    return
                    
                total = len(self.inspection_results)
                passed = sum(1 for r in self.inspection_results if r.get('passou', False))
                
                failed = total - passed
            except Exception as count_error:
                print(f"Erro ao contar resultados de inspeção: {count_error}")
                return
            final_status = "APROVADO" if overall_ok else "REPROVADO"
            
            # Atualizar status de inspeção
            if hasattr(self, 'inspection_status_var'):
                try:
                    if overall_ok:
                        self.inspection_status_var.set("OK")
                    else:
                        self.inspection_status_var.set("NG")
                except Exception as status_error:
                    print(f"Erro ao atualizar status de inspeção: {status_error}")
            
            # Log otimizado estilo Keyence
            if failed_slots:
                print(f"Falhas detectadas em: {', '.join(failed_slots)}")
            print(f"--- Inspeção Keyence Concluída: {final_status} ({passed}/{total}) ---")
            
            # Atualiza interface com estilo industrial Keyence
            if hasattr(self, 'update_results_list') and callable(self.update_results_list):
                try:
                    self.update_results_list()
                except Exception as update_error:
                    print(f"Erro ao atualizar lista de resultados: {update_error}")
            
            # Salva a imagem com os resultados da inspeção no histórico
            if hasattr(self, 'save_inspection_result_to_history') and callable(self.save_inspection_result_to_history):
                try:
                    self.save_inspection_result_to_history(final_status, passed, total)
                except Exception as save_error:
                    print(f"Erro ao salvar resultado no histórico: {save_error}")
            
            # Status com estilo industrial Keyence
            status_text = f"INSPEÇÃO: {final_status} - {passed}/{total} SLOTS OK, {failed} FALHAS"
            if hasattr(self, 'status_var'):
                try:
                    self.status_var.set(status_text)
                except Exception as status_var_error:
                    print(f"Erro ao atualizar texto de status: {status_var_error}")
            
            # Atualiza cor da barra de status baseado no resultado estilo Keyence
            try:
                # Armazenamos uma referência direta ao status_bar durante a criação
                if hasattr(self, 'status_bar'):
                    if overall_ok:
                        self.status_bar.config(background=get_color('colors.status_colors.success_bg'), foreground=get_color('colors.text_color'))
                    else:
                        self.status_bar.config(background=get_color('colors.status_colors.error_bg'), foreground=get_color('colors.text_color'))
                        
                # Atualizar cor do indicador de status de inspeção usando referência direta
                if hasattr(self, 'inspection_status_label'):
                    if overall_ok:
                        self.inspection_status_label.config(foreground=get_color('colors.status_colors.success_bg'))
                    else:
                        self.inspection_status_label.config(foreground=get_color('colors.status_colors.error_bg'))
            except Exception as e:
                print(f"Erro ao atualizar status_bar: {e}")
            
            # Não exibimos mais mensagens, apenas atualizamos o status
            # O status já foi atualizado acima com o texto: f"INSPEÇÃO: {final_status} - {passed}/{total} SLOTS OK, {failed} FALHAS"
        except Exception as final_error:
            print(f"Erro ao processar resultado final: {final_error}")
    
    def create_status_summary_panel(self, parent_frame=None):
        """Cria o painel de resumo de status estilo Keyence IV3"""
        # Se um frame pai for fornecido, criar um painel de resumo geral
        if parent_frame:
            # Frame para o painel de status geral
            status_panel = ttk.Frame(parent_frame, relief="raised", borderwidth=2)
            status_panel.pack(fill=X, pady=5)
            
            # Linha 1: Status geral
            status_row = ttk.Frame(status_panel)
            status_row.pack(fill=X, pady=2)
            
            # Carrega as configurações de estilo
            style_config = load_style_config()
            
            ttk.Label(status_row, text="STATUS:", font=style_config["ok_font"]).pack(side=LEFT, padx=(5, 5))
            
            # Label para status (OK/NG) com estilo industrial Keyence
            self.status_label = ttk.Label(status_row, text="--", font=style_config["ok_font"], 
                                        background=get_color('colors.inspection_colors.pass_bg'), foreground=get_color('colors.special_colors.white_text'), 
                                        width=6, anchor="center", padding=3)
            self.status_label.pack(side=LEFT, padx=5)
            
            # Linha 2: Score e ID
            details_row = ttk.Frame(status_panel)
            details_row.pack(fill=X, pady=2)
            
            # Usa as configurações de estilo já carregadas
            ttk.Label(details_row, text="SCORE:", font=style_config["ok_font"]).pack(side=LEFT, padx=(5, 5))
            
            # Label para score com estilo industrial Keyence
            self.score_label = ttk.Label(details_row, text="--", font=style_config["ok_font"], 
                                       background=get_color('colors.inspection_colors.pass_bg'), foreground=get_color('colors.special_colors.white_text'), 
                                       width=8, anchor="center", padding=3)
            self.score_label.pack(side=LEFT, padx=5)
            
            ttk.Label(details_row, text="ID:", font=style_config["ok_font"]).pack(side=LEFT, padx=(10, 5))
            
            # Label para ID do modelo com estilo industrial Keyence
            self.id_label = ttk.Label(details_row, text="--", font=style_config["ok_font"], 
                                    background=get_color('colors.inspection_colors.pass_bg'), foreground=get_color('colors.special_colors.white_text'), 
                                    anchor="center", padding=3)
            self.id_label.pack(side=LEFT, padx=5, fill=X, expand=True)
            return
        
        # Caso contrário, estamos criando o painel principal de status
        # Primeiro, limpe qualquer widget existente no status_grid_frame
        for widget in self.status_grid_frame.winfo_children():
            widget.destroy()
            
        # Vamos adicionar um cabeçalho mais proeminente
        header_frame = ttk.Frame(self.status_grid_frame)
        header_frame.pack(fill=X, pady=(0, 10))
        

        
        # Caso contrário, criar o painel de resumo de slots
        # Limpar widgets existentes
        for widget in self.status_widgets.values():
            if hasattr(widget, 'frame'):
                widget['frame'].destroy()
        self.status_widgets.clear()
        
        if not self.slots:
            return
        
        # Criar um frame para conter os slots usando pack em vez de grid
        slots_container = ttk.Frame(self.status_grid_frame)
        slots_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Calcular layout (máximo 6 colunas)
        num_slots = len(self.slots)
        cols = min(6, num_slots)
        
        # Criar frames para cada coluna
        column_frames = []
        for i in range(cols):
            col_frame = ttk.Frame(slots_container)
            col_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=2)
            column_frames.append(col_frame)
        
        # Distribuir slots pelas colunas
        for i, slot in enumerate(self.slots):
            col_idx = i % cols
            
            # Frame para cada slot com estilo industrial
            slot_frame = ttk.Frame(column_frames[col_idx], relief="raised", borderwidth=2)
            slot_frame.pack(fill=X, pady=3, padx=2)
            
            # Label do ID do slot com estilo industrial Keyence
            id_label = ttk.Label(slot_frame, text=f"SLOT {slot['id']}", 
                                font=get_font('small_font'), background=get_color('colors.special_colors.black_bg'), foreground=get_color('colors.special_colors.white_text'))
            id_label.pack(pady=2, fill=X)
            
            # Label do status (OK/NG) com estilo industrial Keyence
            status_label = ttk.Label(slot_frame, text="---", 
                                   font=get_font('header_font'),
                                   foreground=get_color('colors.status_colors.inactive_text'),
                                   background=get_color('colors.status_colors.muted_bg'),
                                   anchor="center")
            status_label.pack(pady=2, fill=X)
            
            # Label do score com estilo industrial Keyence
            score_label = ttk.Label(slot_frame, text="", 
                                  font=get_font('small_font'),
                                  background=get_color('colors.special_colors.black_bg'),
                                  foreground=get_color('colors.status_colors.muted_text'))
            score_label.pack(pady=1, fill=X)
            
            # Armazenar referências
            self.status_widgets[slot['id']] = {
                'frame': slot_frame,
                'id_label': id_label,
                'status_label': status_label,
                'score_label': score_label
            }
    
    def update_status_summary_panel(self):
        """Atualiza o painel de resumo com os resultados da inspeção no estilo industrial"""
        if not hasattr(self, 'status_widgets') or not self.status_widgets:
            return
        
        # Resetar todos os status com estilo industrial
        for slot_id, widgets in self.status_widgets.items():
            if all(key in widgets for key in ['status_label', 'score_label', 'frame']):
                try:
                    widgets['status_label'].config(text="---", foreground=get_color('colors.status_colors.inactive_text'), background=get_color('colors.status_colors.muted_bg'))
                    widgets['score_label'].config(text="---", background=get_color('colors.special_colors.black_bg'), foreground=get_color('colors.status_colors.muted_text'))
                    widgets['frame'].config(relief="raised", borderwidth=2, padding=2)
                except Exception as e:
                    print(f"Erro ao resetar widget do slot {slot_id}: {e}")
        
        # Verificar se temos resultados de inspeção
        if not hasattr(self, 'inspection_results') or not self.inspection_results:
            return
            
        # Atualizar com resultados da inspeção usando estilo industrial
        for result in self.inspection_results:
            slot_id = result['slot_id']
            if slot_id in self.status_widgets:
                widgets = self.status_widgets[slot_id]
                
                # Carrega as configurações de estilo
                style_config = load_style_config()
                
                try:
                    if result['passou']:
                        # Estilo industrial para OK (cor personalizada)
                        widgets['status_label'].config(text="OK", foreground=get_color('colors.special_colors.white_text'), background=get_color('colors.ok_color', style_config))
                        widgets['frame'].config(relief="raised", borderwidth=3)
                        widgets['id_label'].config(background=get_color('colors.inspection_colors.ok_detail_bg'), foreground=get_color('colors.special_colors.white_text'))
                    else:
                        # Estilo industrial para NG (cor personalizada)
                        widgets['status_label'].config(text="NG", foreground=get_color('colors.special_colors.white_text'), background=get_color('colors.ng_color', style_config))
                        widgets['frame'].config(relief="raised", borderwidth=3)
                        widgets['id_label'].config(background=get_color('colors.inspection_colors.ng_detail_bg'), foreground=get_color('colors.special_colors.white_text'))
                    
                    # Atualizar score com estilo industrial
                    score_text = f"{result['score']:.3f}"
                    if result['passou']:
                        widgets['score_label'].config(text=score_text, background=get_color('colors.inspection_colors.ok_detail_bg'), foreground=get_color('colors.special_colors.white_text'))
                    else:
                        widgets['score_label'].config(text=score_text, background=get_color('colors.inspection_colors.ng_detail_bg'), foreground=get_color('colors.special_colors.white_text'))
                except Exception as e:
                    print(f"Erro ao atualizar widget do slot {slot_id}: {e}")
    
    def update_results_list(self):
        """Atualiza lista de resultados com estilo industrial Keyence"""
        # === LIMPEZA OTIMIZADA ===
        children = self.results_listbox.get_children()
        if children:
            self.results_listbox.delete(*children)  # Mais eficiente que loop
        
        # === CONFIGURAÇÃO DE TAGS ESTILO KEYENCE ===
        # Carrega as configurações de estilo (uma única vez)
        style_config = load_style_config()
        
        # Estilo OK - cor personalizada
        self.results_listbox.tag_configure("pass", 
                                         foreground=get_color('colors.special_colors.white_text'), 
                                         background=get_color('colors.ok_color', style_config), 
                                         font=style_config["ok_font"])
        
        # Estilo NG - cor personalizada
        self.results_listbox.tag_configure("fail", 
                                         foreground=get_color('colors.special_colors.white_text'), 
                                         background=get_color('colors.ng_color', style_config), 
                                         font=style_config["ng_font"])
        
        # Estilo cabeçalho - cinza industrial Keyence
        self.results_listbox.tag_configure("header", 
                                         foreground=get_color('colors.special_colors.white_text'), 
                                         background=get_color('colors.inspection_colors.pass_bg'), 
                                         font=style_config["ok_font"])
        
        # === VARIÁVEIS PARA RESUMO GERAL ===
        total_slots = len(self.inspection_results) if self.inspection_results else 0
        passed_slots = 0
        total_score = 0
        model_id = "--"
        
        # === INSERÇÃO OTIMIZADA COM ESTILO INDUSTRIAL KEYENCE ===
        for result in self.inspection_results:
            status = "OK" if result['passou'] else "NG"
            score_text = f"{result['score']:.3f}"
            tags = ("pass",) if result['passou'] else ("fail",)
            
            # Atualizar contadores para resumo
            if result['passou']:
                passed_slots += 1
            total_score += result['score']
            
            # Obter ID do modelo se disponível
            if 'model_id' in result and model_id == "--":
                model_id = result['model_id']
            
            # Detalhes formatados para estilo industrial Keyence
            detalhes = result['detalhes'].upper() if result['passou'] else f"⚠ {result['detalhes'].upper()}"
            
            self.results_listbox.insert("", "end",
                                       text=result['slot_id'],
                                       values=(status, score_text, detalhes),
                                       tags=tags)
        
        # Atualizar painel de resumo de status detalhado
        self.update_status_summary_panel()
        
        # Calcular status geral no estilo Keyence (uma única vez)
        overall_status = "OK" if total_slots > 0 and passed_slots == total_slots else "NG"
        
        # Atualizar painel de resumo geral se existir
        if hasattr(self, 'status_label') and hasattr(self, 'score_label') and hasattr(self, 'id_label'):
            if total_slots > 0:
                total_score / total_slots
                
                # Atualizar labels com estilo Keyence
                self.status_label.config(
                    text=overall_status,
                    background=get_color('colors.status_colors.success_bg') if overall_status == "OK" else get_color('colors.status_colors.error_bg'),
                    foreground="#FFFFFF"
                )
                
                self.score_label.config(
                    text=f"{passed_slots}/{total_slots}",
                    background=get_color('colors.status_colors.success_bg') if passed_slots == total_slots else get_color('colors.status_colors.error_bg'),
                    foreground="#FFFFFF"
                )
                
                self.id_label.config(text=model_id)
        
        # Atualizar o label grande de resultado NG/OK
        if hasattr(self, 'result_display_label'):
            if total_slots > 0:
                
                # Carrega as configurações de estilo
                style_config = load_style_config()
                
                if overall_status == "OK":
                    self.result_display_label.config(
                        text="OK",
                        foreground="#FFFFFF",
                        background=get_color('colors.ok_color', style_config)
                    )
                else:
                    self.result_display_label.config(
                        text="NG",
                        foreground="#FFFFFF",
                        background=get_color('colors.ng_color', style_config)
                    )
            else:
                # Resetar para estado inicial quando não há resultados
                self.result_display_label.config(
                    text="--",
                    foreground=get_color('colors.status_colors.muted_text'),
                    background=get_color('colors.status_colors.muted_bg')
                )
    
    def draw_inspection_results(self):
        """Desenha resultados da inspeção no canvas com estilo industrial."""
        if not self.inspection_results:
            return
        
        # Garante que scale_factor e offsets estão atualizados
        if not hasattr(self, 'scale_factor') or not hasattr(self, 'x_offset') or not hasattr(self, 'y_offset'):
            # Recalcula scale_factor e offsets se necessário
            if hasattr(self, 'img_test') and self.img_test is not None:
                try:
                    canvas_width = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 640
                    canvas_height = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else 480
                    
                    img_height, img_width = self.img_test.shape[:2]
                    scale_x = canvas_width / img_width
                    scale_y = canvas_height / img_height
                    self.scale_factor = min(scale_x, scale_y)
                    
                    new_width = int(img_width * self.scale_factor)
                    new_height = int(img_height * self.scale_factor)
                    self.x_offset = max(0, (canvas_width - new_width) // 2)
                    self.y_offset = max(0, (canvas_height - new_height) // 2)
                except Exception as e:
                    print(f"Erro ao recalcular scale_factor: {e}")
                    return
            else:
                return
        
        for result in self.inspection_results:
            slot = result['slot_data']
            
            # Converte coordenadas da imagem para canvas (incluindo offsets)
            x1 = int(slot['x'] * self.scale_factor) + self.x_offset
            y1 = int(slot['y'] * self.scale_factor) + self.y_offset
            x2 = int((slot['x'] + slot['w']) * self.scale_factor) + self.x_offset
            y2 = int((slot['y'] + slot['h']) * self.scale_factor) + self.y_offset
            
            # Carrega as configurações de estilo (uma única vez)
            style_config = load_style_config()
            
            # Cores estilo industrial
            if result['passou']:
                outline_color = get_color('colors.ok_color', style_config)  # Cor de OK personalizada
                fill_color = get_color('colors.ok_color', style_config)     # Mesma cor para o fundo
                text_color = get_color('colors.special_colors.white_text')                    # Texto branco
            else:
                outline_color = get_color('colors.ng_color', style_config)  # Cor de NG personalizada
                fill_color = get_color('colors.ng_color', style_config)     # Mesma cor para o fundo
                text_color = get_color('colors.special_colors.white_text')                    # Texto branco
            
            # Desenha retângulo com estilo industrial
            self.canvas.create_rectangle(x1, y1, x2, y2,
                                       outline=outline_color, width=3, 
                                       dash=(3, 2) if not result['passou'] else None,
                                       tags="inspection")
            
            # Cria fundo para o texto (estilo industrial)
            text_bg_width = 60
            text_bg_height = 20
            self.canvas.create_rectangle(x1, y1, x1 + text_bg_width, y1 + text_bg_height,
                                       fill=fill_color, outline=outline_color, width=1,
                                       tags="inspection")
            
            # Adiciona texto com resultado estilo industrial
            status_text = "OK" if result['passou'] else "NG"
            
            # Escolhe a fonte baseada no resultado
            font_str = style_config["ok_font"] if result['passou'] else style_config["ng_font"]
            
            self.canvas.create_text(x1 + text_bg_width/2, y1 + text_bg_height/2,
                                  text=f"S{slot['id']}: {status_text}",
                                  fill=text_color, font=font_str,
                                  anchor="center", tags="inspection")
            
            # Adiciona score em outra posição
            score_text = f"{result['score']:.2f}"
            # Escolhe a fonte baseada no resultado (já temos style_config carregado)
            font_str = style_config["ok_font"] if result['passou'] else style_config["ng_font"]
            self.canvas.create_text(x2 - 5, y2 - 5,
                                  text=score_text,
                                  fill=outline_color, font=font_str,
                                  anchor="se", tags="inspection")
    
    def update_button_states(self):
        """Atualiza estado dos botões baseado no estado atual."""
        len(self.slots) > 0
        self.img_test is not None
        
        # Botões que dependem de modelo e imagem de teste
        # Nota: btn_inspect foi removido, então não precisamos mais atualizar seu estado
    
    def start_background_frame_capture(self):
        """Inicia a captura contínua de frames em segundo plano."""
        def capture_frames():
            while self.live_capture and self.camera and self.camera.isOpened():
                try:
                    ret, frame = self.camera.read()
                    if ret:
                        self.latest_frame = frame.copy()
                    time.sleep(0.033)  # ~30 FPS
                except Exception as e:
                    print(f"Erro na captura de frame: {e}")
                    break
        
        import threading
        self.capture_thread = threading.Thread(target=capture_frames, daemon=True)
        self.capture_thread.start()
    
    def on_closing_inspection(self):
        """Limpa recursos ao fechar a aplicação de inspeção."""
        if self.live_capture:
            self.stop_live_capture_inspection()
        if self.live_view:
            self.stop_live_view()
        self.master.destroy()
    
    def cleanup_on_exit(self):
        """Método para limpeza de recursos ao sair da aplicação"""
        try:
            # Encerra sistema dual se estiver ativo
            if hasattr(self, 'dual_manager') and self.dual_manager:
                stop_dual_cameras()
                print("Sistema dual de câmeras encerrado com sucesso.")
            
            # Encerra pool persistente
            self.shutdown_persistent_pool()
            print("Pool persistente de câmeras encerrado com sucesso.")
        except Exception as e:
            print(f"Erro ao encerrar recursos: {e}")


# Função create_main_window() removida - agora centralizada em montagem.py
# Esta função foi consolidada e melhorada no módulo principal montagem.py


class DualInspectionDialog:
    """Diálogo simples para seleção de dois modelos para inspeção dual."""
    def __init__(self, parent, db_manager):
        self.parent = parent  # Instância de InspecaoWindow
        self.db_manager = db_manager
        self.top = None
        self.result = None
        self.model_list = []
        self.display_to_id = {}

    def _build_ui(self):
        self.top = Toplevel(self.parent.master)
        self.top.title("Inspeção com 2 Programas")
        self.top.transient(self.parent.master)
        self.top.grab_set()

        # Dimensões básicas
        self.top.geometry("420x220")

        # Container principal
        container = ttk.Frame(self.top, padding=10)
        container.pack(fill=BOTH, expand=True)

        # Título
        title = ttk.Label(container, text="Selecione os 2 programas (modelos) para inspeção sequencial:", font=("Segoe UI", 11, "bold"))
        title.pack(anchor="w", pady=(0, 10))

        # Buscar modelos
        try:
            self.model_list = self.db_manager.list_modelos() or []
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao listar modelos: {e}")
            self.model_list = []

        # Preparar valores de exibição
        values = []
        self.display_to_id.clear()
        for m in self.model_list:
            mid = m.get('id')
            nome = m.get('nome', f"Modelo {mid}")
            cam = m.get('camera_index', '-')
            display = f"{mid} - {nome} (Cam {cam})"
            values.append(display)
            self.display_to_id[display] = mid

        # Linha 1 - Programa 1
        row1 = ttk.Frame(container)
        row1.pack(fill=X, pady=5)
        ttk.Label(row1, text="Programa 1:").pack(side=LEFT)
        self.combo1 = Combobox(row1, values=values, state="readonly")
        self.combo1.pack(side=LEFT, fill=X, expand=True, padx=(10, 0))

        # Linha 2 - Programa 2
        row2 = ttk.Frame(container)
        row2.pack(fill=X, pady=5)
        ttk.Label(row2, text="Programa 2:").pack(side=LEFT)
        self.combo2 = Combobox(row2, values=values, state="readonly")
        self.combo2.pack(side=LEFT, fill=X, expand=True, padx=(10, 0))

        # Pré-selecionar modelo atual, se disponível
        try:
            current_id = getattr(self.parent, 'current_model_id', None)
            if current_id is not None:
                for disp, mid in self.display_to_id.items():
                    if mid == current_id:
                        self.combo1.set(disp)
                        break
        except Exception:
            pass

        # Botões de ação
        btns = ttk.Frame(container)
        btns.pack(fill=X, pady=(15, 0))
        ttk.Button(btns, text="Cancelar", bootstyle="secondary", command=self._on_cancel).pack(side=RIGHT)
        ttk.Button(btns, text="OK", bootstyle="success", command=self._on_ok).pack(side=RIGHT, padx=(0, 8))

        # Manter centralizado em relação à janela pai
        self._center_on_parent()
        self.top.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _center_on_parent(self):
        try:
            self.top.update_idletasks()
            px = self.parent.master.winfo_rootx()
            py = self.parent.master.winfo_rooty()
            pw = self.parent.master.winfo_width()
            ph = self.parent.master.winfo_height()
            tw = self.top.winfo_width()
            th = self.top.winfo_height()
            x = px + (pw - tw) // 2
            y = py + (ph - th) // 2
            self.top.geometry(f"{tw}x{th}+{x}+{y}")
        except Exception:
            pass

    def _on_ok(self):
        sel1 = self.combo1.get()
        sel2 = self.combo2.get()
        if not sel1 or not sel2:
            messagebox.showwarning("Atenção", "Selecione os dois programas.")
            return
        if sel1 == sel2:
            messagebox.showwarning("Atenção", "Os dois programas devem ser diferentes.")
            return
        try:
            model1_id = self.display_to_id.get(sel1)
            model2_id = self.display_to_id.get(sel2)
            if model1_id is None or model2_id is None:
                raise ValueError("Seleção inválida de modelo")
            self.result = {
                'model1_id': model1_id,
                'model2_id': model2_id,
            }
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao validar seleção: {e}")
            return
        self.top.destroy()

    def _on_cancel(self):
        self.result = None
        self.top.destroy()

    def show(self):
        """Exibe o diálogo de forma modal e retorna o resultado."""
        self._build_ui()
        self.top.wait_window()
        return self.result

class MultiProgramDialog:
    """Diálogo para seleção dinâmica de múltiplos programas, com botão + para adicionar slots."""
    def __init__(self, parent, db_manager, preselected_ids=None):
        self.parent = parent
        self.db_manager = db_manager
        self.top = None
        self.result = None
        self.model_list = []
        self.display_to_id = {}
        self.slot_rows = []  # cada item: (frame, label, combobox)
        self.preselected_ids = preselected_ids or []

    def _build_ui(self):
        self.top = Toplevel(self.parent.master)
        self.top.title("Inspeção com Programas")
        self.top.transient(self.parent.master)
        self.top.grab_set()

        # Dimensão inicial, cresce conforme slots
        self.top.geometry("520x360")

        container = ttk.Frame(self.top, padding=10)
        container.pack(fill=BOTH, expand=True)

        title = ttk.Label(container, text="Selecione os programas (1..N) para inspecionar a cada captura:", font=("Segoe UI", 11, "bold"))
        title.pack(anchor="w", pady=(0, 10))

        # Buscar modelos
        try:
            self.model_list = self.db_manager.list_modelos() or []
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao listar modelos: {e}")
            self.model_list = []

        # Preparar valores
        values = []
        self.display_to_id.clear()
        for m in self.model_list:
            mid = m.get('id')
            nome = m.get('nome', f"Modelo {mid}")
            cam = m.get('camera_index', '-')
            display = f"{mid} - {nome} (Cam {cam})"
            values.append(display)
            self.display_to_id[display] = mid

        # Área rolável para slots
        slots_frame = ttk.Frame(container)
        slots_frame.pack(fill=BOTH, expand=True)

        self.slots_container = ttk.Frame(slots_frame)
        self.slots_container.pack(fill=BOTH, expand=True)

        # Botão adicionar slot
        actions_row = ttk.Frame(container)
        actions_row.pack(fill=X, pady=(8, 0))
        ttk.Button(actions_row, text="+ Adicionar Programa", bootstyle="info", command=lambda: self._add_slot(values)).pack(side=LEFT)

        # Pré-popular slots: usa preselected_ids se houver, senão 2 vazios
        if self.preselected_ids:
            for pid in self.preselected_ids:
                self._add_slot(values, preselect_id=pid)
        else:
            self._add_slot(values)
            self._add_slot(values)

        # Botões finais
        btns = ttk.Frame(container)
        btns.pack(fill=X, pady=(12, 0))
        ttk.Button(btns, text="Cancelar", bootstyle="secondary", command=self._on_cancel).pack(side=RIGHT)
        ttk.Button(btns, text="OK", bootstyle="success", command=self._on_ok).pack(side=RIGHT, padx=(0, 8))

        self._center_on_parent()
        self.top.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _add_slot(self, values, preselect_id=None):
        idx = len(self.slot_rows) + 1
        row = ttk.Frame(self.slots_container)
        row.pack(fill=X, pady=5)
        ttk.Label(row, text=f"Programa {idx}:").pack(side=LEFT)
        combo = Combobox(row, values=values, state="readonly")
        combo.pack(side=LEFT, fill=X, expand=True, padx=(10, 0))
        # botão remover
        ttk.Button(row, text="Remover", bootstyle="danger-outline", command=lambda r=row: self._remove_slot(r)).pack(side=LEFT, padx=6)
        self.slot_rows.append((row, combo))
        # pré-seleciona
        if preselect_id is not None:
            for disp, mid in self.display_to_id.items():
                if mid == preselect_id:
                    combo.set(disp)
                    break

    def _remove_slot(self, row):
        # garante pelo menos 1 slot
        if len(self.slot_rows) <= 1:
            messagebox.showwarning("Atenção", "Mantenha ao menos um programa.")
            return
        # remove do array e da UI
        self.slot_rows = [(r, c) for (r, c) in self.slot_rows if r is not row]
        row.destroy()
        # renomeia labels
        for i, (r, _) in enumerate(self.slot_rows, start=1):
            for child in r.winfo_children():
                if isinstance(child, ttk.Label):
                    child.configure(text=f"Programa {i}:")
                    break

    def _on_ok(self):
        selections = []
        seen = set()
        for _, combo in self.slot_rows:
            val = combo.get()
            if not val:
                messagebox.showwarning("Atenção", "Selecione todos os programas.")
                return
            mid = self.display_to_id.get(val)
            if mid in seen:
                messagebox.showwarning("Atenção", "Não selecione o mesmo programa repetido.")
                return
            seen.add(mid)
            selections.append(mid)
        self.result = {'program_ids': selections}
        self.top.destroy()

    def _on_cancel(self):
        self.result = None
        self.top.destroy()

    def _center_on_parent(self):
        try:
            self.top.update_idletasks()
            px = self.parent.master.winfo_rootx()
            py = self.parent.master.winfo_rooty()
            pw = self.parent.master.winfo_width()
            ph = self.parent.master.winfo_height()
            tw = self.top.winfo_width()
            th = self.top.winfo_height()
            x = px + (pw - tw) // 2
            y = py + (ph - th) // 2
            self.top.geometry(f"{tw}x{th}+{x}+{y}")
        except Exception:
            pass

    def show(self):
        self._build_ui()
        self.top.wait_window()
        return self.result


