import cv2
import threading
import time
import platform
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

# Logger do módulo
logger = logging.getLogger(__name__)

class CameraDriver(ABC):
    """Classe abstrata para drivers de câmera."""
    
    def __init__(self, camera_index: int, name: str = None):
        self.camera_index = camera_index
        self.name = name or f"Camera_{camera_index}"
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_active = False
        self.last_frame = None
        self.frame_lock = threading.RLock()
        self.capture_thread = None
        self._stop_flag = False
        self._from_pool = False
        
    @abstractmethod
    def initialize_camera(self) -> bool:
        """Inicializa a câmera com configurações específicas do driver."""
        pass
    
    @abstractmethod
    def configure_camera(self) -> None:
        """Configura parâmetros específicos da câmera."""
        pass
    
    def start_capture(self) -> bool:
        """Inicia a captura de frames em thread separada."""
        if not self.initialize_camera():
            return False
            
        self._stop_flag = False
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"{self.name}_capture"
        )
        self.capture_thread.start()
        self.is_active = True
        logger.info(f"Driver {self.name} iniciado com sucesso")
        return True
    
    def stop_capture(self) -> None:
        """Para a captura e libera recursos."""
        self._stop_flag = True
        self.is_active = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            
        if self.camera:
            try:
                # Se a câmera veio do pool persistente, não liberar aqui
                if not self._from_pool:
                    self.camera.release()
            except Exception as e:
                logger.warning(f"Erro ao liberar câmera {self.name}: {e}")
            finally:
                self.camera = None
                
        logger.info(f"Driver {self.name} parado")
    
    def get_frame(self) -> Optional[Any]:
        """Retorna o último frame capturado."""
        with self.frame_lock:
            return self.last_frame.copy() if self.last_frame is not None else None
    
    def _capture_loop(self) -> None:
        """Loop principal de captura de frames."""
        while not self._stop_flag and self.camera and self.camera.isOpened():
            try:
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    with self.frame_lock:
                        self.last_frame = frame
                else:
                    logger.warning(f"Falha na captura do frame - {self.name}")
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Erro na captura {self.name}: {e}")
                time.sleep(0.1)
                
    def is_healthy(self) -> bool:
        """Verifica se o driver está funcionando corretamente."""
        return (self.is_active and 
                self.camera is not None and 
                self.camera.isOpened() and 
                self.last_frame is not None)


class InternalCameraDriver(CameraDriver):
    """Driver otimizado para câmera interna (webcam integrada)."""
    
    def __init__(self, camera_index: int = 0):
        super().__init__(camera_index, f"InternalCam_{camera_index}")
        
    def initialize_camera(self) -> bool:
        """Inicializa câmera interna com configurações otimizadas."""
        try:
            is_windows = platform.system() == 'Windows'
            
            # Tenta obter do pool persistente para evitar múltiplas aberturas
            try:
                from camera_manager import get_persistent_camera, _open_camera
                cam = get_persistent_camera(self.camera_index)
            except Exception:
                cam = None
            
            if cam and cam.isOpened():
                self.camera = cam
                self._from_pool = True
            else:
                # Usa rotina _open_camera com fallback de backend
                try:
                    from camera_manager import _open_camera as cm_open
                    self.camera = cm_open(self.camera_index)
                except Exception:
                    # Último recurso: backends padrão
                    if is_windows:
                        self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                    else:
                        self.camera = cv2.VideoCapture(self.camera_index)
                
            if not self.camera.isOpened():
                logger.error(f"Não foi possível abrir câmera interna {self.camera_index}")
                return False
                
            self.configure_camera()
            logger.info(f"Câmera interna {self.camera_index} inicializada")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar câmera interna {self.camera_index}: {e}")
            return False
    
    def configure_camera(self) -> None:
        """Configura câmera interna usando configuração centralizada."""
        if not self.camera:
            return
        try:
            # Usa configuração centralizada para manter consistência e evitar superexposição
            from camera_manager import configure_video_capture
            configure_video_capture(self.camera, self.camera_index)
            logger.info(f"Câmera interna {self.camera_index} configurada")
        except Exception as e:
            logger.warning(f"Erro ao configurar câmera interna {self.camera_index}: {e}")


class ExternalCameraDriver(CameraDriver):
    """Driver otimizado para câmeras externas (USB, IP, etc.)."""
    
    def __init__(self, camera_index: int):
        super().__init__(camera_index, f"ExternalCam_{camera_index}")
        
    def initialize_camera(self) -> bool:
        """Inicializa câmera externa com configurações de alta qualidade."""
        try:
            is_windows = platform.system() == 'Windows'
            
            # Preferir pool persistente para compartilhar o handle
            try:
                from camera_manager import get_persistent_camera
                cam = get_persistent_camera(self.camera_index)
            except Exception:
                cam = None
            
            if cam and cam.isOpened():
                self.camera = cam
                self._from_pool = True
            else:
                # Usa rotina _open_camera com fallback de backend
                try:
                    from camera_manager import _open_camera as cm_open
                    self.camera = cm_open(self.camera_index)
                except Exception:
                    if is_windows:
                        self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                    else:
                        self.camera = cv2.VideoCapture(self.camera_index)
                
            if not self.camera.isOpened():
                logger.error(f"Não foi possível abrir câmera externa {self.camera_index}")
                return False
                
            self.configure_camera()
            logger.info(f"Câmera externa {self.camera_index} inicializada")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar câmera externa {self.camera_index}: {e}")
            return False
    
    def configure_camera(self) -> None:
        """Configura câmera externa usando configuração centralizada."""
        if not self.camera:
            return
        try:
            from camera_manager import configure_video_capture
            configure_video_capture(self.camera, self.camera_index)
            logger.info(f"Câmera externa {self.camera_index} configurada")
        except Exception as e:
            logger.warning(f"Erro ao configurar câmera externa {self.camera_index}: {e}")


class DualCameraManager:
    """Gerenciador para duas câmeras com drivers diferentes."""
    
    def __init__(self):
        self.internal_driver: Optional[InternalCameraDriver] = None
        self.external_driver: Optional[ExternalCameraDriver] = None
        self.drivers: Dict[int, CameraDriver] = {}
        self.is_initialized = False
        
    def initialize_cameras(self, internal_index: int = 0, external_index: Optional[int] = 1) -> bool:
        """Inicializa câmeras com drivers específicos. Se external_index for None ou igual ao internal, inicializa apenas a interna."""
        success = True
        
        try:
            # Inicializa driver da câmera interna
            self.internal_driver = InternalCameraDriver(internal_index)
            if self.internal_driver.start_capture():
                self.drivers[internal_index] = self.internal_driver
                logger.info(f"Driver interno inicializado para câmera {internal_index}")
            else:
                logger.error(f"Falha ao inicializar driver interno para câmera {internal_index}")
                success = False
            
            # Inicializa driver da câmera externa somente se for válido e diferente do interno
            if external_index is not None and external_index != internal_index:
                self.external_driver = ExternalCameraDriver(external_index)
                if self.external_driver.start_capture():
                    self.drivers[external_index] = self.external_driver
                    logger.info(f"Driver externo inicializado para câmera {external_index}")
                else:
                    logger.error(f"Falha ao inicializar driver externo para câmera {external_index}")
                    success = False
            else:
                logger.info("Somente câmera interna será utilizada (nenhuma câmera externa válida encontrada)")
                self.external_driver = None
                
            self.is_initialized = success
            return success
            
        except Exception as e:
            logger.error(f"Erro ao inicializar gerenciador dual: {e}")
            return False
    
    def get_frame(self, camera_index: int) -> Optional[Any]:
        """Obtém frame de uma câmera específica."""
        driver = self.drivers.get(camera_index)
        if driver and driver.is_healthy():
            return driver.get_frame()
        return None
    
    def get_all_frames(self) -> Dict[int, Any]:
        """Obtém frames de todas as câmeras ativas."""
        frames = {}
        for camera_index, driver in self.drivers.items():
            if driver.is_healthy():
                frame = driver.get_frame()
                if frame is not None:
                    frames[camera_index] = frame
        return frames
    
    def get_camera_status(self) -> Dict[int, bool]:
        """Retorna status de saúde de todas as câmeras."""
        return {index: driver.is_healthy() for index, driver in self.drivers.items()}
    
    def stop_all(self) -> None:
        """Para todos os drivers e libera recursos."""
        for driver in self.drivers.values():
            driver.stop_capture()
            
        self.drivers.clear()
        self.internal_driver = None
        self.external_driver = None
        self.is_initialized = False
        logger.info("Todos os drivers de câmera foram parados")
    
    def restart_camera(self, camera_index: int) -> bool:
        """Reinicia uma câmera específica."""
        if camera_index in self.drivers:
            driver = self.drivers[camera_index]
            driver.stop_capture()
            
            # Aguarda um pouco antes de reiniciar
            time.sleep(1.0)
            
            if driver.start_capture():
                logger.info(f"Câmera {camera_index} reiniciada com sucesso")
                return True
            else:
                logger.error(f"Falha ao reiniciar câmera {camera_index}")
                del self.drivers[camera_index]
                return False
        return False


# Instância global do gerenciador dual
_dual_manager: Optional[DualCameraManager] = None

def get_dual_camera_manager() -> DualCameraManager:
    """Obtém a instância global do gerenciador dual de câmeras."""
    global _dual_manager
    if _dual_manager is None:
        _dual_manager = DualCameraManager()
    return _dual_manager

def initialize_dual_cameras(internal_index: int = 0, external_index: int = 1) -> bool:
    """Função de conveniência para inicializar o sistema dual de câmeras."""
    manager = get_dual_camera_manager()
    return manager.initialize_cameras(internal_index, external_index)

def get_camera_frame(camera_index: int) -> Optional[Any]:
    """Função de conveniência para obter frame de uma câmera."""
    manager = get_dual_camera_manager()
    return manager.get_frame(camera_index)

def get_all_camera_frames() -> Dict[int, Any]:
    """Função de conveniência para obter frames de todas as câmeras."""
    manager = get_dual_camera_manager()
    return manager.get_all_frames()

def stop_dual_cameras() -> None:
    """Função de conveniência para parar todas as câmeras."""
    global _dual_manager
    if _dual_manager:
        _dual_manager.stop_all()
        _dual_manager = None