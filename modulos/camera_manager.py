import cv2
import time
import platform
import logging
from typing import List, Optional
import threading
import numpy as np

# Logger do módulo
logger = logging.getLogger(__name__)

# Cache global de câmeras e controle de último uso
_camera_cache: dict[int, cv2.VideoCapture] = {}
_camera_last_used: dict[int, float] = {}
_camera_lock = threading.RLock()

# Pool persistente de câmeras ativas
_persistent_camera_pool: dict[int, cv2.VideoCapture] = {}
_pool_health_status: dict[int, bool] = {}
_pool_lock = threading.RLock()
_pool_initialized = False
_health_monitor_thread = None

# Frame pump (captura contínua centralizada)
_frame_pump_threads: dict[int, threading.Thread] = {}
_latest_frames: dict[int, np.ndarray] = {}
_frame_pump_running = False


def _synchronized_pool(func):
    """Decorator para sincronizar acesso ao pool persistente."""
    def wrapper(*args, **kwargs):
        with _pool_lock:
            return func(*args, **kwargs)
    return wrapper

def _synchronized(func):
    """Decorator simples para sincronizar acesso a recursos compartilhados."""
    def wrapper(*args, **kwargs):
        with _camera_lock:
            return func(*args, **kwargs)
    return wrapper


def configure_video_capture(cap: cv2.VideoCapture, camera_index: int) -> None:
    """Aplica configurações padrão de captura (resolução, FPS, exposição, ganho, WB).

    Nota: Alguns parâmetros podem não ser suportados por todas as câmeras/OS. Os erros são ignorados.
    """
    if cap is None or not cap.isOpened():
        return

    try:
        # Importa configurações do sistema (padrões se ausentes)
        from .utils import load_style_config
        cfg = load_style_config().get('system', {})
    except Exception:
        cfg = {}

    # Resolução e FPS
    try:
        width = int(cfg.get('camera_width', 1280 if camera_index > 0 else 640))
        height = int(cfg.get('camera_height', 720 if camera_index > 0 else 480))
        fps = int(cfg.get('camera_fps', 30))
        buffersize = int(cfg.get('buffer_size', 1))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
    except Exception:
        pass

    try:
        auto_exp = bool(cfg.get('auto_exposure', True))
        # Ajuste robusto por backend:
        # - DirectShow (Windows): 0.75 = auto, 0.25 = manual
        # - MSMF (Windows): 1 = auto, 0 = manual
        backend_name = ""
        try:
            backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else ""
        except Exception:
            backend_name = ""
        backend_upper = str(backend_name).upper()

        if "DSHOW" in backend_upper:
            value = 0.75 if auto_exp else 0.25
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
        elif "MSMF" in backend_upper:
            value = 1 if auto_exp else 0
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
        else:
            # Fallback: tenta sequências comuns
            if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto_exp else 0.25):
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if auto_exp else 0)
    except Exception:
        pass

    try:
        # Define exposição apenas se auto-exposure estiver desligado
        if not bool(cfg.get('auto_exposure', True)):
            # Valores típicos de exposição em DSHOW/MSMF são negativos em log2s.
            # Deixe a câmera 1 (externa) um pouco mais clara por padrão.
            base_exposure = float(cfg.get('base_exposure', -5 if camera_index > 0 else -6))
            cap.set(cv2.CAP_PROP_EXPOSURE, base_exposure)
        # Ajuste de ganho com preferência por 0 para evitar escurecimento progressivo por AGC
        gain = float(cfg.get('gain', 0))
        cap.set(cv2.CAP_PROP_GAIN, gain)
    except Exception:
        pass

    try:
        # Garante ganho mínimo quando auto exposure está ON (evita drift de brilho)
        if bool(cfg.get('auto_exposure', True)):
            cap.set(cv2.CAP_PROP_GAIN, 0)
    except Exception:
        pass

    # Log de diagnóstico dos valores efetivos
    try:
        backend_name = ""
        try:
            backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else ""
        except Exception:
            backend_name = ""
        eff_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        eff_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        eff_gain = cap.get(cv2.CAP_PROP_GAIN)
        eff_wb_auto = cap.get(cv2.CAP_PROP_AUTO_WB) if hasattr(cv2, 'CAP_PROP_AUTO_WB') else None
        logger.info(
            f"Camera {camera_index} backend={backend_name} autoExp={eff_auto} exp={eff_exp} gain={eff_gain} autoWB={eff_wb_auto}"
        )
    except Exception:
        pass

    try:
        auto_wb = bool(cfg.get('auto_wb', True))
        cap.set(cv2.CAP_PROP_AUTO_WB, 1 if auto_wb else 0)
        if not auto_wb:
            wb_temp = int(cfg.get('wb_temperature', 4500))
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temp)
    except Exception:
        pass

    try:
        autofocus = 1 if (camera_index == 0 and bool(cfg.get('autofocus_internal', True))) else 0
        cap.set(cv2.CAP_PROP_AUTOFOCUS, autofocus)
    except Exception:
        pass


def _open_camera(camera_index: int) -> Optional[cv2.VideoCapture]:
    """Abre a câmera respeitando diferenças entre Windows e outros SOs."""
    is_windows = platform.system() == 'Windows'
    is_linux = platform.system() == 'Linux'
    machine = platform.machine().lower()
    platform_str = platform.platform().lower()
    is_rpi = is_linux and (('arm' in machine or 'aarch64' in machine) or ('raspbian' in platform_str or 'raspberry' in platform_str))

    def try_open_with_backend(backend_flag) -> Optional[cv2.VideoCapture]:
        cap_try = cv2.VideoCapture(camera_index, backend_flag) if backend_flag is not None else cv2.VideoCapture(camera_index)
        if not cap_try or not cap_try.isOpened():
            if cap_try:
                try:
                    cap_try.release()
                except Exception:
                    pass
            return None
        # Configurar e validar leitura de teste
        configure_video_capture(cap_try, camera_index)
        ok = False
        try:
            # Descarta alguns frames e tenta ler um frame válido
            for _ in range(5):
                ret, frame = cap_try.read()
                if ret and frame is not None and getattr(frame, 'size', 0) > 0:
                    ok = True
                    break
                time.sleep(0.05)
        except Exception:
            ok = False
        if not ok:
            try:
                cap_try.release()
            except Exception:
                pass
            return None
        return cap_try

    cap = None
    if is_windows:
        # Seleciona backend conforme configuração
        try:
            from .utils import load_style_config
            backend_name = load_style_config().get('system', {}).get('camera_backend', 'AUTO').upper()
        except Exception:
            backend_name = 'AUTO'

        name_to_flag = {
            'AUTO': None,
            'DIRECTSHOW': cv2.CAP_DSHOW,
            'MSMF': cv2.CAP_MSMF,
            'V4L2': cv2.CAP_V4L2,
            'GSTREAMER': cv2.CAP_GSTREAMER,
        }
        preferred = name_to_flag.get(backend_name, None)

        tried = []
        order = [preferred] if preferred is not None else []
        # fallback order padrão Windows
        for b in (cv2.CAP_DSHOW, cv2.CAP_MSMF, None):
            if b not in order:
                order.append(b)

        for bflag in order:
            tried.append(bflag)
            cap = try_open_with_backend(bflag)
            if cap is not None:
                break
    else:
        # Em Linux/RPi tenta conforme configuração, com suporte a libcamera via GStreamer
        try:
            from .utils import load_style_config
            backend_name = load_style_config().get('system', {}).get('camera_backend', 'AUTO').upper()
            width_cfg = int(load_style_config().get('system', {}).get('camera_width', 1280))
            height_cfg = int(load_style_config().get('system', {}).get('camera_height', 720))
            fps_cfg = int(load_style_config().get('system', {}).get('camera_fps', 30))
        except Exception:
            backend_name, width_cfg, height_cfg, fps_cfg = 'AUTO', 1280, 720, 30

        def try_open_libcamera_pipeline() -> Optional[cv2.VideoCapture]:
            try:
                # Pipeline GStreamer para libcamera (Bullseye/Bookworm)
                pipeline = (
                    f"libcamerasrc ! video/x-raw,width={width_cfg},height={height_cfg},"
                    f"framerate={max(1, fps_cfg)}/1 ! videoconvert ! appsink"
                )
                cap_try = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if not cap_try or not cap_try.isOpened():
                    try:
                        if cap_try:
                            cap_try.release()
                    except Exception:
                        pass
                    return None
                # Configurar e validar
                configure_video_capture(cap_try, camera_index)
                ok = False
                for _ in range(5):
                    ret, frame = cap_try.read()
                    if ret and frame is not None and getattr(frame, 'size', 0) > 0:
                        ok = True
                        break
                    time.sleep(0.05)
                if not ok:
                    try:
                        cap_try.release()
                    except Exception:
                        pass
                    return None
                return cap_try
            except Exception:
                return None

        # Ordem de tentativas em Linux/RPi
        attempts: list[Optional[cv2.VideoCapture]] = []
        cap = None

        # 1) Em Raspberry Pi, prioriza libcamera quando AUTO ou LIBCAMERA
        if is_rpi and (backend_name in ('AUTO', 'LIBCAMERA')):
            cap = try_open_libcamera_pipeline()
            if cap is None:
                # fallback para V4L2 por índice (USB webcams)
                try:
                    cap = try_open_with_backend(cv2.CAP_V4L2)
                except Exception:
                    cap = None
        else:
            # 2) Em Linux comum, respeita backend configurado (V4L2/GSTREAMER/AUTO)
            name_to_flag = {
                'AUTO': None,
                'V4L2': cv2.CAP_V4L2,
                'GSTREAMER': cv2.CAP_GSTREAMER,
                'DIRECTSHOW': cv2.CAP_DSHOW,
                'MSMF': cv2.CAP_MSMF,
            }
            order = [name_to_flag.get(backend_name, None), cv2.CAP_V4L2, None]
            tried = set()
            for bflag in order:
                if bflag in tried:
                    continue
                tried.add(bflag)
                try:
                    cap = try_open_with_backend(bflag)
                except Exception:
                    cap = None
                if cap is not None:
                    break

        # 3) Fallback final no RPi: tenta libcamera mesmo se backend especificado não for AUTO/LIBCAMERA
        if cap is None and is_rpi:
            cap = try_open_libcamera_pipeline()

    if cap is None:
        logger.error(f"Erro: Não foi possível abrir a câmera {camera_index}")
        return None
    return cap


@_synchronized_pool
def start_frame_pump(camera_indices: List[int] = None, fps: float = 30.0) -> bool:
    """Inicia threads de captura contínua por câmera, armazenando o último frame.

    Requer o pool persistente inicializado. Se não estiver, inicializa automaticamente.
    """
    global _frame_pump_threads, _latest_frames, _frame_pump_running

    if not _pool_initialized:
        initialize_persistent_pool(camera_indices)

    if camera_indices is None:
        camera_indices = list(_persistent_camera_pool.keys())

    if not camera_indices:
        return False

    _frame_pump_running = True
    interval = max(0.0, 1.0 / fps) if fps > 0 else 0.0

    def make_loop(idx: int):
        def loop():
            global _frame_pump_running
            while _frame_pump_running and _pool_initialized:
                try:
                    cam = _persistent_camera_pool.get(idx)
                    if cam is None or not cam.isOpened():
                        # tenta recriar
                        cam = _open_camera(idx)
                        if cam is not None:
                            _persistent_camera_pool[idx] = cam
                    if cam is None:
                        time.sleep(0.2)
                        continue
                    ret, frame = cam.read()
                    if ret and frame is not None and getattr(frame, 'size', 0) > 0:
                        _latest_frames[idx] = frame
                    time.sleep(interval)
                except Exception:
                    time.sleep(0.2)
        return loop

    # Inicia threads por câmera faltante
    for idx in camera_indices:
        if idx in _frame_pump_threads and _frame_pump_threads[idx].is_alive():
            continue
        t = threading.Thread(target=make_loop(idx), daemon=True, name=f"frame_pump_{idx}")
        _frame_pump_threads[idx] = t
        t.start()

    return True


@_synchronized_pool
def stop_frame_pump(camera_indices: List[int] = None) -> None:
    """Para as threads de captura contínua e limpa últimos frames."""
    global _frame_pump_threads, _latest_frames, _frame_pump_running
    _frame_pump_running = False
    threads = list(_frame_pump_threads.items()) if camera_indices is None else [(i, _frame_pump_threads.get(i)) for i in camera_indices]
    for idx, t in threads:
        try:
            if t and t.is_alive():
                t.join(timeout=1.0)
        except Exception:
            pass
        if camera_indices is None or idx in camera_indices:
            _frame_pump_threads.pop(idx, None)
            _latest_frames.pop(idx, None)


@_synchronized_pool
def get_latest_frame(camera_index: int):
    """Obtém o último frame capturado pelo frame pump para a câmera informada."""
    frame = _latest_frames.get(camera_index)
    if frame is None:
        return None
    try:
        return frame.copy()
    except Exception:
        return frame


def detect_cameras(max_cameras: int = 5, callback=None) -> List[int]:
    """
    Detecta webcams disponíveis no sistema, retornando índices que abriram com sucesso.
    Se callback for fornecido, é chamado com o índice encontrado.
    """
    found: List[int] = []
    for idx in range(max_cameras):
        try:
            cap = _open_camera(idx)
            if cap is not None:
                found.append(idx)
                if callback:
                    try:
                        callback(idx)
                    except Exception as cb_err:
                        logger.debug(f"Callback de detect_cameras falhou para índice {idx}: {cb_err}")
                try:
                    cap.release()
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Erro ao testar câmera {idx}: {e}")
    logger.info(f"Câmeras detectadas: {found}")
    return found


@_synchronized
def get_cached_camera(camera_index: int = 0, force_new: bool = False) -> Optional[cv2.VideoCapture]:
    """Obtém uma instância de câmera do cache ou cria uma nova."""
    global _camera_cache, _camera_last_used

    if force_new and camera_index in _camera_cache:
        try:
            _camera_cache[camera_index].release()
        except Exception:
            pass
        _camera_cache.pop(camera_index, None)
        _camera_last_used.pop(camera_index, None)

    cap = _camera_cache.get(camera_index)
    if cap is None or not cap.isOpened():
        cap = _open_camera(camera_index)
        if cap is None:
            return None
        _camera_cache[camera_index] = cap
        logger.info(f"Câmera {camera_index} aberta e armazenada no cache")

    _camera_last_used[camera_index] = time.time()
    return cap


@_synchronized
def release_cached_camera(camera_index: int = 0) -> None:
    """Libera uma câmera específica do cache."""
    global _camera_cache, _camera_last_used
    cap = _camera_cache.pop(camera_index, None)
    _camera_last_used.pop(camera_index, None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        logger.info(f"Câmera {camera_index} liberada do cache")


@_synchronized
def cleanup_unused_cameras(max_idle_time: int = 300) -> None:
    """Limpa câmeras não utilizadas há mais de max_idle_time segundos."""
    now = time.time()
    to_release = [idx for idx, last in _camera_last_used.items() if now - last > max_idle_time]
    for idx in to_release:
        cap = _camera_cache.pop(idx, None)
        _camera_last_used.pop(idx, None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            logger.info(f"Câmera {idx} liberada por inatividade ({max_idle_time}s)")


@_synchronized
def release_all_cached_cameras() -> None:
    """Libera todas as câmeras do cache."""
    cameras_to_remove = list(_camera_cache.keys())
    for camera_index in cameras_to_remove:
        release_cached_camera(camera_index)
    logger.info(f"Todas as câmeras do cache foram liberadas ({len(cameras_to_remove)} câmeras)")


def capture_image_from_camera(camera_index: int = 0, use_cache: bool = True):
    """
    Captura uma única imagem da webcam especificada.
    Retorna a imagem capturada ou None em caso de erro.
    Compatível com Windows e Raspberry Pi.
    """
    try:
        cap = None
        if use_cache:
            cap = get_cached_camera(camera_index)
            if cap is None:
                return None
        else:
            cap = _open_camera(camera_index)
            if cap is None:
                return None

        # Limpa buffer antigo para obter frame mais recente
        for _ in range(3):
            ret, frame = cap.read()
            if not ret:
                break

        ret, frame = cap.read()

        if not use_cache and cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        if ret and frame is not None and getattr(frame, 'size', 0) > 0:
            logger.debug(f"Imagem capturada com sucesso da câmera {camera_index} (cache: {use_cache})")
            if use_cache:
                _camera_last_used[camera_index] = time.time()
            return frame
        else:
            logger.error(f"Não foi possível capturar imagem da câmera {camera_index}")
            return None
    except Exception as e:
        logger.error(f"Erro ao capturar imagem da câmera {camera_index}: {e}")
        if use_cache:
            try:
                logger.info(f"Tentando recriar câmera {camera_index} após erro...")
                cap = get_cached_camera(camera_index, force_new=True)
                if cap:
                    ret, frame = cap.read()
                    if ret and frame is not None and getattr(frame, 'size', 0) > 0:
                        logger.info(f"Câmera {camera_index} recriada com sucesso")
                        _camera_last_used[camera_index] = time.time()
                        return frame
            except Exception as retry_error:
                logger.error(f"Erro ao recriar câmera {camera_index}: {retry_error}")
        return None


 


@_synchronized_pool
def initialize_persistent_pool(camera_indices: List[int] = None) -> bool:
    """Inicializa o pool persistente de câmeras."""
    global _persistent_camera_pool, _pool_health_status, _pool_initialized, _health_monitor_thread
    
    if _pool_initialized:
        logger.info("Pool persistente já inicializado")
        return True
    
    try:
        # Se não especificado, detecta câmeras automaticamente
        if camera_indices is None:
            camera_indices = detect_cameras()
        
        logger.info(f"Inicializando pool persistente para câmeras: {camera_indices}")
        
        # Inicializa cada câmera no pool
        for camera_index in camera_indices:
            try:
                cap = _open_camera(camera_index)
                if cap is not None:
                    _persistent_camera_pool[camera_index] = cap
                    _pool_health_status[camera_index] = True
                    logger.info(f"Câmera {camera_index} adicionada ao pool persistente")
                else:
                    _pool_health_status[camera_index] = False
                    logger.warning(f"Falha ao adicionar câmera {camera_index} ao pool")
            except Exception as e:
                logger.error(f"Erro ao inicializar câmera {camera_index} no pool: {e}")
                _pool_health_status[camera_index] = False
        
        _pool_initialized = True
        
        # Inicia thread de monitoramento de saúde
        if _health_monitor_thread is None or not _health_monitor_thread.is_alive():
            _health_monitor_thread = threading.Thread(
                target=_monitor_pool_health, 
                daemon=True
            )
            _health_monitor_thread.start()
            logger.info("Thread de monitoramento de saúde do pool iniciada")
        
        return len(_persistent_camera_pool) > 0
        
    except Exception as e:
        logger.error(f"Erro ao inicializar pool persistente: {e}")
        return False


@_synchronized_pool
def get_persistent_camera(camera_index: int) -> Optional[cv2.VideoCapture]:
    """Obtém uma câmera do pool persistente."""
    global _persistent_camera_pool, _pool_health_status
    
    if not _pool_initialized:
        logger.warning("Pool persistente não inicializado, inicializando automaticamente...")
        initialize_persistent_pool()
    
    # Verifica se a câmera existe no pool
    if camera_index not in _persistent_camera_pool:
        logger.warning(f"Câmera {camera_index} não encontrada no pool, tentando adicionar...")
        try:
            cap = _open_camera(camera_index)
            if cap is not None:
                _persistent_camera_pool[camera_index] = cap
                _pool_health_status[camera_index] = True
                logger.info(f"Câmera {camera_index} adicionada dinamicamente ao pool")
            else:
                return None
        except Exception as e:
            logger.error(f"Erro ao adicionar câmera {camera_index} dinamicamente: {e}")
            return None
    
    # Verifica saúde da câmera
    camera = _persistent_camera_pool.get(camera_index)
    if camera is None or not camera.isOpened():
        logger.warning(f"Câmera {camera_index} não está saudável, tentando recriar...")
        try:
            # Remove câmera problemática
            if camera:
                camera.release()
            
            # Recria câmera
            new_camera = _open_camera(camera_index)
            if new_camera is not None:
                _persistent_camera_pool[camera_index] = new_camera
                _pool_health_status[camera_index] = True
                logger.info(f"Câmera {camera_index} recriada no pool")
                return new_camera
            else:
                _pool_health_status[camera_index] = False
                return None
        except Exception as e:
            logger.error(f"Erro ao recriar câmera {camera_index}: {e}")
            _pool_health_status[camera_index] = False
            return None
    
    _pool_health_status[camera_index] = True
    return camera


def _monitor_pool_health():
    """Thread de monitoramento contínuo da saúde das câmeras no pool."""
    while _pool_initialized:
        try:
            time.sleep(30)  # Verifica a cada 30 segundos
            
            with _pool_lock:
                cameras_to_check = list(_persistent_camera_pool.keys())
            
            for camera_index in cameras_to_check:
                try:
                    camera = _persistent_camera_pool.get(camera_index)
                    if camera and camera.isOpened():
                        # Tenta ler um frame para verificar se está funcionando
                        ret, _ = camera.read()
                        if ret:
                            _pool_health_status[camera_index] = True
                        else:
                            logger.warning(f"Câmera {camera_index} não conseguiu capturar frame")
                            _pool_health_status[camera_index] = False
                    else:
                        logger.warning(f"Câmera {camera_index} não está aberta")
                        _pool_health_status[camera_index] = False
                except Exception as e:
                    logger.error(f"Erro ao verificar saúde da câmera {camera_index}: {e}")
                    _pool_health_status[camera_index] = False
                    
        except Exception as e:
            logger.error(f"Erro no monitoramento de saúde do pool: {e}")
            time.sleep(10)  # Aguarda antes de tentar novamente


@_synchronized_pool
def get_pool_status() -> dict:
    """Retorna o status atual do pool persistente."""
    return {
        'initialized': _pool_initialized,
        'cameras': list(_persistent_camera_pool.keys()),
        'health_status': _pool_health_status.copy(),
        'active_cameras': len([k for k, v in _pool_health_status.items() if v])
    }


@_synchronized_pool
def shutdown_persistent_pool():
    """Encerra o pool persistente e libera todas as câmeras."""
    global _persistent_camera_pool, _pool_health_status, _pool_initialized, _health_monitor_thread
    
    logger.info("Encerrando pool persistente de câmeras...")
    
    _pool_initialized = False
    
    # Libera todas as câmeras
    for camera_index, camera in _persistent_camera_pool.items():
        try:
            if camera:
                camera.release()
                logger.info(f"Câmera {camera_index} liberada do pool persistente")
        except Exception as e:
            logger.error(f"Erro ao liberar câmera {camera_index}: {e}")
    
    _persistent_camera_pool.clear()
    _pool_health_status.clear()
    
    # Aguarda thread de monitoramento terminar
    if _health_monitor_thread and _health_monitor_thread.is_alive():
        _health_monitor_thread.join(timeout=5)
    
    logger.info("Pool persistente encerrado")


@_synchronized
def capture_image_from_persistent_pool(camera_index: int):
    """Captura uma imagem usando o pool persistente (mais rápido)."""
    try:
        # Tenta usar o frame pump primeiro
        frame = get_latest_frame(camera_index)
        if frame is not None:
            return frame

        camera = get_persistent_camera(camera_index)
        if camera is None:
            logger.error(f"Câmera {camera_index} não disponível no pool persistente")
            return None

        # Como fallback, realiza leitura direta (pode conflitar com outras leituras)
        for _ in range(3):
            ret, frame = camera.read()
            if ret and frame is not None and getattr(frame, 'size', 0) > 0:
                logger.debug(f"Imagem capturada do pool persistente (fallback read) - câmera {camera_index}")
                return frame
        logger.error(f"Falha na captura do pool persistente - câmera {camera_index}")
        return None
            
    except Exception as e:
        logger.error(f"Erro ao capturar do pool persistente - câmera {camera_index}: {e}")
        return None


__all__ = [
    "detect_cameras",
    "get_cached_camera",
    "release_cached_camera",
    "cleanup_unused_cameras",
    "release_all_cached_cameras",
    "capture_image_from_camera",
    "start_frame_pump",
    "stop_frame_pump",
    "get_latest_frame",
    "configure_video_capture",
    "initialize_persistent_pool",
    "get_persistent_camera",
    "get_pool_status",
    "shutdown_persistent_pool",
    "capture_image_from_persistent_pool",
]


