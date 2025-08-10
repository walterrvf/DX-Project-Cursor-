import cv2
import time
import platform
from typing import List, Optional

# Cache global para instâncias de câmera para evitar reinicializações desnecessárias
_camera_cache = {}
_camera_last_used = {}


def detect_cameras(max_cameras: int = 5, callback=None) -> List[int]:
    """
    Detecta webcams disponíveis no sistema.
    Retorna lista de índices de câmeras funcionais.
    Compatível com Windows e Raspberry Pi.
    """
    available_cameras: List[int] = []
    is_windows = platform.system() == 'Windows'

    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if is_windows else cv2.VideoCapture(i)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    available_cameras.append(i)
                cap.release()
        except Exception:
            # Silencia erros de câmeras não encontradas
            continue

    if not available_cameras:
        available_cameras.append(0)
        print("Nenhuma câmera detectada automaticamente. Usando índice 0 como padrão.")
    else:
        print(f"Câmeras detectadas: {available_cameras}")

    if callback:
        callback(available_cameras)

    return available_cameras


def get_cached_camera(camera_index: int = 0, force_new: bool = False):
    """Obtém uma instância de câmera do cache ou cria uma nova."""
    global _camera_cache, _camera_last_used

    if force_new or camera_index not in _camera_cache:
        if camera_index in _camera_cache:
            try:
                _camera_cache[camera_index].release()
            except Exception:
                pass
            del _camera_cache[camera_index]

        try:
            is_windows = platform.system() == 'Windows'
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if is_windows else cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"Erro: Não foi possível abrir a câmera {camera_index}")
                return None

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Resolução padrão
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            _camera_cache[camera_index] = cap
            print(f"Nova instância de câmera criada para índice {camera_index}")
        except Exception as e:
            print(f"Erro ao criar instância da câmera {camera_index}: {e}")
            return None

    _camera_last_used[camera_index] = time.time()
    return _camera_cache[camera_index]


def release_cached_camera(camera_index: int = 0) -> None:
    """Libera uma câmera específica do cache."""
    global _camera_cache, _camera_last_used
    if camera_index in _camera_cache:
        try:
            _camera_cache[camera_index].release()
            print(f"Câmera {camera_index} liberada do cache")
        except Exception as e:
            print(f"Erro ao liberar câmera {camera_index}: {e}")
        finally:
            del _camera_cache[camera_index]
            if camera_index in _camera_last_used:
                del _camera_last_used[camera_index]


def cleanup_unused_cameras(max_idle_time: int = 300) -> None:
    """Limpa câmeras não utilizadas há muito tempo para liberar recursos."""
    global _camera_last_used
    current_time = time.time()
    cameras_to_remove = [idx for idx, last in _camera_last_used.items() if current_time - last > max_idle_time]
    for camera_index in cameras_to_remove:
        release_cached_camera(camera_index)
        print(f"Câmera {camera_index} removida do cache por inatividade")


def schedule_camera_cleanup(window, interval_ms: int = 60000) -> None:
    """Agenda limpeza automática de câmeras não utilizadas."""
    try:
        cleanup_unused_cameras()
        window.after(interval_ms, lambda: schedule_camera_cleanup(window, interval_ms))
    except Exception as e:
        print(f"Erro na limpeza automática de câmeras: {e}")
        window.after(interval_ms, lambda: schedule_camera_cleanup(window, interval_ms))


def release_all_cached_cameras() -> None:
    """Libera todas as câmeras do cache."""
    cameras_to_remove = list(_camera_cache.keys())
    for camera_index in cameras_to_remove:
        release_cached_camera(camera_index)
    print(f"Todas as câmeras do cache foram liberadas ({len(cameras_to_remove)} câmeras)")


def capture_image_from_camera(camera_index: int = 0, use_cache: bool = True):
    """
    Captura uma única imagem da webcam especificada.
    Retorna a imagem capturada ou None em caso de erro.
    Compatível com Windows e Raspberry Pi.
    """
    try:
        if use_cache:
            cap = get_cached_camera(camera_index)
            if cap is None:
                return None
        else:
            is_windows = platform.system() == 'Windows'
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if is_windows else cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"Erro: Não foi possível abrir a câmera {camera_index}")
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)

        # Limpa buffer antigo para obter frame mais recente
        for _ in range(3):
            ret, frame = cap.read()
            if not ret:
                break

        ret, frame = cap.read()

        if not use_cache:
            cap.release()

        if ret and frame is not None and frame.size > 0:
            print(f"Imagem capturada com sucesso da câmera {camera_index} (cache: {use_cache})")
            return frame
        else:
            print(f"Erro: Não foi possível capturar imagem da câmera {camera_index}")
            return None
    except Exception as e:
        print(f"Erro ao capturar imagem da câmera {camera_index}: {e}")
        if use_cache:
            try:
                print(f"Tentando recriar câmera {camera_index} após erro...")
                cap = get_cached_camera(camera_index, force_new=True)
                if cap:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"Câmera {camera_index} recriada com sucesso")
                        return frame
            except Exception as retry_error:
                print(f"Erro ao recriar câmera {camera_index}: {retry_error}")
        return None


__all__ = [
    "detect_cameras",
    "get_cached_camera",
    "release_cached_camera",
    "cleanup_unused_cameras",
    "schedule_camera_cleanup",
    "release_all_cached_cameras",
    "capture_image_from_camera",
]


