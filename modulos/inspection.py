import cv2
import numpy as np
from typing import Tuple, Optional, List, Any

# Parâmetros ORB padrão
ORB_FEATURES = 5000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8

# Inicialização do detector ORB
try:
    orb = cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_N_LEVELS,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
    )
except Exception as e:
    print(f"Erro ao inicializar ORB em inspection: {e}")
    orb = None

_ref_image_cache = {
    'image_hash': None,
    'keypoints': None,
    'descriptors': None,
    'gray_image': None,
}


def find_image_transform(img_ref: np.ndarray, img_test: np.ndarray) -> Tuple[Optional[np.ndarray], int, str]:
    """
    Encontra homografia entre imagem de referência e teste usando ORB + FLANN + RANSAC.
    Retorna (H, num_inliers, error_msg)
    """
    global _ref_image_cache

    if orb is None:
        return None, 0, "Detector ORB não disponível."

    if img_ref is None or img_test is None or img_ref.size == 0 or img_test.size == 0:
        return None, 0, "Imagens de referência ou teste inválidas."

    try:
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY) if len(img_ref.shape) == 3 else img_ref
        gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY) if len(img_test.shape) == 3 else img_test

        ref_hash = hash(gray_ref.tobytes())
        if (_ref_image_cache['image_hash'] == ref_hash and
                _ref_image_cache['keypoints'] is not None and
                _ref_image_cache['descriptors'] is not None):
            kp_ref = _ref_image_cache['keypoints']
            desc_ref = _ref_image_cache['descriptors']
        else:
            kp_ref, desc_ref = orb.detectAndCompute(gray_ref, None)
            _ref_image_cache.update({
                'image_hash': ref_hash,
                'keypoints': kp_ref,
                'descriptors': desc_ref,
                'gray_image': gray_ref.copy(),
            })

        kp_test, desc_test = orb.detectAndCompute(gray_test, None)
        if desc_ref is None or desc_test is None:
            return None, 0, "Não foi possível extrair descritores de uma das imagens."
        if len(desc_ref) < 4 or len(desc_test) < 4:
            return None, 0, f"Poucos descritores encontrados: ref={len(desc_ref)}, test={len(desc_test)}"

        # FLANN para descritores binários
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=64)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_ref, desc_test, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            return None, 0, f"Poucos good_matches: {len(good_matches)}"

        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=2000, confidence=0.995)
        if H is None or mask is None:
            return None, 0, "Homografia não encontrada."

        inliers = int(mask.sum())
        return H, inliers, ""
    except Exception as e:
        return None, 0, f"Erro em find_image_transform: {e}"


def transform_rectangle(img_shape: Tuple[int, int, int], rect: Tuple[int, int, int, int], M: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    """Transforma um retângulo (x,y,w,h) pela homografia M e retorna o bbox na imagem destino.
    Retorna None se o bbox ficar inválido.
    """
    if M is None:
        return rect

    x, y, w, h = rect
    corners = np.float32([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
    ]).reshape(-1, 1, 2)

    try:
        transformed = cv2.perspectiveTransform(corners, M)
        x_coords = transformed[:, 0, 0]
        y_coords = transformed[:, 0, 1]

        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

        img_h, img_w = img_shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)

        new_w = x_max - x_min
        new_h = y_max - y_min
        if new_w <= 0 or new_h <= 0:
            return None
        return x_min, y_min, new_w, new_h
    except Exception:
        return None


def check_slot(img_test: np.ndarray, slot_data: dict, M: Optional[np.ndarray]):
    """Verifica um slot na imagem de teste.
    Retorna: (passou, correlation, pixels, corners, bbox, log_msgs)
    Implementação enxuta baseada em template matching, com suporte a homografia.
    """
    log_msgs: List[str] = []
    corners: Optional[List[Tuple[int, int]]] = None
    bbox: List[int] = [0, 0, 0, 0]

    try:
        x, y, w, h = int(slot_data['x']), int(slot_data['y']), int(slot_data['w']), int(slot_data['h'])
        rect = (x, y, w, h)

        # Aplica transformação se fornecida
        transformed_rect = transform_rectangle(img_test.shape, rect, M)
        if transformed_rect is None:
            log_msgs.append("Retângulo transformado inválido")
            return False, 0.0, 0, None, bbox, log_msgs

        x, y, w, h = transformed_rect
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        bbox = [x, y, w, h]

        # Checagem de limites
        if x < 0 or y < 0 or x + w > img_test.shape[1] or y + h > img_test.shape[0] or w <= 0 or h <= 0:
            log_msgs.append(f"ROI fora dos limites: ({x},{y},{w},{h})")
            return False, 0.0, 0, corners, bbox, log_msgs

        roi = img_test[y:y+h, x:x+w]
        if roi.size == 0:
            log_msgs.append("ROI vazia")
            return False, 0.0, 0, corners, bbox, log_msgs

        # Se houver template_path, faz template matching
        template_path = slot_data.get('template_path') or slot_data.get('template')
        method_name = slot_data.get('template_method', 'TM_CCOEFF_NORMED')
        method = getattr(cv2, method_name, cv2.TM_CCOEFF_NORMED)
        corr_thr = float(slot_data.get('correlation_threshold', 0.5))

        correlation = 0.0
        if template_path:
            try:
                template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                if template is None:
                    log_msgs.append(f"Template não encontrado: {template_path}")
                else:
                    if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]:
                        # Redimensiona template para caber na ROI
                        scale = min(roi.shape[1] / template.shape[1], roi.shape[0] / template.shape[0])
                        new_size = (max(1, int(template.shape[1] * scale)), max(1, int(template.shape[0] * scale)))
                        template = cv2.resize(template, new_size)
                    res = cv2.matchTemplate(roi, template, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    correlation = float(max_val if method in (cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED) else 1.0 - min_val)
                    log_msgs.append(f"Template corr={correlation:.3f}")
            except Exception as e:
                log_msgs.append(f"Erro no template matching: {e}")

        # Caso sem template: simples checagem de variação (pixels não pretos)
        pixels = int(cv2.countNonZero(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
        px_thr = int(slot_data.get('min_pixels', 0))

        passou = True
        if template_path:
            passou = correlation >= corr_thr
        elif px_thr > 0:
            passou = pixels >= px_thr

        return passou, correlation, pixels, corners, bbox, log_msgs

    except Exception as e:
        log_msgs.append(f"Erro em check_slot: {e}")
        return False, 0.0, 0, corners, bbox, log_msgs


__all__ = ["find_image_transform", "transform_rectangle", "check_slot"]


