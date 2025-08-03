import cv2
import numpy as np
from pathlib import Path
import ttkbootstrap as ttk
from ttkbootstrap.constants import (LEFT, BOTH, YES, DISABLED, NORMAL, END, TOP, X, Y, BOTTOM, RIGHT,
                                    HORIZONTAL, VERTICAL, NW)
from tkinter import (Canvas, filedialog, messagebox, simpledialog, Toplevel, Label, StringVar,
                     PhotoImage as tkPhotoImage, Text, scrolledtext, colorchooser)
from tkinter.ttk import Combobox
from PIL import Image, ImageTk
from datetime import datetime
import sys
import os
import json

# Importa módulos do sistema de banco de dados
try:
    # Quando importado como módulo
    from .database_manager import DatabaseManager
    from .model_selector import ModelSelectorDialog, SaveModelDialog
except ImportError:
    # Quando executado diretamente
    from database_manager import DatabaseManager
    from model_selector import ModelSelectorDialog, SaveModelDialog

# ---------- parâmetros globais ------------------------------------------------
# Caminho absoluto para a pasta de modelos na raiz do projeto
MODEL_DIR = Path(__file__).parent.parent / "modelos"
MODEL_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR = MODEL_DIR / "_templates"
TEMPLATE_DIR.mkdir(exist_ok=True)

# Limiares de inspeção
THR_CORR = 0.1  # Limiar para template matching (clips)
MIN_PX = 10      # Contagem mínima de pixels para template matching (clips)

# Parâmetros do Canvas e Preview
PREVIEW_W = 800  # Largura máxima do canvas para exibição inicial
PREVIEW_H = 600  # Altura máxima do canvas para exibição inicial

# Parâmetros ORB para registro de imagem
ORB_FEATURES = 5000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8

# Cores para desenho no canvas (Editor) - ALTERADAS PARA NOVO LAYOUT
COLOR_CLIP = "#FF6B6B"  # Vermelho coral
COLOR_SELECTED = "#FFE66D"  # Amarelo dourado
COLOR_DRAWING = "#A8E6CF"  # Verde claro

# Cores para desenho no canvas (Inspeção) - ALTERADAS PARA NOVO LAYOUT
COLOR_PASS = "#95E1D3"  # Verde menta
COLOR_FAIL = "#F38BA8"  # Rosa
COLOR_ALIGN_FAIL = "#FECA57"  # Laranja claro

# Configurações de estilo padrão
DEFAULT_STYLES = {
    "background_color": "#1E1E1E",  # Cor de fundo dos diálogos
    "text_color": "#FFFFFF",       # Cor do texto
    "button_color": "#4CAF50",     # Cor dos botões
    "ng_color": "#F38BA8",         # Cor do texto NG
    "ok_color": "#95E1D3",         # Cor do texto OK
    "ng_font": "Arial 12 bold",    # Fonte do texto NG
    "ok_font": "Arial 12 bold",    # Fonte do texto OK
    "selection_color": "#FFE66D",  # Cor do quadro de seleção
}

# Caminho para o arquivo de configurações de estilo
STYLE_CONFIG_PATH = Path(__file__).parent.parent / "config" / "style_config.json"


# ---------- utilidades --------------------------------------------------------
def load_style_config():
    """
    Carrega as configurações de estilo do arquivo JSON.
    Se o arquivo não existir, cria um novo com as configurações padrão.
    """
    try:
        # Cria o diretório de configuração se não existir
        config_dir = STYLE_CONFIG_PATH.parent
        config_dir.mkdir(exist_ok=True)
        
        # Se o arquivo não existir, cria com as configurações padrão
        if not STYLE_CONFIG_PATH.exists():
            save_style_config(DEFAULT_STYLES)
            return DEFAULT_STYLES.copy()
        
        # Carrega as configurações do arquivo
        with open(STYLE_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # Verifica se todas as chaves necessárias estão presentes
        for key in DEFAULT_STYLES.keys():
            if key not in config:
                config[key] = DEFAULT_STYLES[key]
        
        return config
    except Exception as e:
        print(f"Erro ao carregar configurações de estilo: {e}")
        return DEFAULT_STYLES.copy()

def save_style_config(config):
    """
    Salva as configurações de estilo em um arquivo JSON.
    """
    try:
        # Cria o diretório de configuração se não existir
        config_dir = STYLE_CONFIG_PATH.parent
        config_dir.mkdir(exist_ok=True)
        
        # Salva as configurações no arquivo
        with open(STYLE_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Configurações de estilo salvas em {STYLE_CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"Erro ao salvar configurações de estilo: {e}")
        return False

def detect_cameras(max_cameras=5):
    """
    Detecta webcams disponíveis no sistema.
    Retorna lista de índices de câmeras funcionais.
    Compatível com Windows e Raspberry Pi.
    """
    available_cameras = []
    
    # Detecta o sistema operacional
    import platform
    is_windows = platform.system() == 'Windows'
    
    for i in range(max_cameras):
        try:
            # Usa DirectShow no Windows para evitar erros do obsensor
            # No Raspberry Pi, usa a API padrão
            if is_windows:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(i)
                
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if cap is not None and cap.isOpened():
                # Testa se consegue ler um frame
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    available_cameras.append(i)
                cap.release()
        except Exception as e:
            # Silencia erros de câmeras não encontradas
            print(f"Erro ao testar câmera {i}: {e}")
            continue
    
    # Se não encontrar nenhuma câmera, adiciona índice 0 como padrão
    if not available_cameras:
        available_cameras.append(0)
        print("Nenhuma câmera detectada automaticamente. Usando índice 0 como padrão.")
    else:
        print(f"Câmeras detectadas: {available_cameras}")
    
    return available_cameras

def capture_image_from_camera(camera_index=0):
    """
    Captura uma única imagem da webcam especificada.
    Retorna a imagem capturada ou None em caso de erro.
    Compatível com Windows e Raspberry Pi.
    """
    try:
        # Detecta o sistema operacional
        import platform
        is_windows = platform.system() == 'Windows'
        
        # Usa DirectShow no Windows para melhor compatibilidade
        # No Raspberry Pi, usa a API padrão
        if is_windows:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_index)
        
        # Usa resolução nativa para câmeras externas (1920x1080) ou padrão para webcam interna
        if camera_index > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir a câmera {camera_index}")
            return None
        
        # Aguarda um pouco para a câmera se estabilizar
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                break
        
        # Captura a imagem final
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None and frame.size > 0:
            print(f"Imagem capturada com sucesso da câmera {camera_index}")
            return frame
        else:
            print(f"Erro: Não foi possível capturar imagem da câmera {camera_index}")
            return None
            
    except Exception as e:
        print(f"Erro ao capturar imagem da câmera {camera_index}: {e}")
        return None

def cv2_to_tk(img_bgr, max_w=None, max_h=None):
    """
    Converte imagem OpenCV BGR para formato Tkinter PhotoImage,
    redimensionando para caber em max_w x max_h, se fornecido.
    
    Otimizada para evitar lógica duplicada de redimensionamento.
    """
    # Validação de entrada
    if img_bgr is None or img_bgr.size == 0:
        return None, 1.0
    
    h, w = img_bgr.shape[:2]
    scale = 1.0

    # Calcula escala necessária de forma otimizada
    if max_w and w > max_w:
        scale = min(scale, max_w / w)
    if max_h and h > max_h:
        scale = min(scale, max_h / h)

    # Redimensiona apenas se necessário
    if scale != 1.0:
        new_w = max(1, int(w * scale))  # Garante dimensão mínima
        new_h = max(1, int(h * scale))
        
        try:
            # Usa INTER_AREA para redução e INTER_LINEAR para ampliação
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            img_bgr_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interpolation)
        except cv2.error as e:
             print(f"Erro ao redimensionar imagem: {e}. Dimensões: ({new_w}x{new_h})")
             return None, 1.0
    else:
        img_bgr_resized = img_bgr

    # Conversão para Tkinter
    try:
        img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
        photo_image = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        return photo_image, scale
    except Exception as e:
        print(f"Erro ao converter imagem para Tkinter: {e}")
        return None, scale




# Inicialização otimizada do detector ORB
try:
    # Configurações otimizadas para melhor performance
    orb = cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_N_LEVELS,
        edgeThreshold=31,  # Reduz detecção em bordas para melhor performance
        firstLevel=0,      # Nível inicial da pirâmide
        WTA_K=2,          # Número de pontos para comparação
        scoreType=cv2.ORB_HARRIS_SCORE,  # Usa Harris score para melhor qualidade
        patchSize=31      # Tamanho do patch para descritores
    )
    print("Detector ORB inicializado com sucesso (configuração otimizada).")
except Exception as e:
    print(f"Erro ao inicializar ORB: {e}. O registro de imagem não funcionará.")
    orb = None

# Cache para descritores de imagem de referência (otimização)
_ref_image_cache = {
    'image_hash': None,
    'keypoints': None,
    'descriptors': None,
    'gray_image': None
}


def find_image_transform(img_ref, img_test):
    """
    Encontra a transformação entre duas imagens usando ORB.
    
    Otimizada com:
    - Cache para imagem de referência
    - Validação de entrada mais eficiente
    - Matching otimizado
    
    Retorna: (matriz_homografia, matches_count, error_message)
    """
    global _ref_image_cache
    
    if orb is None:
        error_msg = "Detector ORB não disponível."
        print(error_msg)
        return None, 0, error_msg
    
    # Validação de entrada otimizada
    if img_ref is None or img_test is None or img_ref.size == 0 or img_test.size == 0:
        error_msg = "Imagens de referência ou teste inválidas."
        print(error_msg)
        return None, 0, error_msg
    
    try:
        # Converte para escala de cinza
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY) if len(img_ref.shape) == 3 else img_ref
        gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY) if len(img_test.shape) == 3 else img_test
        
        # === CACHE PARA IMAGEM DE REFERÊNCIA ===
        # Calcula hash simples da imagem de referência
        ref_hash = hash(gray_ref.tobytes())
        
        # Verifica se pode usar cache
        if (_ref_image_cache['image_hash'] == ref_hash and 
            _ref_image_cache['keypoints'] is not None and 
            _ref_image_cache['descriptors'] is not None):
            # Usa dados do cache
            kp_ref = _ref_image_cache['keypoints']
            desc_ref = _ref_image_cache['descriptors']
            print("Usando cache para imagem de referência")
        else:
            # Detecta keypoints e descritores para referência
            kp_ref, desc_ref = orb.detectAndCompute(gray_ref, None)
            # Atualiza cache
            _ref_image_cache.update({
                'image_hash': ref_hash,
                'keypoints': kp_ref,
                'descriptors': desc_ref,
                'gray_image': gray_ref.copy()
            })
            print("Cache atualizado para imagem de referência")
        
        # Detecta keypoints e descritores para teste (sempre novo)
        kp_test, desc_test = orb.detectAndCompute(gray_test, None)
        
        # Validação de descritores
        if desc_ref is None or desc_test is None:
            error_msg = "Não foi possível extrair descritores de uma das imagens."
            print(error_msg)
            return None, 0, error_msg
        
        if len(desc_ref) < 4 or len(desc_test) < 4:
            error_msg = f"Poucos descritores encontrados: ref={len(desc_ref)}, test={len(desc_test)}"
            print(error_msg)
            return None, 0, error_msg
        
        # === MATCHING OTIMIZADO ===
        # Usa BFMatcher otimizado com crossCheck
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc_ref, desc_test)
        
        if len(matches) < 4:
            error_msg = f"Poucos matches encontrados: {len(matches)}"
            print(error_msg)
            return None, len(matches), error_msg
        
        # Ordena matches por distância e filtra os melhores
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Usa apenas os melhores matches para melhor performance
        max_matches = min(len(matches), 100)  # Limita a 100 melhores matches
        good_matches = matches[:max_matches]
        
        # Extrai pontos correspondentes
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # === CÁLCULO DE HOMOGRAFIA OTIMIZADO ===
        # Usa parâmetros otimizados para RANSAC
        M, mask = cv2.findHomography(
            src_pts, dst_pts, 
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,  # Threshold mais restritivo
            maxIters=2000,              # Máximo de iterações
            confidence=0.99             # Confiança desejada
        )
        
        if M is None:
            error_msg = "Não foi possível calcular a homografia."
            print(error_msg)
            return None, len(good_matches), error_msg
        
        inliers_count = np.sum(mask)
        inlier_ratio = inliers_count / len(good_matches)
        
        print(f"Homografia calculada: {inliers_count}/{len(good_matches)} inliers ({inlier_ratio:.2%})")
        return M, inliers_count, None
        
    except Exception as e:
        error_msg = f"Erro em find_image_transform: {e}"
        print(error_msg)
        return None, 0, error_msg


def transform_rectangle(rect, M, img_shape):
    """
    Transforma um retângulo usando uma matriz de homografia.
    rect: (x, y, w, h)
    M: matriz de homografia 3x3
    img_shape: (height, width) da imagem de destino
    Retorna: (x, y, w, h) transformado ou None se inválido
    """
    if M is None:
        return None
    
    x, y, w, h = rect
    
    # Define os 4 cantos do retângulo
    corners = np.float32([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ]).reshape(-1, 1, 2)
    
    try:
        # Transforma os cantos
        transformed_corners = cv2.perspectiveTransform(corners, M)
        
        # Calcula o bounding box dos cantos transformados
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Garante que está dentro dos limites da imagem
        img_h, img_w = img_shape[:2]
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(img_w, int(x_max))
        y_max = min(img_h, int(y_max))
        
        new_w = x_max - x_min
        new_h = y_max - y_min
        
        if new_w <= 0 or new_h <= 0:
            print(f"Retângulo transformado inválido: ({x_min}, {y_min}, {new_w}, {new_h})")
            return None
        
        return (x_min, y_min, new_w, new_h)
        
    except Exception as e:
        print(f"Erro ao transformar retângulo: {e}")
        return None


def check_slot(img_test, slot_data, M):
    """
    Verifica um slot na imagem de teste.
    Retorna: (passou, correlation, pixels, corners, bbox, log_msgs)
    """
    log_msgs = []
    corners = None
    bbox = [0, 0, 0, 0]
    
    try:
        slot_type = slot_data.get('tipo', 'clip')
        x, y, w, h = slot_data['x'], slot_data['y'], slot_data['w'], slot_data['h']
        
        # Calcula os cantos originais do slot
        original_corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        
        # Transforma o retângulo se temos matriz de homografia
        if M is not None:
            # Transforma os cantos usando a matriz de homografia
            corners_array = np.array(original_corners, dtype=np.float32).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners_array, M)
            corners = [(int(pt[0][0]), int(pt[0][1])) for pt in transformed_corners]
            
            # Calcula bounding box dos cantos transformados
            x_coords = [pt[0] for pt in corners]
            y_coords = [pt[1] for pt in corners]
            x, y = max(0, min(x_coords)), max(0, min(y_coords))
            w = min(img_test.shape[1] - x, max(x_coords) - x)
            h = min(img_test.shape[0] - y, max(y_coords) - y)
            
            log_msgs.append(f"Slot transformado para ({x}, {y}, {w}, {h})")
        else:
            corners = original_corners
            log_msgs.append("Usando coordenadas originais (sem transformação)")
        
        bbox = [x, y, w, h]
        
        # Verifica se a ROI está dentro dos limites da imagem
        if x < 0 or y < 0 or x + w > img_test.shape[1] or y + h > img_test.shape[0]:
            log_msgs.append(f"ROI fora dos limites da imagem: ({x}, {y}, {w}, {h})")
            return False, 0.0, 0, corners, bbox, log_msgs
        
        # Extrai ROI
        roi = img_test[y:y+h, x:x+w]
        if roi.size == 0:
            log_msgs.append("ROI vazia")
            return False, 0.0, 0, corners, bbox, log_msgs
        
        if slot_type == 'clip':
            # Verifica método de detecção
            detection_method = slot_data.get('detection_method', 'template_matching')
            
            if detection_method == 'histogram_analysis':
                # === ANÁLISE POR HISTOGRAMA ===
                try:
                    # Calcula histograma da ROI em HSV
                    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    
                    # Parâmetros do histograma
                    h_bins = 50
                    s_bins = 60
                    hist_range = [0, 180, 0, 256]  # H: 0-179, S: 0-255
                    
                    # Calcula histograma 2D (H-S)
                    hist = cv2.calcHist([roi_hsv], [0, 1], None, [h_bins, s_bins], hist_range)
                    
                    # Normaliza histograma
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    
                    # Calcula métricas do histograma
                    hist_sum = np.sum(hist)
                    hist_mean = np.mean(hist)
                    hist_std = np.std(hist)
                    hist_max = np.max(hist)
                    
                    # Calcula entropia do histograma
                    hist_flat = hist.flatten()
                    hist_flat = hist_flat[hist_flat > 0]  # Remove zeros
                    entropy = -np.sum(hist_flat * np.log2(hist_flat + 1e-10))
                    
                    # Score baseado em múltiplas métricas
                    # Combina entropia (diversidade de cores) e distribuição
                    entropy_score = min(entropy / 10.0, 1.0)  # Normaliza entropia
                    distribution_score = min(hist_std * 10, 1.0)  # Penaliza distribuições muito uniformes
                    intensity_score = min(hist_max * 2, 1.0)  # Considera picos de intensidade
                    
                    # Score final combinado
                    histogram_score = (entropy_score * 0.5 + distribution_score * 0.3 + intensity_score * 0.2)
                    
                    # Usa limiar personalizado do slot ou padrão
                    if 'correlation_threshold' in slot_data:
                        threshold = slot_data.get('correlation_threshold', 0.3)
                        threshold_source = "correlation_threshold"
                    else:
                        threshold = slot_data.get('detection_threshold', 30.0) / 100.0  # Converte % para decimal
                        threshold_source = "detection_threshold"
                    
                    passou = histogram_score >= threshold
                    
                    log_msgs.append(f"Histograma: {histogram_score:.3f} (limiar: {threshold:.2f} [{threshold_source}], entropia: {entropy:.2f}, std: {hist_std:.3f})")
                    return passou, histogram_score, 0, corners, bbox, log_msgs
                    
                except Exception as e:
                    log_msgs.append(f"Erro na análise por histograma: {str(e)}")
                    return False, 0.0, 0, corners, bbox, log_msgs
            
            else:  # template_matching (método padrão)
                # === TEMPLATE MATCHING PARA CLIPS ===
                template_path = slot_data.get('template_path')
                if not template_path or not Path(template_path).exists():
                    log_msgs.append("Template não encontrado")
                    return False, 0.0, 0, corners, bbox, log_msgs
                
                template = cv2.imread(str(template_path))
                if template is None:
                    log_msgs.append("Erro ao carregar template")
                    return False, 0.0, 0, corners, bbox, log_msgs
                
                # === TEMPLATE MATCHING OTIMIZADO ===
                correlation_threshold = slot_data.get('correlation_threshold', 0.7)
                template_method_str = slot_data.get('template_method', 'TM_CCOEFF_NORMED')
                scale_tolerance = slot_data.get('scale_tolerance', 10.0) / 100.0
                
                # Mapeamento otimizado de métodos
                method_map = {
                    'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
                    'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
                    'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
                }
                template_method = method_map.get(template_method_str, cv2.TM_CCOEFF_NORMED)
                
                max_val = 0.0
                best_scale = 1.0
                
                # Otimização: reduz número de escalas testadas
                if scale_tolerance > 0:
                    # Testa apenas 3 escalas para melhor performance
                    scales = [1.0 - scale_tolerance, 1.0, 1.0 + scale_tolerance]
                else:
                    scales = [1.0]  # Apenas escala original
                
                # Pré-converte template para escala de cinza se necessário (otimização)
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            for scale in scales:
                # Calcula dimensões da escala
                scaled_w = int(template_gray.shape[1] * scale)
                scaled_h = int(template_gray.shape[0] * scale)
                
                # Validação de dimensões otimizada
                if (scaled_w <= 0 or scaled_h <= 0 or 
                    scaled_w > roi_gray.shape[1] or scaled_h > roi_gray.shape[0]):
                    continue
                
                # Redimensiona template (usa INTER_AREA para redução, INTER_LINEAR para ampliação)
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                scaled_template = cv2.resize(template_gray, (scaled_w, scaled_h), interpolation=interpolation)
                
                # Template matching otimizado (usa imagens em escala de cinza)
                result = cv2.matchTemplate(roi_gray, scaled_template, template_method)
                
                # Extrai valor de correlação
                if template_method == cv2.TM_SQDIFF_NORMED:
                    min_val, _, _, _ = cv2.minMaxLoc(result)
                    current_val = 1.0 - min_val  # Inverte para SQDIFF
                else:
                    _, current_val, _, _ = cv2.minMaxLoc(result)
                
                # Atualiza melhor resultado
                if current_val > max_val:
                    max_val = current_val
                    best_scale = scale
            
            # Usa limiar personalizado do slot ou padrão
            # Prioridade: correlation_threshold > detection_threshold > padrão global
            if 'correlation_threshold' in slot_data:
                threshold = slot_data.get('correlation_threshold', 0.1)
                threshold_source = "correlation_threshold"
            else:
                threshold = slot_data.get('detection_threshold', 70.0) / 100.0  # Converte % para decimal
                threshold_source = "detection_threshold"
            
            passou = max_val >= threshold
            
            log_msgs.append(f"Correlação: {max_val:.3f} (limiar: {threshold:.2f} [{threshold_source}], escala: {best_scale:.2f}, método: {template_method_str})")
            return passou, max_val, 0, corners, bbox, log_msgs
        
        else:  # fita - tipo removido, apenas clips são suportados
            log_msgs.append("Tipo 'fita' não é mais suportado - apenas template matching para 'clip'")
            return False, 0.0, 0, corners, bbox, log_msgs
    
    except Exception as e:
        log_msgs.append(f"Erro: {str(e)}")
        print(f"Erro em check_slot: {e}")
        return False, 0.0, 0, corners, bbox, log_msgs


class EditSlotDialog(Toplevel):
    def __init__(self, parent, slot_data, malha_frame_instance):
        """Inicializa diálogo de edição com configuração otimizada"""
        try:
            super().__init__(parent)
            
            # Verifica se os parâmetros são válidos
            if not parent or not slot_data or not malha_frame_instance:
                raise ValueError("Parâmetros inválidos para EditSlotDialog")
            
            # Verifica se o slot_data tem as chaves necessárias
            basic_keys = ['id', 'x', 'y', 'w', 'h', 'tipo']
            required_keys = basic_keys.copy()
            
            # Para slots do tipo 'clip', adiciona campos específicos
            if slot_data.get('tipo') == 'clip':
                clip_keys = ['cor', 'detection_threshold']
                required_keys.extend(clip_keys)
            
            missing_keys = [key for key in required_keys if key not in slot_data]
            if missing_keys:
                raise ValueError(f"Dados do slot incompletos. Chaves ausentes: {missing_keys}")
            
            # === INICIALIZAÇÃO DE DADOS ===
            self.slot_data = slot_data.copy()
            self.malha_frame = malha_frame_instance
            self.result = None
            self._is_destroyed = False
            
            # Inicializa configurações de estilo se não existirem
            if 'style_config' not in self.slot_data:
                self.slot_data['style_config'] = {
                    'bg_color': '#1E1E1E',  # Cor de fundo padrão
                    'text_color': '#FFFFFF',  # Cor do texto padrão
                    'ok_color': '#95E1D3',  # Cor para OK
                    'ng_color': '#F38BA8',  # Cor para NG
                    'selection_color': '#FFE66D',  # Cor de seleção
                    'ok_font': 'Arial 12 bold',  # Fonte para OK
                    'ng_font': 'Arial 12 bold'   # Fonte para NG
                }
            
            # === CONFIGURAÇÃO DA JANELA ===
            self.title(f"Editando Slot {slot_data['id']}")
            self.geometry("400x650")
            self.resizable(False, False)
            self.configure(bg='#2E2E2E')  # Cor de fundo escura para toda a janela
            
            # Configuração modal otimizada
            self.transient(parent)
            self.protocol("WM_DELETE_WINDOW", self.cancel)
            
            # Verifica se a janela pai ainda existe
            if not parent.winfo_exists():
                raise RuntimeError("Janela pai não existe mais")
            
            try:
                # === CONFIGURAÇÃO DA INTERFACE ===
                print("Iniciando setup_ui...")
                self.setup_ui()
                print("setup_ui concluído")
                
                print("Iniciando load_slot_data...")
                self.load_slot_data()
                print("load_slot_data concluído")
                
                print("Iniciando center_window...")
                self.center_window()
                print("center_window concluído")
                
                # Aplica modalidade diretamente
                print("Aplicando modalidade...")
                self.apply_modal_grab()
                print("Modalidade aplicada")
                
            except Exception as ui_error:
                print(f"Erro ao configurar interface: {ui_error}")
                import traceback
                traceback.print_exc()
                raise ui_error
                
        except Exception as e:
            print(f"Erro ao inicializar EditSlotDialog: {e}")
            import traceback
            traceback.print_exc()
            
            # Tenta mostrar erro se possível
            try:
                messagebox.showerror("Erro", f"Erro ao abrir editor: {str(e)}")
            except:
                print("Não foi possível mostrar messagebox de erro")
            
            # Destrói a janela se foi criada
            try:
                if hasattr(self, 'winfo_exists') and self.winfo_exists():
                    self.destroy()
            except:
                pass
            
            # Re-levanta a exceção para que o chamador saiba que houve erro
            raise e
    
    def apply_modal_grab(self):
        """Aplica grab_set() após a janela estar completamente inicializada"""
        try:
            # Temporariamente removendo grab_set() para evitar travamentos
            # self.grab_set()
            self.focus_set()
            print("Modal grab aplicado com sucesso (sem grab_set)")
        except Exception as e:
            print(f"Erro ao aplicar modal grab: {e}")
    
    def center_window(self):
        try:
            print("Iniciando centralização da janela...")
            self.update_idletasks()
            
            # Centralização direta sem delay
            width = 500  # largura padrão
            height = 400  # altura padrão
            
            x = (self.winfo_screenwidth() // 2) - (width // 2)
            y = (self.winfo_screenheight() // 2) - (height // 2)
            
            self.geometry(f"{width}x{height}+{x}+{y}")
            print(f"Janela centralizada: {width}x{height}+{x}+{y}")
        except Exception as e:
            print(f"Erro ao centralizar janela: {e}")
    
    def setup_ui(self):
        """Configura interface otimizada do diálogo de edição"""
        try:
            print("Criando frame principal...")
            # === FRAME PRINCIPAL ===
            main_frame = ttk.Frame(self)
            main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
            print("Frame principal criado")
            
            print("Criando seção de informações...")
            # === INFORMAÇÕES DO SLOT ===
            info_frame = ttk.LabelFrame(main_frame, text="Informações do Slot")
            info_frame.pack(fill=X, pady=(0, 10))
            
            # Labels de informação otimizadas
            slot_info = f"ID: {self.slot_data['id']} | Tipo: {self.slot_data['tipo']}"
            ttk.Label(info_frame, text=slot_info).pack(anchor="w", padx=5, pady=5)
            print("Seção de informações criada")
            
            print("Criando seção de edição de malha...")
            # === EDIÇÃO DE MALHA ===
            mesh_frame = ttk.LabelFrame(main_frame, text="Posição e Dimensões")
            mesh_frame.pack(fill=X, pady=(0, 10))
            
            mesh_grid = ttk.Frame(mesh_frame)
            mesh_grid.pack(fill=X, padx=5, pady=5)
            
            # Inicialização otimizada de variáveis
            print("Inicializando variáveis...")
            self.x_var = StringVar()
            self.y_var = StringVar()
            self.w_var = StringVar()
            self.h_var = StringVar()
            self.detection_threshold_var = StringVar()
            print("Variáveis inicializadas")
            
            print("Criando campos de entrada...")
            # Grid otimizado para campos de entrada
            fields = [
                ("X:", self.x_var, 0, 0),
                ("Y:", self.y_var, 0, 2),
                ("Largura:", self.w_var, 1, 0),
                ("Altura:", self.h_var, 1, 2)
            ]
            
            for label_text, var, row, col in fields:
                ttk.Label(mesh_grid, text=label_text).grid(
                    row=row, column=col, sticky="w", padx=(0, 5), pady=2
                )
                ttk.Entry(mesh_grid, textvariable=var, width=8).grid(
                    row=row, column=col+1, padx=(0, 10), pady=2
                )
            print("Campos de entrada criados")
            
            print("Criando seção de configurações...")
            # === CONFIGURAÇÕES BÁSICAS ===
            config_frame = ttk.LabelFrame(main_frame, text="Configurações")
            config_frame.pack(fill=X, pady=(0, 10))
            
            # Campo de limiar otimizado
            threshold_frame = ttk.Frame(config_frame)
            threshold_frame.pack(fill=X, padx=5, pady=5)
            
            ttk.Label(threshold_frame, text="Limiar de Detecção (%):").pack(side=LEFT)
            ttk.Entry(threshold_frame, textvariable=self.detection_threshold_var, width=10).pack(side=LEFT, padx=(5, 0))
            print("Seção de configurações criada")
            
            print("Criando botões de ação...")
            # === BOTÕES DE AÇÃO ===
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=X, pady=(10, 0))
            
            # Botões otimizados
            ttk.Button(button_frame, text="Salvar", command=self.save_changes).pack(side=LEFT, padx=(0, 5))
            ttk.Button(button_frame, text="Cancelar", command=self.cancel).pack(side=LEFT)
            print("Botões de ação criados")
        
            print("setup_ui concluído com sucesso")
            
        except Exception as e:
            print(f"Erro na configuração da UI: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erro", f"Falha ao configurar interface: {e}")
            self.destroy()
    
    def load_slot_data(self):
        try:
            print(f"Carregando dados do slot: {self.slot_data}")
            # Carrega os dados do slot nos campos da interface
            if self.slot_data:
                self.x_var.set(str(self.slot_data.get('x', 0)))
                self.y_var.set(str(self.slot_data.get('y', 0)))
                self.w_var.set(str(self.slot_data.get('w', 100)))
                self.h_var.set(str(self.slot_data.get('h', 100)))
                self.detection_threshold_var.set(str(self.slot_data.get('detection_threshold', 50)))
                print("Dados carregados com sucesso nos campos")
            else:
                print("Nenhum dado de slot para carregar")
        except Exception as e:
            print(f"Erro ao carregar dados do slot: {e}")
            messagebox.showerror("Erro", f"Erro ao carregar dados do slot: {str(e)}")
    
    def get_hex_color(self, bgr):
        """Converte cor BGR para hexadecimal."""
        try:
            b, g, r = bgr
            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return "#000000"
    
    def update_template_visibility(self, event=None):
        """Mostra ou oculta o template do clip."""
        if self.slot_data['tipo'] != 'clip':
            return
        
        template_path = self.slot_data.get('template_path')
        if not template_path or not Path(template_path).exists():
            messagebox.showwarning("Aviso", "Template não encontrado.")
            return
        
        if self.show_template_var.get():
            # Mostra template
            template = cv2.imread(str(template_path))
            if template is not None:
                cv2.imshow(f"Template - Slot {self.slot_data['id']}", template)
        else:
            # Oculta template
            cv2.destroyWindow(f"Template - Slot {self.slot_data['id']}")
    
    def pick_new_color(self):
        """Função simplificada para escolher nova cor."""
        try:
            messagebox.showinfo("Info", "Função de seleção de cor simplificada.")
        except Exception as e:
            print(f"Erro na seleção de cor: {e}")
            messagebox.showerror("Erro", f"Erro na seleção de cor: {str(e)}")
    
    def save_changes(self):
        """Salva as alterações feitas no slot."""
        try:
            print(f"\n=== SALVANDO ALTERAÇÕES DO SLOT {self.slot_data.get('id', 'N/A')} ===")
            
            # Validação e conversão dos valores
            try:
                x_val = int(self.x_var.get().strip())
                y_val = int(self.y_var.get().strip())
                w_val = int(self.w_var.get().strip())
                h_val = int(self.h_var.get().strip())
                threshold_val = float(self.detection_threshold_var.get().strip())
                
                # Validações básicas
                if w_val <= 0 or h_val <= 0:
                    raise ValueError("Largura e altura devem ser maiores que zero")
                if threshold_val < 0 or threshold_val > 100:
                    raise ValueError("Limiar deve estar entre 0 e 100")
                    
            except ValueError as ve:
                messagebox.showerror("Erro de Validação", f"Valores inválidos: {str(ve)}")
                return
            
            # Salva alterações de malha (posição e tamanho)
            self.slot_data['x'] = x_val
            self.slot_data['y'] = y_val
            self.slot_data['w'] = w_val
            self.slot_data['h'] = h_val
            
            # Salva limiar de detecção
            self.slot_data['detection_threshold'] = threshold_val
            
            print(f"Dados salvos: posição ({self.slot_data['x']},{self.slot_data['y']}), tamanho ({self.slot_data['w']},{self.slot_data['h']}), limiar {self.slot_data['detection_threshold']}%")
            
            # Chama o método update_slot_data da instância malha_frame
            print("Chamando update_slot_data...")
            self.malha_frame.update_slot_data(self.slot_data)
            print("Alterações salvas com sucesso!")
            
            self.destroy()
        
        except Exception as e:
            print(f"ERRO ao salvar alterações: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erro", f"Erro ao salvar alterações: {str(e)}")
    
    def cancel(self):
        """Cancela a edição com proteções contra travamentos."""
        try:
            print("Cancelando edição...")
            
            # Verifica se a janela já foi destruída
            if hasattr(self, '_is_destroyed') and self._is_destroyed:
                print("Janela já foi destruída anteriormente")
                return
            
            # Marca como destruída para evitar múltiplas chamadas
            self._is_destroyed = True
            
            # Remove grab modal se estiver ativo
            try:
                self.grab_release()
            except:
                pass
            
            # Fecha janela de template se estiver aberta
            if hasattr(self, 'slot_data') and self.slot_data and self.slot_data.get('tipo') == 'clip':
                try:
                    cv2.destroyWindow(f"Template - Slot {self.slot_data['id']}")
                except:
                    pass  # Ignora erro se janela não existir
            
            # Limpa referências
            self.result = None
            if hasattr(self, 'malha_frame'):
                self.malha_frame = None
            
            print("Destruindo janela...")
            
            # Verifica se a janela ainda existe antes de destruir
            if self.winfo_exists():
                self.destroy()
                print("Janela destruída com sucesso")
            else:
                print("Janela já foi destruída")
                
        except Exception as e:
            print(f"Erro ao cancelar: {e}")
            import traceback
            traceback.print_exc()
            
            # Tentativa final de destruir a janela
            try:
                if hasattr(self, 'winfo_exists') and self.winfo_exists():
                    self.destroy()
            except:
                print("Não foi possível destruir a janela na tentativa final")
                pass


class SlotTrainingDialog(Toplevel):
    """Diálogo para treinamento de slots com feedback OK/NG."""
    
    def __init__(self, parent, slot_data, montagem_instance):
        super().__init__(parent)
        self.slot_data = slot_data
        self.montagem_instance = montagem_instance
        self.training_samples = []  # Lista de amostras de treinamento
        
        # Define o diretório para salvar as amostras
        template_path = self.slot_data.get('template_path')
        if template_path:
            template_dir = os.path.dirname(template_path)
            self.samples_dir = os.path.join(template_dir, f"slot_{slot_data['id']}_samples")
            
            # Cria diretórios se não existirem
            os.makedirs(os.path.join(self.samples_dir, "ok"), exist_ok=True)
            os.makedirs(os.path.join(self.samples_dir, "ng"), exist_ok=True)
        else:
            self.samples_dir = None
            print("AVISO: Não foi possível definir o diretório de amostras (template_path não definido)")
        
        self.title(f"Treinamento - Slot {slot_data['id']}")
        self.geometry("800x600")
        self.resizable(True, True)
        
        # Variáveis
        self.current_image = None
        self.current_roi = None
        
        self.setup_ui()
        self.center_window()
        self.apply_modal_grab()
        
    def apply_modal_grab(self):
        """Aplica grab modal para manter foco na janela."""
        self.transient(self.master)
        self.grab_set()
        
    def center_window(self):
        """Centraliza a janela na tela."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        
    def setup_ui(self):
        """Configura a interface do diálogo de treinamento."""
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(main_frame, text=f"🎯 Treinamento do Slot {self.slot_data['id']}", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 15))
        
        # Frame superior - controles
        controls_frame = ttk.LabelFrame(main_frame, text="📷 Controles de Captura")
        controls_frame.pack(fill=X, pady=(0, 10))
        
        # Botões de captura
        capture_frame = ttk.Frame(controls_frame)
        capture_frame.pack(fill=X, padx=10, pady=10)
        
        self.btn_capture_webcam = ttk.Button(capture_frame, text="📷 Capturar da Webcam", 
                                           command=self.capture_from_webcam, width=20)
        self.btn_capture_webcam.pack(side=LEFT, padx=(0, 10))
        
        self.btn_load_image = ttk.Button(capture_frame, text="📁 Carregar Imagem", 
                                       command=self.load_image_file, width=20)
        self.btn_load_image.pack(side=LEFT, padx=(0, 10))
        
        # Botão para limpar histórico
        self.btn_clear_history = ttk.Button(capture_frame, text="🗑️ Limpar Histórico", 
                                          command=self.clear_training_history, width=20)
        self.btn_clear_history.pack(side=RIGHT)
        
        # Frame central dividido em duas colunas
        central_frame = ttk.Frame(main_frame)
        central_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # Coluna esquerda - visualização atual
        left_frame = ttk.LabelFrame(central_frame, text="🖼️ Visualização Atual")
        left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
        
        # Canvas para exibir imagem atual
        self.canvas = Canvas(left_frame, bg="#1E1E1E", width=400, height=300)
        self.canvas.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Botões de feedback para imagem atual
        feedback_buttons_frame = ttk.Frame(left_frame)
        feedback_buttons_frame.pack(fill=X, padx=10, pady=(0, 10))
        
        self.btn_mark_ok = ttk.Button(feedback_buttons_frame, text="✅ Marcar como OK", 
                                    command=self.mark_as_ok, state=DISABLED, width=15)
        self.btn_mark_ok.pack(side=LEFT, padx=(0, 10))
        
        self.btn_mark_ng = ttk.Button(feedback_buttons_frame, text="❌ Marcar como NG", 
                                    command=self.mark_as_ng, state=DISABLED, width=15)
        self.btn_mark_ng.pack(side=LEFT)
        
        # Coluna direita - histórico de treinamento
        right_frame = ttk.LabelFrame(central_frame, text="📊 Histórico de Treinamento")
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=(5, 0))
        
        # Notebook para separar OK e NG
        self.history_notebook = ttk.Notebook(right_frame)
        self.history_notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Aba OK
        self.ok_frame = ttk.Frame(self.history_notebook)
        self.history_notebook.add(self.ok_frame, text="✅ Amostras OK (0)")
        
        # Scrollable frame para amostras OK
        self.ok_canvas = Canvas(self.ok_frame, bg="#f0f8f0")
        self.ok_scrollbar = ttk.Scrollbar(self.ok_frame, orient="vertical", command=self.ok_canvas.yview)
        self.ok_scrollable_frame = ttk.Frame(self.ok_canvas)
        
        self.ok_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.ok_canvas.configure(scrollregion=self.ok_canvas.bbox("all"))
        )
        
        self.ok_canvas.create_window((0, 0), window=self.ok_scrollable_frame, anchor="nw")
        self.ok_canvas.configure(yscrollcommand=self.ok_scrollbar.set)
        
        self.ok_canvas.pack(side="left", fill="both", expand=True)
        self.ok_scrollbar.pack(side="right", fill="y")
        
        # Adiciona suporte para scroll com mouse wheel
        self.ok_canvas.bind("<MouseWheel>", lambda e: self.ok_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Aba NG
        self.ng_frame = ttk.Frame(self.history_notebook)
        self.history_notebook.add(self.ng_frame, text="❌ Amostras NG (0)")
        
        # Scrollable frame para amostras NG
        self.ng_canvas = Canvas(self.ng_frame, bg="#f8f0f0")
        self.ng_scrollbar = ttk.Scrollbar(self.ng_frame, orient="vertical", command=self.ng_canvas.yview)
        self.ng_scrollable_frame = ttk.Frame(self.ng_canvas)
        
        self.ng_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.ng_canvas.configure(scrollregion=self.ng_canvas.bbox("all"))
        )
        
        self.ng_canvas.create_window((0, 0), window=self.ng_scrollable_frame, anchor="nw")
        self.ng_canvas.configure(yscrollcommand=self.ng_scrollbar.set)
        
        self.ng_canvas.pack(side="left", fill="both", expand=True)
        self.ng_scrollbar.pack(side="right", fill="y")
        
        # Adiciona suporte para scroll com mouse wheel
        self.ng_canvas.bind("<MouseWheel>", lambda e: self.ng_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Frame inferior - informações e ações
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=X, pady=(10, 0))
        
        # Informações de treinamento
        info_frame = ttk.LabelFrame(bottom_frame, text="📈 Estatísticas")
        info_frame.pack(fill=X, pady=(0, 10))
        
        stats_frame = ttk.Frame(info_frame)
        stats_frame.pack(fill=X, padx=10, pady=10)
        
        self.info_label = ttk.Label(stats_frame, text="Amostras coletadas: 0 OK, 0 NG", 
                                   font=("Arial", 10, "bold"))
        self.info_label.pack(side=LEFT)
        
        self.threshold_label = ttk.Label(stats_frame, text="Threshold atual: N/A", 
                                        font=("Arial", 10))
        self.threshold_label.pack(side=RIGHT)
        
        # Botões de ação
        action_frame = ttk.Frame(bottom_frame)
        action_frame.pack(fill=X)
        
        self.btn_apply_training = ttk.Button(action_frame, text="🚀 Aplicar Treinamento", 
                                           command=self.apply_training, state=DISABLED, width=20)
        self.btn_apply_training.pack(side=LEFT, padx=(0, 10))
        
        self.btn_cancel = ttk.Button(action_frame, text="❌ Cancelar", 
                                   command=self.cancel, width=15)
        self.btn_cancel.pack(side=RIGHT)
        
        # Atualiza threshold atual se existir
        current_threshold = self.slot_data.get('correlation_threshold', 
                                             self.slot_data.get('detection_threshold', 'N/A'))
        if current_threshold != 'N/A':
            self.threshold_label.config(text=f"Threshold atual: {current_threshold:.3f}")
        
        # Carrega amostras existentes se houver
        self.load_existing_samples()
        
    def capture_from_webcam(self):
        """Captura imagem da webcam para treinamento."""
        try:
            # Usa a mesma função de captura da montagem
            camera_index = 0
            if hasattr(self.montagem_instance, 'camera_combo') and self.montagem_instance.camera_combo.get():
                camera_index = int(self.montagem_instance.camera_combo.get())
            
            captured_image = capture_image_from_camera(camera_index)
            
            if captured_image is not None:
                self.process_captured_image(captured_image)
            else:
                messagebox.showerror("Erro", "Falha ao capturar imagem da webcam.")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao capturar da webcam: {str(e)}")
            
    def load_image_file(self):
        """Carrega imagem de arquivo para treinamento."""
        try:
            file_path = filedialog.askopenfilename(
                title="Selecionar Imagem",
                filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff")]
            )
            
            if file_path:
                image = cv2.imread(file_path)
                if image is not None:
                    self.process_captured_image(image)
                else:
                    messagebox.showerror("Erro", "Falha ao carregar a imagem.")
                    
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar imagem: {str(e)}")
            
    def process_captured_image(self, image):
        """Processa a imagem capturada e extrai a ROI do slot."""
        try:
            self.current_image = image.copy()
            
            # Encontra a transformação entre a imagem de referência e a capturada
            if not hasattr(self.montagem_instance, 'img_original') or self.montagem_instance.img_original is None:
                messagebox.showerror("Erro", "Imagem de referência não carregada.")
                return
                
            M, inliers_count, error_msg = find_image_transform(self.montagem_instance.img_original, image)
            
            if M is None:
                messagebox.showwarning("Aviso", "Não foi possível alinhar a imagem. Usando coordenadas diretas.")
                # Usa coordenadas diretas se não conseguir alinhar
                x, y, w, h = self.slot_data['x'], self.slot_data['y'], self.slot_data['w'], self.slot_data['h']
            else:
                # Transforma as coordenadas do slot
                original_corners = np.array([[
                    [self.slot_data['x'], self.slot_data['y']], 
                    [self.slot_data['x'] + self.slot_data['w'], self.slot_data['y']],
                    [self.slot_data['x'] + self.slot_data['w'], self.slot_data['y'] + self.slot_data['h']],
                    [self.slot_data['x'], self.slot_data['y'] + self.slot_data['h']]
                ]], dtype=np.float32)
                
                transformed_corners = cv2.perspectiveTransform(original_corners, M)[0]
                
                # Calcula bounding box
                x = int(min(corner[0] for corner in transformed_corners))
                y = int(min(corner[1] for corner in transformed_corners))
                w = int(max(corner[0] for corner in transformed_corners) - x)
                h = int(max(corner[1] for corner in transformed_corners) - y)
            
            # Valida e ajusta coordenadas
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                messagebox.showerror("Erro", "ROI inválida detectada.")
                return
                
            # Extrai ROI
            self.current_roi = image[y:y+h, x:x+w].copy()
            
            # Exibe a imagem com a ROI destacada
            self.display_image_with_roi(image, x, y, w, h)
            
            # Habilita botões de feedback
            self.btn_mark_ok.config(state=NORMAL)
            self.btn_mark_ng.config(state=NORMAL)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar imagem: {str(e)}")
            
    def display_image_with_roi(self, image, roi_x, roi_y, roi_w, roi_h):
        """Exibe a imagem com a ROI destacada no canvas."""
        try:
            # Cria cópia da imagem para desenhar
            display_image = image.copy()
            
            # Desenha retângulo da ROI
            cv2.rectangle(display_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 3)
            
            # Converte para exibição no canvas
            tk_image, _ = cv2_to_tk(display_image, max_w=580, max_h=380)
            
            # Limpa canvas e exibe imagem
            self.canvas.delete("all")
            self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, 
                                   image=tk_image, anchor="center")
            
            # Mantém referência da imagem
            self.canvas.image = tk_image
            
        except Exception as e:
            print(f"Erro ao exibir imagem: {e}")
            
    def mark_as_ok(self):
        """Marca a amostra atual como OK."""
        if self.current_roi is not None:
            timestamp = datetime.now()
            self.training_samples.append({
                'roi': self.current_roi.copy(),
                'label': 'OK',
                'timestamp': timestamp
            })
            
            # Salva a amostra em disco
            if self.samples_dir:
                try:
                    # Cria diretório se não existir
                    ok_dir = os.path.join(self.samples_dir, "ok")
                    os.makedirs(ok_dir, exist_ok=True)
                    
                    # Formata o timestamp para o nome do arquivo
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    filename = f"ok_sample_{timestamp_str}.png"
                    file_path = os.path.join(ok_dir, filename)
                    
                    # Salva a imagem
                    cv2.imwrite(file_path, self.current_roi.copy())
                    print(f"Amostra OK salva em: {file_path}")
                except Exception as e:
                    print(f"Erro ao salvar amostra OK: {e}")
            
            # Adiciona ao histórico visual
            self.add_sample_to_history(self.current_roi.copy(), "OK", timestamp)
            
            self.update_info_label()
            self.update_tab_titles()
            self.reset_capture_state()
            messagebox.showinfo("Sucesso", "Amostra marcada como OK!")
            
    def mark_as_ng(self):
        """Marca a amostra atual como NG."""
        if self.current_roi is not None:
            timestamp = datetime.now()
            self.training_samples.append({
                'roi': self.current_roi.copy(),
                'label': 'NG',
                'timestamp': timestamp
            })
            
            # Salva a amostra em disco
            if self.samples_dir:
                try:
                    # Cria diretório se não existir
                    ng_dir = os.path.join(self.samples_dir, "ng")
                    os.makedirs(ng_dir, exist_ok=True)
                    
                    # Formata o timestamp para o nome do arquivo
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    filename = f"ng_sample_{timestamp_str}.png"
                    file_path = os.path.join(ng_dir, filename)
                    
                    # Salva a imagem
                    cv2.imwrite(file_path, self.current_roi.copy())
                    print(f"Amostra NG salva em: {file_path}")
                except Exception as e:
                    print(f"Erro ao salvar amostra NG: {e}")
            
            # Adiciona ao histórico visual
            self.add_sample_to_history(self.current_roi.copy(), "NG", timestamp)
            
            self.update_info_label()
            self.update_tab_titles()
            self.reset_capture_state()
            messagebox.showinfo("Sucesso", "Amostra marcada como NG!")
            
    def reset_capture_state(self):
        """Reseta o estado de captura."""
        self.current_image = None
        self.current_roi = None
        self.btn_mark_ok.config(state=DISABLED)
        self.btn_mark_ng.config(state=DISABLED)
        self.canvas.delete("all")
        
    def add_sample_to_history(self, roi_image, label, timestamp):
        """Adiciona uma amostra ao histórico visual."""
        try:
            # Redimensiona a imagem para miniatura (100x100)
            thumbnail_size = (100, 100)
            roi_resized = cv2.resize(roi_image, thumbnail_size)
            
            # Converte para formato Tkinter
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            roi_pil = Image.fromarray(roi_rgb)
            roi_tk = ImageTk.PhotoImage(roi_pil)
            
            # Seleciona o frame correto
            if label == "OK":
                parent_frame = self.ok_scrollable_frame
                bg_color = "#e8f5e8"
            else:
                parent_frame = self.ng_scrollable_frame
                bg_color = "#f5e8e8"
            
            # Cria frame para a amostra
            sample_frame = ttk.Frame(parent_frame)
            sample_frame.pack(fill=X, padx=5, pady=2)
            
            # Frame interno com borda colorida
            inner_frame = ttk.Frame(sample_frame, relief="solid", borderwidth=1)
            inner_frame.pack(fill=X, padx=2, pady=2)
            
            # Frame para imagem e informações
            content_frame = ttk.Frame(inner_frame)
            content_frame.pack(fill=X, padx=5, pady=5)
            
            # Label para a imagem
            img_label = ttk.Label(content_frame, image=roi_tk)
            img_label.image = roi_tk  # Mantém referência
            img_label.pack(side=LEFT, padx=(0, 10))
            
            # Frame para informações
            info_frame = ttk.Frame(content_frame)
            info_frame.pack(side=LEFT, fill=BOTH, expand=True)
            
            # Informações da amostra
            time_str = timestamp.strftime("%H:%M:%S")
            date_str = timestamp.strftime("%d/%m/%Y")
            
            ttk.Label(info_frame, text=f"🕒 {time_str}", font=("Arial", 9)).pack(anchor="w")
            ttk.Label(info_frame, text=f"📅 {date_str}", font=("Arial", 8)).pack(anchor="w")
            ttk.Label(info_frame, text=f"📏 {roi_image.shape[1]}x{roi_image.shape[0]}", 
                     font=("Arial", 8)).pack(anchor="w")
            
            # Botão para remover amostra
            remove_btn = ttk.Button(info_frame, text="🗑️", width=3,
                                   command=lambda: self.remove_sample_from_history(sample_frame, label, timestamp))
            remove_btn.pack(anchor="e", pady=(5, 0))
            
        except Exception as e:
            print(f"Erro ao adicionar amostra ao histórico: {e}")
    
    def remove_sample_from_history(self, sample_frame, label, timestamp):
        """Remove uma amostra do histórico visual e da lista."""
        try:
            # Remove da lista de amostras
            self.training_samples = [s for s in self.training_samples 
                                   if not (s['label'] == label and s['timestamp'] == timestamp)]
            
            # Remove o arquivo de amostra do disco
            if self.samples_dir:
                try:
                    # Determina o diretório correto (ok ou ng)
                    sample_dir = os.path.join(self.samples_dir, "ok" if label == "OK" else "ng")
                    if os.path.exists(sample_dir):
                        # Formata o timestamp para o nome do arquivo
                        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                        filename = f"{label.lower()}_sample_{timestamp_str}.png"
                        file_path = os.path.join(sample_dir, filename)
                        
                        # Remove o arquivo se existir
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Arquivo de amostra removido: {file_path}")
                except Exception as e:
                    print(f"Erro ao remover arquivo de amostra: {e}")
            
            # Remove o frame visual
            sample_frame.destroy()
            
            # Atualiza contadores
            self.update_info_label()
            self.update_tab_titles()
            
        except Exception as e:
            print(f"Erro ao remover amostra: {e}")
    
    def update_tab_titles(self):
        """Atualiza os títulos das abas com o número de amostras."""
        ok_count = sum(1 for sample in self.training_samples if sample['label'] == 'OK')
        ng_count = sum(1 for sample in self.training_samples if sample['label'] == 'NG')
        
        self.history_notebook.tab(0, text=f"✅ Amostras OK ({ok_count})")
        self.history_notebook.tab(1, text=f"❌ Amostras NG ({ng_count})")
    
    def clear_training_history(self):
        """Limpa todo o histórico de treinamento."""
        if messagebox.askyesno("Confirmar", "Deseja realmente limpar todo o histórico de treinamento?"):
            # Limpa a lista de amostras
            self.training_samples.clear()
            
            # Limpa os frames visuais
            for widget in self.ok_scrollable_frame.winfo_children():
                widget.destroy()
            for widget in self.ng_scrollable_frame.winfo_children():
                widget.destroy()
            
            # Remove arquivos de amostra do disco
            if self.samples_dir:
                try:
                    # Remove amostras OK
                    ok_dir = os.path.join(self.samples_dir, "ok")
                    if os.path.exists(ok_dir):
                        for filename in os.listdir(ok_dir):
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                os.remove(os.path.join(ok_dir, filename))
                    
                    # Remove amostras NG
                    ng_dir = os.path.join(self.samples_dir, "ng")
                    if os.path.exists(ng_dir):
                        for filename in os.listdir(ng_dir):
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                os.remove(os.path.join(ng_dir, filename))
                                
                    print("Arquivos de amostra removidos do disco")
                except Exception as e:
                    print(f"Erro ao remover arquivos de amostra: {e}")
            
            # Atualiza interface
            self.update_info_label()
            self.update_tab_titles()
            
            messagebox.showinfo("Sucesso", "Histórico de treinamento limpo!")
    
    def load_existing_samples(self):
        """Carrega amostras existentes do diretório de treinamento."""
        try:
            # Verifica se existem amostras OK
            ok_samples_dir = os.path.join(self.samples_dir, "ok")
            if os.path.exists(ok_samples_dir):
                for filename in sorted(os.listdir(ok_samples_dir)):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        sample_path = os.path.join(ok_samples_dir, filename)
                        try:
                            # Carrega a imagem
                            roi_image = cv2.imread(sample_path)
                            if roi_image is not None:
                                # Extrai timestamp do nome do arquivo
                                timestamp_str = filename.split('_')[2:4]  # ok_sample_YYYYMMDD_HHMMSS
                                if len(timestamp_str) >= 2:
                                    date_part = timestamp_str[0]
                                    time_part = timestamp_str[1].split('.')[0]  # Remove extensão
                                    timestamp = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                                else:
                                    timestamp = datetime.now()
                                
                                # Adiciona à lista de amostras
                                self.training_samples.append({
                                    'roi': roi_image,
                                    'label': 'OK',
                                    'timestamp': timestamp
                                })
                                
                                # Adiciona ao histórico visual
                                self.add_sample_to_history(roi_image, "OK", timestamp)
                        except Exception as e:
                            print(f"Erro ao carregar amostra OK {filename}: {e}")
            
            # Verifica se existem amostras NG
            ng_samples_dir = os.path.join(self.samples_dir, "ng")
            if os.path.exists(ng_samples_dir):
                for filename in sorted(os.listdir(ng_samples_dir)):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        sample_path = os.path.join(ng_samples_dir, filename)
                        try:
                            # Carrega a imagem
                            roi_image = cv2.imread(sample_path)
                            if roi_image is not None:
                                # Extrai timestamp do nome do arquivo
                                timestamp_str = filename.split('_')[2:4]  # ng_sample_YYYYMMDD_HHMMSS
                                if len(timestamp_str) >= 2:
                                    date_part = timestamp_str[0]
                                    time_part = timestamp_str[1].split('.')[0]  # Remove extensão
                                    timestamp = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                                else:
                                    timestamp = datetime.now()
                                
                                # Adiciona à lista de amostras
                                self.training_samples.append({
                                    'roi': roi_image,
                                    'label': 'NG',
                                    'timestamp': timestamp
                                })
                                
                                # Adiciona ao histórico visual
                                self.add_sample_to_history(roi_image, "NG", timestamp)
                        except Exception as e:
                            print(f"Erro ao carregar amostra NG {filename}: {e}")
            
            # Atualiza interface
            self.update_info_label()
            self.update_tab_titles()
            
        except Exception as e:
            print(f"Erro ao carregar amostras existentes: {e}")
    
    def update_info_label(self):
        """Atualiza o label de informações."""
        ok_count = sum(1 for sample in self.training_samples if sample['label'] == 'OK')
        ng_count = sum(1 for sample in self.training_samples if sample['label'] == 'NG')
        
        self.info_label.config(text=f"Amostras coletadas: {ok_count} OK, {ng_count} NG")
        
        # Habilita botão de aplicar se há amostras suficientes
        if len(self.training_samples) >= 2:  # Pelo menos 2 amostras
            self.btn_apply_training.config(state=NORMAL)
            
    def apply_training(self):
        """Aplica o treinamento coletado para melhorar a precisão do slot."""
        try:
            if len(self.training_samples) < 2:
                messagebox.showwarning("Aviso", "São necessárias pelo menos 2 amostras para treinamento.")
                return
                
            # Analisa as amostras para ajustar parâmetros
            ok_samples = [s['roi'] for s in self.training_samples if s['label'] == 'OK']
            ng_samples = [s['roi'] for s in self.training_samples if s['label'] == 'NG']
            
            if not ok_samples:
                messagebox.showwarning("Aviso", "É necessária pelo menos uma amostra OK.")
                return
                
            # Calcula novo limiar baseado nas amostras
            new_threshold = self.calculate_optimal_threshold(ok_samples, ng_samples)
            
            if new_threshold is not None:
                # Atualiza o slot com o novo limiar
                old_threshold = self.slot_data.get('correlation_threshold', self.slot_data.get('detection_threshold', 0.8))
                self.slot_data['correlation_threshold'] = new_threshold
                
                # Salva um template melhorado se há amostras OK
                if ok_samples:
                    self.update_template_with_best_sample(ok_samples)
                
                # Atualiza o slot na instância principal
                self.montagem_instance.update_slot_data(self.slot_data)
                
                # Marca modelo como modificado
                self.montagem_instance.mark_model_modified()
                
                messagebox.showinfo("Sucesso", 
                    f"Treinamento aplicado!\n\n"
                    f"Limiar anterior: {old_threshold:.3f}\n"
                    f"Novo limiar: {new_threshold:.3f}\n\n"
                    f"Amostras utilizadas: {len(self.training_samples)}")
                
                self.destroy()
            else:
                messagebox.showerror("Erro", "Não foi possível calcular novo limiar.")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao aplicar treinamento: {str(e)}")
            
    def calculate_optimal_threshold(self, ok_samples, ng_samples):
        """Calcula o limiar ótimo baseado nas amostras de treinamento."""
        try:
            # Carrega template atual
            template_path = self.slot_data.get('template_path')
            if not template_path or not Path(template_path).exists():
                return None
                
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                return None
                
            # Calcula correlações para amostras OK
            ok_correlations = []
            for roi in ok_samples:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Redimensiona template se necessário
                if roi_gray.shape != template.shape:
                    template_resized = cv2.resize(template, (roi_gray.shape[1], roi_gray.shape[0]))
                else:
                    template_resized = template
                    
                # Template matching
                result = cv2.matchTemplate(roi_gray, template_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                ok_correlations.append(max_val)
                
            # Calcula correlações para amostras NG (se existirem)
            ng_correlations = []
            for roi in ng_samples:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                if roi_gray.shape != template.shape:
                    template_resized = cv2.resize(template, (roi_gray.shape[1], roi_gray.shape[0]))
                else:
                    template_resized = template
                    
                result = cv2.matchTemplate(roi_gray, template_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                ng_correlations.append(max_val)
                
            # Calcula limiar ótimo
            if ok_correlations:
                min_ok = min(ok_correlations)
                
                if ng_correlations:
                    max_ng = max(ng_correlations)
                    # Limiar entre o máximo NG e mínimo OK
                    new_threshold = (min_ok + max_ng) / 2
                    # Garante que está dentro de limites razoáveis
                    new_threshold = max(0.3, min(0.95, new_threshold))
                else:
                    # Se não há amostras NG, usa um valor conservador
                    new_threshold = max(0.5, min_ok * 0.9)
                    
                return new_threshold
                
            return None
            
        except Exception as e:
            print(f"Erro ao calcular limiar: {e}")
            return None
            
    def update_template_with_best_sample(self, ok_samples):
        """Atualiza o template com a melhor amostra OK."""
        try:
            template_path = self.slot_data.get('template_path')
            if not template_path:
                return
                
            # Encontra a melhor amostra (maior correlação com template atual)
            current_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if current_template is None:
                return
                
            best_sample = None
            best_correlation = -1
            
            for roi in ok_samples:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Redimensiona para comparar
                if roi_gray.shape != current_template.shape:
                    roi_resized = cv2.resize(roi_gray, (current_template.shape[1], current_template.shape[0]))
                else:
                    roi_resized = roi_gray
                    
                # Calcula correlação
                result = cv2.matchTemplate(roi_resized, current_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_correlation:
                    best_correlation = max_val
                    best_sample = roi_gray
                    
            # Salva o melhor template
            if best_sample is not None:
                # Redimensiona para o tamanho original do template
                if best_sample.shape != current_template.shape:
                    best_sample = cv2.resize(best_sample, (current_template.shape[1], current_template.shape[0]))
                    
                cv2.imwrite(template_path, best_sample)
                print(f"Template atualizado com melhor amostra (correlação: {best_correlation:.3f})")
                
        except Exception as e:
            print(f"Erro ao atualizar template: {e}")
            
    def cancel(self):
        """Cancela o treinamento."""
        self.destroy()


class SystemConfigDialog(Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Configurações do Sistema")
        self.geometry("500x600")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.result = False
        self.center_window()
        self.setup_ui()
    
    def center_window(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.winfo_screenheight() // 2) - (600 // 2)
        self.geometry(f"500x600+{x}+{y}")
    
    def setup_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Configurações ORB
        orb_frame = ttk.LabelFrame(main_frame, text="Configurações ORB (Alinhamento de Imagem)")
        orb_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(orb_frame, text="Número de Features:").pack(anchor="w", padx=5, pady=2)
        self.orb_features_var = ttk.IntVar(value=ORB_FEATURES)
        features_frame = ttk.Frame(orb_frame)
        features_frame.pack(fill=X, padx=5, pady=5)
        
        self.features_scale = ttk.Scale(features_frame, from_=1000, to=10000, variable=self.orb_features_var, orient=HORIZONTAL)
        self.features_scale.pack(side=LEFT, fill=X, expand=True)
        
        self.features_label = ttk.Label(features_frame, text=f"{self.orb_features_var.get()}", width=8)
        self.features_label.pack(side=RIGHT, padx=(5, 0))
        
        def update_features_label(val):
            self.features_label.config(text=f"{int(float(val))}")
        self.features_scale.config(command=update_features_label)
        
        ttk.Label(orb_frame, text="Fator de Escala:").pack(anchor="w", padx=5, pady=(10, 2))
        self.scale_factor_var = ttk.DoubleVar(value=ORB_SCALE_FACTOR)
        scale_frame = ttk.Frame(orb_frame)
        scale_frame.pack(fill=X, padx=5, pady=5)
        
        self.scale_scale = ttk.Scale(scale_frame, from_=1.1, to=2.0, variable=self.scale_factor_var, orient=HORIZONTAL)
        self.scale_scale.pack(side=LEFT, fill=X, expand=True)
        
        self.scale_label = ttk.Label(scale_frame, text=f"{self.scale_factor_var.get():.2f}", width=8)
        self.scale_label.pack(side=RIGHT, padx=(5, 0))
        
        def update_scale_label(val):
            self.scale_label.config(text=f"{float(val):.2f}")
        self.scale_scale.config(command=update_scale_label)
        
        ttk.Label(orb_frame, text="Número de Níveis:").pack(anchor="w", padx=5, pady=(10, 2))
        self.n_levels_var = ttk.IntVar(value=ORB_N_LEVELS)
        levels_spin = ttk.Spinbox(orb_frame, from_=4, to=16, textvariable=self.n_levels_var, width=10)
        levels_spin.pack(anchor="w", padx=5, pady=5)
        
        # Configurações de Canvas
        canvas_frame = ttk.LabelFrame(main_frame, text="Configurações de Visualização")
        canvas_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(canvas_frame, text="Largura Máxima do Preview:").pack(anchor="w", padx=5, pady=2)
        self.preview_w_var = ttk.IntVar(value=PREVIEW_W)
        w_spin = ttk.Spinbox(canvas_frame, from_=400, to=1600, increment=100, textvariable=self.preview_w_var, width=10)
        w_spin.pack(anchor="w", padx=5, pady=5)
        
        ttk.Label(canvas_frame, text="Altura Máxima do Preview:").pack(anchor="w", padx=5, pady=(10, 2))
        self.preview_h_var = ttk.IntVar(value=PREVIEW_H)
        h_spin = ttk.Spinbox(canvas_frame, from_=300, to=1200, increment=100, textvariable=self.preview_h_var, width=10)
        h_spin.pack(anchor="w", padx=5, pady=5)
        
        # Configurações Padrão de Detecção
        detection_frame = ttk.LabelFrame(main_frame, text="Configurações Padrão de Detecção")
        detection_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(detection_frame, text="Limiar de Correlação Padrão (Clips):").pack(anchor="w", padx=5, pady=2)
        self.thr_corr_var = ttk.DoubleVar(value=THR_CORR)
        corr_frame = ttk.Frame(detection_frame)
        corr_frame.pack(fill=X, padx=5, pady=5)
        
        self.corr_scale = ttk.Scale(corr_frame, from_=0.1, to=1.0, variable=self.thr_corr_var, orient=HORIZONTAL)
        self.corr_scale.pack(side=LEFT, fill=X, expand=True)
        
        self.corr_label = ttk.Label(corr_frame, text=f"{self.thr_corr_var.get():.2f}", width=8)
        self.corr_label.pack(side=RIGHT, padx=(5, 0))
        
        def update_corr_label(val):
            self.corr_label.config(text=f"{float(val):.2f}")
        self.corr_scale.config(command=update_corr_label)
        
        ttk.Label(detection_frame, text="Pixels Mínimos Padrão (Template Matching):").pack(anchor="w", padx=5, pady=(10, 2))
        self.min_px_var = ttk.IntVar(value=MIN_PX)
        px_spin = ttk.Spinbox(detection_frame, from_=1, to=1000, textvariable=self.min_px_var, width=10)
        px_spin.pack(anchor="w", padx=5, pady=5)
        
        # Botões
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=X, pady=(20, 0))
        
        ttk.Button(button_frame, text="Salvar", command=self.save_config).pack(side=LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Restaurar Padrões", command=self.restore_defaults).pack(side=LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancelar", command=self.cancel).pack(side=LEFT)
    
    def save_config(self):
        """Salva as configurações do sistema"""
        global ORB_FEATURES, ORB_SCALE_FACTOR, ORB_N_LEVELS, PREVIEW_W, PREVIEW_H, THR_CORR, MIN_PX, orb
        
        try:
            # Atualiza variáveis globais
            ORB_FEATURES = int(self.orb_features_var.get())
            ORB_SCALE_FACTOR = float(self.scale_factor_var.get())
            ORB_N_LEVELS = int(self.n_levels_var.get())
            PREVIEW_W = int(self.preview_w_var.get())
            PREVIEW_H = int(self.preview_h_var.get())
            THR_CORR = float(self.thr_corr_var.get())
            MIN_PX = int(self.min_px_var.get())
            
            # Reinicializa detector ORB com novos parâmetros
            try:
                orb = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=ORB_SCALE_FACTOR, nlevels=ORB_N_LEVELS)
                print(f"Detector ORB reinicializado: features={ORB_FEATURES}, scale={ORB_SCALE_FACTOR}, levels={ORB_N_LEVELS}")
            except Exception as e:
                print(f"Erro ao reinicializar ORB: {e}")
                messagebox.showwarning("Aviso", "Erro ao reinicializar detector ORB. O alinhamento pode não funcionar.")
            
            self.result = True
            messagebox.showinfo("Sucesso", "Configurações salvas com sucesso!")
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar configurações: {str(e)}")
    
    def restore_defaults(self):
        """Restaura configurações padrão"""
        self.orb_features_var.set(5000)
        self.scale_factor_var.set(1.2)
        self.n_levels_var.set(8)
        self.preview_w_var.set(800)
        self.preview_h_var.set(600)
        self.thr_corr_var.set(0.1)
        self.min_px_var.set(10)
        
        # Atualiza labels
        self.features_label.config(text="5000")
        self.scale_label.config(text="1.20")
        self.corr_label.config(text="0.10")
    
    def cancel(self):
        """Cancela a edição"""
        self.destroy()


class MontagemWindow(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        
        # Inicializa o gerenciador de banco de dados
        # Usa caminho absoluto baseado na raiz do projeto
        db_path = MODEL_DIR / "models.db"
        self.db_manager = DatabaseManager(str(db_path))
        
        # Dados da aplicação
        self.img_original = None
        self.img_display = None
        self.scale_factor = 1.0
        self.slots = []
        self.selected_slot_id = None
        self.current_model_id = None  # ID do modelo atual no banco
        self.model_modified = False  # Flag para indicar se o modelo foi modificado
        
        # Estado do desenho
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None
        
        # Flag para prevenir loop infinito na seleção
        self._selecting_slot = False
        
        # Flag para prevenir múltiplos cliques simultâneos no botão de edição
        self._processing_edit_click = False
        
        # Controle de webcam
        self.available_cameras = detect_cameras()
        self.selected_camera = 0
        self.camera = None
        self.live_capture = False
        self.latest_frame = None
        
        # Variáveis de ferramentas de edição
        self.current_drawing_mode = "rectangle"
        self.current_rotation = 0
        self.editing_handle = None
        
        self.setup_ui()
        self.update_button_states()
    
    def mark_model_modified(self):
        """Marca o modelo como modificado e atualiza o status."""
        if not self.model_modified:
            self.model_modified = True
            self.update_status_display()
    
    def mark_model_saved(self):
        """Marca o modelo como salvo e atualiza o status."""
        if self.model_modified:
            self.model_modified = False
            self.update_status_display()
    
    def update_status_display(self):
        """Atualiza a exibição do status baseado no estado atual."""
        if self.img_original is None:
            self.status_var.set("Carregue uma imagem para começar")
        elif not self.slots:
            self.status_var.set("Imagem carregada - Desenhe slots para criar o modelo")
        elif self.model_modified:
            self.status_var.set("Modelo modificado - Salve as alterações")
        else:
            model_name = "Modelo atual"
            if self.current_model_id:
                try:
                    modelo = self.db_manager.load_modelo(self.current_model_id)
                    model_name = modelo['nome']
                except:
                    pass
            self.status_var.set(f"Modelo: {model_name} - {len(self.slots)} slots")
    
    def setup_ui(self):
        # Frame principal com layout horizontal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Painel esquerdo - Controles
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 10))
        
        # Painel direito - Canvas
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)
        
        # === PAINEL ESQUERDO ===
        
        # Seção de Imagem
        img_frame = ttk.LabelFrame(left_panel, text="Imagem")
        img_frame.pack(fill=X, pady=(0, 10))
        
        self.btn_load_image = ttk.Button(img_frame, text="Carregar Imagem", 
                                        command=self.load_image)
        self.btn_load_image.pack(fill=X, padx=5, pady=2)
        
        # Seção de Webcam
        webcam_frame = ttk.LabelFrame(left_panel, text="Webcam")
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
        
        # Botão para iniciar/parar captura contínua
        self.btn_live_capture = ttk.Button(webcam_frame, text="Iniciar Captura Contínua", 
                                          command=self.toggle_live_capture)
        self.btn_live_capture.pack(fill=X, padx=5, pady=2)
        
        self.btn_capture = ttk.Button(webcam_frame, text="Capturar Imagem", 
                                     command=self.capture_from_webcam)
        self.btn_capture.pack(fill=X, padx=5, pady=2)
        
        # Seção de Modelo
        model_frame = ttk.LabelFrame(left_panel, text="Modelo")
        model_frame.pack(fill=X, pady=(0, 10))
        
        self.btn_load_model = ttk.Button(model_frame, text="Carregar Modelo", 
                                        command=self.load_model_dialog)
        self.btn_load_model.pack(fill=X, padx=5, pady=2)
        
        self.btn_save_model = ttk.Button(model_frame, text="Salvar Modelo", 
                                        command=self.save_model)
        self.btn_save_model.pack(fill=X, padx=5, pady=2)
        
        # Seção de Ferramentas de Edição
        tools_frame = ttk.LabelFrame(left_panel, text="Ferramentas de Edição")
        tools_frame.pack(fill=X, pady=(0, 10))
        
        # Modo de desenho
        mode_frame = ttk.Frame(tools_frame)
        mode_frame.pack(fill=X, padx=5, pady=5)
        
        ttk.Label(mode_frame, text="Modo de Desenho:").pack(anchor="w")
        
        self.drawing_mode = StringVar(value="rectangle")
        
        mode_buttons_frame = ttk.Frame(mode_frame)
        mode_buttons_frame.pack(fill=X, pady=2)
        
        # Configurando estilo para os botões de rádio com fundo escuro
        self.style = ttk.Style()
        self.style.configure("TRadiobutton", background="#1E1E1E", foreground="white")
        # Configurando mapeamento para garantir que o fundo permaneça escuro em todos os estados
        self.style.map("TRadiobutton",
                      background=[('active', '#1E1E1E'), ('selected', '#1E1E1E')],
                      foreground=[('active', 'white'), ('selected', 'white')])
        
        self.btn_rect_mode = ttk.Radiobutton(mode_buttons_frame, text="Retângulo", 
                                           variable=self.drawing_mode, value="rectangle",
                                           command=self.set_drawing_mode,
                                           style="TRadiobutton")
        self.btn_rect_mode.pack(side=LEFT, padx=(0, 5))
        
        self.btn_exclusion_mode = ttk.Radiobutton(mode_buttons_frame, text="Exclusão", 
                                                variable=self.drawing_mode, value="exclusion",
                                                command=self.set_drawing_mode,
                                                style="TRadiobutton")
        self.btn_exclusion_mode.pack(side=LEFT)
        
        # Controles de rotação
        rotation_frame = ttk.Frame(tools_frame)
        rotation_frame.pack(fill=X, padx=5, pady=5)
        
        ttk.Label(rotation_frame, text="Rotação (graus):").pack(anchor="w")
        
        rotation_control_frame = ttk.Frame(rotation_frame)
        rotation_control_frame.pack(fill=X, pady=2)
        
        self.rotation_var = StringVar(value="0")
        self.rotation_entry = ttk.Entry(rotation_control_frame, textvariable=self.rotation_var, width=8)
        self.rotation_entry.pack(side=LEFT, padx=(0, 5))
        
        self.btn_rotate_left = ttk.Button(rotation_control_frame, text="↺ -15°", 
                                        command=lambda: self.adjust_rotation(-15), width=8)
        self.btn_rotate_left.pack(side=LEFT, padx=(0, 2))
        
        self.btn_rotate_right = ttk.Button(rotation_control_frame, text="↻ +15°", 
                                         command=lambda: self.adjust_rotation(15), width=8)
        self.btn_rotate_right.pack(side=LEFT)
        
        # Status da ferramenta
        self.tool_status_var = StringVar(value="Modo: Retângulo")
        ttk.Label(tools_frame, textvariable=self.tool_status_var, 
                 font=("Arial", 8), foreground="#666").pack(padx=5, pady=(0, 5))
        
        # Seção de Slots
        slots_frame = ttk.LabelFrame(left_panel, text="Slots")
        slots_frame.pack(fill=X, pady=(0, 10))
        
        self.btn_clear_slots = ttk.Button(slots_frame, text="Limpar Todos os Slots", 
                                         command=self.clear_slots)
        self.btn_clear_slots.pack(fill=X, padx=5, pady=2)
        
        self.btn_edit_slot = ttk.Button(slots_frame, text="Editar Slot Selecionado", 
                                       command=self.edit_selected_slot)
        self.btn_edit_slot.pack(fill=X, padx=5, pady=2)
        
        self.btn_delete_slot = ttk.Button(slots_frame, text="Deletar Slot Selecionado", 
                                         command=self.delete_selected_slot)
        self.btn_delete_slot.pack(fill=X, padx=5, pady=2)
        
        self.btn_train_slot = ttk.Button(slots_frame, text="Treinar Slot Selecionado", 
                                        command=self.train_selected_slot)
        self.btn_train_slot.pack(fill=X, padx=5, pady=2)
        
        # Lista de slots
        slots_list_frame = ttk.Frame(slots_frame)
        slots_list_frame.pack(fill=X, padx=5, pady=5)
        
        # Scrollbar para lista de slots
        scrollbar_slots = ttk.Scrollbar(slots_list_frame)
        scrollbar_slots.pack(side=RIGHT, fill=Y)
        
        # Treeview para slots
        self.slots_listbox = ttk.Treeview(slots_list_frame, yscrollcommand=scrollbar_slots.set, height=6)
        self.slots_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar_slots.config(command=self.slots_listbox.yview)
        


        
        # Seção de Ajuda
        help_frame = ttk.LabelFrame(left_panel, text="Ajuda")
        help_frame.pack(fill=X, pady=(0, 10))
        
        self.btn_help = ttk.Button(help_frame, text="Mostrar Ajuda", 
                                  command=self.show_help)
        self.btn_help.pack(fill=X, padx=5, pady=5)
        
        self.btn_config = ttk.Button(help_frame, text="Configurações do Sistema", 
                                    command=self.open_system_config)
        self.btn_config.pack(fill=X, padx=5, pady=(0, 5))
        
        # === PAINEL DIREITO ===
        
        # Canvas com scrollbars
        canvas_frame = ttk.LabelFrame(right_panel, text="Editor")
        canvas_frame.pack(fill=BOTH, expand=True)
        
        # Frame para canvas e scrollbars
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=VERTICAL)
        v_scrollbar.pack(side=RIGHT, fill=Y)
        
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=HORIZONTAL)
        h_scrollbar.pack(side=BOTTOM, fill=X)
        
        # Canvas
        self.canvas = Canvas(canvas_container, bg="#2C3E50",  # Cor de fundo alterada
                           yscrollcommand=v_scrollbar.set,
                           xscrollcommand=h_scrollbar.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Configurar scrollbars
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        
        # Binds do canvas
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Status bar
        self.status_var = StringVar()
        self.status_var.set("Carregue uma imagem para começar")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken")
        status_bar.pack(side=BOTTOM, fill=X, padx=5, pady=2)
    
    def clear_all(self):
        """Limpa todos os dados do editor."""
        self.img_original = None
        self.img_display = None
        self.scale_factor = 1.0
        self.slots = []
        self.selected_slot_id = None
        self.model_path = None
        self.drawing = False
        self.current_rect = None
        
        # Reset das flags de controle
        self._selecting_slot = False
        self._processing_edit_click = False
        
        # Limpa canvas
        self.canvas.delete("all")
        
        # Limpa lista de slots
        for item in self.slots_listbox.get_children():
            self.slots_listbox.delete(item)
        
        # Atualiza status
        self.status_var.set("Dados limpos")
        self.update_button_states()
    
    def load_image_data(self, image_path):
        """Carrega dados da imagem e calcula escala."""
        try:
            # Carrega imagem
            self.img_original = cv2.imread(str(image_path))
            if self.img_original is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
            
            print(f"Imagem carregada: {image_path}")
            print(f"Dimensões: {self.img_original.shape}")
            
            # Converte para exibição no canvas
            self.img_display, self.scale_factor = cv2_to_tk(self.img_original, PREVIEW_W, PREVIEW_H)
            
            if self.img_display is None:
                raise ValueError("Erro ao converter imagem para exibição")
            
            print(f"Escala aplicada: {self.scale_factor:.3f}")
            
            # Configura canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=NW, image=self.img_display)
            
            # Atualiza região de scroll
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            return True
            
        except Exception as e:
            print(f"Erro em load_image_data: {e}")
            messagebox.showerror("Erro", f"Erro ao carregar imagem: {str(e)}")
            return False
    
    def load_image(self):
        """Carrega uma nova imagem."""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.clear_all()
            
            if self.load_image_data(file_path):
                self.status_var.set(f"Imagem carregada: {Path(file_path).name}")
                self.update_button_states()
    
    def start_live_capture(self):
        """Inicia captura contínua da câmera em segundo plano."""
        if self.live_capture:
            return
            
        try:
            camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
            
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            # Configurações otimizadas para inicialização mais rápida
            # Usa DirectShow no Windows para melhor compatibilidade
            # No Raspberry Pi, usa a API padrão
            if is_windows:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações otimizadas para performance e inicialização rápida
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Usa resolução nativa para câmeras externas (1920x1080) ou padrão para webcam interna
            if camera_index > 0:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            else:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.live_capture = True
            self.process_live_frame()
            self.status_var.set(f"Câmera {camera_index} ativa em segundo plano")
            
        except Exception as e:
            print(f"Erro ao iniciar câmera: {e}")
            messagebox.showerror("Erro", f"Erro ao iniciar câmera: {str(e)}")
    
    def stop_live_capture(self):
        """Para a captura contínua da câmera."""
        self.live_capture = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.latest_frame = None
        self.status_var.set("Câmera desconectada")
    
    def toggle_live_capture(self):
        """Alterna entre iniciar e parar a captura contínua."""
        if not self.live_capture:
            self.start_live_capture()
            if self.live_capture:  # Se iniciou com sucesso
                self.btn_live_capture.config(text="Parar Captura Contínua")
        else:
            self.stop_live_capture()
            self.btn_live_capture.config(text="Iniciar Captura Contínua")
    
    def process_live_frame(self):
        """Processa frames da câmera em segundo plano."""
        if not self.live_capture or not self.camera:
            return
        
        try:
            ret, frame = self.camera.read()
            if ret:
                self.latest_frame = frame.copy()
        except Exception as e:
            print(f"Erro ao capturar frame: {e}")
            # Para a captura em caso de erro
            self.stop_live_capture()
            return
        
        # Agenda próximo frame (100ms para melhor estabilidade)
        if self.live_capture:
            self.master.after(100, self.process_live_frame)
    
    def capture_from_webcam(self):
        """Captura instantânea da imagem mais recente da câmera."""
        try:
            if not self.live_capture or self.latest_frame is None:
                # Fallback para captura única se não há captura contínua
                camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
                captured_image = capture_image_from_camera(camera_index)
            else:
                # Usa o frame mais recente da captura contínua
                captured_image = self.latest_frame.copy()
            
            if captured_image is not None:
                # Limpa dados anteriores
                self.clear_all()
                
                # Carrega a imagem capturada
                self.img_original = captured_image
                
                # Converte para exibição
                self.img_display, self.scale_factor = cv2_to_tk(self.img_original, PREVIEW_W, PREVIEW_H)
                
                if self.img_display:
                    # Limpa o canvas e exibe a nova imagem
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, anchor=NW, image=self.img_display)
                    
                    # Atualiza a região de scroll
                    self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                    
                    # Atualiza estado dos botões
                    self.update_button_states()
                    
                    camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
                    self.status_var.set(f"Imagem capturada da câmera {camera_index}")
                    messagebox.showinfo("Sucesso", "Imagem capturada instantaneamente!")
                else:
                    messagebox.showerror("Erro", "Erro ao processar a imagem capturada.")
            else:
                messagebox.showerror("Erro", "Nenhuma imagem disponível para captura.")
                
        except Exception as e:
            print(f"Erro ao capturar da webcam: {e}")
            messagebox.showerror("Erro", f"Erro ao capturar da webcam: {str(e)}")
    
    def load_model_dialog(self):
        """Abre diálogo para carregar modelo do banco de dados."""
        dialog = ModelSelectorDialog(self.master, self.db_manager)
        result = dialog.show()
        
        if result:
            if result['action'] == 'load':
                self.load_model_from_db(result['model_id'])
            elif result['action'] == 'new':
                self.create_new_model(result['name'])
    
    def load_model_from_db(self, model_id):
        """Carrega um modelo do banco de dados."""
        try:
            # Carrega dados do modelo
            model_data = self.db_manager.load_modelo(model_id)
            
            # Limpa dados atuais
            self.clear_all()
            
            # Carrega imagem de referência
            image_path = model_data['image_path']
            
            # Tenta caminho absoluto primeiro
            if not Path(image_path).exists():
                # Tenta caminho relativo ao diretório de modelos
                relative_path = MODEL_DIR / Path(image_path).name
                if relative_path.exists():
                    image_path = str(relative_path)
                else:
                    raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
            
            if not self.load_image_data(image_path):
                return
            
            # Carrega slots
            self.slots = model_data['slots']
            self.current_model_id = model_id
            
            # Atualiza interface
            self.update_slots_list()
            self.redraw_slots()
            
            self.status_var.set(f"Modelo carregado: {model_data['nome']} ({len(self.slots)} slots)")
            self.update_button_states()
            
            # Marca modelo como salvo (recém carregado)
            self.mark_model_saved()
            
            print(f"Modelo '{model_data['nome']}' carregado com sucesso: {len(self.slots)} slots")
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            messagebox.showerror("Erro", f"Erro ao carregar modelo: {str(e)}")
    
    def create_new_model(self, model_name):
        """Cria um novo modelo vazio."""
        try:
            # Limpa dados atuais
            self.clear_all()
            
            # Define como novo modelo (sem ID ainda)
            self.current_model_id = None
            self.slots = []
            
            self.status_var.set(f"Novo modelo criado: {model_name}")
            self.update_button_states()
            
            # Marca modelo como salvo (novo modelo vazio)
            self.mark_model_saved()
            
            print(f"Novo modelo '{model_name}' criado")
            
        except Exception as e:
            print(f"Erro ao criar novo modelo: {e}")
            messagebox.showerror("Erro", f"Erro ao criar novo modelo: {str(e)}")
    
    def update_slots_list(self):
        """Atualiza a lista de slots na interface."""
        # Limpa lista atual
        for item in self.slots_listbox.get_children():
            self.slots_listbox.delete(item)
        
        # Adiciona slots
        for slot in self.slots:
            self.slots_listbox.insert("", "end", 
                                    text=slot['id'],
                                    values=(slot['tipo'], f"({slot['x']}, {slot['y']})"))
    
    def redraw_slots(self):
        """Redesenha todos os slots no canvas."""
        try:
            if self.img_display is None or not hasattr(self, 'canvas'):
                return
            
            # Remove retângulos existentes
            self.canvas.delete("slot")
            
            # Desenha cada slot
            for slot in self.slots:
                if slot and 'id' in slot:
                    self.draw_slot(slot)
        except Exception as e:
            print(f"Erro ao redesenhar slots: {e}")
            self.status_var.set("Erro ao atualizar visualização")
    
    def draw_slot(self, slot):
        """Desenha um slot no canvas."""
        try:
            # Verifica se o slot tem todos os campos necessários
            required_fields = ['x', 'y', 'w', 'h', 'id', 'tipo']
            if not all(field in slot for field in required_fields):
                print(f"Slot inválido: campos obrigatórios ausentes {slot}")
                return
            
            # Verifica se scale_factor existe
            if not hasattr(self, 'scale_factor') or self.scale_factor <= 0:
                print("Scale factor inválido")
                return
            
            # Converte coordenadas da imagem para canvas
            x1 = int(slot['x'] * self.scale_factor)
            y1 = int(slot['y'] * self.scale_factor)
            x2 = int((slot['x'] + slot['w']) * self.scale_factor)
            y2 = int((slot['y'] + slot['h']) * self.scale_factor)
            
            # Calcula centro para rotação
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Carrega as configurações de estilo
            style_config = load_style_config()
            
            # Escolhe cor baseada na seleção
            if slot['id'] == self.selected_slot_id:
                color = style_config["selection_color"]
                width = 3
            else:
                color = COLOR_CLIP
                width = 2
            
            # Obtém rotação do slot
            rotation = slot.get('rotation', 0)
            
            # Desenha forma baseada no tipo do slot
            shape = slot.get('shape', 'rectangle')
            # Para retângulos, aplica rotação se necessário
            if True:
                # Para retângulos, aplica rotação se necessário
                if rotation != 0:
                    # Calcula pontos do retângulo rotacionado
                    import math
                    rad = math.radians(rotation)
                    cos_r = math.cos(rad)
                    sin_r = math.sin(rad)
                    
                    # Pontos relativos ao centro
                    w_half = (x2 - x1) / 2
                    h_half = (y2 - y1) / 2
                    
                    # Calcula os 4 cantos rotacionados
                    points = []
                    corners = [(-w_half, -h_half), (w_half, -h_half), 
                              (w_half, h_half), (-w_half, h_half)]
                    
                    for dx, dy in corners:
                        # Rotaciona ponto
                        rx = dx * cos_r - dy * sin_r
                        ry = dx * sin_r + dy * cos_r
                        # Translada para posição final
                        points.extend([center_x + rx, center_y + ry])
                    
                    shape_id = self.canvas.create_polygon(points, outline=color, 
                                                 width=width, fill="", tags="slot")
                else:
                    # Retângulo sem rotação
                    shape_id = self.canvas.create_rectangle(x1, y1, x2, y2, 
                                               outline=color, width=width, tags="slot")
            
            # Desenha áreas de exclusão se existirem
            exclusion_areas = slot.get('exclusion_areas', [])
            for exclusion in exclusion_areas:
                ex_x1 = int(exclusion['x'] * self.scale_factor)
                ex_y1 = int(exclusion['y'] * self.scale_factor)
                ex_x2 = int((exclusion['x'] + exclusion['w']) * self.scale_factor)
                ex_y2 = int((exclusion['y'] + exclusion['h']) * self.scale_factor)
                
                # Desenha área de exclusão em vermelho
                self.canvas.create_rectangle(ex_x1, ex_y1, ex_x2, ex_y2,
                                            outline="#FF4444", width=2, tags="slot")
            
            # Adiciona texto com ID
            # Carrega as configurações de estilo
            style_config = load_style_config()
            self.canvas.create_text(x1 + 5, y1 + 5, text=slot['id'],
                                   fill="white", font=style_config["ok_font"], tags="slot")
            
            # Adiciona botão de edição (pequeno quadrado no canto superior direito)
            edit_size = 12
            edit_x1 = x2 - edit_size - 2
            edit_y1 = y1 + 2
            edit_x2 = x2 - 2
            edit_y2 = y1 + edit_size + 2
            
            edit_btn = self.canvas.create_rectangle(edit_x1, edit_y1, edit_x2, edit_y2,
                                                   fill="#4CAF50", outline="white", width=1,
                                                   tags=("slot", f"edit_btn_{slot['id']}"))
            
            # Adiciona ícone de edição (pequeno "E")
            # Carrega as configurações de estilo se ainda não foi carregado
            if 'style_config' not in locals():
                style_config = load_style_config()
            self.canvas.create_text((edit_x1 + edit_x2) // 2, (edit_y1 + edit_y2) // 2,
                                   text="E", fill="white", font=style_config["ok_font"],
                                   tags=("slot", f"edit_text_{slot['id']}"))
        except Exception as e:
            print(f"Erro ao desenhar slot {slot.get('id', 'desconhecido')}: {e}")
    
    def on_canvas_press(self, event):
        """Inicia desenho de novo slot ou edita slot existente."""
        try:
            if self.img_original is None:
                return
            
            # Verifica se o canvas existe e está válido
            if not hasattr(self, 'canvas') or not self.canvas.winfo_exists():
                print("Canvas não existe ou foi destruído")
                return
            
            # Converte coordenadas do canvas para coordenadas da tela
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            # Verifica se clicou em um handle de edição primeiro
            try:
                closest_items = self.canvas.find_closest(canvas_x, canvas_y)
                if closest_items:
                    clicked_item = closest_items[0]
                    tags = self.canvas.gettags(clicked_item)
                    
                    # Verifica se é um handle de edição
                    for tag in tags:
                        if tag == "edit_handle" or tag.startswith("resize_handle_") or tag == "rotation_handle":
                            # Deixa o evento ser processado pelos handles
                            return
            except Exception as e:
                print(f"Erro ao verificar handles: {e}")
            
            # Verifica se clicou em um botão de edição
            try:
                closest_items = self.canvas.find_closest(canvas_x, canvas_y)
                if not closest_items:
                    print("Nenhum item encontrado no canvas")
                    return
                
                clicked_item = closest_items[0]
                tags = self.canvas.gettags(clicked_item)
                
                if not tags:
                    print("Item clicado não possui tags")
                    # Continua para verificar slots existentes
                else:
                    for tag in tags:
                        if tag.startswith('edit_btn_') or tag.startswith('edit_text_'):
                            try:
                                # Extrai o slot_id da tag
                                tag_parts = tag.split('_')
                                if len(tag_parts) < 3:
                                    print(f"Tag inválida: {tag}")
                                    continue
                                
                                slot_id = int(tag_parts[-1])
                                
                                # Verifica se o slot existe
                                if not any(s['id'] == slot_id for s in self.slots):
                                    print(f"Slot {slot_id} não encontrado na lista")
                                    return
                                
                                # Previne múltiplas chamadas simultâneas
                                if hasattr(self, '_processing_edit_click') and self._processing_edit_click:
                                    print("Já processando clique de edição")
                                    return
                                
                                self._processing_edit_click = True
                                
                                try:
                                    print(f"Processando clique no botão de edição do slot {slot_id}")
                                    self.select_slot(slot_id)
                                    # Removido chamada automática para edit_selected_slot() para evitar travamento
                                    print(f"Slot {slot_id} selecionado. Use o botão 'Editar Slot Selecionado' para editar.")
                                    return
                                finally:
                                    self._processing_edit_click = False
                                    
                            except ValueError as ve:
                                print(f"Erro ao converter slot_id: {ve}")
                                continue
                            except Exception as e:
                                print(f"Erro ao processar clique no botão de edição: {e}")
                                import traceback
                                traceback.print_exc()
                                return
                            
            except Exception as e:
                 print(f"Erro ao verificar item clicado: {e}")
                 import traceback
                 traceback.print_exc()
            
            # Verifica se clicou em um slot existente
            try:
                clicked_slot = self.find_slot_at(canvas_x, canvas_y)
                if clicked_slot:
                    print(f"Clicou no slot {clicked_slot['id']}")
                    
                    # Se está no modo de exclusão e há um slot selecionado, permite desenhar área de exclusão
                    if self.current_drawing_mode == "exclusion" and self.selected_slot_id is not None:
                        print("Iniciando desenho de área de exclusão")
                        self.drawing = True
                        self.start_x = canvas_x
                        self.start_y = canvas_y
                        self.canvas.delete("drawing")
                        return
                    else:
                        # Seleciona o slot e mostra handles de edição
                        self.select_slot(clicked_slot['id'])
                        self.show_edit_handles(clicked_slot)
                        return
            except Exception as e:
                print(f"Erro ao verificar slot existente: {e}")
                import traceback
                traceback.print_exc()
            
            # Se está no modo de exclusão mas não há slot selecionado, mostra mensagem
            if self.current_drawing_mode == "exclusion":
                self.status_var.set("Selecione um slot primeiro para criar área de exclusão")
                return
            
            # Inicia desenho de novo slot
            try:
                print("Iniciando desenho de novo slot")
                self.deselect_all_slots()
                self.hide_edit_handles()
                self.drawing = True
                self.start_x = canvas_x
                self.start_y = canvas_y
                
                # Remove retângulo de desenho anterior
                self.canvas.delete("drawing")
            except Exception as e:
                print(f"Erro ao iniciar desenho de novo slot: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Erro geral em on_canvas_press: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Erro ao processar clique no canvas")
    
    def on_canvas_drag(self, event):
        """Atualiza desenho do slot durante arraste."""
        if not self.drawing:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Remove forma anterior
        self.canvas.delete("drawing")
        
        # Define cor baseada no modo
        if self.current_drawing_mode == "exclusion":
            outline_color = "#FF4444"  # Vermelho para exclusão
        else:
            outline_color = COLOR_DRAWING
        
        # Desenha retângulo (para rectangle e exclusion)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, canvas_x, canvas_y,
            outline=outline_color, width=2, tags="drawing"
        )
    
    def on_canvas_release(self, event):
        """Finaliza desenho do slot."""
        if not self.drawing:
            return
        
        self.drawing = False
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Remove forma de desenho
        self.canvas.delete("drawing")
        
        # Calcula dimensões
        x1, y1 = min(self.start_x, canvas_x), min(self.start_y, canvas_y)
        x2, y2 = max(self.start_x, canvas_x), max(self.start_y, canvas_y)
        
        w = x2 - x1
        h = y2 - y1
        
        # Verifica se a área é válida
        if w < 10 or h < 10:
            self.status_var.set("Área muito pequena (mínimo 10x10 pixels)")
            return
        
        # Converte coordenadas do canvas para imagem original
        img_x = int(x1 / self.scale_factor)
        img_y = int(y1 / self.scale_factor)
        img_w = int(w / self.scale_factor)
        img_h = int(h / self.scale_factor)
        
        # Verifica se é área de exclusão
        if self.current_drawing_mode == "exclusion":
            self.add_exclusion_area(img_x, img_y, img_w, img_h)
        else:
            # Adiciona slot normal
            self.add_slot(img_x, img_y, img_w, img_h)
    
    def find_slot_at(self, canvas_x, canvas_y):
        """Encontra slot nas coordenadas do canvas."""
        for slot in self.slots:
            x1 = slot['x'] * self.scale_factor
            y1 = slot['y'] * self.scale_factor
            x2 = (slot['x'] + slot['w']) * self.scale_factor
            y2 = (slot['y'] + slot['h']) * self.scale_factor
            
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                return slot
        return None
    
    def select_slot(self, slot_id):
        """Seleciona um slot."""
        try:
            # Previne loop infinito
            if hasattr(self, '_selecting_slot') and self._selecting_slot:
                return
            
            self._selecting_slot = True
            
            # Verifica se o slot existe
            slot_exists = any(s['id'] == slot_id for s in self.slots)
            if not slot_exists:
                print(f"Erro: Slot {slot_id} não encontrado")
                return
            
            self.selected_slot_id = slot_id
            
            # Atualiza seleção na lista
            for item in self.slots_listbox.get_children():
                item_text = self.slots_listbox.item(item, "text")
                if str(item_text) == str(slot_id):
                    self.slots_listbox.selection_set(item)
                    self.slots_listbox.focus(item)
                    break
            
            self.redraw_slots()
            self.update_button_states()
            self.status_var.set(f"Slot {slot_id} selecionado")
        except Exception as e:
            print(f"Erro ao selecionar slot {slot_id}: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Erro na seleção do slot")
        finally:
            self._selecting_slot = False
    
    def deselect_all_slots(self):
        """Remove seleção de todos os slots."""
        self.selected_slot_id = None
        self.slots_listbox.selection_remove(self.slots_listbox.selection())
        self.hide_edit_handles()
        self.redraw_slots()
        self.update_button_states()
    
    def add_slot(self, xa, ya, wa, ha):
        """Adiciona um novo slot."""
        if self.img_original is None:
            messagebox.showerror("Erro", "Nenhuma imagem carregada.")
            return
        
        # Converte coordenadas do canvas para imagem original
        x = int(xa)
        y = int(ya)
        w = int(wa)
        h = int(ha)
        
        # Valida coordenadas
        img_h, img_w = self.img_original.shape[:2]
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            messagebox.showerror("Erro", "Slot está fora dos limites da imagem.")
            return
        
        # Extrai ROI
        roi = self.img_original[y:y+h, x:x+w]
        if roi.size == 0:
            messagebox.showerror("Erro", "ROI do slot está vazia.")
            return
        
        print(f"add_slot: Adicionando slot na posição ({x}, {y}), tamanho ({w}, {h})")
        
        # Apenas slots do tipo 'clip' são suportados
        slot_type = 'clip'
        
        # Valores padrão (não utilizados para clips, mas mantidos para compatibilidade)
        bgr_color = [0, 0, 255]  # Vermelho padrão
        h_tolerance = 10
        s_tolerance = 50
        v_tolerance = 50
        
        # Gera ID único
        existing_ids = [slot['id'] for slot in self.slots]
        slot_id = 1
        while slot_id in existing_ids:
            slot_id += 1
        
        # Cria dados do slot com configurações padrão específicas por tipo
        slot_data = {
            'id': slot_id,
            'tipo': slot_type,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cor': bgr_color,
            'h_tolerance': h_tolerance,
            's_tolerance': s_tolerance,
            'v_tolerance': v_tolerance,
            'detection_threshold': 0.8,  # Limiar padrão para detecção
            'shape': self.current_drawing_mode,  # Forma: rectangle, exclusion
            'rotation': self.current_rotation,   # Rotação em graus
            'exclusion_areas': []               # Lista de áreas de exclusão
        }
        
        # Configurações específicas para clips
        slot_data.update({
            'correlation_threshold': THR_CORR,
            'template_method': 'TM_CCOEFF_NORMED',
            'scale_tolerance': 0.1
        })
        
        # Salva template para o clip
        template_filename = f"slot_{slot_id}_template.png"
        template_path = TEMPLATE_DIR / template_filename
        
        try:
            cv2.imwrite(str(template_path), roi)
            slot_data['template_path'] = str(template_path)
            print(f"Template salvo: {template_path}")
        except Exception as e:
            print(f"Erro ao salvar template: {e}")
            messagebox.showerror("Erro", f"Erro ao salvar template: {str(e)}")
            return
        
        # Adiciona slot à lista
        self.slots.append(slot_data)
        
        # Atualiza interface
        self.update_slots_list()
        self.redraw_slots()
        
        self.status_var.set(f"Slot {slot_id} ({slot_type}) adicionado")
        self.update_button_states()
        
        # Marca modelo como modificado
        self.mark_model_modified()
        
        print(f"Slot {slot_id} adicionado com sucesso: {slot_data}")
    
    def on_slot_select(self, event):
        """Callback para seleção na lista de slots."""
        try:
            # Previne loop infinito - não processa se já estamos selecionando
            if hasattr(self, '_selecting_slot') and self._selecting_slot:
                return
                
            selection = self.slots_listbox.selection()
            if selection:
                item = selection[0]
                slot_id_text = self.slots_listbox.item(item, "text")
                if slot_id_text and str(slot_id_text).isdigit():
                    slot_id = int(slot_id_text)
                    self.select_slot(slot_id)
                else:
                    print(f"Erro: ID do slot inválido: {slot_id_text}")
        except Exception as e:
            print(f"Erro na seleção do slot: {e}")
            self.status_var.set("Erro na seleção do slot")
    
    def on_slot_double_click(self, event):
        """Callback para duplo-clique na lista de slots - abre edição."""
        try:
            # Verifica se há uma seleção válida
            selection = self.slots_listbox.selection()
            if not selection:
                print("Nenhum slot selecionado para edição")
                return
            
            # Verifica se há slots disponíveis
            if not self.slots:
                print("Nenhum slot disponível para edição")
                messagebox.showinfo("Aviso", "Nenhum slot disponível para edição.")
                return
            
            # Obtém o item selecionado
            item = selection[0]
            slot_id_text = self.slots_listbox.item(item, "text")
            
            # Valida o ID do slot
            if not slot_id_text or not str(slot_id_text).isdigit():
                print(f"Erro: ID do slot inválido: {slot_id_text}")
                messagebox.showerror("Erro", f"ID do slot inválido: {slot_id_text}")
                return
            
            slot_id = int(slot_id_text)
            
            # Verifica se o slot existe na lista
            slot_exists = any(slot['id'] == slot_id for slot in self.slots)
            if not slot_exists:
                print(f"Erro: Slot {slot_id} não encontrado na lista")
                messagebox.showerror("Erro", f"Slot {slot_id} não encontrado.")
                return
            
            # Previne múltiplas janelas de edição
            if hasattr(self, '_editing_slot') and self._editing_slot:
                print("Janela de edição já está aberta")
                messagebox.showinfo("Aviso", "Uma janela de edição já está aberta.")
                return
            
            # Marca que está editando
            self._editing_slot = True
            
            try:
                # Seleciona o slot apenas - removido chamada automática para edit_selected_slot()
                self.select_slot(slot_id)
                print(f"Slot {slot_id} selecionado via duplo-clique. Use o botão 'Editar Slot Selecionado' para editar.")
            finally:
                # Garante que a flag seja limpa mesmo se houver erro
                self._editing_slot = False
                
        except Exception as e:
            print(f"Erro no duplo-clique do slot: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Erro ao abrir edição do slot")
            messagebox.showerror("Erro", f"Erro ao abrir edição: {str(e)}")
            # Limpa a flag em caso de erro
            if hasattr(self, '_editing_slot'):
                self._editing_slot = False
    
    def clear_slots(self):
        """Remove todos os slots."""
        if not self.slots:
            messagebox.showinfo("Info", "Nenhum slot para remover.")
            return
        
        if messagebox.askyesno("Confirmar", f"Remover todos os {len(self.slots)} slots?"):
            self.slots = []
            self.selected_slot_id = None
            self.update_slots_list()
            self.redraw_slots()
            self.status_var.set("Todos os slots removidos")
            self.update_button_states()
            
            # Marca modelo como modificado
            self.mark_model_modified()
    
    def edit_selected_slot(self):
        """Edita o slot selecionado usando menu inline."""
        print(f"\n=== INICIANDO EDIÇÃO DO SLOT ===")
        print(f"Selected slot ID: {self.selected_slot_id}")
        
        # Verifica se há um slot selecionado
        if self.selected_slot_id is None:
            print("ERRO: Nenhum slot selecionado")
            messagebox.showinfo("Aviso", "Nenhum slot selecionado.")
            return
        
        # Verifica se a lista de slots não está vazia
        if not self.slots:
            print("ERRO: Lista de slots vazia")
            messagebox.showinfo("Aviso", "Nenhum slot disponível para edição.")
            return
        
        # Busca o slot na lista
        slot_to_edit = next((s for s in self.slots if s['id'] == self.selected_slot_id), None)
        if not slot_to_edit:
            print(f"ERRO: Slot {self.selected_slot_id} não encontrado na lista")
            messagebox.showerror("Erro", f"Dados do slot {self.selected_slot_id} não encontrados.")
            return
        
        # Verifica se os dados do slot são válidos
        required_keys = ['id', 'x', 'y', 'w', 'h', 'tipo']
        # Para slots do tipo 'clip', verifica campos específicos
        if slot_to_edit.get('tipo') == 'clip':
            clip_keys = ['cor', 'detection_threshold']
            required_keys.extend(clip_keys)
        
        missing_keys = [key for key in required_keys if key not in slot_to_edit]
        if missing_keys:
            print(f"ERRO: Dados do slot incompletos. Chaves ausentes: {missing_keys}")
            print(f"Dados do slot: {slot_to_edit}")
            messagebox.showerror("Erro", f"Dados do slot estão incompletos. Chaves ausentes: {missing_keys}")
            return
        
        print(f"Slot encontrado: {slot_to_edit}")
        print("Criando menu de edição inline...")
        
        try:
            # Edita usando menu inline completo
            self.create_inline_edit_menu(slot_to_edit)
            print("Menu de edição criado com sucesso")
            
        except Exception as e:
            print(f"ERRO ao editar slot: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erro", f"Erro ao editar slot: {str(e)}")
    
    def edit_slot_with_simple_dialogs(self, slot_data):
        """Edita o slot usando diálogos simples do tkinter"""
        from tkinter import simpledialog
        
        print(f"Editando slot {slot_data['id']} com diálogos simples")
        
        # Edita X
        new_x = simpledialog.askinteger(
            "Editar Slot", 
            f"Posição X atual: {slot_data['x']}\nNova posição X:",
            initialvalue=slot_data['x'],
            minvalue=0
        )
        if new_x is None:  # Usuário cancelou
            return
        
        # Edita Y
        new_y = simpledialog.askinteger(
            "Editar Slot", 
            f"Posição Y atual: {slot_data['y']}\nNova posição Y:",
            initialvalue=slot_data['y'],
            minvalue=0
        )
        if new_y is None:
            return
        
        # Edita Largura
        new_w = simpledialog.askinteger(
            "Editar Slot", 
            f"Largura atual: {slot_data['w']}\nNova largura:",
            initialvalue=slot_data['w'],
            minvalue=1
        )
        if new_w is None:
            return
        
        # Edita Altura
        new_h = simpledialog.askinteger(
            "Editar Slot", 
            f"Altura atual: {slot_data['h']}\nNova altura:",
            initialvalue=slot_data['h'],
            minvalue=1
        )
        if new_h is None:
            return
        
        # Para slots do tipo clip, edita o limiar de detecção
        new_threshold = None
        if slot_data.get('tipo') == 'clip':
            current_threshold = slot_data.get('detection_threshold', 0.8)
            new_threshold = simpledialog.askfloat(
                "Editar Slot", 
                f"Limiar de detecção atual: {current_threshold}\nNovo limiar (0.0 - 1.0):",
                initialvalue=current_threshold,
                minvalue=0.0,
                maxvalue=1.0
            )
            if new_threshold is None:
                return
        
        # Aplica as alterações
        slot_data['x'] = new_x
        slot_data['y'] = new_y
        slot_data['w'] = new_w
        slot_data['h'] = new_h
        
        if new_threshold is not None:
            slot_data['detection_threshold'] = new_threshold
        
        # Atualiza a exibição
        self.redraw_slots()
        self.update_slots_list()
        
        print(f"Slot {slot_data['id']} atualizado: X={new_x}, Y={new_y}, W={new_w}, H={new_h}")
        if new_threshold is not None:
            print(f"Limiar de detecção: {new_threshold}")
        
        messagebox.showinfo("Sucesso", f"Slot {slot_data['id']} atualizado com sucesso!")
    
    def create_inline_edit_menu(self, slot_data):
        """Cria uma janela simples de edição para o slot"""
        print("Criando janela de edição simples...")
        
        # Carrega as configurações de estilo
        self.style_config = load_style_config()
        
        # Remove janela anterior se existir
        if hasattr(self, 'edit_menu_frame') and self.edit_menu_frame:
            try:
                self.edit_menu_frame.destroy()
            except:
                pass
        
        # Cria uma janela simples não-modal
        from tkinter import Toplevel
        self.edit_menu_frame = Toplevel(self.master)
        self.edit_menu_frame.title(f"Editar Slot {slot_data['id']}")
        self.edit_menu_frame.geometry("400x700")
        self.edit_menu_frame.resizable(True, True)
        
        # Aplica a cor de fundo da janela
        self.edit_menu_frame.configure(bg=self.style_config["background_color"])
        
        # Centraliza a janela
        self.edit_menu_frame.transient(self.master)
        self.edit_menu_frame.update_idletasks()
        x = (self.edit_menu_frame.winfo_screenwidth() // 2) - (200)
        y = (self.edit_menu_frame.winfo_screenheight() // 2) - (350)
        self.edit_menu_frame.geometry(f"400x700+{x}+{y}")
        
        # Frame principal com scrollbar
        main_frame = ttk.Frame(self.edit_menu_frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(main_frame, text=f"Editando Slot {slot_data['id']}", 
                               font=('Arial', 12, 'bold'),
                               foreground=self.style_config["text_color"])
        title_label.pack(pady=(0, 10))
        
        # Canvas e Scrollbar para campos
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill='both', expand=True)
        
        canvas = Canvas(canvas_frame, highlightthickness=0, bg=self.style_config["background_color"])
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="TFrame")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Frame para campos de entrada (agora dentro do scrollable_frame)
        fields_frame = ttk.Frame(scrollable_frame)
        fields_frame.pack(fill='x', pady=(0, 10), padx=10)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Variáveis para os campos
        self.edit_vars = {}
        
        # Campos básicos
        basic_fields = [
            ('X:', 'x', slot_data['x']),
            ('Y:', 'y', slot_data['y']),
            ('Largura:', 'w', slot_data['w']),
            ('Altura:', 'h', slot_data['h'])
        ]
        
        for i, (label_text, key, value) in enumerate(basic_fields):
            row_frame = ttk.Frame(fields_frame)
            row_frame.pack(fill=X, pady=2)
            
            label = ttk.Label(row_frame, text=label_text, width=10)
            label.pack(side=LEFT)
            
            var = ttk.StringVar(value=str(value))
            self.edit_vars[key] = var
            entry = ttk.Entry(row_frame, textvariable=var, width=15)
            entry.pack(side=LEFT, padx=(5, 0))
        
        # Campos específicos para clips
        if slot_data.get('tipo') == 'clip':
            # Separador
            separator = ttk.Separator(fields_frame, orient='horizontal')
            separator.pack(fill=X, pady=(10, 5))
            
            # Título para parâmetros avançados
            advanced_label = ttk.Label(fields_frame, text="Parâmetros de Detecção:", 
                                     font=('Arial', 9, 'bold'),
                                     foreground=self.style_config["text_color"])
            advanced_label.pack(pady=(0, 5))
            
            # Método de Detecção
            detection_method_frame = ttk.Frame(fields_frame)
            detection_method_frame.pack(fill=X, pady=2)
            
            detection_method_label = ttk.Label(detection_method_frame, text="Método Detecção:", width=12)
            detection_method_label.pack(side=LEFT)
            
            detection_methods = [
                "template_matching",
                "histogram_analysis"
            ]
            
            detection_method_var = ttk.StringVar(value=slot_data.get('detection_method', 'template_matching'))
            self.edit_vars['detection_method'] = detection_method_var
            detection_method_combo = ttk.Combobox(detection_method_frame, textvariable=detection_method_var, 
                                                values=detection_methods, width=18, state="readonly")
            detection_method_combo.pack(side=LEFT, padx=(5, 0))
            
            # Tooltip para detection_method
            detection_tip_label = ttk.Label(detection_method_frame, text="ℹ️", foreground="blue")
            detection_tip_label.pack(side=LEFT, padx=(2, 0))
            detection_tip_label.bind("<Button-1>", lambda e: messagebox.showinfo("Método de Detecção", 
                "Escolha o método de análise para este slot:\n\n" +
                "• template_matching: Usa template matching tradicional (padrão)\n" +
                "• histogram_analysis: Usa análise de histograma de cores"))
            
            # Detection Threshold
            det_threshold_frame = ttk.Frame(fields_frame)
            det_threshold_frame.pack(fill=X, pady=2)
            
            det_threshold_label = ttk.Label(det_threshold_frame, text="Det. Threshold:", width=12)
            det_threshold_label.pack(side=LEFT)
            
            det_threshold_var = ttk.StringVar(value=str(slot_data.get('detection_threshold', 0.9)))
            self.edit_vars['detection_threshold'] = det_threshold_var
            det_threshold_entry = ttk.Entry(det_threshold_frame, textvariable=det_threshold_var, width=10)
            det_threshold_entry.pack(side=LEFT, padx=(5, 0))
            
            # Tooltip para detection_threshold
            det_tip_label = ttk.Label(det_threshold_frame, text="ℹ️", foreground="blue")
            det_tip_label.pack(side=LEFT, padx=(2, 0))
            det_tip_label.bind("<Button-1>", lambda e: messagebox.showinfo("Detection Threshold", 
                "Limiar mínimo de confiança (0.0-1.0) para considerar uma detecção válida.\nValor padrão: 0.9 (90%)"))
            
            # Correlation Threshold
            corr_threshold_frame = ttk.Frame(fields_frame)
            corr_threshold_frame.pack(fill=X, pady=2)
            
            corr_threshold_label = ttk.Label(corr_threshold_frame, text="Corr. Threshold:", width=12)
            corr_threshold_label.pack(side=LEFT)
            
            corr_threshold_var = ttk.StringVar(value=str(slot_data.get('correlation_threshold', 0.1)))
            self.edit_vars['correlation_threshold'] = corr_threshold_var
            corr_threshold_entry = ttk.Entry(corr_threshold_frame, textvariable=corr_threshold_var, width=10)
            corr_threshold_entry.pack(side=LEFT, padx=(5, 0))
            
            # Tooltip para correlation_threshold
            corr_tip_label = ttk.Label(corr_threshold_frame, text="ℹ️", foreground="blue")
            corr_tip_label.pack(side=LEFT, padx=(2, 0))
            corr_tip_label.bind("<Button-1>", lambda e: messagebox.showinfo("Correlation Threshold", 
                "Limiar mínimo de correlação (0.0-1.0) para matching de templates.\nValor padrão: 0.1 (10%)"))
            
            # Template Method
            template_method_frame = ttk.Frame(fields_frame)
            template_method_frame.pack(fill=X, pady=2)
            
            template_method_label = ttk.Label(template_method_frame, text="Template Method:", width=12)
            template_method_label.pack(side=LEFT)
            
            template_methods = [
                "TM_CCOEFF_NORMED",
                "TM_CCORR_NORMED", 
                "TM_SQDIFF_NORMED",
                "TM_CCOEFF",
                "TM_CCORR",
                "TM_SQDIFF"
            ]
            
            template_method_var = ttk.StringVar(value=slot_data.get('template_method', 'TM_CCOEFF_NORMED'))
            self.edit_vars['template_method'] = template_method_var
            template_method_combo = ttk.Combobox(template_method_frame, textvariable=template_method_var, 
                                               values=template_methods, width=18, state="readonly")
            template_method_combo.pack(side=LEFT, padx=(5, 0))
            
            # Tooltip para template_method
            method_tip_label = ttk.Label(template_method_frame, text="ℹ️", foreground="blue")
            method_tip_label.pack(side=LEFT, padx=(2, 0))
            method_tip_label.bind("<Button-1>", lambda e: messagebox.showinfo("Template Method", 
                "Algoritmo do OpenCV para comparar templates:\n\n" +
                "• TM_CCOEFF_NORMED: Coeficientes de correlação normalizados (recomendado)\n" +
                "• TM_CCORR_NORMED: Correlação cruzada normalizada\n" +
                "• TM_SQDIFF_NORMED: Diferença quadrática normalizada\n" +
                "• TM_CCOEFF: Coeficientes de correlação\n" +
                "• TM_CCORR: Correlação cruzada\n" +
                "• TM_SQDIFF: Diferença quadrática"))
            
            # Scale Tolerance
            scale_tolerance_frame = ttk.Frame(fields_frame)
            scale_tolerance_frame.pack(fill=X, pady=2)
            
            scale_tolerance_label = ttk.Label(scale_tolerance_frame, text="Scale Tolerance:", width=12,
                                            foreground=self.style_config["text_color"])
            scale_tolerance_label.pack(side=LEFT)
            
            scale_tolerance_var = ttk.StringVar(value=str(slot_data.get('scale_tolerance', 0.1)))
            self.edit_vars['scale_tolerance'] = scale_tolerance_var
            scale_tolerance_entry = ttk.Entry(scale_tolerance_frame, textvariable=scale_tolerance_var, width=10)
            scale_tolerance_entry.pack(side=LEFT, padx=(5, 0))
            
            # Tooltip para scale_tolerance
            scale_tip_label = ttk.Label(scale_tolerance_frame, text="ℹ️", foreground="blue")
            scale_tip_label.pack(side=LEFT, padx=(2, 0))
            scale_tip_label.bind("<Button-1>", lambda e: messagebox.showinfo("Scale Tolerance", 
                "Permite variação no tamanho do objeto detectado em relação ao template.\nValor 0.1 = ±10% de variação\nValor padrão: 0.1"))
        
        # Separador para seção de personalização
        separator = ttk.Separator(fields_frame, orient='horizontal')
        separator.pack(fill=X, pady=(15, 5))
        
        # Título para personalização
        style_label = ttk.Label(fields_frame, text="Personalização de Aparência:", 
                              font=('Arial', 9, 'bold'),
                              foreground=self.style_config["text_color"])
        style_label.pack(pady=(0, 5))
        
        # Cor de fundo
        bg_color_frame = ttk.Frame(fields_frame)
        bg_color_frame.pack(fill=X, pady=2)
        
        bg_color_label = ttk.Label(bg_color_frame, text="Cor de Fundo:", width=12,
                                 foreground=self.style_config["text_color"])
        bg_color_label.pack(side=LEFT)
        
        bg_color_var = ttk.StringVar(value=self.style_config["background_color"])
        self.edit_vars['background_color'] = bg_color_var
        bg_color_entry = ttk.Entry(bg_color_frame, textvariable=bg_color_var, width=10)
        bg_color_entry.pack(side=LEFT, padx=(5, 0))
        
        # Botão para escolher cor de fundo
        bg_color_btn = ttk.Button(bg_color_frame, text="Escolher", 
                                command=lambda: self.choose_color("background_color"))
        bg_color_btn.pack(side=LEFT, padx=(5, 0))
        
        # Cor do texto
        text_color_frame = ttk.Frame(fields_frame)
        text_color_frame.pack(fill=X, pady=2)
        
        text_color_label = ttk.Label(text_color_frame, text="Cor do Texto:", width=12,
                                   foreground=self.style_config["text_color"])
        text_color_label.pack(side=LEFT)
        
        text_color_var = ttk.StringVar(value=self.style_config["text_color"])
        self.edit_vars['text_color'] = text_color_var
        text_color_entry = ttk.Entry(text_color_frame, textvariable=text_color_var, width=10)
        text_color_entry.pack(side=LEFT, padx=(5, 0))
        
        # Botão para escolher cor do texto
        text_color_btn = ttk.Button(text_color_frame, text="Escolher", 
                                  command=lambda: self.choose_color("text_color"))
        text_color_btn.pack(side=LEFT, padx=(5, 0))
        
        # Cor do texto NG
        ng_color_frame = ttk.Frame(fields_frame)
        ng_color_frame.pack(fill=X, pady=2)
        
        ng_color_label = ttk.Label(ng_color_frame, text="Cor do NG:", width=12,
                                 foreground=self.style_config["text_color"])
        ng_color_label.pack(side=LEFT)
        
        ng_color_var = ttk.StringVar(value=self.style_config["ng_color"])
        self.edit_vars['ng_color'] = ng_color_var
        ng_color_entry = ttk.Entry(ng_color_frame, textvariable=ng_color_var, width=10)
        ng_color_entry.pack(side=LEFT, padx=(5, 0))
        
        # Botão para escolher cor do NG
        ng_color_btn = ttk.Button(ng_color_frame, text="Escolher", 
                                command=lambda: self.choose_color("ng_color"))
        ng_color_btn.pack(side=LEFT, padx=(5, 0))
        
        # Cor do texto OK
        ok_color_frame = ttk.Frame(fields_frame)
        ok_color_frame.pack(fill=X, pady=2)
        
        ok_color_label = ttk.Label(ok_color_frame, text="Cor do OK:", width=12,
                                 foreground=self.style_config["text_color"])
        ok_color_label.pack(side=LEFT)
        
        ok_color_var = ttk.StringVar(value=self.style_config["ok_color"])
        self.edit_vars['ok_color'] = ok_color_var
        ok_color_entry = ttk.Entry(ok_color_frame, textvariable=ok_color_var, width=10)
        ok_color_entry.pack(side=LEFT, padx=(5, 0))
        
        # Botão para escolher cor do OK
        ok_color_btn = ttk.Button(ok_color_frame, text="Escolher", 
                                command=lambda: self.choose_color("ok_color"))
        ok_color_btn.pack(side=LEFT, padx=(5, 0))
        
        # Cor do quadro de seleção
        selection_color_frame = ttk.Frame(fields_frame)
        selection_color_frame.pack(fill=X, pady=2)
        
        selection_color_label = ttk.Label(selection_color_frame, text="Cor Seleção:", width=12,
                                        foreground=self.style_config["text_color"])
        selection_color_label.pack(side=LEFT)
        
        selection_color_var = ttk.StringVar(value=self.style_config["selection_color"])
        self.edit_vars['selection_color'] = selection_color_var
        selection_color_entry = ttk.Entry(selection_color_frame, textvariable=selection_color_var, width=10)
        selection_color_entry.pack(side=LEFT, padx=(5, 0))
        
        # Botão para escolher cor do quadro de seleção
        selection_color_btn = ttk.Button(selection_color_frame, text="Escolher", 
                                       command=lambda: self.choose_color("selection_color"))
        selection_color_btn.pack(side=LEFT, padx=(5, 0))
        
        # Fonte do NG
        ng_font_frame = ttk.Frame(fields_frame)
        ng_font_frame.pack(fill=X, pady=2)
        
        ng_font_label = ttk.Label(ng_font_frame, text="Fonte do NG:", width=12,
                                foreground=self.style_config["text_color"])
        ng_font_label.pack(side=LEFT)
        
        ng_font_var = ttk.StringVar(value=self.style_config["ng_font"])
        self.edit_vars['ng_font'] = ng_font_var
        ng_font_entry = ttk.Entry(ng_font_frame, textvariable=ng_font_var, width=20)
        ng_font_entry.pack(side=LEFT, padx=(5, 0))
        
        # Botão para escolher fonte do NG
        ng_font_btn = ttk.Button(ng_font_frame, text="Escolher", 
                               command=lambda: self.choose_font("ng_font"))
        ng_font_btn.pack(side=LEFT, padx=(5, 0))
        
        # Fonte do OK
        ok_font_frame = ttk.Frame(fields_frame)
        ok_font_frame.pack(fill=X, pady=2)
        
        ok_font_label = ttk.Label(ok_font_frame, text="Fonte do OK:", width=12,
                                foreground=self.style_config["text_color"])
        ok_font_label.pack(side=LEFT)
        
        ok_font_var = ttk.StringVar(value=self.style_config["ok_font"])
        self.edit_vars['ok_font'] = ok_font_var
        ok_font_entry = ttk.Entry(ok_font_frame, textvariable=ok_font_var, width=20)
        ok_font_entry.pack(side=LEFT, padx=(5, 0))
        
        # Botão para escolher fonte do OK
        ok_font_btn = ttk.Button(ok_font_frame, text="Escolher", 
                               command=lambda: self.choose_font("ok_font"))
        ok_font_btn.pack(side=LEFT, padx=(5, 0))
        
        # Frame para botões (fora do canvas, fixo na parte inferior)
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='x', pady=(10, 0), side='bottom')
        
        # Botão Salvar
        save_btn = ttk.Button(buttons_frame, text="Salvar", 
                             command=lambda: self.save_inline_edit(slot_data),
                             style='success.TButton')
        save_btn.pack(side=LEFT, padx=(0, 5))
        
        # Botão Cancelar
        cancel_btn = ttk.Button(buttons_frame, text="Cancelar", 
                               command=self.cancel_inline_edit,
                               style='secondary.TButton')
        cancel_btn.pack(side=LEFT)
        
        # Limpa o bind do mousewheel quando a janela for fechada
        def on_close():
            canvas.unbind_all("<MouseWheel>")
            self.cancel_inline_edit()
        
        self.edit_menu_frame.protocol("WM_DELETE_WINDOW", on_close)
    
    def choose_color(self, color_key):
        """Abre o seletor de cores e atualiza o campo correspondente"""
        try:
            # Obtém a cor atual
            current_color = self.edit_vars[color_key].get()
            
            # Abre o seletor de cores
            color = colorchooser.askcolor(initialcolor=current_color, title=f"Escolher cor para {color_key}")
            
            # Se o usuário selecionou uma cor (não cancelou)
            if color and color[1]:
                # Atualiza o campo com a cor selecionada (formato hexadecimal)
                self.edit_vars[color_key].set(color[1])
                
                # Atualiza a interface para refletir a nova cor
                if color_key == "background_color":
                    self.edit_menu_frame.configure(bg=color[1])
                    for widget in self.edit_menu_frame.winfo_children():
                        if isinstance(widget, Canvas):
                            widget.configure(bg=color[1])
        except Exception as e:
            print(f"Erro ao escolher cor: {e}")
            messagebox.showerror("Erro", f"Erro ao escolher cor: {str(e)}")
    
    def choose_font(self, font_key):
        """Abre um diálogo para escolher a fonte"""
        try:
            # Obtém a fonte atual
            current_font = self.edit_vars[font_key].get()
            
            # Lista de fontes disponíveis
            available_fonts = [
                "Arial", "Arial Black", "Calibri", "Cambria", "Comic Sans MS", 
                "Courier New", "Georgia", "Impact", "Tahoma", "Times New Roman", 
                "Trebuchet MS", "Verdana"
            ]
            
            # Lista de tamanhos de fonte
            font_sizes = ["8", "9", "10", "11", "12", "14", "16", "18", "20", "22", "24", "28", "32", "36", "48", "72"]
            
            # Lista de estilos de fonte
            font_styles = ["normal", "bold", "italic", "bold italic"]
            
            # Cria uma janela de diálogo para escolher a fonte
            font_dialog = Toplevel(self.edit_menu_frame)
            font_dialog.title(f"Escolher fonte para {font_key}")
            font_dialog.geometry("400x300")
            font_dialog.transient(self.edit_menu_frame)
            font_dialog.grab_set()
            
            # Centraliza a janela
            font_dialog.update_idletasks()
            x = (font_dialog.winfo_screenwidth() // 2) - (200)
            y = (font_dialog.winfo_screenheight() // 2) - (150)
            font_dialog.geometry(f"400x300+{x}+{y}")
            
            # Frame principal
            main_frame = ttk.Frame(font_dialog, padding=10)
            main_frame.pack(fill='both', expand=True)
            
            # Variáveis para armazenar a seleção
            font_family_var = ttk.StringVar()
            font_size_var = ttk.StringVar()
            font_style_var = ttk.StringVar()
            
            # Tenta extrair os componentes da fonte atual
            try:
                # Formato esperado: "família tamanho estilo"
                font_parts = current_font.split()
                if len(font_parts) >= 3:
                    font_family_var.set(font_parts[0])
                    font_size_var.set(font_parts[1])
                    font_style_var.set(" ".join(font_parts[2:]))
                else:
                    # Valores padrão se não conseguir extrair
                    font_family_var.set("Arial")
                    font_size_var.set("12")
                    font_style_var.set("bold")
            except:
                # Valores padrão em caso de erro
                font_family_var.set("Arial")
                font_size_var.set("12")
                font_style_var.set("bold")
            
            # Frame para família da fonte
            family_frame = ttk.Frame(main_frame)
            family_frame.pack(fill='x', pady=5)
            
            ttk.Label(family_frame, text="Família:").pack(side=LEFT)
            family_combo = ttk.Combobox(family_frame, textvariable=font_family_var, values=available_fonts, width=20)
            family_combo.pack(side=LEFT, padx=(5, 0))
            
            # Frame para tamanho da fonte
            size_frame = ttk.Frame(main_frame)
            size_frame.pack(fill='x', pady=5)
            
            ttk.Label(size_frame, text="Tamanho:").pack(side=LEFT)
            size_combo = ttk.Combobox(size_frame, textvariable=font_size_var, values=font_sizes, width=10)
            size_combo.pack(side=LEFT, padx=(5, 0))
            
            # Frame para estilo da fonte
            style_frame = ttk.Frame(main_frame)
            style_frame.pack(fill='x', pady=5)
            
            ttk.Label(style_frame, text="Estilo:").pack(side=LEFT)
            style_combo = ttk.Combobox(style_frame, textvariable=font_style_var, values=font_styles, width=15)
            style_combo.pack(side=LEFT, padx=(5, 0))
            
            # Frame para visualização
            preview_frame = ttk.Frame(main_frame, height=100)
            preview_frame.pack(fill='x', pady=10)
            preview_frame.pack_propagate(False)
            
            preview_label = ttk.Label(preview_frame, text="Texto de exemplo AaBbCcDd 123")
            preview_label.pack(expand=True)
            
            # Função para atualizar a visualização
            def update_preview(*args):
                try:
                    font_family = font_family_var.get()
                    font_size = int(font_size_var.get())
                    font_style = font_style_var.get()
                    
                    # Configura a fonte para o preview
                    preview_font = (font_family, font_size, font_style)
                    preview_label.configure(font=preview_font)
                except Exception as e:
                    print(f"Erro ao atualizar preview: {e}")
            
            # Vincula as variáveis à função de atualização
            font_family_var.trace_add("write", update_preview)
            font_size_var.trace_add("write", update_preview)
            font_style_var.trace_add("write", update_preview)
            
            # Atualiza o preview inicialmente
            update_preview()
            
            # Frame para botões
            buttons_frame = ttk.Frame(main_frame)
            buttons_frame.pack(fill='x', pady=10)
            
            # Função para aplicar a fonte selecionada
            def apply_font():
                try:
                    font_family = font_family_var.get()
                    font_size = font_size_var.get()
                    font_style = font_style_var.get()
                    
                    # Formata a string da fonte
                    font_string = f"{font_family} {font_size} {font_style}"
                    
                    # Atualiza a variável
                    self.edit_vars[font_key].set(font_string)
                    
                    # Fecha o diálogo
                    font_dialog.destroy()
                except Exception as e:
                    print(f"Erro ao aplicar fonte: {e}")
                    messagebox.showerror("Erro", f"Erro ao aplicar fonte: {str(e)}")
            
            # Botão OK
            ttk.Button(buttons_frame, text="OK", command=apply_font).pack(side=LEFT, padx=(0, 5))
            
            # Botão Cancelar
            ttk.Button(buttons_frame, text="Cancelar", command=font_dialog.destroy).pack(side=LEFT)
            
            # Torna a janela modal
            font_dialog.wait_window()
            
        except Exception as e:
            print(f"Erro ao escolher fonte: {e}")
            messagebox.showerror("Erro", f"Erro ao escolher fonte: {str(e)}")
    
    def save_inline_edit(self, slot_data):
        """Salva as alterações do menu inline"""
        try:
            # Atualiza os dados básicos do slot
            slot_data['x'] = int(self.edit_vars['x'].get())
            slot_data['y'] = int(self.edit_vars['y'].get())
            slot_data['w'] = int(self.edit_vars['w'].get())
            slot_data['h'] = int(self.edit_vars['h'].get())
            
            # Atualiza os parâmetros específicos para clips
            if slot_data.get('tipo') == 'clip':
                if 'detection_method' in self.edit_vars:
                    slot_data['detection_method'] = self.edit_vars['detection_method'].get()
                if 'detection_threshold' in self.edit_vars:
                    slot_data['detection_threshold'] = float(self.edit_vars['detection_threshold'].get())
                if 'correlation_threshold' in self.edit_vars:
                    slot_data['correlation_threshold'] = float(self.edit_vars['correlation_threshold'].get())
                if 'template_method' in self.edit_vars:
                    slot_data['template_method'] = self.edit_vars['template_method'].get()
                if 'scale_tolerance' in self.edit_vars:
                    slot_data['scale_tolerance'] = float(self.edit_vars['scale_tolerance'].get())
            
            # Salva no banco de dados se há um modelo carregado
            if self.current_model_id is not None:
                try:
                    self.db_manager.update_slot(slot_data['db_id'], slot_data)
                except Exception as e:
                    print(f"Erro ao salvar slot no banco: {e}")
            
            # Atualiza as configurações de estilo
            style_config = {}
            style_keys = ["background_color", "text_color", "ng_color", "ok_color", 
                        "ng_font", "ok_font", "selection_color"]
            
            for key in style_keys:
                if key in self.edit_vars:
                    style_config[key] = self.edit_vars[key].get()
            
            # Salva as configurações de estilo
            if style_config:
                # Carrega as configurações atuais para manter valores não editados
                current_config = load_style_config()
                # Atualiza com os novos valores
                current_config.update(style_config)
                # Salva no arquivo
                save_style_config(current_config)
                print("Configurações de estilo salvas com sucesso")
            
            # Atualiza a exibição
            self.redraw_slots()
            self.update_slots_list()
            
            # Remove o menu de edição
            self.cancel_inline_edit()
            
            # Marca modelo como modificado
            self.mark_model_modified()
            
            print(f"Slot {slot_data['id']} atualizado com sucesso")
            messagebox.showinfo("Sucesso", f"Slot {slot_data['id']} foi atualizado com sucesso!")
            
        except ValueError as e:
            messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos.")
        except Exception as e:
            print(f"Erro ao salvar: {e}")
            messagebox.showerror("Erro", f"Erro ao salvar alterações: {str(e)}")
    
    def cancel_inline_edit(self):
        """Cancela a edição inline"""
        if hasattr(self, 'edit_menu_frame') and self.edit_menu_frame:
            self.edit_menu_frame.destroy()
            self.edit_menu_frame = None
        if hasattr(self, 'edit_vars'):
            self.edit_vars = None
    
    def update_slot_data(self, updated_slot_data):
        """Atualiza os dados de um slot específico."""
        slot_id_to_update = updated_slot_data.get('id')
        if slot_id_to_update is None:
            print("ERRO: ID do slot não encontrado nos dados atualizados")
            return
        
        print(f"\n=== ATUALIZANDO SLOT {slot_id_to_update} NA LISTA ===")
        print(f"Dados recebidos: {updated_slot_data}")
        
        found = False
        for i, slot in enumerate(self.slots):
            if slot['id'] == slot_id_to_update:
                print(f"Slot encontrado na posição {i}")
                print(f"Dados antigos: {slot}")
                
                # Preserva canvas_id se existir
                updated_slot_data['canvas_id'] = slot.get('canvas_id')
                
                # Substitui o slot na lista
                self.slots[i] = updated_slot_data
                found = True
                
                print(f"Dados novos: {self.slots[i]}")
                print(f"Slot {slot_id_to_update} atualizado com sucesso na lista.")
                break
        
        if not found:
            print(f"ERRO: Slot {slot_id_to_update} não encontrado na lista para update.")
            print(f"Slots disponíveis: {[s.get('id') for s in self.slots]}")
            return
        
        print("Atualizando interface...")
        self.deselect_all_slots()
        self.redraw_slots()
        self.update_slots_list()
        
        # Marca o modelo como modificado
        self.mark_model_modified()
        
        print("Interface atualizada com sucesso!")    
    def delete_selected_slot(self):
        """Remove o slot selecionado."""
        if self.selected_slot_id is None:
            messagebox.showwarning("Aviso", "Selecione um slot para deletar.")
            return
        
        if messagebox.askyesno("Confirmar", f"Deletar slot {self.selected_slot_id}?"):
            # Encontra o slot a ser removido
            slot_to_remove = None
            for slot in self.slots:
                if slot['id'] == self.selected_slot_id:
                    slot_to_remove = slot
                    break
            
            # Remove do banco de dados se há um modelo carregado
            if self.current_model_id is not None and slot_to_remove and 'db_id' in slot_to_remove:
                try:
                    self.db_manager.delete_slot(slot_to_remove['db_id'])
                except Exception as e:
                    print(f"Erro ao remover slot do banco: {e}")
            
            # Remove slot da lista
            self.slots = [slot for slot in self.slots if slot['id'] != self.selected_slot_id]
            
            # Remove seleção
            self.selected_slot_id = None
            
            # Atualiza interface
            self.update_slots_list()
            self.redraw_slots()
            self.status_var.set("Slot deletado")
            self.update_button_states()
            
            # Marca modelo como modificado
            self.mark_model_modified()
    
    def train_selected_slot(self):
        """Abre o diálogo de treinamento para o slot selecionado."""
        if self.selected_slot_id is None:
            messagebox.showwarning("Aviso", "Nenhum slot selecionado.")
            return
        
        # Encontra o slot
        selected_slot = None
        for slot in self.slots:
            if slot['id'] == self.selected_slot_id:
                selected_slot = slot
                break
        
        if selected_slot is None:
            messagebox.showerror("Erro", "Slot não encontrado.")
            return
        
        if selected_slot.get('tipo') != 'clip':
            messagebox.showwarning("Aviso", "Treinamento disponível apenas para slots do tipo 'clip'.")
            return
        
        # Abre diálogo de treinamento
        dialog = SlotTrainingDialog(self.master, selected_slot, self)
        dialog.wait_window()
        
        # Atualiza interface após treinamento
        self.redraw_slots()
        self.update_slots_list()
    
    def save_model(self):
        """Salva o modelo atual no banco de dados."""
        if self.img_original is None:
            messagebox.showerror("Erro", "Nenhuma imagem carregada.")
            return
        
        if not self.slots:
            messagebox.showwarning("Aviso", "Nenhum slot definido para salvar.")
            return
        
        # Abre diálogo para salvar modelo
        dialog = SaveModelDialog(self, self.db_manager, self.current_model_id)
        result = dialog.show()
        
        if not result:
            return
        
        try:
            # Determina o nome do modelo
            if 'name' in result:
                model_name = result['name']
            elif result['action'] == 'overwrite' and 'model_id' in result:
                # Para sobrescrever, busca o nome do modelo existente
                existing_model = self.db_manager.load_modelo(result['model_id'])
                model_name = existing_model['nome']
            else:
                raise ValueError("Nome do modelo não encontrado")
            
            if result['action'] in ['update', 'overwrite']:
                # Atualiza modelo existente
                model_id = result['model_id']
                
                # Obtém pasta específica do modelo
                model_folder = self.db_manager.get_model_folder_path(model_name, model_id)
                
                # Salva imagem de referência na pasta do modelo
                image_filename = f"{model_name}_reference.jpg"
                image_path = model_folder / image_filename
                cv2.imwrite(str(image_path), self.img_original)
                
                self.db_manager.update_modelo(
                    model_id,
                    nome=model_name,
                    image_path=str(image_path),
                    slots=self.slots
                )
                
                self.current_model_id = model_id
                
            else:
                # Cria novo modelo primeiro para obter o ID
                # Salva temporariamente com caminho vazio
                model_id = self.db_manager.save_modelo(
                    nome=model_name,
                    image_path="",  # Será atualizado depois
                    slots=self.slots
                )
                
                # Obtém pasta específica do modelo (já criada pelo save_modelo)
                model_folder = self.db_manager.get_model_folder_path(model_name, model_id)
                
                # Salva imagem de referência na pasta do modelo
                image_filename = f"{model_name}_reference.jpg"
                image_path = model_folder / image_filename
                cv2.imwrite(str(image_path), self.img_original)
                
                # Atualiza o caminho da imagem no banco
                self.db_manager.update_modelo(
                    model_id,
                    image_path=str(image_path)
                )
                
                self.current_model_id = model_id
            
            # Marca o modelo como salvo
            self.mark_model_saved()
            
            print(f"Modelo '{model_name}' salvo com sucesso no banco de dados")
            messagebox.showinfo("Sucesso", f"Modelo '{model_name}' salvo com {len(self.slots)} slots.")
            
        except Exception as e:
            print(f"Erro ao salvar modelo: {e}")
            messagebox.showerror("Erro", f"Erro ao salvar modelo: {str(e)}")
    
    def update_button_states(self):
        """Atualiza estado dos botões baseado no estado atual."""
        has_image = self.img_original is not None
        has_slots = len(self.slots) > 0
        has_selection = self.selected_slot_id is not None
        
        # Botões que dependem de imagem
        self.btn_save_model.config(state=NORMAL if has_image and has_slots else DISABLED)
        
        # Botões que dependem de slots
        self.btn_clear_slots.config(state=NORMAL if has_slots else DISABLED)
        
        # Botões que dependem de seleção
        self.btn_edit_slot.config(state=NORMAL if has_selection else DISABLED)
        self.btn_delete_slot.config(state=NORMAL if has_selection else DISABLED)
        self.btn_train_slot.config(state=NORMAL if has_selection else DISABLED)
    
    def set_drawing_mode(self):
        """Define o modo de desenho atual."""
        self.current_drawing_mode = self.drawing_mode.get()
        mode_names = {
            "rectangle": "Retângulo",
            "exclusion": "Área de Exclusão"
        }
        self.tool_status_var.set(f"Modo: {mode_names.get(self.current_drawing_mode, 'Desconhecido')}")
        print(f"Modo de desenho alterado para: {self.current_drawing_mode}")
    
    def adjust_rotation(self, delta):
        """Ajusta a rotação em graus."""
        try:
            current = float(self.rotation_var.get())
            new_rotation = (current + delta) % 360
            self.rotation_var.set(str(int(new_rotation)))
            self.current_rotation = new_rotation
            print(f"Rotação ajustada para: {new_rotation}°")
        except ValueError:
            self.rotation_var.set("0")
            self.current_rotation = 0
    
    def add_exclusion_area(self, x, y, w, h):
        """Adiciona área de exclusão ao slot selecionado."""
        if self.selected_slot_id is None:
            messagebox.showwarning("Aviso", "Selecione um slot primeiro para adicionar área de exclusão.")
            return
        
        # Encontra o slot selecionado
        selected_slot = None
        for slot in self.slots:
            if slot['id'] == self.selected_slot_id:
                selected_slot = slot
                break
        
        if selected_slot is None:
            messagebox.showerror("Erro", "Slot selecionado não encontrado.")
            return
        
        # Adiciona área de exclusão (sem verificação de limites)
        exclusion_area = {
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'shape': self.current_drawing_mode
        }
        
        selected_slot['exclusion_areas'].append(exclusion_area)
        self.mark_model_modified()
        self.redraw_slots()
        
        print(f"Área de exclusão adicionada ao slot {self.selected_slot_id}: ({x}, {y}, {w}, {h})")
        self.status_var.set(f"Área de exclusão adicionada ao slot {self.selected_slot_id}")
    
    def show_edit_handles(self, slot):
        """Mostra handles de edição para o slot selecionado."""
        self.hide_edit_handles()  # Remove handles anteriores
        
        x = slot['x'] * self.scale_factor
        y = slot['y'] * self.scale_factor
        w = slot['w'] * self.scale_factor
        h = slot['h'] * self.scale_factor
        
        handle_size = 8
        handle_color = "#FF4444"
        
        # Handles de redimensionamento (cantos e meio das bordas)
        handles = [
            # Cantos
            (x - handle_size//2, y - handle_size//2, "nw"),  # Canto superior esquerdo
            (x + w - handle_size//2, y - handle_size//2, "ne"),  # Canto superior direito
            (x - handle_size//2, y + h - handle_size//2, "sw"),  # Canto inferior esquerdo
            (x + w - handle_size//2, y + h - handle_size//2, "se"),  # Canto inferior direito
            # Meio das bordas
            (x + w//2 - handle_size//2, y - handle_size//2, "n"),  # Meio superior
            (x + w//2 - handle_size//2, y + h - handle_size//2, "s"),  # Meio inferior
            (x - handle_size//2, y + h//2 - handle_size//2, "w"),  # Meio esquerdo
            (x + w - handle_size//2, y + h//2 - handle_size//2, "e"),  # Meio direito
        ]
        
        # Handle de rotação (acima do slot)
        rotation_handle_y = y - 30
        rotation_handle = self.canvas.create_oval(
            x + w//2 - handle_size//2, rotation_handle_y - handle_size//2,
            x + w//2 + handle_size//2, rotation_handle_y + handle_size//2,
            fill="#4444FF", outline="white", width=2,
            tags=("edit_handle", "rotation_handle")
        )
        
        # Linha conectando o handle de rotação ao slot
        self.canvas.create_line(
            x + w//2, y, x + w//2, rotation_handle_y,
            fill="#4444FF", width=2, tags="edit_handle"
        )
        
        # Cria handles de redimensionamento
        for hx, hy, direction in handles:
            handle = self.canvas.create_rectangle(
                hx, hy, hx + handle_size, hy + handle_size,
                fill=handle_color, outline="white", width=2,
                tags=("edit_handle", f"resize_handle_{direction}")
            )
        
        # Bind eventos para os handles
        self.canvas.tag_bind("edit_handle", "<Button-1>", self.on_handle_press)
        self.canvas.tag_bind("edit_handle", "<B1-Motion>", self.on_handle_drag)
        self.canvas.tag_bind("edit_handle", "<ButtonRelease-1>", self.on_handle_release)
    
    def hide_edit_handles(self):
        """Esconde todos os handles de edição."""
        self.canvas.delete("edit_handle")
        self.editing_handle = None
    
    def on_handle_press(self, event):
        """Inicia edição com handle."""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Encontra qual handle foi clicado
        closest_items = self.canvas.find_closest(canvas_x, canvas_y)
        if closest_items:
            item = closest_items[0]
            tags = self.canvas.gettags(item)
            
            for tag in tags:
                if tag.startswith("resize_handle_"):
                    self.editing_handle = {
                        'type': 'resize',
                        'direction': tag.split('_')[-1],
                        'start_x': canvas_x,
                        'start_y': canvas_y
                    }
                    break
                elif tag == "rotation_handle":
                    self.editing_handle = {
                        'type': 'rotation',
                        'start_x': canvas_x,
                        'start_y': canvas_y
                    }
                    break
    
    def on_handle_drag(self, event):
        """Processa arrastar do handle."""
        if not self.editing_handle or self.selected_slot_id is None:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Encontra o slot selecionado
        selected_slot = None
        for slot in self.slots:
            if slot['id'] == self.selected_slot_id:
                selected_slot = slot
                break
        
        if not selected_slot:
            return
        
        if self.editing_handle['type'] == 'resize':
            self.handle_resize_drag(selected_slot, canvas_x, canvas_y)
        elif self.editing_handle['type'] == 'rotation':
            self.handle_rotation_drag(selected_slot, canvas_x, canvas_y)
    
    def handle_resize_drag(self, slot, canvas_x, canvas_y):
        """Lida com redimensionamento do slot."""
        direction = self.editing_handle['direction']
        
        # Converte coordenadas do canvas para coordenadas da imagem
        img_x = canvas_x / self.scale_factor
        img_y = canvas_y / self.scale_factor
        
        # Calcula novas dimensões baseadas na direção do handle
        new_x, new_y = slot['x'], slot['y']
        new_w, new_h = slot['w'], slot['h']
        
        if 'w' in direction:  # Lado esquerdo
            new_w = slot['x'] + slot['w'] - img_x
            new_x = img_x
        elif 'e' in direction:  # Lado direito
            new_w = img_x - slot['x']
        
        if 'n' in direction:  # Lado superior
            new_h = slot['y'] + slot['h'] - img_y
            new_y = img_y
        elif 's' in direction:  # Lado inferior
            new_h = img_y - slot['y']
        
        # Garante dimensões mínimas
        if new_w < 10:
            new_w = 10
        if new_h < 10:
            new_h = 10
        
        # Atualiza o slot
        slot['x'] = max(0, new_x)
        slot['y'] = max(0, new_y)
        slot['w'] = new_w
        slot['h'] = new_h
        
        # Marca modelo como modificado e atualiza interface
        self.mark_model_modified()
        self.redraw_slots()
        self.show_edit_handles(slot)
        self.update_slots_list()
    
    def handle_rotation_drag(self, slot, canvas_x, canvas_y):
        """Lida com rotação do slot."""
        # Calcula o centro do slot
        center_x = (slot['x'] + slot['w'] / 2) * self.scale_factor
        center_y = (slot['y'] + slot['h'] / 2) * self.scale_factor
        
        # Calcula o ângulo baseado na posição do mouse
        import math
        angle = math.degrees(math.atan2(canvas_y - center_y, canvas_x - center_x))
        
        # Arredonda para incrementos de 15 graus
        angle = round(angle / 15) * 15
        
        # Atualiza a rotação do slot
        slot['rotation'] = angle
        
        # Marca modelo como modificado e atualiza interface
        self.mark_model_modified()
        self.redraw_slots()
        self.show_edit_handles(slot)
        self.update_slots_list()
        
        # Atualiza o campo de rotação na interface
        self.rotation_var.set(str(int(angle)))
    
    def on_handle_release(self, event):
        """Finaliza edição com handle."""
        if self.editing_handle:
            self.mark_model_modified()
            self.editing_handle = None
     
    def show_help(self):
        """Mostra janela de ajuda."""
        help_window = Toplevel(self.master)
        help_window.title("Ajuda - Editor de Malha")
        help_window.geometry("600x500")
        help_window.resizable(True, True)
        
        # Torna a janela modal
        help_window.transient(self.master)
        help_window.grab_set()
        
        # Texto de ajuda
        help_text = """
# Editor de Malha - Ajuda

## Como usar:

### 1. Carregar Imagem
- Clique em "Carregar Imagem" para selecionar uma imagem de referência
- Formatos suportados: JPG, PNG, BMP, TIFF

### 2. Criar Slots
- Clique e arraste no canvas para desenhar um retângulo
- Apenas slots do tipo 'clip' são suportados
- Será salvo um template da região para template matching

### 3. Gerenciar Slots
- Clique em um slot para selecioná-lo
- Use "Editar Slot" para modificar configurações
- Use "Deletar Slot" para remover um slot
- Use "Limpar Slots" para remover todos os slots

### 4. Salvar/Carregar Modelos
- Use "Salvar Modelo" para salvar a configuração atual
- Use "Carregar Modelo" para carregar uma configuração existente
- Os modelos são salvos em formato JSON

### 5. Cores dos Slots
- Clips: Vermelho coral
- Selecionado: Amarelo dourado
- Desenhando: Verde claro

## Dicas:
- Slots muito pequenos (< 10x10 pixels) não são aceitos
- Templates de clips são salvos automaticamente
- Use zoom e scroll para trabalhar com imagens grandes
- Modelos salvam caminhos relativos para portabilidade
"""
        
        # Frame principal
        main_frame = ttk.Frame(help_window)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Área de texto com scroll
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Texto
        text_widget = Text(text_frame, wrap="word", yscrollcommand=scrollbar.set,
                          font=("Consolas", 10), bg="#F8F9FA", fg="#2C3E50")
        text_widget.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Insere texto
        text_widget.insert("1.0", help_text)
        text_widget.config(state=DISABLED)
        
        # Botão fechar
        ttk.Button(main_frame, text="Fechar", 
                  command=help_window.destroy).pack(pady=(10, 0))
        
        # Centralizar janela
        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() // 2) - (help_window.winfo_width() // 2)
        y = (help_window.winfo_screenheight() // 2) - (help_window.winfo_height() // 2)
        help_window.geometry(f"+{x}+{y}")
    
    def open_system_config(self):
        """Abre a janela de configuração do sistema."""
        config_dialog = SystemConfigDialog(self.master)
        config_dialog.wait_window()
    
    def on_closing(self):
        """Limpa recursos ao fechar a aplicação."""
        if self.live_capture:
            self.stop_live_capture()
        self.master.destroy()


# Classe para aba de Inspeção
class InspecaoWindow(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        
        # Dados da aplicação
        self.img_reference = None
        self.img_test = None
        self.img_display = None
        self.scale_factor = 1.0
        self.slots = []
        self.current_model_id = None
        self.inspection_results = []
        
        # Inicializa gerenciador de banco de dados
        # Usa caminho absoluto baseado na raiz do projeto
        db_path = MODEL_DIR / "models.db"
        self.db_manager = DatabaseManager(str(db_path))
        
        # Estado da inspeção
        self.live_view = False
        self.camera = None
        self.live_capture = False
        self.latest_frame = None
        
        # Controle de webcam
        self.available_cameras = detect_cameras()
        self.selected_camera = 0
        
        self.setup_ui()
        self.update_button_states()
    
    def setup_ui(self):
        # Configuração de estilo industrial Keyence
        self.style = ttk.Style()
        
        # Carrega as configurações de estilo personalizadas
        style_config = load_style_config()
        
        # Cores industriais Keyence com personalização
        self.bg_color = style_config["background_color"]  # Fundo escuro mais profundo
        self.panel_color = "#2A2A2A"  # Cor dos painéis
        self.accent_color = style_config["button_color"]  # Cor de destaque
        self.success_color = style_config["ok_color"]  # Verde brilhante industrial
        self.warning_color = "#FFCC00"  # Amarelo industrial
        self.danger_color = style_config["ng_color"]  # Vermelho industrial
        self.text_color = style_config["text_color"]  # Texto branco
        self.button_bg = "#3A3A3A"  # Cor de fundo dos botões
        self.button_active = "#4A4A4A"  # Cor quando botão ativo
        
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
                       background=[('active', '#FF7733'), ('pressed', '#CC4400')])
        
        # Estilos para resultados
        self.style.configure('Success.TFrame', background='#004400')
        self.style.configure('Danger.TFrame', background='#440000')
        
        # Estilos para Entry e Combobox
        self.style.configure('TEntry', fieldbackground='#2A2A2A', foreground=self.text_color)
        self.style.map('TEntry',
                       fieldbackground=[('readonly', '#2A2A2A')],
                       foreground=[('readonly', self.text_color)])
        
        self.style.configure('TCombobox', fieldbackground='#2A2A2A', foreground=self.text_color, selectbackground='#3A3A3A')
        self.style.map('TCombobox',
                       fieldbackground=[('readonly', '#2A2A2A')],
                       foreground=[('readonly', self.text_color)])
        
        # Configurar cores para a interface - usando style em vez de configure diretamente
        # Nota: widgets ttk não suportam configuração direta de background
        # self.configure(background=self.bg_color) # Esta linha causava erro
        
        # Frame principal com layout horizontal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Painel esquerdo - Controles
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 10))
        
        # Painel direito - Canvas
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)
        
        # === PAINEL ESQUERDO ===
        
        # Cabeçalho com título estilo Keyence
        header_frame = ttk.Frame(left_panel, style='Header.TFrame')
        header_frame.pack(fill=X, pady=(0, 15))
        
        # Estilo para o cabeçalho
        self.style.configure('Header.TFrame', background=self.accent_color)
        
        # Logo e título
        header_label = ttk.Label(header_frame, text="KEYENCE VISION SYSTEM", 
                                font=style_config["ok_font"], foreground="white",
                                background=self.accent_color)
        header_label.pack(pady=10, fill=X)
        
        # Versão do sistema
        version_label = ttk.Label(header_frame, text="V1.0.0 - INDUSTRIAL INSPECTION", 
                                font=style_config["ok_font"].replace("12", "8"), foreground="white",
                                background=self.accent_color)
        version_label.pack(pady=(0, 10))
        
        # Seção de Modelo - Estilo industrial Keyence
        model_frame = ttk.LabelFrame(left_panel, text="MODELO DE INSPEÇÃO")
        model_frame.pack(fill=X, pady=(0, 10))
        
        # Indicador de modelo carregado
        model_indicator_frame = ttk.Frame(model_frame)
        model_indicator_frame.pack(fill=X, padx=5, pady=2)
        
        ttk.Label(model_indicator_frame, text="STATUS:", font=("Arial", 8, "bold")).pack(side=LEFT, padx=(0, 5))
        
        self.model_status_var = StringVar(value="NÃO CARREGADO")
        model_status = ttk.Label(model_indicator_frame, textvariable=self.model_status_var, 
                                foreground=self.danger_color, font=("Arial", 8, "bold"))
        model_status.pack(side=LEFT)
        
        # Botão com ícone industrial
        self.btn_load_model = ttk.Button(model_frame, text="CARREGAR MODELO ▼", 
                                       command=self.load_model_dialog, style="Action.TButton")
        self.btn_load_model.pack(fill=X, padx=5, pady=5)
        
        # Seção de Imagem de Teste - Estilo industrial
        test_frame = ttk.LabelFrame(left_panel, text="IMAGEM DE TESTE")
        test_frame.pack(fill=X, pady=(0, 10))
        
        self.btn_load_test = ttk.Button(test_frame, text="CARREGAR IMAGEM", 
                                       command=self.load_test_image)
        self.btn_load_test.pack(fill=X, padx=5, pady=2)
        
        # Seção de Webcam - Estilo industrial
        webcam_frame = ttk.LabelFrame(left_panel, text="CÂMERA")
        webcam_frame.pack(fill=X, pady=(0, 10))
        
        # Combobox para seleção de câmera
        camera_selection_frame = ttk.Frame(webcam_frame)
        camera_selection_frame.pack(fill=X, padx=5, pady=2)
        
        ttk.Label(camera_selection_frame, text="ID:").pack(side=LEFT)
        self.camera_combo = Combobox(camera_selection_frame, 
                                   values=[str(i) for i in self.available_cameras],
                                   state="readonly", width=5)
        self.camera_combo.pack(side=RIGHT)
        if self.available_cameras:
            self.camera_combo.set(str(self.available_cameras[0]))
        
        # Botão para iniciar/parar captura contínua
        self.btn_live_capture_inspection = ttk.Button(webcam_frame, text="INICIAR CAPTURA CONTÍNUA", 
                                                     command=self.toggle_live_capture_inspection)
        self.btn_live_capture_inspection.pack(fill=X, padx=5, pady=2)
        
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
        
        # Botão de inspeção com ícone industrial
        self.btn_inspect = ttk.Button(inspection_frame, text="▶ EXECUTAR INSPEÇÃO", 
                                    command=self.run_inspection, style="Inspect.TButton")
        self.btn_inspect.pack(fill=X, padx=5, pady=5)
        
        # Botão de inspeção contínua
        self.btn_continuous_inspect = ttk.Button(inspection_frame, text="⟳ INSPEÇÃO CONTÍNUA", 
                                              command=self.toggle_live_capture_inspection)
        self.btn_continuous_inspect.pack(fill=X, padx=5, pady=5)
        
        # Resultados - Estilo industrial Keyence
        results_frame = ttk.LabelFrame(left_panel, text="RESULTADOS DE INSPEÇÃO")
        results_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # Painel de resumo de resultados
        summary_frame = ttk.Frame(results_frame)
        summary_frame.pack(fill=X, padx=5, pady=5)
        
        # Criar painel de resumo de status
        self.create_status_summary_panel(summary_frame)
        
        # Lista de resultados com estilo industrial
        list_container = ttk.Frame(results_frame)
        list_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        scrollbar_results = ttk.Scrollbar(list_container)
        scrollbar_results.pack(side=RIGHT, fill=Y)
        
        # Configurar estilo da Treeview para parecer com sistemas Keyence
        self.style.configure("Treeview", 
                           background="#222222", 
                           foreground=self.text_color, 
                           fieldbackground="#222222",
                           borderwidth=1,
                           relief="solid")
        self.style.configure("Treeview.Heading", 
                           font=style_config["ok_font"], 
                           background="#444444", 
                           foreground="#FFFFFF")
        self.style.map("Treeview", 
                      background=[("selected", style_config["selection_color"])],
                      foreground=[("selected", "#000000")])
        
        self.results_listbox = ttk.Treeview(list_container, yscrollcommand=scrollbar_results.set, height=8)
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
        self.results_listbox.tag_configure("pass", background="#004400", foreground="#FFFFFF")
        self.results_listbox.tag_configure("fail", background="#440000", foreground="#FFFFFF")
        
        # === PAINEL DIREITO ===
        
        # Dividir painel direito em duas seções
        # Seção superior - Canvas de inspeção com estilo industrial
        canvas_frame = ttk.LabelFrame(right_panel, text="VISUALIZAÇÃO DE INSPEÇÃO")
        canvas_frame.pack(fill=BOTH, expand=True, pady=(0, 5))
        
        # Frame para canvas e scrollbars
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=VERTICAL)
        v_scrollbar.pack(side=RIGHT, fill=Y)
        
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=HORIZONTAL)
        h_scrollbar.pack(side=BOTTOM, fill=X)
        
        # Canvas com fundo escuro estilo industrial
        self.canvas = Canvas(canvas_container, bg="#121212",
                           yscrollcommand=v_scrollbar.set,
                           xscrollcommand=h_scrollbar.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Configurar scrollbars
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        
        # Seção inferior - Painel de resumo de status estilo Keyence IV3
        status_summary_frame = ttk.LabelFrame(right_panel, text="PAINEL DE STATUS")
        status_summary_frame.pack(fill=X, pady=(5, 0))
        
        # Frame interno para o grid de status
        self.status_grid_frame = ttk.Frame(status_summary_frame)
        self.status_grid_frame.pack(fill=X, padx=10, pady=10)
        
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
            
            # Criar painel de resumo de status
            self.create_status_summary_panel()
            
            self.status_var.set(f"Modelo carregado: {model_data['nome']} ({len(self.slots)} slots)")
            self.update_button_states()
            
            print(f"Modelo de inspeção '{model_data['nome']}' carregado com sucesso: {len(self.slots)} slots")
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            messagebox.showerror("Erro", f"Erro ao carregar modelo: {str(e)}")
    
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
                
                # Para de captura ao vivo se estiver ativa
                if self.live_view:
                    self.stop_live_view()
                
                # Limpa resultados de inspeção anteriores
                self.inspection_results = []
                
                self.update_display()
                self.status_var.set(f"Imagem de teste carregada: {Path(file_path).name}")
                self.update_button_states()
                
            except Exception as e:
                print(f"Erro ao carregar imagem de teste: {e}")
                messagebox.showerror("Erro", f"Erro ao carregar imagem: {str(e)}")
    
    def start_live_capture_inspection(self):
        """Inicia captura contínua da câmera em segundo plano para inspeção."""
        if self.live_capture:
            return
            
        try:
            camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
            
            # Para live view se estiver ativo
            if self.live_view:
                self.stop_live_view()
            
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            # Configurações otimizadas para inicialização mais rápida
            # Usa DirectShow no Windows para melhor compatibilidade
            # No Raspberry Pi, usa a API padrão
            if is_windows:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações otimizadas para performance e inicialização rápida
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Usa resolução nativa para câmeras externas (1920x1080) ou padrão para webcam interna
            if camera_index > 0:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            else:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.live_capture = True
            self.process_live_frame_inspection()
            self.status_var.set(f"Câmera {camera_index} ativa em segundo plano")
            
        except Exception as e:
            print(f"Erro ao iniciar câmera: {e}")
            messagebox.showerror("Erro", f"Erro ao iniciar câmera: {str(e)}")
    
    def stop_live_capture_inspection(self):
        """Para a captura contínua da câmera para inspeção."""
        self.live_capture = False
        if self.camera and not self.live_view:  # Não libera se live_view estiver usando
            self.camera.release()
            self.camera = None
        self.latest_frame = None
        if not self.live_view:
            self.status_var.set("Câmera desconectada")
    
    def toggle_live_capture_inspection(self):
        """Alterna entre iniciar e parar a captura contínua para inspeção."""
        if not self.live_capture:
            self.start_live_capture_inspection()
            if self.live_capture:  # Se iniciou com sucesso
                self.btn_live_capture_inspection.config(text="Parar Captura Contínua")
        else:
            self.stop_live_capture_inspection()
            self.btn_live_capture_inspection.config(text="Iniciar Captura Contínua")
    
    def process_live_frame_inspection(self):
        """Processa frames da câmera em segundo plano para inspeção."""
        if not self.live_capture or not self.camera:
            return
        
        try:
            ret, frame = self.camera.read()
            if ret:
                self.latest_frame = frame.copy()
        except Exception as e:
            print(f"Erro ao capturar frame: {e}")
            # Para a captura em caso de erro
            self.stop_live_capture_inspection()
            return
        
        # Agenda próximo frame (100ms para melhor estabilidade)
        if self.live_capture:
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
                if self.live_view:
                    self.stop_live_view()
                
                # Carrega a imagem capturada
                self.img_test = captured_image
                
                # Limpa resultados de inspeção anteriores
                self.inspection_results = []
                
                # Atualiza a exibição
                self.update_display()
                
                # Atualiza estado dos botões
                self.update_button_states()
                
                camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
                self.status_var.set(f"Imagem capturada da câmera {camera_index}")
                messagebox.showinfo("Sucesso", "Imagem capturada instantaneamente!")
                
                # Executa inspeção automática se modelo carregado
                if hasattr(self, 'slots') and self.slots and hasattr(self, 'img_reference') and self.img_reference is not None:
                    self.run_inspection()
            else:
                messagebox.showerror("Erro", "Nenhuma imagem disponível para captura.")
                
        except Exception as e:
            print(f"Erro ao capturar da webcam: {e}")
            messagebox.showerror("Erro", f"Erro ao capturar da webcam: {str(e)}")
    

    
    def stop_live_view(self):
        """Para a captura ao vivo."""
        self.live_view = False
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def process_live_frame(self):
        """Processa frame da câmera de forma otimizada"""
        if not self.live_view or not self.camera:
            return
        
        ret, frame = self.camera.read()
        if ret:
            self.img_test = frame
            self.update_display()
            
            # Inspeção automática otimizada (menos frequente)
            if self.slots and hasattr(self, '_frame_count'):
                self._frame_count += 1
                # Executa inspeção a cada 5 frames para melhor performance
                if self._frame_count % 5 == 0:
                    self.run_inspection(show_message=False)
            elif self.slots:
                self._frame_count = 0
        
        # Agenda próximo frame
        if self.live_view:
            self.master.after(100, self.process_live_frame)
    
    def update_display(self):
        """Atualiza exibição no canvas de forma otimizada"""
        if self.img_test is None:
            return
        
        # === CONVERSÃO OTIMIZADA ===
        self.img_display, self.scale_factor = cv2_to_tk(self.img_test, PREVIEW_W, PREVIEW_H)
        
        if self.img_display is None:
            return
        
        # === ATUALIZAÇÃO EFICIENTE DO CANVAS ===
        # Remove apenas overlays, mantém imagem base quando possível
        self.canvas.delete("result_overlay")
        self.canvas.delete("inspection")
        
        # Cria ou atualiza imagem
        if not hasattr(self, '_canvas_image_id'):
            self._canvas_image_id = self.canvas.create_image(0, 0, anchor=NW, image=self.img_display)
        else:
            self.canvas.itemconfig(self._canvas_image_id, image=self.img_display)
        
        # Desenha resultados se disponíveis
        if hasattr(self, 'inspection_results') and self.inspection_results:
            self.draw_inspection_results()
        
        # Atualiza scroll region apenas se necessário
        bbox = self.canvas.bbox("all")
        if bbox != self.canvas.cget("scrollregion"):
            self.canvas.configure(scrollregion=bbox)
    
    def run_inspection(self, show_message=True):
        """Executa inspeção otimizada com estilo industrial Keyence"""
        # === ATUALIZAÇÃO DE STATUS ===
        try:
            self.inspection_status_var.set("PROCESSANDO...")
            self.update_idletasks()  # Força atualização da UI
        except Exception as e:
            print(f"Erro ao atualizar status: {e}")
        
        # === VALIDAÇÃO INICIAL ===
        if not self.slots or self.img_reference is None or self.img_test is None:
            if show_message:
                messagebox.showerror("Erro", "Carregue o modelo de referência E a imagem de teste antes de inspecionar.", parent=self)
            self.inspection_status_var.set("ERRO")
            return
        
        print("--- Iniciando Inspeção Keyence ---")
        
        # Limpa resultados anteriores
        self.canvas.delete("result_overlay")
        
        # === 1. ALINHAMENTO DE IMAGEM ===
        try:
            self.inspection_status_var.set("ALINHANDO...")
            self.update_idletasks()  # Força atualização da UI
            M, _, align_error = find_image_transform(self.img_reference, self.img_test)
        except Exception as e:
            self.inspection_status_var.set("ERRO")
            if show_message:
                messagebox.showerror("Erro de Processamento", f"Erro durante alinhamento: {e}", parent=self)
            return
        
        if M is None:
            print(f"FALHA no Alinhamento: {align_error}")
            self.inspection_status_var.set("FALHA DE ALINHAMENTO")
            if show_message:
                messagebox.showerror("Falha no Alinhamento", f"Não foi possível alinhar as imagens.\nErro: {align_error}", parent=self)
            
            # Desenha slots de referência em cor de erro (estilo Keyence)
            for slot in self.slots:
                xr, yr, wr, hr = slot['x'], slot['y'], slot['w'], slot['h']
                xa, ya = xr * self.scale_factor, yr * self.scale_factor
                wa, ha = wr * self.scale_factor, hr * self.scale_factor
                self.canvas.create_rectangle(xa, ya, xa+wa, ya+ha, outline=COLOR_ALIGN_FAIL, width=2, tags="result_overlay")
                # Carrega as configurações de estilo
                style_config = load_style_config()
                self.canvas.create_text(xa + wa/2, ya + ha/2, text=f"S{slot['id']}\nFAIL", fill=COLOR_ALIGN_FAIL, font=style_config["ng_font"], tags="result_overlay", justify="center")
            return
        
        # === 2. VERIFICAÇÃO DOS SLOTS (ESTILO KEYENCE) ===
        try:
            self.inspection_status_var.set("INSPECIONANDO...")
            self.update_idletasks()  # Força atualização da UI
            
            overall_ok = True
            self.inspection_results = []
            failed_slots = []  # Para log otimizado
            
            # Adicionar modelo_id aos resultados se disponível
            model_id = getattr(self, 'current_model_id', '--')
            
            for i, slot in enumerate(self.slots):
                # Atualizar status com progresso
                progress = f"SLOT {i+1}/{len(self.slots)}"
                self.inspection_status_var.set(progress)
                self.update_idletasks()  # Força atualização da UI
                
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
        except Exception as e:
            self.inspection_status_var.set("ERRO")
            if show_message:
                messagebox.showerror("Erro de Processamento", f"Erro durante inspeção: {e}", parent=self)
            return
            
        # === 3. DESENHO OTIMIZADO NO CANVAS COM ESTILO KEYENCE ===
        for result in self.inspection_results:
            is_ok = result['passou']
            corners = result['corners']
            bbox = result['bbox']
            slot_id = result['slot_id']
            
            # Cores no estilo Keyence
            fill_color = COLOR_PASS if is_ok else COLOR_FAIL
            
            if corners is not None:
                # Conversão otimizada de coordenadas
                canvas_corners = [(int(pt[0] * self.scale_factor), int(pt[1] * self.scale_factor)) for pt in corners]
                
                # Desenha polígono transformado estilo Keyence
                self.canvas.create_polygon(canvas_corners, outline=fill_color, fill="", width=2, tags="result_overlay")
                
                # Adiciona um pequeno retângulo de status no canto estilo Keyence
                status_x, status_y = canvas_corners[0][0], canvas_corners[0][1] - 20
                self.canvas.create_rectangle(status_x, status_y, status_x + 40, status_y + 16, 
                                           fill=fill_color, outline="", tags="result_overlay")
                
                # Label otimizado estilo Keyence
                # Carrega as configurações de estilo
                style_config = load_style_config()
                self.canvas.create_text(status_x + 20, status_y + 8,
                                      text=f"S{slot_id}", fill="#FFFFFF", anchor="center", tags="result_overlay",
                                      font=style_config["ok_font"])
                
                # Adiciona indicador de status
                status_text = "OK" if is_ok else "NG"
                # Carrega as configurações de estilo se ainda não foi carregado
                if 'style_config' not in locals():
                    style_config = load_style_config()
                # Escolhe a fonte baseada no resultado
                font_str = style_config["ok_font"] if is_ok else style_config["ng_font"]
                self.canvas.create_text(canvas_corners[0][0] + 60, canvas_corners[0][1] - 12,
                                      text=status_text, fill=fill_color, anchor="nw", tags="result_overlay",
                                      font=font_str)
            elif bbox != [0,0,0,0]:  # Fallback para bbox com estilo Keyence
                xa, ya = bbox[0] * self.scale_factor, bbox[1] * self.scale_factor
                wa, ha = bbox[2] * self.scale_factor, bbox[3] * self.scale_factor
                self.canvas.create_rectangle(xa, ya, xa+wa, ya+ha, outline=COLOR_FAIL, width=1, dash=(4, 2), tags="result_overlay")
                
                # Adiciona indicador de erro estilo Keyence
                # Carrega as configurações de estilo
                style_config = load_style_config()
                self.canvas.create_text(xa + wa/2, ya + ha/2, text=f"S{slot_id}\nERRO", fill=COLOR_FAIL, 
                                      font=style_config["ng_font"], tags="result_overlay", justify="center")
        
        # === 4. RESULTADO FINAL ESTILO KEYENCE ===
        total = len(self.inspection_results)
        passed = sum(1 for r in self.inspection_results if r['passou'])
        failed = total - passed
        final_status = "APROVADO" if overall_ok else "REPROVADO"
        
        # Atualizar status de inspeção
        if overall_ok:
            self.inspection_status_var.set("OK")
        else:
            self.inspection_status_var.set("NG")
        
        # Log otimizado estilo Keyence
        if failed_slots:
            print(f"Falhas detectadas em: {', '.join(failed_slots)}")
        print(f"--- Inspeção Keyence Concluída: {final_status} ({passed}/{total}) ---")
        
        # Atualiza interface com estilo industrial Keyence
        self.update_results_list()
        
        # Status com estilo industrial Keyence
        status_text = f"INSPEÇÃO: {final_status} - {passed}/{total} SLOTS OK, {failed} FALHAS"
        self.status_var.set(status_text)
        
        # Atualiza cor da barra de status baseado no resultado estilo Keyence
        try:
            # Armazenamos uma referência direta ao status_bar durante a criação
            if hasattr(self, 'status_bar'):
                if overall_ok:
                    self.status_bar.config(background="#00AA00", foreground="#FFFFFF")
                else:
                    self.status_bar.config(background="#CC0000", foreground="#FFFFFF")
                    
            # Atualizar cor do indicador de status de inspeção usando referência direta
            if hasattr(self, 'inspection_status_label'):
                if overall_ok:
                    self.inspection_status_label.config(foreground="#00AA00")
                else:
                    self.inspection_status_label.config(foreground="#CC0000")
        except Exception as e:
            print(f"Erro ao atualizar status_bar: {e}")
        
        if show_message:
            # Mensagem estilo industrial Keyence
            if overall_ok:
                messagebox.showinfo("RESULTADO DA INSPEÇÃO KEYENCE", 
                                  f"✓ INSPEÇÃO CONCLUÍDA COM SUCESSO\n\nRESULTADO: {final_status}\n{passed}/{total} SLOTS APROVADOS", 
                                  parent=self)
            else:
                messagebox.showerror("RESULTADO DA INSPEÇÃO KEYENCE", 
                                   f"⚠ FALHA NA INSPEÇÃO\n\nRESULTADO: {final_status}\n{passed}/{total} SLOTS APROVADOS\n{failed} SLOTS REPROVADOS", 
                                   parent=self)
    
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
                                        background="#333333", foreground="#FFFFFF", 
                                        width=6, anchor="center", padding=3)
            self.status_label.pack(side=LEFT, padx=5)
            
            # Linha 2: Score e ID
            details_row = ttk.Frame(status_panel)
            details_row.pack(fill=X, pady=2)
            
            # Usa as configurações de estilo já carregadas
            ttk.Label(details_row, text="SCORE:", font=style_config["ok_font"]).pack(side=LEFT, padx=(5, 5))
            
            # Label para score com estilo industrial Keyence
            self.score_label = ttk.Label(details_row, text="--", font=style_config["ok_font"], 
                                       background="#333333", foreground="#FFFFFF", 
                                       width=8, anchor="center", padding=3)
            self.score_label.pack(side=LEFT, padx=5)
            
            ttk.Label(details_row, text="ID:", font=style_config["ok_font"]).pack(side=LEFT, padx=(10, 5))
            
            # Label para ID do modelo com estilo industrial Keyence
            self.id_label = ttk.Label(details_row, text="--", font=style_config["ok_font"], 
                                    background="#333333", foreground="#FFFFFF", 
                                    anchor="center", padding=3)
            self.id_label.pack(side=LEFT, padx=5, fill=X, expand=True)
            return
        
        # Caso contrário, criar o painel de resumo de slots
        # Limpar widgets existentes
        for widget in self.status_widgets.values():
            if hasattr(widget, 'frame'):
                widget['frame'].destroy()
        self.status_widgets.clear()
        
        if not self.slots:
            return
        
        # Calcular layout do grid (máximo 6 colunas)
        num_slots = len(self.slots)
        cols = min(6, num_slots)
        rows = (num_slots + cols - 1) // cols
        
        # Criar widgets para cada slot com estilo industrial Keyence
        for i, slot in enumerate(self.slots):
            row = i // cols
            col = i % cols
            
            # Frame para cada slot com estilo industrial
            slot_frame = ttk.Frame(self.status_grid_frame, relief="raised", borderwidth=2)
            slot_frame.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            
            # Configurar peso das colunas para expansão uniforme
            self.status_grid_frame.columnconfigure(col, weight=1)
            
            # Label do ID do slot com estilo industrial Keyence
            id_label = ttk.Label(slot_frame, text=f"SLOT {slot['id']}", 
                                font=('Arial', 8, 'bold'), background="#1E1E1E", foreground="#FFFFFF")
            id_label.pack(pady=2, fill=X)
            
            # Label do status (OK/NG) com estilo industrial Keyence
            status_label = ttk.Label(slot_frame, text="---", 
                                   font=('Arial', 14, 'bold'),
                                   foreground="#7F8C8D",
                                   background="#2A2A2A",
                                   anchor="center")
            status_label.pack(pady=2, fill=X)
            
            # Label do score com estilo industrial Keyence
            score_label = ttk.Label(slot_frame, text="", 
                                  font=('Arial', 9, 'bold'),
                                  background="#1E1E1E",
                                  foreground="#CCCCCC")
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
        if not self.status_widgets:
            return
        
        # Resetar todos os status com estilo industrial
        for slot_id, widgets in self.status_widgets.items():
            widgets['status_label'].config(text="---", foreground="#7F8C8D", background="#2A2A2A")
            widgets['score_label'].config(text="---", background="#1E1E1E", foreground="#CCCCCC")
            widgets['frame'].config(relief="raised", borderwidth=2, padding=2)
        
        # Atualizar com resultados da inspeção usando estilo industrial
        for result in self.inspection_results:
            slot_id = result['slot_id']
            if slot_id in self.status_widgets:
                widgets = self.status_widgets[slot_id]
                
                # Carrega as configurações de estilo
                style_config = load_style_config()
                
                if result['passou']:
                    # Estilo industrial para OK (cor personalizada)
                    widgets['status_label'].config(text="OK", foreground="#FFFFFF", background=style_config["ok_color"])
                    widgets['frame'].config(relief="raised", borderwidth=3)
                    widgets['id_label'].config(background="#003300", foreground="#FFFFFF")
                else:
                    # Estilo industrial para NG (cor personalizada)
                    widgets['status_label'].config(text="NG", foreground="#FFFFFF", background=style_config["ng_color"])
                    widgets['frame'].config(relief="raised", borderwidth=3)
                    widgets['id_label'].config(background="#330000", foreground="#FFFFFF")
                
                # Atualizar score com estilo industrial
                score_text = f"{result['score']:.3f}"
                if result['passou']:
                    widgets['score_label'].config(text=score_text, background="#003300", foreground="#FFFFFF")
                else:
                    widgets['score_label'].config(text=score_text, background="#330000", foreground="#FFFFFF")
    
    def update_results_list(self):
        """Atualiza lista de resultados com estilo industrial Keyence"""
        # === LIMPEZA OTIMIZADA ===
        children = self.results_listbox.get_children()
        if children:
            self.results_listbox.delete(*children)  # Mais eficiente que loop
        
        # === CONFIGURAÇÃO DE TAGS ESTILO KEYENCE ===
        # Carrega as configurações de estilo
        style_config = load_style_config()
        
        # Estilo OK - cor personalizada
        self.results_listbox.tag_configure("pass", 
                                         foreground="#FFFFFF", 
                                         background=style_config["ok_color"], 
                                         font=style_config["ok_font"])
        
        # Estilo NG - cor personalizada
        self.results_listbox.tag_configure("fail", 
                                         foreground="#FFFFFF", 
                                         background=style_config["ng_color"], 
                                         font=style_config["ng_font"])
        
        # Estilo cabeçalho - cinza industrial Keyence
        # Carrega as configurações de estilo
        style_config = load_style_config()
        self.results_listbox.tag_configure("header", 
                                         foreground="#FFFFFF", 
                                         background="#333333", 
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
        
        # Atualizar painel de resumo geral se existir
        if hasattr(self, 'status_label') and hasattr(self, 'score_label') and hasattr(self, 'id_label'):
            # Calcular status geral no estilo Keyence
            if total_slots > 0:
                avg_score = total_score / total_slots
                overall_status = "OK" if passed_slots == total_slots else "NG"
                
                # Atualizar labels com estilo Keyence
                self.status_label.config(
                    text=overall_status,
                    background="#00AA00" if overall_status == "OK" else "#CC0000",
                    foreground="#FFFFFF"
                )
                
                self.score_label.config(
                    text=f"{passed_slots}/{total_slots}",
                    background="#00AA00" if passed_slots == total_slots else "#CC0000",
                    foreground="#FFFFFF"
                )
                
                self.id_label.config(text=model_id)
    
    def draw_inspection_results(self):
        """Desenha resultados da inspeção no canvas com estilo industrial."""
        if not self.inspection_results:
            return
        
        for result in self.inspection_results:
            slot = result['slot_data']
            
            # Converte coordenadas da imagem para canvas
            x1 = int(slot['x'] * self.scale_factor)
            y1 = int(slot['y'] * self.scale_factor)
            x2 = int((slot['x'] + slot['w']) * self.scale_factor)
            y2 = int((slot['y'] + slot['h']) * self.scale_factor)
            
            # Carrega as configurações de estilo
            style_config = load_style_config()
            
            # Cores estilo industrial
            if result['passou']:
                outline_color = style_config["ok_color"]  # Cor de OK personalizada
                fill_color = style_config["ok_color"]     # Mesma cor para o fundo
                text_color = "#FFFFFF"                    # Texto branco
            else:
                outline_color = style_config["ng_color"]  # Cor de NG personalizada
                fill_color = style_config["ng_color"]     # Mesma cor para o fundo
                text_color = "#FFFFFF"                    # Texto branco
            
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
            
            # Carrega as configurações de estilo
            style_config = load_style_config()
            
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
        has_model = len(self.slots) > 0
        has_test_image = self.img_test is not None
        
        # Botões que dependem de modelo e imagem de teste
        self.btn_inspect.config(state=NORMAL if has_model and has_test_image else DISABLED)
    
    def on_closing_inspection(self):
        """Limpa recursos ao fechar a aplicação de inspeção."""
        if self.live_capture:
            self.stop_live_capture_inspection()
        if self.live_view:
            self.stop_live_view()
        self.master.destroy()


def create_main_window():
    """Cria e configura a janela principal da aplicação."""
    # Inicializa ttkbootstrap com tema mais claro
    root = ttk.Window(themename="flatly")  # Tema alterado de "cyborg" para "flatly"
    root.title("Sistema de Visão Computacional - Inspeção de Montagem")
    
    # Configurar para abrir em tela cheia
    root.state('zoomed')  # Maximiza a janela no Windows
    
    # Configura fechamento de janelas OpenCV
    def on_closing():
        cv2.destroyAllWindows()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Cria notebook para abas
    notebook = ttk.Notebook(root)
    notebook.pack(fill=BOTH, expand=True, padx=5, pady=5)
    
    # Aba Editor de Malha
    montagem_frame = MontagemWindow(notebook)
    montagem_frame.pack(fill=BOTH, expand=True)
    notebook.add(montagem_frame, text="Editor de Malha")
    
    # Aba Inspeção
    inspecao_frame = InspecaoWindow(notebook)
    inspecao_frame.pack(fill=BOTH, expand=True)
    notebook.add(inspecao_frame, text="Inspeção")
    
    return root


def main():
    """Função principal do módulo montagem."""
    app = create_main_window()
    app.mainloop()


if __name__ == "__main__":
    main()