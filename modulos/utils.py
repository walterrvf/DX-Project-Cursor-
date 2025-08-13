import json
from pathlib import Path
from tkinter import ttk

# Configurações de estilo padrão (fallback caso o arquivo de config não exista)
DEFAULT_STYLES = {
    "slot_font_size": 11,
    "result_font_size": 11,
    "button_font_size": 10,
    "hud_font_size": 13,
    "hud_opacity": 85,
    "hud_position": "top-right",
    "show_fps": True,
    "show_timestamp": True,
    
    "fonts": {
        "ok_font": "Segoe UI 11 bold",
        "ng_font": "Segoe UI 11 bold",
        "title_font": "Inter 26 bold",
        "subtitle_font": "Inter 19 normal",
        "header_font": "Inter 15 bold",
        "small_font": "Segoe UI 10",
        "tiny_font": "Segoe UI 9",
        "console_font": "JetBrains Mono 10",
        "large_font": "Inter 16 bold",
        "medium_font": "Segoe UI 12"
    },
    
    "colors": {
        "background_color": "#0B1220",
        "text_color": "#E6EAF2",
        "ok_color": "#22C55E",
        "ng_color": "#EF4444",
        "selection_color": "#93C5FD",
        "button_color": "#FFFFFF",
        
        "canvas_colors": {
            "canvas_bg": "#0B1220",
            "canvas_dark_bg": "#0B1220",
            "panel_bg": "#111827",
            "dark_panel_bg": "#0B1220",
            "button_bg": "#334155",
            "button_active": "#475569",
            "modern_bg": "#0F172A"
        },
        
        "editor_colors": {
            "clip_color": "#6366F1",
            "selected_color": "#F59E0B",
            "drawing_color": "#10B981",
            "delete_color": "#FF4444",
            "handle_color": "#FF4444"
        },
        
        "inspection_colors": {
            "pass_color": "#22C55E",
            "fail_color": "#EF4444",
            "align_fail_color": "#F97316",
            "pass_bg": "#333333",
            "fail_bg": "#440000",
            "ok_detail_bg": "#003300",
            "ng_detail_bg": "#330000"
        },
        
        "status_colors": {
            "success_bg": "#00AA00",
            "error_bg": "#CC0000",
            "warning_bg": "#FFCC00",
            "info_bg": "#0000AA",
            "neutral_bg": "#AAAAAA",
            "inactive_text": "#7F8C8D",
            "inactive_bg": "#2A2A2A",
            "muted_text": "#CCCCCC",
            "muted_bg": "#2A2A2A"
        },
        
        "ui_colors": {
            "primary": "#6366F1",
            "secondary": "#F59E0B",
            "success": "#22C55E",
            "danger": "#EF4444",
            "warning": "#F97316",
            "info": "#3B82F6",
            "light": "#F8F9FA",
            "dark": "#1E1E1E",
            "muted": "#6C757D"
        },
        
        "dialog_colors": {
            "window_bg": "#2E2E2E",
            "frame_bg": "#1a1d29",
            "left_panel_bg": "#252837",
            "center_panel_bg": "#2d3142",
            "right_panel_bg": "#252837",
            "listbox_bg": "#252837",
            "listbox_fg": "#e5e7eb",
            "listbox_select_bg": "#6366F1",
            "listbox_active_bg": "#374151",
            "entry_bg": "#2A2A2A",
            "entry_readonly_bg": "#2A2A2A",
            "combobox_select_bg": "#3A3A3A"
        },
        
        "button_colors": {
            "modern_bg": "#6366F1",
            "modern_active": "#818CF8",
            "modern_pressed": "#4F46E5",
            "success_bg": "#22C55E",
            "success_active": "#16A34A",
            "success_pressed": "#15803D",
            "danger_bg": "#EF4444",
            "danger_active": "#DC2626",
            "danger_pressed": "#B91C1C",
            "inspect_active": "#FB923C",
            "inspect_pressed": "#EA580C"
        },
        
        "special_colors": {
            "ok_canvas_bg": "#f0f8f0",
            "ng_canvas_bg": "#f8f0f0",
            "ok_result_bg": "#e8f5e8",
            "ng_result_bg": "#f5e8e8",
            "preview_canvas_bg": "#1E1E1E",
            "console_bg": "#F8F9FA",
            "console_fg": "#2C3E50",
            "tooltip_fg": "#888888",
            "green_text": "#28a745",
            "gray_text": "#AAAAAA",
            "white_text": "#FFFFFF",
            "black_bg": "#000000"
        }
    },
    
    # Fontes (mantidas para compatibilidade)
    "ng_font": "Arial 10 bold",
    "ok_font": "Arial 10 bold"
}

def get_project_root():
    """Retorna o caminho da raiz do projeto."""
    return Path(__file__).parent.parent

def get_style_config_path():
    """Retorna o caminho para o arquivo de configuração de estilo."""
    config_dir = get_project_root() / "config"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "style_config.json"

def load_style_config():
    """
    Carrega as configurações de estilo do arquivo JSON.
    Se o arquivo não existir, cria um novo com as configurações padrão.
    """
    try:
        # Obtém o caminho absoluto para o arquivo de configuração
        config_path = get_style_config_path()
        
        # Cria o diretório de configuração se não existir
        config_dir = config_path.parent
        config_dir.mkdir(exist_ok=True)
        
        # Se o arquivo não existir, cria com as configurações padrão
        if not config_path.exists():
            save_style_config(DEFAULT_STYLES)
            return DEFAULT_STYLES.copy()
        
        # Carrega as configurações do arquivo
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Verifica se todas as chaves necessárias estão presentes
        for key in DEFAULT_STYLES.keys():
            if key not in config:
                config[key] = DEFAULT_STYLES[key]
        
        return config
    except Exception as e:
        print(f"Erro ao carregar style_config.json: {e}")
        return DEFAULT_STYLES.copy()

def save_style_config(config):
    """
    Salva as configurações de estilo em um arquivo JSON.
    """
    try:
        # Obtém o caminho absoluto para o arquivo de configuração
        config_path = get_style_config_path()
        
        # Cria o diretório de configuração se não existir
        config_dir = config_path.parent
        config_dir.mkdir(exist_ok=True)
        
        # Salva as configurações no arquivo
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Configurações de estilo salvas em {config_path}")
        return True
    except Exception as e:
        print(f"Erro ao salvar configurações de estilo: {e}")
        return False

def apply_style_config(config):
    """
    Aplica as configurações de estilo aos widgets ttk.
    
    Args:
        config: Dicionário com as configurações de estilo
    """
    try:
        # Configura o estilo dos widgets ttk
        style = ttk.Style()
        
        # Obtém as cores do config
        colors = config.get('colors', {})
        
        # Aplica as configurações básicas
        bg_color = colors.get('background_color', '#222222')
        text_color = colors.get('text_color', '#ffffff')
        
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=text_color)
        style.configure('TButton', foreground=text_color, padding=(10, 6))
        style.map('TButton', background=[('active', colors.get('button_colors', {}).get('modern_active', '#3B82F6'))])
        style.configure('Accent.TButton', foreground='#FFFFFF', padding=(12, 8))
        style.map('Accent.TButton', background=[('!disabled', colors.get('ui_colors', {}).get('primary', '#6366F1')),
                                                ('active', colors.get('button_colors', {}).get('modern_active', '#818CF8'))])
        style.configure('Success.TButton', foreground='#FFFFFF', padding=(12, 8))
        style.map('Success.TButton', background=[('!disabled', colors.get('button_colors', {}).get('success_bg', '#22C55E')),
                                                 ('active', colors.get('button_colors', {}).get('success_active', '#16A34A'))])
        style.configure('Danger.TButton', foreground='#FFFFFF', padding=(12, 8))
        style.map('Danger.TButton', background=[('!disabled', colors.get('button_colors', {}).get('danger_bg', '#EF4444')),
                                                ('active', colors.get('button_colors', {}).get('danger_active', '#DC2626'))])

        style.configure('TNotebook', background=bg_color)
        style.configure('TNotebook.Tab', padding=(14, 8), font=('Segoe UI', 11))

        panel_bg = colors.get('canvas_colors', {}).get('panel_bg', bg_color)
        style.configure('TLabelframe', background=panel_bg, foreground=text_color)
        style.configure('TLabelframe.Label', background=panel_bg, foreground=text_color, font=('Segoe UI', 11, 'bold'))

        # Seção moderna com borda sutil
        style.configure('Section.TLabelframe', background=panel_bg, bordercolor='#0F172A', relief='solid')
        style.configure('Section.TLabelframe.Label', background=panel_bg, foreground=text_color, font=('Segoe UI', 11, 'bold'))

        # Cards utilitários
        style.configure('Card.TFrame', background=panel_bg, relief='ridge')
        style.configure('Treeview', background=colors.get('canvas_colors', {}).get('panel_bg', '#111827'),
                        fieldbackground=colors.get('canvas_colors', {}).get('panel_bg', '#111827'),
                        foreground=text_color, rowheight=28, bordercolor='#0F172A')
        style.map('Treeview', background=[('selected', colors.get('selection_color', '#93C5FD'))],
                  foreground=[('selected', '#0B1220')])
        
        # Aplica configurações de fonte
        if 'slot_font_size' in config:
            style.configure('Slot.TLabel', font=('Segoe UI', config['slot_font_size']))
        
        if 'result_font_size' in config:
            style.configure('Result.TLabel', font=('Segoe UI', config['result_font_size']))
        
        if 'button_font_size' in config:
            style.configure('TButton', font=('Segoe UI', config['button_font_size']))
            
    except Exception as e:
        print(f"Erro ao aplicar configurações de estilo: {e}")


def get_color(color_path, config=None):
    """
    Obtém uma cor específica do arquivo de configuração.
    
    Args:
        color_path: Caminho para a cor (ex: 'colors.ok_color' ou 'colors.canvas_colors.canvas_bg')
        config: Configuração opcional. Se None, carrega do arquivo.
    
    Returns:
        str: Código hexadecimal da cor
    """
    if config is None:
        config = load_style_config()
    
    try:
        # Navega pelo caminho da cor
        keys = color_path.split('.')
        value = config
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        # Fallback para cores padrão
        fallback_colors = {
            'colors.background_color': '#222222',
            'colors.text_color': '#ffffff',
            'colors.ok_color': '#8cde81',
            'colors.ng_color': '#e7472c',
            'colors.selection_color': '#FFE66D',
            'colors.canvas_colors.canvas_bg': '#1E1E1E',
            'colors.canvas_colors.panel_bg': '#2A2A2A',
            'colors.editor_colors.clip_color': '#6366F1',
            'colors.editor_colors.selected_color': '#F59E0B',
            'colors.editor_colors.drawing_color': '#10B981',
            'colors.inspection_colors.pass_color': '#22C55E',
            'colors.inspection_colors.fail_color': '#EF4444',
            'colors.status_colors.success_bg': '#00AA00',
            'colors.status_colors.error_bg': '#CC0000'
        }
        return fallback_colors.get(color_path, '#FFFFFF')


def get_colors_group(group_name, config=None):
    """
    Obtém um grupo completo de cores.
    
    Args:
        group_name: Nome do grupo (ex: 'canvas_colors', 'editor_colors')
        config: Configuração opcional. Se None, carrega do arquivo.
    
    Returns:
        dict: Dicionário com as cores do grupo
    """
    if config is None:
        config = load_style_config()
    
    try:
        return config['colors'][group_name]
    except (KeyError, TypeError):
        return {}


def get_font(font_path, config=None):
    """
    Obtém uma fonte específica do arquivo de configuração.
    
    Args:
        font_path: Caminho para a fonte (ex: 'fonts.ok_font' ou 'ok_font')
        config: Configuração opcional. Se None, carrega do arquivo.
    
    Returns:
        str: String da fonte (ex: 'Arial 10 bold')
    """
    if config is None:
        config = load_style_config()
    
    try:
        # Se não tem ponto, assume que está em 'fonts'
        if '.' not in font_path:
            font_path = f'fonts.{font_path}'
            
        # Navega pelo caminho da fonte
        keys = font_path.split('.')
        value = config
        for key in keys:
            value = value[key]
        # Converte strings "Family Size [style...]" em tupla Tk-friendly
        if isinstance(value, str):
            parsed = _parse_font_string(value)
            return parsed
        return value
    except (KeyError, TypeError):
        # Fallback para fontes padrão
        fallback_fonts = {
            'fonts.ok_font': 'Arial 10 bold',
            'fonts.ng_font': 'Arial 10 bold',
            'fonts.title_font': 'Arial 28 bold',
            'fonts.subtitle_font': 'Arial 24',
            'fonts.header_font': 'Arial 16 bold',
            'fonts.small_font': 'Arial 7',
            'fonts.tiny_font': 'Arial 8',
            'fonts.console_font': 'Consolas 10'
        }
        return _parse_font_string(fallback_fonts.get(font_path, 'Arial 10'))


def _parse_font_string(font_value: str):
    """Converte string de fonte (ex: 'Segoe UI 14 bold') para tupla Tk ('Segoe UI', 14, 'bold')."""
    try:
        if not isinstance(font_value, str):
            return font_value
        tokens = font_value.strip().split()
        # Procura primeiro token que é número (tamanho)
        size_index = None
        for i, tk in enumerate(tokens):
            try:
                int(tk)
                size_index = i
                break
            except ValueError:
                continue
        if size_index is None:
            # Sem tamanho explícito: retorna string original
            return font_value
        family = ' '.join(tokens[:size_index]) or 'Arial'
        size = int(tokens[size_index])
        styles = tokens[size_index + 1:]
        # Retorna tupla: (family, size, *styles)
        return (family, size, *styles) if styles else (family, size)
    except Exception:
        return ('Arial', 10)


def update_color(color_path, new_color, save_to_file=True):
    """
    Atualiza uma cor específica na configuração.
    
    Args:
        color_path: Caminho para a cor (ex: 'colors.ok_color')
        new_color: Nova cor em formato hexadecimal
        save_to_file: Se True, salva as alterações no arquivo
    
    Returns:
        bool: True se a atualização foi bem-sucedida
    """
    try:
        config = load_style_config()
        
        # Navega até o local da cor
        keys = color_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Atualiza a cor
        current[keys[-1]] = new_color
        
        # Salva se solicitado
        if save_to_file:
            save_style_config(config)
        
        return True
    except Exception as e:
        print(f"Erro ao atualizar cor {color_path}: {e}")
        return False


def update_font(font_path, new_font, save_to_file=True):
    """
    Atualiza uma fonte específica no arquivo de configuração.
    
    Args:
        font_path: Caminho para a fonte (ex: 'fonts.ok_font')
        new_font: Nova fonte (ex: 'Arial 12 bold')
        save_to_file: Se deve salvar no arquivo (padrão: True)
    """
    try:
        config = load_style_config()
        
        # Se não tem ponto, assume que está em 'fonts'
        if '.' not in font_path:
            font_path = f'fonts.{font_path}'
            
        # Navega pelo caminho da fonte
        keys = font_path.split('.')
        current = config
        
        # Navega até o penúltimo nível
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Define a nova fonte
        current[keys[-1]] = new_font
        
        if save_to_file:
            save_style_config(config)
            
        return True
        
    except Exception as e:
        print(f"Erro ao atualizar fonte {font_path}: {e}")
        return False
