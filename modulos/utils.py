import json
from pathlib import Path
from tkinter import ttk

# Configurações de estilo padrão
DEFAULT_STYLES = {
    "background_color": "#1E1E1E",  # Cor de fundo dos diálogos
    "text_color": "#FFFFFF",       # Cor do texto
    "button_color": "#FFFFFF",     # Cor dos botões
    "ng_color": "#F38BA8",         # Cor do texto NG
    "ok_color": "#95E1D3",         # Cor do texto OK
    "ng_font": "Arial 10 bold",    # Fonte do texto NG
    "ok_font": "Arial 10 bold",    # Fonte do texto OK
    "selection_color": "#FFE66D",  # Cor do quadro de seleção
    "slot_font_size": 10,          # Tamanho da fonte para slots
    "result_font_size": 10,        # Tamanho da fonte para resultados
    "button_font_size": 9,         # Tamanho da fonte para botões
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
    Aplica as configurações de estilo à interface atual.
    
    Args:
        config: Dicionário com as configurações de estilo
    """
    try:
        # Obtém o estilo atual
        style = ttk.Style()
        
        # Configura as cores e fontes
        bg_color = config.get("background_color", DEFAULT_STYLES["background_color"])
        text_color = config.get("text_color", DEFAULT_STYLES["text_color"])
        ok_color = config.get("ok_color", DEFAULT_STYLES["ok_color"])
        ng_color = config.get("ng_color", DEFAULT_STYLES["ng_color"])
        
        # Tamanhos de fonte
        slot_font_size = config.get("slot_font_size", DEFAULT_STYLES["slot_font_size"])
        result_font_size = config.get("result_font_size", DEFAULT_STYLES["result_font_size"])
        button_font_size = config.get("button_font_size", DEFAULT_STYLES["button_font_size"])
        
        # Cria as fontes com os tamanhos configurados
        ok_font = f"Arial {result_font_size} bold"
        ng_font = f"Arial {result_font_size} bold"
        slot_font = f"Arial {slot_font_size}"
        button_font = f"Arial {button_font_size}"
        
        # Atualiza as configurações no dicionário
        config["ok_font"] = ok_font
        config["ng_font"] = ng_font
        
        # Aplica as configurações ao estilo
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", foreground=text_color, background=bg_color, font=slot_font)
        style.configure("TButton", font=button_font)
        style.configure("OK.TLabel", foreground=ok_color, font=ok_font)
        style.configure("NG.TLabel", foreground=ng_color, font=ng_font)
        
        print("Configurações de estilo aplicadas com sucesso")
        return True
    except Exception as e:
        print(f"Erro ao aplicar configurações de estilo: {e}")
        return False
