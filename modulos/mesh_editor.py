"""
Módulo para a janela de edição de malha (MontagemWindow).
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

# Valores padrão compartilhados (fallback quando não carregados de outro lugar)
PREVIEW_W = 1200
PREVIEW_H = 900
THR_CORR = 0.1
MIN_PX = 10
ORB_FEATURES = 5000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8

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
    )
    from image_utils import cv2_to_tk
    from paths import get_model_dir, get_template_dir, get_model_template_dir
    from inspection import find_image_transform
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
    from inspection import find_image_transform

# Variáveis globais
MODEL_DIR = get_model_dir()

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
        self.x_offset = 0
        self.y_offset = 0
        self.slots = []
        self.selected_slot_id = None
        self.current_model_id = None  # ID do modelo atual no banco
        self.current_model = None  # Dados do modelo atual
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
        
        # Controle de webcam - Sistema de múltiplas câmeras ativas
        self.available_cameras = detect_cameras()
        self.selected_camera = 0
        self.camera = None
        self.live_capture = False
        self.live_view = False
        self.latest_frame = None
        
        # Gerenciador de múltiplas câmeras ativas
        self.active_cameras = {}  # Dicionário para manter câmeras ativas
        self.camera_frames = {}   # Frames mais recentes de cada câmera
        
        # Variáveis de ferramentas de edição
        self.current_drawing_mode = "rectangle"
        self.editing_handle = None
        
        # Configura a interface primeiro
        self.setup_ui()
        self.update_button_states()
        
        # Inicializa variável de controle da câmera atual
        self.current_camera_index = self.available_cameras[0] if self.available_cameras else 0
        
        # Inicia múltiplas câmeras em segundo plano após inicialização completa
        if self.available_cameras:
            self.after(500, self.initialize_multiple_cameras)
        
        # Opcional: executar limpeza periódica de câmeras não utilizadas
        try:
            self.after(60_000, cleanup_unused_cameras)
        except Exception:
            pass
    
    def configure_modern_styles(self):
        """Configura estilos modernos para a interface."""
        style = ttk.Style()
        
        # Estilo para frames principais
        style.configure("Modern.TFrame", 
                       background=get_color('colors.dialog_colors.frame_bg'),
                       relief="flat")
        
        # Estilo para cards/painéis
        style.configure("Card.TFrame",
                       background=get_color('colors.dialog_colors.left_panel_bg'),
                       relief="flat",
                       borderwidth=1)
        
        # Estilo para painel principal do canvas
        style.configure("Canvas.TFrame",
                       background=get_color('colors.dialog_colors.center_panel_bg'),
                       relief="flat")
        
        # Estilo para painel direito
        style.configure("RightPanel.TFrame",
                       background=get_color('colors.dialog_colors.right_panel_bg'),
                       relief="flat")
        
        # Estilo para botões modernos
        style.configure("Modern.TButton",
                       background=get_color('colors.button_colors.modern_bg'),
                       foreground="white",
                       borderwidth=0,
                       focuscolor="none",
                       padding=(12, 8))
        
        style.map("Modern.TButton",
                 background=[("active", get_color('colors.button_colors.modern_active')),
                       ("pressed", get_color('colors.button_colors.modern_pressed'))])
        
        # Estilo para botões de sucesso
        style.configure("Success.TButton",
                       background=get_color('colors.button_colors.success_bg'),
                       foreground="white",
                       borderwidth=0,
                       focuscolor="none",
                       padding=(12, 8))
        
        style.map("Success.TButton",
                 background=[("active", get_color('colors.button_colors.success_active')),
                       ("pressed", get_color('colors.button_colors.success_pressed'))])
        
        # Estilo para botões de perigo
        style.configure("Danger.TButton",
                       background=get_color('colors.button_colors.danger_bg'),
                       foreground="white",
                       borderwidth=0,
                       focuscolor="none",
                       padding=(12, 8))
        
        style.map("Danger.TButton",
                 background=[("active", get_color('colors.button_colors.danger_active')),
                       ("pressed", get_color('colors.button_colors.danger_pressed'))])
        
        # Estilo para labels modernos
        style.configure("Modern.TLabel",
                       background=get_color('colors.dialog_colors.listbox_bg'),
            foreground=get_color('colors.dialog_colors.listbox_fg'),
                       font=("Segoe UI", 10))
        
        # Estilo para LabelFrames modernos
        style.configure("Modern.TLabelframe",
                       background=get_color('colors.dialog_colors.listbox_bg'),
            foreground=get_color('colors.dialog_colors.listbox_fg'),
                       borderwidth=1,
                       relief="solid",
                       labelmargins=(10, 5, 10, 5))
        
        style.configure("Modern.TLabelframe.Label",
                       background=get_color('colors.dialog_colors.listbox_bg'),
            foreground=get_color('colors.dialog_colors.listbox_fg'),
                       font=("Segoe UI", 10, "bold"))
        
    def start_background_camera_direct(self, camera_index):
        """Inicia a câmera diretamente em segundo plano com índice específico."""
        try:
            # Usa pool persistente para evitar múltiplas aberturas do dispositivo
            from camera_manager import get_persistent_camera
            self.camera = get_persistent_camera(camera_index)
            if not self.camera or not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            self.live_capture = True
            print(f"Webcam {camera_index} inicializada com sucesso em segundo plano")
            
            # Inicia captura de frames em thread separada
            self.start_background_frame_capture()
            
        except Exception as e:
            print(f"Erro ao inicializar webcam {camera_index}: {e}")
            self.camera = None
            self.live_capture = False
    
    def initialize_multiple_cameras(self):
        """Inicializa múltiplas câmeras simultaneamente para troca rápida."""
        try:
            print("Inicializando sistema de múltiplas câmeras no editor de malha...")
            
            # Inicializa todas as câmeras disponíveis
            for camera_index in self.available_cameras:
                try:
                    self.start_camera_connection(camera_index)
                    print(f"Câmera {camera_index} inicializada com sucesso no editor de malha")
                except Exception as e:
                    print(f"Erro ao inicializar câmera {camera_index} no editor de malha: {e}")
            
            # Define a câmera principal ativa
            if self.available_cameras:
                self.current_camera_index = self.available_cameras[0]
                self.camera = self.active_cameras.get(self.current_camera_index)
                
                # Inicia captura de frames para todas as câmeras
                self.start_multi_camera_capture()
                
        except Exception as e:
            print(f"Erro ao inicializar múltiplas câmeras no editor de malha: {e}")
    
    def start_camera_connection(self, camera_index):
        """Inicia conexão com uma câmera específica."""
        try:
            # Usa pool persistente para compartilhar o mesmo handle do dispositivo
            from camera_manager import get_persistent_camera
            camera = get_persistent_camera(camera_index)
            if not camera or not camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Armazena a câmera ativa
            self.active_cameras[camera_index] = camera
            
        except Exception as e:
            print(f"Erro ao conectar câmera {camera_index}: {e}")
            raise
    
    def start_multi_camera_capture(self):
        """Inicia captura de frames para todas as câmeras ativas."""
        try:
            # Preferir frame pump centralizado do camera_manager
            from camera_manager import start_frame_pump
            start_frame_pump(list(self.active_cameras.keys()), fps=30.0)
        except Exception as e:
            print(f"Erro ao iniciar frame pump no editor de malha: {e}")
    
    def capture_camera_frames(self, camera_index):
        """Captura frames continuamente de uma câmera específica."""
        try:
            import time
            camera = self.active_cameras.get(camera_index)
            if not camera:
                return
                
            while camera_index in self.active_cameras and camera.isOpened():
                try:
                    ret, frame = camera.read()
                    if ret:
                        self.camera_frames[camera_index] = frame.copy()
                        
                        # Atualiza latest_frame se esta é a câmera ativa
                        if camera_index == self.current_camera_index:
                            self.latest_frame = frame.copy()
                            
                    time.sleep(0.033)  # ~30 FPS
                except Exception as e:
                    print(f"Erro na captura da câmera {camera_index} no editor de malha: {e}")
                    break
                    
        except Exception as e:
            print(f"Erro geral na captura da câmera {camera_index} no editor de malha: {e}")

    
    
    def start_background_frame_capture(self):
        """Inicia captura de frames em segundo plano sem exibir no canvas."""
        def capture_loop():
            while self.live_capture and self.camera and self.camera.isOpened():
                try:
                    ret, frame = self.camera.read()
                    if ret:
                        self.latest_frame = frame.copy()
                    time.sleep(0.033)  # ~30 FPS
                except Exception as e:
                    print(f"Erro na captura em segundo plano: {e}")
                    break
        
        # Inicia thread para captura contínua
        import threading
        self.background_thread = threading.Thread(target=capture_loop, daemon=True)
        self.background_thread.start()
    
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
        """Configura a interface moderna com design responsivo."""
        # Frame principal com gradiente visual
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=15, pady=15)
        
        # Configura grid para layout responsivo
        main_frame.grid_columnconfigure(0, weight=0, minsize=300)  # Painel esquerdo - largura fixa mínima
        main_frame.grid_columnconfigure(1, weight=1)  # Painel central - expansível
        main_frame.grid_columnconfigure(2, weight=0, minsize=380)  # Painel direito - largura fixa
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Painel esquerdo - Controles com design card
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        
        # Painel central - Editor de malha com bordas arredondadas
        center_panel = ttk.Frame(main_frame)
        center_panel.grid(row=0, column=1, sticky="nsew")
        
        # Painel direito - Editor de slot com design moderno
        self.right_panel = ttk.Frame(main_frame)
        self.right_panel.grid(row=0, column=2, sticky="nsew", padx=(15, 0))
        
        # === PAINEL ESQUERDO - DESIGN MODERNO ===
        
        # Seção de Imagem com ícones
        img_frame = ttk.LabelFrame(left_panel, text="Imagem")
        img_frame.pack(fill=X, pady=(0, 15))
        
        self.btn_load_image = ttk.Button(img_frame, text="Carregar Imagem", 
                                        command=self.load_image)
        self.btn_load_image.pack(fill=X, padx=10, pady=8)
        
        # Seção de Webcam com design moderno
        webcam_frame = ttk.LabelFrame(left_panel, text="Webcam")
        webcam_frame.pack(fill=X, pady=(0, 15))
        
        # Combobox para seleção de câmera com estilo moderno
        camera_selection_frame = ttk.Frame(webcam_frame)
        camera_selection_frame.pack(fill=X, padx=10, pady=8)
        
        ttk.Label(camera_selection_frame, text="Câmera:").pack(side=LEFT)
        self.camera_combo = Combobox(camera_selection_frame, 
                                   values=[str(i) for i in self.available_cameras],
                                   state="readonly", width=8,
                                   font=("Segoe UI", 9))
        self.camera_combo.pack(side=RIGHT, padx=(10, 0))
        if self.available_cameras:
            self.camera_combo.set(str(self.available_cameras[0]))
        
        # Adiciona callback para mudança de câmera
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_changed)
        
        # Botão para capturar imagem da webcam
        self.btn_capture = ttk.Button(webcam_frame, text="Capturar Imagem", 
                                     command=self.capture_from_webcam)
        self.btn_capture.pack(fill=X, padx=10, pady=(8, 8))
        
        # Seção de Modelo com design moderno
        model_frame = ttk.LabelFrame(left_panel, text="Modelo")
        model_frame.pack(fill=X, pady=(0, 15))
        
        self.btn_load_model = ttk.Button(model_frame, text="Carregar Modelo", 
                                        command=self.load_model_dialog)
        self.btn_load_model.pack(fill=X, padx=10, pady=(8, 4))
        
        self.btn_save_model = ttk.Button(model_frame, text="Salvar Modelo", 
                                        command=self.save_model)
        self.btn_save_model.pack(fill=X, padx=10, pady=(4, 8))
        
        # Seção de Ferramentas de Edição com design moderno
        tools_frame = ttk.LabelFrame(left_panel, text="Ferramentas de Edição", )
        tools_frame.pack(fill=X, pady=(0, 15))
        
        # Modo de desenho com cards
        mode_frame = ttk.Frame(tools_frame, )
        mode_frame.pack(fill=X, padx=10, pady=8)
        
        ttk.Label(mode_frame, text="Modo de Desenho:", ).pack(anchor="w", pady=(5, 8))
        
        self.drawing_mode = StringVar(value="rectangle")
        
        mode_buttons_frame = ttk.Frame(mode_frame, )
        mode_buttons_frame.pack(fill=X, pady=(0, 5))
        
        # Configurando estilo moderno para os botões de rádio
        self.style = ttk.Style()
        self.style.configure("Modern.TRadiobutton", 
                           background=get_color('colors.dialog_colors.listbox_bg'),
            foreground=get_color('colors.dialog_colors.listbox_fg'),
                           font=("Segoe UI", 9))
        self.style.map("Modern.TRadiobutton",
                      background=[('active', get_color('colors.dialog_colors.listbox_active_bg')), ('selected', get_color('colors.dialog_colors.listbox_select_bg'))],
                      foreground=[('active', 'white'), ('selected', 'white')])
        
        self.btn_rect_mode = ttk.Radiobutton(mode_buttons_frame, text="Retângulo", 
                                           variable=self.drawing_mode, value="rectangle",
                                           command=self.set_drawing_mode,
                                           )
        self.btn_rect_mode.pack(side=LEFT, padx=(5, 10))
        
        # Removido modo de "Exclusão"
        
        # Status da ferramenta com design moderno
        self.tool_status_var = StringVar(value="Modo: Retângulo")
        status_label = ttk.Label(tools_frame, textvariable=self.tool_status_var, 
                               font=("Segoe UI", 8), 
                               foreground=get_color('colors.status_colors.muted_text'),
            background=get_color('colors.dialog_colors.listbox_bg'))
        status_label.pack(padx=10, pady=(0, 8))
        
        # Seção de Slots com design moderno
        slots_frame = ttk.LabelFrame(left_panel, text="Slots", )
        slots_frame.pack(fill=X, pady=(0, 15))
        
        self.btn_clear_slots = ttk.Button(slots_frame, text="Limpar Todos os Slots", 
                                         command=self.clear_slots,
                                         )
        self.btn_clear_slots.pack(fill=X, padx=10, pady=(8, 4))
        
        self.btn_delete_slot = ttk.Button(slots_frame, text="Deletar Slot Selecionado", 
                                         command=self.delete_selected_slot,
                                         )
        self.btn_delete_slot.pack(fill=X, padx=10, pady=(4, 4))
        
        self.btn_train_slot = ttk.Button(slots_frame, text="Treinar Slot Selecionado", 
                                        command=self.train_selected_slot,
                                        )
        self.btn_train_slot.pack(fill=X, padx=10, pady=(4, 8))
        
        # Informações dos slots com design moderno
        self.slot_info_frame = ttk.LabelFrame(slots_frame, text="Informações do Slot", )
        self.slot_info_frame.pack(fill=X, padx=10, pady=(8, 8))
        
        # Label para mostrar informações do slot selecionado
        self.slot_info_label = ttk.Label(self.slot_info_frame, 
                                       text="Nenhum slot selecionado", 
                                       justify=LEFT,
                                       font=("Segoe UI", 9))
        self.slot_info_label.pack(fill=X, padx=8, pady=8)
        
        # Seção de Ajuda com design moderno
        help_frame = ttk.LabelFrame(left_panel, text="Ajuda & Configurações", )
        help_frame.pack(fill=X, pady=(0, 15))
        
        self.btn_help = ttk.Button(help_frame, text="Mostrar Ajuda", 
                                  command=self.show_help,
                                  )
        self.btn_help.pack(fill=X, padx=10, pady=(8, 4))
        
        # Botão de configurações com design moderno
        self.btn_config = ttk.Button(help_frame, text="Configurações do Sistema", 
                                    command=self.open_system_config,
                                    )
        self.btn_config.pack(fill=X, padx=10, pady=(4, 8))
        
        # === PAINEL CENTRAL - Editor de Malha ===
        
        # Canvas com scrollbars e design moderno
        canvas_frame = ttk.LabelFrame(center_panel, text="Editor de Malha", )
        canvas_frame.pack(fill=BOTH, expand=True)
        
        # Frame para canvas e scrollbars
        canvas_container = ttk.Frame(canvas_frame, )
        canvas_container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=VERTICAL)
        v_scrollbar.pack(side=RIGHT, fill=Y)
        
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=HORIZONTAL)
        h_scrollbar.pack(side=BOTTOM, fill=X)
        
        # Canvas com design moderno
        self.canvas = Canvas(canvas_container, 
                           bg=get_color('colors.canvas_colors.modern_bg'),  # Cor de fundo moderna
                           highlightthickness=0,
                           relief="flat",
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
        
        # Binds para zoom e pan
        self.canvas.bind("<MouseWheel>", self.on_canvas_zoom)
        self.canvas.bind("<Button-2>", self.on_canvas_pan_start)  # Botão do meio
        self.canvas.bind("<B2-Motion>", self.on_canvas_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_canvas_pan_end)
        
        # Variáveis para zoom e pan
        self.zoom_level = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Status bar com design moderno
        status_frame = ttk.Frame(self)
        status_frame.pack(side=BOTTOM, fill=X, padx=10, pady=(5, 10))
        
        self.status_var = StringVar()
        self.status_var.set("📋 Carregue uma imagem para começar")
        status_bar = ttk.Label(status_frame, 
                              textvariable=self.status_var, 
                              font=("Segoe UI", 9))
        status_bar.pack(padx=15, pady=8)
        
        # Inicializa o painel direito com mensagem padrão
        self.show_default_right_panel()
        
        # Configura tamanho inicial responsivo
        self.configure_responsive_window()
    
    def configure_responsive_window(self):
        """Configura o tamanho da janela de forma responsiva baseado na resolução da tela."""
        # Proteção contra recursão infinita
        if getattr(self, '_configuring_window', False):
            return
        
        try:
            self._configuring_window = True
            
            # Verifica se é uma janela top-level (tem métodos geometry e minsize)
            if not hasattr(self, 'geometry') or not hasattr(self, 'minsize'):
                # Se não é uma janela top-level, tenta configurar a janela pai
                if hasattr(self, 'master') and hasattr(self.master, 'geometry'):
                    parent = self.master
                else:
                    print("Configuração responsiva não aplicável para este tipo de widget")
                    return
            else:
                parent = self
            
            # Verifica se a janela já foi inicializada
            try:
                parent.update_idletasks()
                if parent.winfo_width() <= 1 or parent.winfo_height() <= 1:
                    # Janela ainda não foi renderizada, agenda para depois
                    parent.after(200, self.configure_responsive_window)
                    return
            except Exception:
                pass
            
            # Obtém dimensões da tela
            screen_width = parent.winfo_screenwidth()
            screen_height = parent.winfo_screenheight()
            
            # Calcula tamanho ideal (80% da tela, mas com limites)
            ideal_width = min(max(int(screen_width * 0.8), 1200), screen_width - 100)
            ideal_height = min(max(int(screen_height * 0.8), 800), screen_height - 100)
            
            # Centraliza a janela
            x = (screen_width - ideal_width) // 2
            y = (screen_height - ideal_height) // 2
            
            # Aplica a geometria apenas se diferente da atual
            try:
                current_geometry = parent.geometry()
                new_geometry = f"{ideal_width}x{ideal_height}+{x}+{y}"
                
                if current_geometry != new_geometry:
                    parent.geometry(new_geometry)
                    
                    # Define tamanho mínimo
                    if hasattr(parent, 'minsize'):
                        parent.minsize(1000, 700)
                    
                    print(f"Janela configurada: {ideal_width}x{ideal_height} (Tela: {screen_width}x{screen_height})")
            except Exception as geo_error:
                print(f"Erro ao aplicar geometria da janela: {geo_error}")
            
        except Exception as e:
            print(f"Erro ao configurar janela responsiva: {e}")
        finally:
            # Libera o flag de configuração
            self._configuring_window = False

    def toggle_live_capture_manual_inspection(self):
        """Alterna o modo de captura contínua com inspeção manual (ativada pelo Enter)."""
        if hasattr(self, 'manual_inspection_mode') and self.manual_inspection_mode:
            self.stop_live_capture_manual_inspection()
        else:
            self.start_live_capture_manual_inspection()

    def start_live_capture_manual_inspection(self):
        """Inicia captura contínua com inspeção manual."""
        if self.live_capture:
            self.stop_live_capture()
        try:
            camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
            import platform
            is_windows = platform.system() == 'Windows'
            if is_windows:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            try:
                from camera_manager import configure_video_capture
                configure_video_capture(self.camera, camera_index)
            except Exception:
                pass
            self.live_capture = True
            self.manual_inspection_mode = True
            self.latest_frame = None
            self.status_var.set(f"Modo Inspeção Manual ativado - Câmera {camera_index} ativa")
            self.process_live_frame()
        except Exception as e:
            print(f"Erro ao iniciar inspeção manual: {e}")
            messagebox.showerror("Erro", f"Erro ao iniciar inspeção manual: {str(e)}")

    def stop_live_capture_manual_inspection(self):
        """Para a captura contínua com inspeção manual."""
        self.live_capture = False
        self.manual_inspection_mode = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.latest_frame = None
        self.status_var.set("Modo Inspeção Manual desativado")
    
    def on_camera_changed(self, event=None):
        """Callback para quando o usuário muda a seleção da câmera - Troca instantânea."""
        try:
            new_camera_index = int(self.camera_combo.get())
            if new_camera_index != self.current_camera_index:
                print(f"Trocando instantaneamente da câmera {self.current_camera_index} para câmera {new_camera_index} no editor de malha")
                
                # Verifica se a nova câmera está disponível no sistema de múltiplas câmeras
                if hasattr(self, 'active_cameras') and new_camera_index in self.active_cameras:
                    # Troca instantânea - apenas atualiza referências
                    self.current_camera_index = new_camera_index
                    self.camera = self.active_cameras[new_camera_index]
                    
                    # Atualiza latest_frame com o frame mais recente da nova câmera
                    if hasattr(self, 'camera_frames') and new_camera_index in self.camera_frames and self.camera_frames[new_camera_index] is not None:
                        try:
                            self.latest_frame = self.camera_frames[new_camera_index].copy()
                        except Exception:
                            self.latest_frame = self.camera_frames[new_camera_index]
                    
                    print(f"Câmera {new_camera_index} ativada instantaneamente no editor de malha")
                else:
                    # Fallback para o método tradicional se a câmera não estiver no sistema múltiplo
                    print(f"Câmera {new_camera_index} não encontrada no sistema múltiplo, usando método tradicional")
                    
                    # Para todas as capturas ativas
                    if hasattr(self, 'live_capture') and self.live_capture:
                        self.stop_live_capture()
                    if hasattr(self, 'manual_inspection_mode') and self.manual_inspection_mode:
                        self.stop_live_capture_manual_inspection()
                    
                    # Libera câmera atual
                    if hasattr(self, 'camera') and self.camera:
                        self.camera.release()
                        self.camera = None
                    
                    # Atualiza índice da câmera
                    self.current_camera_index = new_camera_index
                    
                    # Usa o pool de câmeras em vez de inicializar em segundo plano
                    if hasattr(self, 'camera_pool') and new_camera_index in self.camera_pool:
                        self.camera = self.camera_pool[new_camera_index]
                        print(f"Câmera {new_camera_index} obtida do pool na troca")
                    elif hasattr(self, 'active_cameras') and new_camera_index in self.active_cameras:
                        self.camera = self.active_cameras[new_camera_index]
                        print(f"Câmera {new_camera_index} obtida do sistema ativo na troca")
                    else:
                        # Tenta obter do cache do camera_manager
                        try:
                            from camera_manager import get_cached_camera
                            cached_camera = get_cached_camera(new_camera_index)
                            if cached_camera:
                                self.camera = cached_camera
                                print(f"Câmera {new_camera_index} obtida do cache na troca")
                            else:
                                print(f"Câmera {new_camera_index} não disponível na troca")
                        except Exception as cache_error:
                            print(f"Erro ao obter câmera do cache na troca: {cache_error}")
                
        except (ValueError, AttributeError) as e:
            print(f"Erro ao trocar câmera: {e}")
    
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
        self.current_model = None
        
        # Reset das flags de controle
        self._selecting_slot = False
        self._processing_edit_click = False
        
        # Limpa canvas
        self.canvas.delete("all")
        
        # Limpa informações do slot
        self.slot_info_label.config(text="Nenhum slot selecionado")
        
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
            # Preserva o modelo atual se estiver em criação
            current_model_backup = None
            if hasattr(self, 'current_model') and self.current_model and self.current_model.get('id') is None:
                current_model_backup = self.current_model.copy()
            
            self.clear_all()
            
            # Restaura o modelo em criação
            if current_model_backup:
                self.current_model = current_model_backup
                self.current_model['image_path'] = file_path  # Atualiza o caminho da imagem
            
            if self.load_image_data(file_path):
                self.status_var.set(f"Imagem carregada: {Path(file_path).name}")
                self.update_button_states()
    
    def auto_start_webcam(self):
        """Inicia automaticamente a webcam em segundo plano se houver câmeras disponíveis."""
        try:
            if self.available_cameras and len(self.available_cameras) > 0:
                # Seleciona a primeira câmera disponível
                self.camera_combo.set(str(self.available_cameras[0]))
                # Inicia a câmera em segundo plano (sem exibir no canvas)
                self.start_background_camera()
                print(f"Webcam iniciada em segundo plano: Câmera {self.available_cameras[0]}")
            else:
                print("Nenhuma câmera disponível para inicialização automática")
        except Exception as e:
            print(f"Erro na inicialização automática da webcam: {e}")
    
    def start_background_camera(self):
        """Inicia a câmera em segundo plano para captura quando solicitado."""
        try:
            camera_index = int(self.camera_combo.get()) if self.camera_combo.get() else 0
            
            # Detecta o sistema operacional
            import platform
            is_windows = platform.system() == 'Windows'
            
            # Configurações otimizadas para inicialização mais rápida
            if is_windows:
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise ValueError(f"Não foi possível abrir a câmera {camera_index}")
            
            # Configurações otimizadas para performance
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Usa resolução padrão para inicialização rápida
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Marca que a câmera está disponível em segundo plano
            self.live_capture = False  # Não está fazendo live capture
            self.live_view = False     # Não está exibindo no canvas
            
            self.status_var.set(f"Câmera {camera_index} pronta em segundo plano")
            
        except Exception as e:
            print(f"Erro ao iniciar câmera em segundo plano: {e}")
            if self.camera:
                self.camera.release()
                self.camera = None
    
    def on_canvas_zoom(self, event):
        """Implementa zoom no canvas com a roda do mouse."""
        try:
            # Determinar direção do zoom
            if event.delta > 0:
                # Zoom in
                zoom_factor = 1.1
            else:
                # Zoom out
                zoom_factor = 0.9
            
            # Aplicar zoom
            old_zoom = self.zoom_level
            self.zoom_level *= zoom_factor
            
            # Limitar zoom entre 0.1x e 5.0x
            self.zoom_level = max(0.1, min(self.zoom_level, 5.0))
            
            # Se o zoom mudou, redimensionar a imagem
            if self.zoom_level != old_zoom and hasattr(self, 'current_image') and self.current_image is not None:
                self.update_canvas_image()
                
        except Exception as e:
            print(f"Erro no zoom: {e}")
    
    def on_canvas_pan_start(self, event):
        """Inicia o pan com o botão do meio do mouse."""
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")
    
    def on_canvas_pan_drag(self, event):
        """Executa o pan arrastando com o botão do meio."""
        try:
            # Calcular deslocamento
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            # Mover a visualização do canvas
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            
        except Exception as e:
            print(f"Erro no pan: {e}")
    
    def on_canvas_pan_end(self, event):
        """Finaliza o pan."""
        self.canvas.config(cursor="")
    
    def update_canvas_image(self):
        """Atualiza a imagem no canvas com o nível de zoom atual."""
        try:
            if hasattr(self, 'current_image') and self.current_image is not None:
                # Calcular novo tamanho
                original_height, original_width = self.current_image.shape[:2]
                new_width = int(original_width * self.zoom_level)
                new_height = int(original_height * self.zoom_level)
                
                # Redimensionar imagem
                resized_image = cv2.resize(self.current_image, (new_width, new_height))
                
                # Converter para formato do Tkinter
                image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                self.photo = ImageTk.PhotoImage(image_pil)
                
                # Atualizar canvas
                self.canvas.delete("image")
                self.canvas.create_image(0, 0, anchor="nw", image=self.photo, tags="image")
                
                # Atualizar região de scroll
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                
                # Redesenhar slots
                self.draw_slots()
                
        except Exception as e:
            print(f"Erro ao atualizar imagem do canvas: {e}")
    
    def start_live_capture(self):
        """Inicia captura contínua da câmera em segundo plano."""
        if self.live_capture:
            return
            
        try:
            # Desativa outros modos de captura se estiverem ativos
            if hasattr(self, 'manual_inspection_mode') and self.manual_inspection_mode:
                self.stop_live_capture_manual_inspection()
                
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
            
            # Configurações centralizadas (inclui exposição/ganho/WB)
            try:
                from camera_manager import configure_video_capture
                configure_video_capture(self.camera, camera_index)
            except Exception:
                pass
            
            self.live_capture = True
            # Garante que o modo de inspeção manual está desativado
            self.manual_inspection_mode = False
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
                # Reseta os outros botões de captura
                # btn_continuous_inspect e btn_manual_inspect removidos
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
                # Usa cache de câmera para evitar reinicializações
                captured_image = capture_image_from_camera(camera_index, use_cache=True)
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
            # Define o modelo atual para uso em outras funções
            self.current_model = model_data
            
            # Configurar câmera padrão associada ao modelo
            prev_camera_index = None
            if hasattr(self, 'camera_combo') and self.camera_combo.get():
                try:
                    prev_camera_index = int(self.camera_combo.get())
                except Exception:
                    prev_camera_index = None
            camera_index = model_data.get('camera_index', 0)
            if hasattr(self, 'camera_combo') and str(camera_index) in [self.camera_combo['values'][i] for i in range(len(self.camera_combo['values']))]:
                self.camera_combo.set(str(camera_index))
            # Reinicia a câmera se o índice mudou
            if prev_camera_index is not None and prev_camera_index != camera_index:
                try:
                    # Para captura ao vivo, se ativa
                    try:
                        self.stop_live_capture()
                    except Exception:
                        pass
                    # Libera câmera atual
                    if self.camera:
                        try:
                            self.live_capture = False
                            self.camera.release()
                        except Exception:
                            pass
                        self.camera = None
                    # Atualiza índice selecionado
                    self.selected_camera = camera_index
                    # Usa o pool de câmeras em vez de inicializar em segundo plano
                    if hasattr(self, 'camera_pool') and camera_index in self.camera_pool:
                        self.camera = self.camera_pool[camera_index]
                        print(f"Câmera {camera_index} obtida do pool no editor de malha")
                    elif hasattr(self, 'active_cameras') and camera_index in self.active_cameras:
                        self.camera = self.active_cameras[camera_index]
                        print(f"Câmera {camera_index} obtida do sistema ativo no editor de malha")
                    else:
                        # Tenta obter do cache do camera_manager
                        try:
                            cached_camera = get_cached_camera(camera_index)
                            if cached_camera:
                                self.camera = cached_camera
                                print(f"Câmera {camera_index} obtida do cache no editor de malha")
                            else:
                                print(f"Câmera {camera_index} não disponível no editor de malha")
                        except Exception as cache_error:
                            print(f"Erro ao obter câmera do cache no editor de malha: {cache_error}")
                except Exception as cam_e:
                    print(f"Erro ao reiniciar câmera para o modelo: {cam_e}")
            
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
            self.status_var.set(f"Erro ao carregar modelo: {str(e)}")
    
    def create_new_model(self, model_name):
        """Cria um novo modelo vazio."""
        try:
            # Limpa dados atuais
            self.clear_all()
            
            # Define como novo modelo (sem ID ainda)
            self.current_model_id = None
            self.slots = []
            
            # Define dados temporários do modelo para permitir adição de slots
            # IMPORTANTE: Deve ser definido APÓS clear_all() que limpa current_model
            self.current_model = {
                'id': None,  # Será definido quando salvar
                'nome': model_name,
                'image_path': self.image_path if hasattr(self, 'image_path') else None,
                'slots': []
            }
            
            self.status_var.set(f"Novo modelo criado: {model_name}")
            self.update_button_states()
            
            # Marca modelo como salvo (novo modelo vazio)
            self.mark_model_saved()
            
            print(f"Novo modelo '{model_name}' criado")
            
        except Exception as e:
            print(f"Erro ao criar novo modelo: {e}")
            messagebox.showerror("Erro", f"Erro ao criar novo modelo: {str(e)}")
    
    def update_slots_tree(self):
        """Atualiza as informações de slots na interface."""
        try:
            # Atualiza a informação do slot selecionado
            self.update_slot_info_display()
        except Exception as e:
            print(f"Erro ao atualizar informações de slots: {e}")
            import traceback
            traceback.print_exc()
            
    def update_slot_info_display(self):
        """Atualiza o display de informações do slot selecionado."""
        if self.selected_slot_id is None:
            self.slot_info_label.config(text="Nenhum slot selecionado")
            return
            
        # Busca o slot selecionado
        selected_slot = next((s for s in self.slots if s['id'] == self.selected_slot_id), None)
        if not selected_slot:
            self.slot_info_label.config(text=f"Erro: Slot {self.selected_slot_id} não encontrado")
            return
            
        # Formata as informações do slot
        info_text = f"ID: {selected_slot['id']}\n"
        info_text += f"Tipo: {selected_slot.get('tipo', 'N/A')}\n"
        info_text += f"Posição: ({selected_slot.get('x', 0)}, {selected_slot.get('y', 0)})\n"
        info_text += f"Tamanho: {selected_slot.get('w', 0)}x{selected_slot.get('h', 0)}"
        
        # Adiciona informações específicas do tipo de slot
        if selected_slot.get('tipo') == 'clip':
            corr_thr = selected_slot.get('correlation_threshold', selected_slot.get('detection_threshold', 0.5))
            info_text += f"\nCorrelação (limiar): {corr_thr}"
            
        self.slot_info_label.config(text=info_text)
    
    def update_slots_list(self):
        """Função legada para compatibilidade - redireciona para update_slots_tree."""
        self.update_slots_tree()
    
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
            
            # Converte coordenadas da imagem para canvas (incluindo offsets)
            x1 = int(slot['x'] * self.scale_factor) + self.x_offset
            y1 = int(slot['y'] * self.scale_factor) + self.y_offset
            x2 = int((slot['x'] + slot['w']) * self.scale_factor) + self.x_offset
            y2 = int((slot['y'] + slot['h']) * self.scale_factor) + self.y_offset
            
            # Calcula centro para rotação
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Carrega as configurações de estilo
            style_config = load_style_config()
            
            # Escolhe cor baseada na seleção
            if slot['id'] == self.selected_slot_id:
                color = get_color('colors.selection_color', style_config)
                width = 3
            else:
                color = get_color('colors.editor_colors.clip_color')
                width = 2
            
            # Obtém rotação do slot
            # Desenha retângulo simples (rotação removida)
            shape_id = self.canvas.create_rectangle(x1, y1, x2, y2, 
                                       outline=color, width=width, tags="slot")
            
            # Removido desenho de áreas de exclusão
            
            # Adiciona texto com ID (já usando x1, y1 corrigidos com offsets)
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
                                                   fill=get_color('colors.inspection_colors.pass_color'), outline=get_color('colors.special_colors.white_text'), width=1,
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
                        if tag == "edit_handle" or tag.startswith("resize_handle_"):
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
                    
                    # Seleciona o slot e mostra handles de edição
                    self.select_slot(clicked_slot['id'])
                    self.show_edit_handles(clicked_slot)
                    return
            except Exception as e:
                print(f"Erro ao verificar slot existente: {e}")
                import traceback
                traceback.print_exc()
            
            # Modo de exclusão removido
            
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
        
        # Define cor (modo de exclusão removido)
        outline_color = get_color('colors.editor_colors.drawing_color')
        
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
        
        # Adiciona slot normal (exclusão removida)
        self.add_slot(img_x, img_y, img_w, img_h)
    
    def find_slot_at(self, canvas_x, canvas_y):
        """Encontra slot nas coordenadas do canvas."""
        for slot in self.slots:
            x1 = slot['x'] * self.scale_factor
            y1 = slot['y'] * self.scale_factor
            x2 = (slot['x'] + slot['w']) * self.scale_factor
            y2 = (slot['y'] + slot['h']) * self.scale_factor
            
            # Verificação simples de retângulo
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
            
            # Atualiza informações do slot selecionado
            self.update_slot_info_display()
            
            # Mostra automaticamente o editor de slot no painel direito
            slot_to_edit = next((s for s in self.slots if s['id'] == slot_id), None)
            if slot_to_edit:
                self.show_slot_editor_in_right_panel(slot_to_edit)
            
            self.redraw_slots()
            self.update_button_states()
            self.status_var.set(f"Slot {slot_id} selecionado - Editor aberto no painel direito")
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
        self.slot_info_label.config(text="Nenhum slot selecionado")
        
        # Exibe mensagem padrão no painel direito quando nenhum slot está selecionado
        self.show_default_right_panel()
        
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
            'shape': 'rectangle',
            'use_alignment': True
        }
        
        # Configurações específicas para clips
        slot_data.update({
            'correlation_threshold': THR_CORR,
            'template_method': 'TM_CCOEFF_NORMED',
            'scale_tolerance': 0.1
        })
        
        # Armazena o ROI em memória para salvar depois quando o modelo for salvo
        slot_data['roi_data'] = roi.copy()  # Armazena uma cópia do ROI
        slot_data['template_filename'] = f"slot_{slot_id}_template.png"
        
        print(f"ROI do slot {slot_id} armazenado em memória para salvamento posterior")
        
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
    
    # Funções on_slot_select e on_slot_double_click foram removidas
    # pois não são mais necessárias sem o slots_listbox
    
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
    
    # Função edit_selected_slot removida - editor aparece automaticamente quando slot é selecionado
    
    def clear_right_panel(self):
        """Limpa o painel direito."""
        for widget in self.right_panel.winfo_children():
            widget.destroy()
    
    def show_default_right_panel(self):
        """Exibe mensagem padrão no painel direito quando nenhum slot está selecionado."""
        self.clear_right_panel()
        
        # Título do painel com design moderno
        title_label = ttk.Label(self.right_panel, 
                               text="🎯 Editor de Slot", 
                               font=("Segoe UI", 14, "bold"),
                               )
        title_label.pack(pady=(20, 15))
        
        # Card de mensagem informativa
        info_card = ttk.Frame(self.right_panel, )
        info_card.pack(fill=X, padx=15, pady=10)
        
        info_label = ttk.Label(info_card, 
                              text="💡 Selecione um slot no\nEditor de Malha para\neditar suas propriedades",
                              justify=CENTER,
                              font=("Segoe UI", 10),
                              )
        info_label.pack(pady=15)
        
        # Card de instruções com design moderno
        instructions_frame = ttk.LabelFrame(self.right_panel, 
                                          text="📋 Instruções", 
                                          )
        instructions_frame.pack(fill=X, padx=15, pady=(15, 0))
        
        instructions_text = (
            "🖱️ Clique em um slot no canvas\n"
            "   para selecioná-lo\n\n"
            "⚡ O editor aparecerá\n"
            "   automaticamente\n\n"
            "⚙️ Ajuste posição, tamanho\n"
            "   e parâmetros de detecção"
        )
        
        instructions_label = ttk.Label(instructions_frame, 
                                     text=instructions_text,
                                     justify=LEFT,
                                     font=("Segoe UI", 9),
                                     )
        instructions_label.pack(padx=15, pady=12)
    
    def save_slot_changes(self, slot_data):
        """Salva as alterações feitas no slot"""
        try:
            # Obtém os valores dos campos
            new_x = int(self.edit_vars['x'].get())
            new_y = int(self.edit_vars['y'].get())
            new_w = int(self.edit_vars['w'].get())
            new_h = int(self.edit_vars['h'].get())
            
            # Atualiza os dados do slot
            for slot in self.slots:
                if slot['id'] == slot_data['id']:
                    slot['x'] = new_x
                    slot['y'] = new_y
                    slot['w'] = new_w
                    slot['h'] = new_h
                    
                    # Para slots do tipo clip, atualiza parâmetros de detecção
                    if slot.get('tipo') == 'clip' and 'detection_method' in self.edit_vars:
                        # Método de detecção
                        old_method = slot.get('detection_method', 'template_matching')
                        new_method = self.edit_vars['detection_method'].get()
                        
                        # Atualiza o método de detecção
                        slot['detection_method'] = new_method
                        print(f"Método de detecção alterado de {old_method} para {new_method}")
                        
                        # Limiar de correlação
                        if 'detection_threshold' in self.edit_vars:
                            # Compat: se ainda existir este campo, trata como correlação
                            slot['correlation_threshold'] = float(self.edit_vars['detection_threshold'].get())
                        if 'correlation_threshold' in self.edit_vars:
                            slot['correlation_threshold'] = float(self.edit_vars['correlation_threshold'].get())
                        # Alinhamento por slot
                        if 'use_alignment' in self.edit_vars:
                            slot['use_alignment'] = (self.edit_vars['use_alignment'].get() not in ("0","False","false","no","nao"))
                        # Usar ML se disponível
                        if 'use_ml' in self.edit_vars and slot.get('ml_model_path'):
                            slot['use_ml'] = (self.edit_vars['use_ml'].get() not in ("0","False","false","no","nao"))
                    
                    # Salva no banco de dados se há um modelo carregado
                    if self.current_model_id is not None:
                        try:
                            self.db_manager.update_slot(self.current_model_id, slot)
                        except Exception as e:
                            print(f"Erro ao salvar slot no banco: {e}")
                            messagebox.showwarning("Aviso", "Slot atualizado na interface, mas não foi salvo no banco de dados.")
                    
                    break
            
            # Redesenha o canvas
            self.redraw_slots()
            
            # Atualiza a lista de slots
            self.update_slots_list()
            
            # Limpa o painel direito
            self.clear_right_panel()
            
            # Marca o modelo como modificado
            self.mark_model_modified()
            
            # Atualiza a mensagem de status
            self.status_var.set(f"Slot {slot_data['id']} atualizado com sucesso")
            
            print(f"Slot {slot_data['id']} atualizado com sucesso")
            
        except ValueError as e:
            messagebox.showerror("Erro", f"Valores inválidos: {str(e)}\nVerifique se todos os campos contêm números válidos.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar alterações: {str(e)}")
    
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
    
    def show_slot_editor_in_right_panel(self, slot_data):
        """Cria um editor de slot simplificado no painel direito"""
        print("Criando editor de slot no painel direito...")
        
        # Carrega as configurações de estilo
        self.style_config = load_style_config()
        
        # Limpa o painel direito
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        
        # Título do editor
        title_frame = ttk.Frame(self.right_panel)
        title_frame.pack(fill='x', pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text=f"Editar Slot {slot_data['id']}", 
                               font=('Arial', 10, 'bold'),
                               foreground=get_color('colors.text_color', self.style_config))
        title_label.pack(pady=(0, 5))
        
        # Frame com scrollbar para os campos
        editor_frame = ttk.Frame(self.right_panel)
        editor_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Variáveis para os campos
        self.edit_vars = {}
        
        # Seção: Posição e Tamanho
        position_frame = ttk.LabelFrame(editor_frame, text="Posição e Tamanho")
        position_frame.pack(fill='x', pady=(0, 10), padx=5)
        
        # Campos básicos com descrições simples
        basic_fields = [
            ('X:', 'x', slot_data['x'], "Posição horizontal"),
            ('Y:', 'y', slot_data['y'], "Posição vertical"),
            ('Largura:', 'w', slot_data['w'], "Largura do slot"),
            ('Altura:', 'h', slot_data['h'], "Altura do slot")
        ]
        
        for i, (label_text, key, value, tooltip) in enumerate(basic_fields):
            row_frame = ttk.Frame(position_frame)
            row_frame.pack(fill='x', pady=2, padx=5)
            
            label = ttk.Label(row_frame, text=label_text, width=8)
            label.pack(side='left')
            
            var = StringVar(value=str(value))
            self.edit_vars[key] = var
            entry = ttk.Entry(row_frame, textvariable=var, width=8)
            entry.pack(side='left', padx=(5, 0))
            
            # Tooltip simples
            tip_label = ttk.Label(row_frame, text=tooltip, font=get_font('tiny_font'), foreground=get_color('colors.special_colors.tooltip_fg'))
            tip_label.pack(side='left', padx=(5, 0))
        
        # Seção: Detecção (para slots do tipo clip)
        if slot_data.get('tipo') == 'clip':
            detection_frame = ttk.LabelFrame(editor_frame, text="Parâmetros de Detecção")
            detection_frame.pack(fill='x', pady=(0, 10), padx=5)
            
            # Método de detecção
            method_frame = ttk.Frame(detection_frame)
            method_frame.pack(fill='x', pady=2, padx=5)
            
            method_label = ttk.Label(method_frame, text="Método:", width=8)
            method_label.pack(side='left')
            
            detection_methods = [
                "template_matching", # Comparação de imagem
                "histogram_analysis", # Análise de histograma
                "contour_analysis", # Análise de contorno
                "image_comparison" # Comparação direta de imagem
            ]
            method_var = StringVar(value=slot_data.get('detection_method', 'template_matching'))
            self.edit_vars['detection_method'] = method_var
            
            method_combo = ttk.Combobox(method_frame, textvariable=method_var, 
                                      values=detection_methods, width=15, state="readonly")
            method_combo.pack(side='left', padx=(5, 0))
            
            # Tooltip para explicar cada método
            method_tip = ttk.Label(method_frame, text="Selecione o método de detecção", 
                                  font=get_font('tiny_font'), foreground=get_color('colors.special_colors.tooltip_fg'))
            method_tip.pack(side='left', padx=(5, 0))
            
            # Atualiza o tooltip baseado na seleção
            def update_method_tip(*args):
                method = method_var.get()
                if method == "template_matching":
                    method_tip.config(text="Comparação de imagem com template")
                elif method == "histogram_analysis":
                    method_tip.config(text="Análise de distribuição de cores")
                elif method == "contour_analysis":
                    method_tip.config(text="Análise de contornos e formas")
                elif method == "image_comparison":
                    method_tip.config(text="Comparação direta entre imagens")
            
            method_var.trace("w", update_method_tip)
            update_method_tip()  # Inicializa o tooltip
            
            # Adiciona um canvas para preview do filtro
            preview_frame = ttk.LabelFrame(detection_frame, text="Preview do Filtro")
            preview_frame.pack(fill='x', pady=5, padx=5)
            
            # Canvas para exibir o preview
            self.preview_canvas = Canvas(preview_frame, bg=get_color('colors.special_colors.preview_canvas_bg'), width=200, height=150)
            self.preview_canvas.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Definir variáveis antes da função update_preview_filter
            threshold_var = StringVar(value=str(slot_data.get('correlation_threshold', slot_data.get('detection_threshold', 0.5))))
            
            # Função para atualizar o preview quando o método de detecção mudar
            def update_preview_filter(*args):
                if not hasattr(self, 'img_original') or self.img_original is None:
                    return
                
                # Obtém o slot atual
                if not slot_data or 'x' not in slot_data or 'y' not in slot_data or 'w' not in slot_data or 'h' not in slot_data:
                    return
                
                # Extrai a ROI do slot
                x, y, w, h = slot_data['x'], slot_data['y'], slot_data['w'], slot_data['h']
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > self.img_original.shape[1] or y + h > self.img_original.shape[0]:
                    return
                
                roi = self.img_original[y:y+h, x:x+w].copy()
                
                # Aplica o filtro selecionado
                method = method_var.get()
                filtered_roi = roi.copy()
                
                try:
                    if method == "histogram_analysis":
                        # Converte para HSV e visualiza o histograma
                        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        h_bins = 50
                        s_bins = 60
                        hist_range = [0, 180, 0, 256]  # H: 0-179, S: 0-255
                        hist = cv2.calcHist([roi_hsv], [0, 1], None, [h_bins, s_bins], hist_range)
                        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                        
                        # Cria uma visualização do histograma
                        hist_img = np.zeros((h_bins, s_bins), np.uint8)
                        for h in range(h_bins):
                            for s in range(s_bins):
                                hist_img[h, s] = min(255, int(hist[h, s] * 255))
                        
                        # Redimensiona para melhor visualização
                        hist_img = cv2.resize(hist_img, (w, h))
                        hist_img = cv2.applyColorMap(hist_img, cv2.COLORMAP_JET)
                        filtered_roi = hist_img
                        
                    elif method == "contour_analysis":
                        # Converte para escala de cinza e aplica detecção de bordas
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                        edges = cv2.Canny(roi_blur, 50, 150)
                        
                        # Encontra contornos
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Desenha contornos em uma imagem colorida
                        contour_img = np.zeros_like(roi)
                        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
                        filtered_roi = contour_img
                        
                    elif method == "image_comparison":
                        # Verifica se há um template para comparação
                        template_path = slot_data.get('template_path')
                        if template_path and Path(template_path).exists():
                            template = cv2.imread(str(template_path))
                            if template is not None:
                                # Redimensiona o template para o tamanho da ROI
                                template_resized = cv2.resize(template, (roi.shape[1], roi.shape[0]))
                                
                                # Calcula a diferença absoluta
                                diff = cv2.absdiff(roi, template_resized)
                                filtered_roi = diff
                        else:
                            # Se não há template, mostra mensagem
                            filtered_roi = np.zeros_like(roi)
                            cv2.putText(filtered_roi, "Sem template", (10, h//2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    elif method == "template_matching":
                        # Verifica se há um template para comparação
                        template_path = slot_data.get('template_path')
                        if template_path and Path(template_path).exists():
                            template = cv2.imread(str(template_path))
                            if template is not None:
                                # Mostra o template
                                template_resized = cv2.resize(template, (roi.shape[1], roi.shape[0]))
                                filtered_roi = template_resized
                                
                                # Adiciona texto indicando que é o template
                                cv2.putText(filtered_roi, "Template", (10, 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            # Se não há template, mostra mensagem
                            filtered_roi = np.zeros_like(roi)
                            cv2.putText(filtered_roi, "Sem template", (10, h//2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Erro ao aplicar filtro: {e}")
                    filtered_roi = roi.copy()
                    cv2.putText(filtered_roi, "Erro no filtro", (10, h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Adiciona informações sobre os parâmetros atuais
                try:
                    detection_threshold = float(threshold_var.get())
                    # Adiciona texto com o valor atual da correlação
                    cv2.putText(filtered_roi, f"Correlação: {detection_threshold:.2f}", (10, h-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Erro ao adicionar informações ao preview: {e}")
                
                # Converte para exibição no canvas
                filtered_roi_rgb = cv2.cvtColor(filtered_roi, cv2.COLOR_BGR2RGB)
                filtered_roi_pil = Image.fromarray(filtered_roi_rgb)
                filtered_roi_tk = ImageTk.PhotoImage(filtered_roi_pil)
                
                # Atualiza o canvas
                self.preview_canvas.delete("all")
                
                # Usa as dimensões reais do canvas ou as dimensões configuradas se ainda não foi renderizado
                canvas_width = self.preview_canvas.winfo_width() if self.preview_canvas.winfo_width() > 1 else 200
                canvas_height = self.preview_canvas.winfo_height() if self.preview_canvas.winfo_height() > 1 else 150
                
                self.preview_canvas.create_image(canvas_width//2, 
                                               canvas_height//2, 
                                               image=filtered_roi_tk, anchor="center")
                self.preview_canvas.image = filtered_roi_tk  # Mantém referência
            
            # Vincula a função de atualização ao combobox
            method_var.trace("w", update_preview_filter)
            
            # Inicializa o preview
            update_preview_filter()
            
            # Limiar de detecção
            threshold_frame = ttk.Frame(detection_frame)
            threshold_frame.pack(fill='x', pady=2, padx=5)
            
            threshold_label = ttk.Label(threshold_frame, text="Correlação:", width=8)
            threshold_label.pack(side='left')
            
            self.edit_vars['detection_threshold'] = threshold_var
            threshold_entry = ttk.Entry(threshold_frame, textvariable=threshold_var, width=8)
            threshold_entry.pack(side='left', padx=(5, 0))
            
            # Vincula a função de atualização do preview ao limiar de detecção
            threshold_var.trace("w", update_preview_filter)
            
            threshold_tip = ttk.Label(threshold_frame, text="Limiar de correlação (0.0–1.0)", 
                                    font=get_font('tiny_font'), foreground=get_color('colors.special_colors.tooltip_fg'))
            threshold_tip.pack(side='left', padx=(5, 0))
            
            # Limiar de correlação
            correlation_threshold_frame = ttk.Frame(detection_frame)
            correlation_threshold_frame.pack(fill='x', pady=2, padx=5)
            
            correlation_threshold_label = ttk.Label(correlation_threshold_frame, text="Correlação:", width=8)
            correlation_threshold_label.pack(side='left')
            
            correlation_threshold_var = StringVar(value=str(slot_data.get('correlation_threshold', 0.5)))
            self.edit_vars['correlation_threshold'] = correlation_threshold_var
            correlation_threshold_entry = ttk.Entry(correlation_threshold_frame, textvariable=correlation_threshold_var, width=8)
            correlation_threshold_entry.pack(side='left', padx=(5, 0))
            
            # Vincula a função de atualização do preview ao limiar de correlação
            correlation_threshold_var.trace("w", update_preview_filter)
            
            correlation_threshold_tip = ttk.Label(correlation_threshold_frame, text="Limiar de correlação (0.0-1.0)", 
                                                 font=get_font('tiny_font'), foreground=get_color('colors.special_colors.tooltip_fg'))
            correlation_threshold_tip.pack(side='left', padx=(5, 0))

            # Novo: Alinhamento por slot (autoajuste vs fixo)
            align_frame = ttk.Frame(detection_frame)
            align_frame.pack(fill='x', pady=2, padx=5)
            ttk.Label(align_frame, text="Alinhamento:", width=8).pack(side='left')
            align_var = StringVar(value=("1" if bool(slot_data.get('use_alignment', True)) else "0"))
            self.edit_vars['use_alignment'] = align_var
            align_combo = ttk.Combobox(align_frame, values=["Autoajuste (alinha)", "Fixo (sem alinhar)"], state='readonly', width=18)
            align_combo.pack(side='left', padx=(5, 0))
            if bool(slot_data.get('use_alignment', True)):
                align_combo.set("Autoajuste (alinha)")
            else:
                align_combo.set("Fixo (sem alinhar)")
            def _on_align_change(*args):
                align_var.set("1" if align_combo.get().startswith("Auto") else "0")
            align_combo.bind("<<ComboboxSelected>>", _on_align_change)

            # Novo: Usar ML (visível somente se houver modelo treinado)
            if slot_data.get('ml_model_path'):
                ml_frame = ttk.Frame(detection_frame)
                ml_frame.pack(fill='x', pady=2, padx=5)
                ttk.Label(ml_frame, text="Usar ML:", width=8).pack(side='left')
                use_ml_var = StringVar(value=("1" if bool(slot_data.get('use_ml', True)) else "0"))
                self.edit_vars['use_ml'] = use_ml_var
                ml_combo = ttk.Combobox(ml_frame, values=["Ativado", "Desativado"], state='readonly', width=18)
                ml_combo.pack(side='left', padx=(5, 0))
                if bool(slot_data.get('use_ml', True)):
                    ml_combo.set("Ativado")
                else:
                    ml_combo.set("Desativado")
                def _on_ml_change(*args):
                    use_ml_var.set("1" if ml_combo.get().startswith("Ativado") else "0")
                ml_combo.bind("<<ComboboxSelected>>", _on_ml_change)
        
        # Botões de ação
        buttons_frame = ttk.Frame(self.right_panel)
        buttons_frame.pack(fill='x', pady=10, padx=5)
        
        save_btn = ttk.Button(buttons_frame, text="Salvar", 
                             command=lambda: self.save_slot_changes(slot_data))
        save_btn.pack(side='left', padx=(0, 5), fill='x', expand=True)
        
        cancel_btn = ttk.Button(buttons_frame, text="Cancelar", 
                               command=lambda: self.clear_right_panel())
        cancel_btn.pack(side='left', fill='x', expand=True)
        
        # Adiciona mais informações úteis
        info_frame = ttk.LabelFrame(self.right_panel, text="Informações")
        info_frame.pack(fill='x', pady=(10, 0), padx=5)
        
        # Tipo de slot
        tipo_frame = ttk.Frame(info_frame)
        tipo_frame.pack(fill='x', pady=2, padx=5)
        
        ttk.Label(tipo_frame, text="Tipo:", width=8).pack(side='left')
        tipo_value = ttk.Label(tipo_frame, text=slot_data.get('tipo', 'desconhecido'))
        tipo_value.pack(side='left', padx=(5, 0))
        
        # ID do slot
        id_frame = ttk.Frame(info_frame)
        id_frame.pack(fill='x', pady=2, padx=5)
        
        ttk.Label(id_frame, text="ID:", width=8).pack(side='left')
        id_value = ttk.Label(id_frame, text=str(slot_data['id']))
        id_value.pack(side='left', padx=(5, 0))
            


    
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
                # Tratar sempre correlação como fonte única
                if 'correlation_threshold' in self.edit_vars:
                    slot_data['correlation_threshold'] = float(self.edit_vars['correlation_threshold'].get())
                elif 'detection_threshold' in self.edit_vars:
                    # Compat: se vier deste campo, usa como correlação
                    slot_data['correlation_threshold'] = float(self.edit_vars['detection_threshold'].get())
                if 'template_method' in self.edit_vars:
                    slot_data['template_method'] = self.edit_vars['template_method'].get()
                if 'scale_tolerance' in self.edit_vars:
                    slot_data['scale_tolerance'] = float(self.edit_vars['scale_tolerance'].get())
            
            # Salva no banco de dados se há um modelo carregado
            if self.current_model_id is not None:
                try:
                    self.db_manager.update_slot(self.current_model_id, slot_data)
                except Exception as e:
                    print(f"Erro ao salvar slot no banco: {e}")
            
            # Nota: Removido o processamento de configurações de estilo
            # Essas configurações agora são gerenciadas apenas pelo menu de configurações do sistema
            
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
        
        # Salva no banco de dados se há um modelo carregado
        if self.current_model_id is not None:
            try:
                print(f"Salvando slot {slot_id_to_update} no banco de dados...")
                self.db_manager.update_slot(self.current_model_id, updated_slot_data)
                print(f"Slot {slot_id_to_update} salvo no banco com sucesso!")
            except Exception as e:
                print(f"Erro ao salvar slot no banco: {e}")
        else:
            print("Aviso: Modelo não foi salvo ainda, dados atualizados apenas na memória.")
        
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
    
    def save_templates_to_model_folder(self, model_name, model_id):
        """Salva todos os templates dos slots na pasta do modelo."""
        try:
            # Obtém pasta de templates do modelo
            model_folder = self.db_manager.get_model_folder_path(model_name, model_id)
            templates_folder = model_folder / "templates"
            templates_folder.mkdir(parents=True, exist_ok=True)
            
            # Salva cada template dos slots
            for slot_data in self.slots:
                if 'roi_data' in slot_data and 'template_filename' in slot_data:
                    template_path = templates_folder / slot_data['template_filename']
                    cv2.imwrite(str(template_path), slot_data['roi_data'])
                    
                    # Atualiza o caminho do template no slot
                    slot_data['template_path'] = str(template_path)
                    
                    # Remove os dados temporários
                    del slot_data['roi_data']
                    del slot_data['template_filename']
                    
                    print(f"Template salvo: {template_path}")
                    
        except Exception as e:
            print(f"Erro ao salvar templates: {e}")
            raise e
    
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
            
            # Obtém índice da câmera selecionada para salvar com o modelo
            camera_index = int(self.camera_combo.get()) if hasattr(self, 'camera_combo') and self.camera_combo.get() else 0
            
            if result['action'] in ['update', 'overwrite']:
                # Atualiza modelo existente
                model_id = result['model_id']
                
                # Salva templates primeiro
                self.save_templates_to_model_folder(model_name, model_id)
                
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
                    slots=self.slots,
                    camera_index=camera_index
                )
                
                self.current_model_id = model_id
                # Define o modelo atual para uso em outras funções
                self.current_model = self.db_manager.load_modelo(model_id)
                
            else:
                # Cria novo modelo primeiro para obter o ID
                # Salva temporariamente com caminho vazio
                model_id = self.db_manager.save_modelo(
                    nome=model_name,
                    image_path="",  # Será atualizado depois
                    slots=[],
                    camera_index=camera_index
                )
                
                # Agora salva os templates na pasta correta do modelo
                self.save_templates_to_model_folder(model_name, model_id)
                
                # Obtém pasta específica do modelo (já criada pelo save_modelo)
                model_folder = self.db_manager.get_model_folder_path(model_name, model_id)
                
                # Salva imagem de referência na pasta do modelo
                image_filename = f"{model_name}_reference.jpg"
                image_path = model_folder / image_filename
                cv2.imwrite(str(image_path), self.img_original)
                
                # Atualiza o modelo com os slots e caminho da imagem
                self.db_manager.update_modelo(
                    model_id,
                    image_path=str(image_path),
                    slots=self.slots,
                    camera_index=camera_index
                )
                
                self.current_model_id = model_id
                # Define o modelo atual para uso em outras funções
                self.current_model = self.db_manager.load_modelo(model_id)
            
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
        if hasattr(self, 'btn_save_model'):
            self.btn_save_model.config(state=NORMAL if has_image and has_slots else DISABLED)
        
        # Botões que dependem de slots
        if hasattr(self, 'btn_clear_slots'):
            self.btn_clear_slots.config(state=NORMAL if has_slots else DISABLED)
        
        # Botões que dependem de seleção
        if hasattr(self, 'btn_delete_slot'):
            self.btn_delete_slot.config(state=NORMAL if has_selection else DISABLED)
        if hasattr(self, 'btn_train_slot'):
            self.btn_train_slot.config(state=NORMAL if has_selection else DISABLED)
    
    def set_drawing_mode(self):
        """Define o modo de desenho atual."""
        self.current_drawing_mode = self.drawing_mode.get()
        mode_names = {
            "rectangle": "Retângulo"
        }
        self.tool_status_var.set(f"Modo: {mode_names.get(self.current_drawing_mode, 'Retângulo')}")
        print(f"Modo de desenho alterado para: {self.current_drawing_mode}")
    
    # Função de rotação removida
    
    # Modo de exclusão removido
    
    def show_edit_handles(self, slot):
        """Mostra handles de edição para o slot selecionado."""
        self.hide_edit_handles()  # Remove handles anteriores
        
        x = slot['x'] * self.scale_factor
        y = slot['y'] * self.scale_factor
        w = slot['w'] * self.scale_factor
        h = slot['h'] * self.scale_factor
        
        handle_size = 8
        handle_color = get_color('colors.editor_colors.handle_color')
        
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
        
        # Handle de rotação removido
        
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
                # Tratamento de handle de rotação removido
    
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
        # Tratamento de arrastar handle de rotação removido
    
    def handle_resize_drag(self, slot, canvas_x, canvas_y):
        """Lida com redimensionamento do slot sem recriar handles durante o arraste."""
        direction = self.editing_handle['direction']

        # Converte coordenadas do canvas para coordenadas da imagem (considerando offsets)
        img_x = (canvas_x - self.x_offset) / self.scale_factor
        img_y = (canvas_y - self.y_offset) / self.scale_factor

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

        # Marca modelo como modificado e atualiza apenas o desenho do slot
        # Evita recriar handles durante o arraste (isso fazia o "soltar")
        self.mark_model_modified()
        self.redraw_slots()
    
    # Função de rotação removida
    
    def on_handle_release(self, event):
        """Finaliza edição com handle e reposiciona handles/redesenha lista."""
        if self.editing_handle:
            # Ao finalizar, atualiza UI completa e reposiciona handles
            self.mark_model_modified()
            # Redesenha slots
            self.redraw_slots()
            # Reposiciona handles do slot selecionado
            try:
                selected_slot = next((s for s in self.slots if s['id'] == self.selected_slot_id), None)
                if selected_slot:
                    self.show_edit_handles(selected_slot)
            except Exception:
                pass
            # Atualiza lista/árvore de slots
            try:
                self.update_slots_list()
            except Exception:
                pass
            # Finaliza estado de edição
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
        try:
            from modulos.utils import get_font, get_color
            text_widget = Text(text_frame, wrap="word", yscrollcommand=scrollbar.set,
                              font=get_font('console_font'), bg=get_color('colors.special_colors.console_bg'), fg=get_color('colors.special_colors.console_fg'))
        except:
            text_widget = Text(text_frame, wrap="word", yscrollcommand=scrollbar.set,
                              font=("Consolas", 10), bg="#2b2b2b", fg="#ffffff")
        
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
        y = (help_window.winfo_screenwidth() // 2) - (help_window.winfo_height() // 2)
        help_window.geometry(f"+{x}+{y}")

    def validate_slot_reference(self, slot_id):
        """Valida se a referência do slot está correta para o modelo atual."""
        try:
            current_model = getattr(self, 'current_model', None)
            if not current_model:
                return False
            
            if slot_id not in self.slots:
                return False
            
            slot_data = self.slots[slot_id]
            template_path = slot_data.get('template_path')
            
            if template_path:
                # Verifica se o template pertence ao modelo atual
                expected_prefix = f"modelos/{current_model['nome']}_{current_model['id']}/templates/"
                if not template_path.startswith(expected_prefix):
                    print(f"⚠️ Template {template_path} não pertence ao modelo {current_model['nome']}")
                    return False
                
                # Verifica se o arquivo existe
                abs_path = get_project_root() / template_path
                if not abs_path.exists():
                    print(f"⚠️ Template não encontrado: {template_path}")
                    return False
            
            return True
        except Exception as e:
            print(f"❌ Erro na validação do slot {slot_id}: {e}")
            return False
    
    def cleanup_orphaned_templates(self):
        """Remove templates órfãos que não pertencem ao modelo atual."""
        try:
            current_model = getattr(self, 'current_model', None)
            if not current_model:
                return
            
            orphaned_count = 0
            for slot_id, slot_data in self.slots.items():
                if not self.validate_slot_reference(slot_id):
                    # Remove referência inválida
                    if 'template_path' in slot_data:
                        print(f"🧹 Removendo referência órfã do slot {slot_id}: {slot_data['template_path']}")
                        del slot_data['template_path']
                        orphaned_count += 1
            
            if orphaned_count > 0:
                print(f"✅ {orphaned_count} referências órfãs removidas")
                
        except Exception as e:
            print(f"❌ Erro na limpeza de templates órfãos: {e}")


    
    def open_system_config(self):
        """Abre a janela de configuração do sistema."""
        def on_save(cfg):
            global ORB_FEATURES, ORB_SCALE_FACTOR, ORB_N_LEVELS, PREVIEW_W, PREVIEW_H, THR_CORR, MIN_PX
            ORB_FEATURES = cfg['ORB_FEATURES']
            ORB_SCALE_FACTOR = cfg['ORB_SCALE_FACTOR']
            ORB_N_LEVELS = cfg['ORB_N_LEVELS']
            PREVIEW_W = cfg['PREVIEW_W']
            PREVIEW_H = cfg['PREVIEW_H']
            THR_CORR = cfg['THR_CORR']
            MIN_PX = cfg['MIN_PX']
            
            # Aplicar configurações de aparência
            try:
                style_config = load_style_config()
                
                # Atualizar cores do sistema
                if 'BG_COLOR' in cfg:
                    style_config['colors']['canvas_colors']['modern_bg'] = cfg['BG_COLOR']
                    style_config['colors']['canvas_colors']['canvas_bg'] = cfg['BG_COLOR']
                
                if 'TEXT_COLOR' in cfg:
                    style_config['colors']['text_color'] = cfg['TEXT_COLOR']
                    style_config['colors']['special_colors']['gray_text'] = cfg['TEXT_COLOR']
                
                if 'ACCENT_COLOR' in cfg:
                    style_config['colors']['accent_color'] = cfg['ACCENT_COLOR']
                    style_config['colors']['selection_color'] = cfg['ACCENT_COLOR']
                
                # Atualizar configurações de fonte
                if 'FONT_SIZE' in cfg:
                    style_config['fonts']['console_font'] = f"{cfg['FONT_FAMILY']} {cfg['FONT_SIZE']}"
                    style_config['fonts']['small_font'] = f"{cfg['FONT_FAMILY']} {max(8, cfg['FONT_SIZE'] - 2)}"
                    style_config['fonts']['subtitle_font'] = f"{cfg['FONT_FAMILY']} {cfg['FONT_SIZE'] + 6}"
                
                # Salvar configurações
                save_style_config(style_config)
                
                # Aplicar configurações imediatamente
                apply_style_config(style_config)
                
                print("✅ Configurações de aparência aplicadas com sucesso!")
                
            except Exception as e:
                print(f"❌ Erro ao aplicar configurações de aparência: {e}")
        
        config_dialog = SystemConfigDialog(
            self.master,
            ORB_FEATURES,
            ORB_SCALE_FACTOR,
            ORB_N_LEVELS,
            PREVIEW_W,
            PREVIEW_H,
            THR_CORR,
            MIN_PX,
            on_save
        )
        config_dialog.wait_window()
    
    def set_drawing_mode(self):
        """Define o modo de desenho atual (exclusão removida)."""
        mode = self.drawing_mode.get()
        self.tool_status_var.set("Modo: Retângulo")
        self.current_drawing_mode = "rectangle"
    
    # Funções de rotação removidas

    def on_closing(self):
        """Limpa recursos ao fechar a aplicação."""
        if self.live_capture:
            self.stop_live_capture()
        
        # Libera todas as câmeras em cache
        try:
            release_all_cached_cameras()
            print("Cache de câmeras limpo ao fechar aplicação")
        except Exception as e:
            print(f"Erro ao limpar cache de câmeras: {e}")
        
        self.master.destroy()


# Classe para aba de Inspeção
