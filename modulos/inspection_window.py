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
        schedule_camera_cleanup,
        release_all_cached_cameras,
        capture_image_from_camera,
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
        schedule_camera_cleanup,
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
        
        # Controle de webcam
        self.available_cameras = detect_cameras()
        self.selected_camera = 0
        
        self.setup_ui()
        self.update_button_states()
        
        # Inicia câmera em segundo plano após inicialização completa
        if self.available_cameras:
            self.after(500, lambda: self.start_background_camera_direct(self.available_cameras[0]))
            
    def start_background_camera_direct(self, camera_index):
        """Inicia a câmera diretamente em segundo plano com índice específico."""
        try:
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
            
            self.live_capture = True
            print(f"Webcam {camera_index} inicializada com sucesso em segundo plano")
            
            # Inicia captura de frames em thread separada
            self.start_background_frame_capture()
            
        except Exception as e:
            print(f"Erro ao inicializar webcam {camera_index}: {e}")
            self.camera = None
            self.live_capture = False
    
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
            
            # Usa a função cv2_to_tk para manter consistência com o resto do código
            self.img_display, self.scale_factor = cv2_to_tk(self.img_test, max_w=canvas_width, max_h=canvas_height)
            
            if self.img_display is None:
                return
            
            # Calcula dimensões da imagem redimensionada
            img_height, img_width = self.img_test.shape[:2]
            new_width = int(img_width * self.scale_factor)
            new_height = int(img_height * self.scale_factor)
            
            # Limpa o canvas e exibe a imagem centralizada
            self.canvas.delete("all")
            self.x_offset = max(0, (self.canvas.winfo_width() - new_width) // 2)
            self.y_offset = max(0, (self.canvas.winfo_height() - new_height) // 2)
            self.canvas.create_image(self.x_offset, self.y_offset, anchor=NW, image=self.img_display)
            
            # Atualiza o canvas
            self.canvas.update()
            
            # Aguarda um momento para que o usuário veja a imagem
            self.master.update()
            
        except Exception as e:
            print(f"Erro ao exibir imagem em tela cheia: {e}")
    
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
                if not hasattr(self, '_canvas_image_id'):
                    self._canvas_image_id = self.canvas.create_image(self.x_offset, self.y_offset, anchor=NW, image=self.img_display)
                else:
                    self.canvas.itemconfig(self._canvas_image_id, image=self.img_display)
                    self.canvas.coords(self._canvas_image_id, self.x_offset, self.y_offset)
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
        # Carrega as configurações de estilo
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
        # Carrega as configurações de estilo
        style_config = load_style_config()
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
        
        # Atualizar painel de resumo geral se existir
        if hasattr(self, 'status_label') and hasattr(self, 'score_label') and hasattr(self, 'id_label'):
            # Calcular status geral no estilo Keyence
            if total_slots > 0:
                total_score / total_slots
                overall_status = "OK" if passed_slots == total_slots else "NG"
                
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
                overall_status = "OK" if passed_slots == total_slots else "NG"
                
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
        
        for result in self.inspection_results:
            slot = result['slot_data']
            
            # Converte coordenadas da imagem para canvas (incluindo offsets)
            x1 = int(slot['x'] * self.scale_factor) + self.x_offset
            y1 = int(slot['y'] * self.scale_factor) + self.y_offset
            x2 = int((slot['x'] + slot['w']) * self.scale_factor) + self.x_offset
            y2 = int((slot['y'] + slot['h']) * self.scale_factor) + self.y_offset
            
            # Carrega as configurações de estilo
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


# Função create_main_window() removida - agora centralizada em montagem.py
# Esta função foi consolidada e melhorada no módulo principal montagem.py


