"""
Módulo contendo a classe SlotTrainingDialog para treinamento de slots com feedback OK/NG.
"""

import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import LEFT, RIGHT, BOTH, DISABLED, NORMAL, X
from tkinter import Canvas, filedialog, messagebox, Toplevel, StringVar
from PIL import Image, ImageTk

try:
    from utils import get_color, get_font
    from ml_classifier import MLSlotClassifier
    from camera_manager import capture_image_from_camera
    from paths import get_template_dir, get_model_template_dir
    from inspection import find_image_transform
    from image_utils import cv2_to_tk
except ImportError:
    from utils import get_color, get_font
    from ml_classifier import MLSlotClassifier
    from camera_manager import capture_image_from_camera
    from paths import get_template_dir, get_model_template_dir
    from inspection import find_image_transform
    from image_utils import cv2_to_tk


class SlotTrainingDialog(Toplevel):
    """Diálogo para treinamento de slots com feedback OK/NG."""
    
    def __init__(self, parent, slot_data, montagem_instance):
        super().__init__(parent)
        self.slot_data = slot_data
        self.montagem_instance = montagem_instance
        self.training_samples = []  # Lista de amostras de treinamento
        
        # Inicializa classificador ML
        self.ml_classifier = MLSlotClassifier(slot_id=str(slot_data['id']))
        self.use_ml = False  # Flag para usar ML ou método tradicional
        
        # Define o diretório para salvar as amostras (escopo por PROGRAMA/MODELO)
        template_path = self.slot_data.get('template_path')
        if template_path:
            # Se o slot já tem template, usa a pasta do template (que é específica do modelo)
            template_dir = os.path.dirname(template_path)
            self.samples_dir = os.path.join(template_dir, f"slot_{slot_data['id']}_samples")
        else:
            # Sem template ainda: usa a pasta de templates do MODELO atual para isolar por programa
            try:
                model_name = None
                model_id = None
                if hasattr(self.montagem_instance, 'current_model') and self.montagem_instance.current_model:
                    model = self.montagem_instance.current_model
                    model_name = model.get('nome') or model.get('name')
                if hasattr(self.montagem_instance, 'current_model_id') and self.montagem_instance.current_model_id:
                    model_id = self.montagem_instance.current_model_id
                if model_name and model_id is not None:
                    # Usa helpers do módulo para garantir caminho consistente por modelo
                    model_templates_dir = get_model_template_dir(model_name, model_id)
                    self.samples_dir = os.path.join(str(model_templates_dir), f"slot_{slot_data['id']}_samples")
                else:
                    # Fallback final: ainda isola por model_id se disponível, para evitar mistura
                    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    base = os.path.join(project_dir, "modelos", "_samples")
                    suffix = f"model_{model_id}" if model_id is not None else "model_unknown"
                    self.samples_dir = os.path.join(base, suffix, f"slot_{slot_data['id']}_samples")
                print(f"Diretório de amostras configurado: {self.samples_dir}")
            except Exception as e:
                print(f"Erro ao resolver diretório de amostras por modelo: {e}")
                # Recuo para um diretório local por segurança
                project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                base = os.path.join(project_dir, "modelos", "_samples", "model_unknown")
                self.samples_dir = os.path.join(base, f"slot_{slot_data['id']}_samples")
                print(f"AVISO: usando diretório de amostras de fallback: {self.samples_dir}")
            
        # Cria diretórios se não existirem
        try:
            os.makedirs(os.path.join(self.samples_dir, "ok"), exist_ok=True)
            os.makedirs(os.path.join(self.samples_dir, "ng"), exist_ok=True)
        except Exception as e:
            print(f"Erro ao criar diretórios de amostras: {e}")
            self.samples_dir = None
        
        self.title(f"Treinamento - Slot {slot_data['id']}")
        self.geometry("1200x800")  # Tamanho inicial maior
        self.resizable(True, True)
        self.minsize(1000, 700)  # Tamanho mínimo para evitar elementos sobrepostos
        
        # Variáveis
        self.current_image = None
        self.current_roi = None
        
        self.setup_ui()
        self.center_window()
        self.apply_modal_grab()
        
        # Configura protocolo de fechamento para limpeza de recursos
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def apply_modal_grab(self):
        """Aplica grab modal para manter foco na janela."""
        self.transient(self.master)
        self.grab_set()
        
    def center_window(self):
        """Centraliza a janela na tela."""
        self.update_idletasks()
        # Usa as dimensões definidas em geometry() se a janela ainda não foi renderizada
        width = max(self.winfo_width(), 1200)
        height = max(self.winfo_height(), 800)
        
        # Calcula posição central considerando a barra de tarefas
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2 - 30)  # Ajuste para barra de tarefas
        
        self.geometry(f"{width}x{height}+{x}+{y}")
    
    def on_closing(self):
        """Método chamado quando o diálogo é fechado - não manipula driver/câmera."""
        try:
            # Não libera nem reinicializa câmeras aqui para não interferir no driver
            # Limpa grab modal
            try:
                self.grab_release()
            except Exception:
                pass
            
            # Fecha o diálogo
            self.destroy()
        except Exception as e:
            print(f"Erro ao fechar diálogo de treinamento: {e}")
            # Força fechamento mesmo com erro
            try:
                self.destroy()
            except Exception:
                pass
        
    def setup_ui(self):
        """Configura a interface do diálogo de treinamento."""
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(main_frame, text=f"Treinamento do Slot {self.slot_data['id']}", 
                               font=get_font('header_font'))
        title_label.pack(pady=(0, 15))
        
        # Frame superior - controles
        controls_frame = ttk.LabelFrame(main_frame, text="Controles de Captura")
        controls_frame.pack(fill=X, pady=(0, 10))
        
        # Seleção do método de treinamento
        method_frame = ttk.Frame(controls_frame)
        method_frame.pack(fill=X, padx=10, pady=(10, 5))
        
        ttk.Label(method_frame, text="Método de Treinamento:", font=get_font('tiny_font')).pack(side=LEFT)
        
        self.training_method_var = StringVar(value="traditional")
        self.radio_traditional = ttk.Radiobutton(method_frame, text="Tradicional (Limiar)", 
                                               variable=self.training_method_var, value="traditional",
                                               command=self.on_method_change)
        self.radio_traditional.pack(side=LEFT, padx=(10, 5))
        
        self.radio_ml = ttk.Radiobutton(method_frame, text="Machine Learning (Scikit-learn)", 
                                      variable=self.training_method_var, value="ml",
                                      command=self.on_method_change)
        self.radio_ml.pack(side=LEFT, padx=(5, 0))
        
        # Botões de captura
        capture_frame = ttk.Frame(controls_frame)
        capture_frame.pack(fill=X, padx=10, pady=10)
        
        self.btn_capture_webcam = ttk.Button(capture_frame, text="Capturar da Câmera", 
                                           command=self.capture_from_webcam, width=20)
        self.btn_capture_webcam.pack(side=LEFT, padx=(0, 10))
        
        self.btn_load_image = ttk.Button(capture_frame, text="Carregar Imagem", 
                                       command=self.load_image_file, width=20)
        self.btn_load_image.pack(side=LEFT, padx=(0, 10))
        
        # Botão para limpar histórico
        self.btn_clear_history = ttk.Button(capture_frame, text="Limpar Histórico", 
                                          command=self.clear_training_history, width=20)
        self.btn_clear_history.pack(side=RIGHT)
        
        # Frame central dividido em duas colunas com proporções otimizadas
        central_frame = ttk.Frame(main_frame)
        central_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # Configura grid para melhor controle de layout
        central_frame.grid_columnconfigure(0, weight=2)  # Coluna esquerda maior
        central_frame.grid_columnconfigure(1, weight=1)  # Coluna direita menor
        central_frame.grid_rowconfigure(0, weight=1)
        
        # Coluna esquerda - visualização atual
        left_frame = ttk.LabelFrame(central_frame, text="Visualização Atual")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Frame para canvas com scrollbars
        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Canvas para exibir imagem atual com scrollbars
        self.canvas = Canvas(canvas_frame, bg=get_color('colors.canvas_colors.canvas_bg'))
        
        # Scrollbars para o canvas
        v_scrollbar_canvas = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        h_scrollbar_canvas = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar_canvas.set, xscrollcommand=h_scrollbar_canvas.set)
        
        # Pack dos elementos do canvas
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar_canvas.grid(row=0, column=1, sticky="ns")
        h_scrollbar_canvas.grid(row=1, column=0, sticky="ew")
        
        # Configura grid do canvas_frame
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
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
        right_frame = ttk.LabelFrame(central_frame, text="Histórico de Treinamento")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # Notebook para separar OK e NG
        self.history_notebook = ttk.Notebook(right_frame)
        self.history_notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Aba OK
        self.ok_frame = ttk.Frame(self.history_notebook)
        self.history_notebook.add(self.ok_frame, text="Amostras OK (0)")
        
        # Scrollable frame para amostras OK
        self.ok_canvas = Canvas(self.ok_frame, bg=get_color('colors.special_colors.ok_canvas_bg'))  # Cor específica para OK
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
        self.history_notebook.add(self.ng_frame, text="Amostras NG (0)")
        
        # Scrollable frame para amostras NG
        self.ng_canvas = Canvas(self.ng_frame, bg=get_color('colors.special_colors.ng_canvas_bg'))  # Cor específica para NG
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
        info_frame = ttk.LabelFrame(bottom_frame, text="Estatísticas")
        info_frame.pack(fill=X, pady=(0, 10))
        
        stats_frame = ttk.Frame(info_frame)
        stats_frame.pack(fill=X, padx=10, pady=10)
        
        self.info_label = ttk.Label(stats_frame, text="Amostras coletadas: 0 OK, 0 NG", 
                                   font=get_font('tiny_font'))
        self.info_label.pack(side=LEFT)
        
        self.threshold_label = ttk.Label(stats_frame, text="Limiar atual: N/A", 
                                        font=get_font('tiny_font'))
        self.threshold_label.pack(side=RIGHT)
        
        # Botões de ação
        action_frame = ttk.Frame(bottom_frame)
        action_frame.pack(fill=X)
        
        self.btn_apply_training = ttk.Button(action_frame, text="Aplicar Treinamento", 
                                           command=self.apply_training, state=DISABLED, width=20)
        self.btn_apply_training.pack(side=LEFT, padx=(0, 10))
        
        # Botão para treinar ML (inicialmente oculto)
        self.btn_train_ml = ttk.Button(action_frame, text="Treinar ML", 
                                     command=self.train_ml_model, state=DISABLED, width=15)
        self.btn_train_ml.pack(side=LEFT, padx=(0, 10))
        self.btn_train_ml.pack_forget()  # Oculta inicialmente
        
        # Botão para salvar modelo ML
        self.btn_save_ml = ttk.Button(action_frame, text="Salvar Modelo ML", 
                                    command=self.save_ml_model, state=DISABLED, width=18)
        self.btn_save_ml.pack(side=LEFT, padx=(0, 10))
        self.btn_save_ml.pack_forget()  # Oculta inicialmente
        
        self.btn_cancel = ttk.Button(action_frame, text="Cancelar", 
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
        """Captura imagem da webcam para treinamento usando frame em segundo plano quando disponível."""
        try:
            # Preferir o frame em segundo plano da janela de montagem para evitar mexer no driver
            if (hasattr(self.montagem_instance, 'live_capture') and 
                self.montagem_instance.live_capture and 
                hasattr(self.montagem_instance, 'latest_frame') and 
                self.montagem_instance.latest_frame is not None):
                captured_image = self.montagem_instance.latest_frame.copy()
                print("Usando frame de segundo plano da montagem para captura de treinamento")
            else:
                # Fallback: usa cache de câmera (não reinicia o driver)
                camera_index = 0
                if hasattr(self.montagem_instance, 'camera_combo') and self.montagem_instance.camera_combo.get():
                    camera_index = int(self.montagem_instance.camera_combo.get())
                print("Captura em segundo plano indisponível, usando cache da câmera para captura pontual")
                captured_image = capture_image_from_camera(camera_index, use_cache=True)
            
            if captured_image is not None:
                self.process_captured_image(captured_image)
                print(f"Imagem capturada para treinamento do slot {self.slot_data['id']}")
            else:
                messagebox.showerror("Erro", "Falha ao capturar imagem da webcam.")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao capturar da webcam: {str(e)}")
            print(f"Erro detalhado na captura para treinamento: {e}")
            
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
            
            # === AJUSTE AUTOMÁTICO AO CANVAS ===
            try:
                # Força atualização do canvas
                self.canvas.update_idletasks()
                
                # Obtém o tamanho atual do canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                # Se o canvas ainda não foi renderizado, usa valores baseados na janela
                if canvas_width <= 1 or canvas_height <= 1:
                    # Calcula baseado no tamanho da janela
                    window_width = self.winfo_width()
                    window_height = self.winfo_height()
                    
                    # Estima o espaço disponível para o canvas (60% da largura, 50% da altura)
                    canvas_width = max(int(window_width * 0.6), 800)
                    canvas_height = max(int(window_height * 0.5), 400)
                    
            except Exception as canvas_error:
                print(f"Erro ao obter dimensões do canvas: {canvas_error}")
                canvas_width = 800
                canvas_height = 400
            
            # Converte para exibição no canvas
            tk_image, _ = cv2_to_tk(display_image, max_w=canvas_width, max_h=canvas_height)
            
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
                bg_color = get_color('colors.special_colors.ok_result_bg')
            else:
                parent_frame = self.ng_scrollable_frame
                bg_color = get_color('colors.special_colors.ng_result_bg')
            
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
        
        self.history_notebook.tab(0, text=f"Amostras OK ({ok_count})")
        self.history_notebook.tab(1, text=f"Amostras NG ({ng_count})")
    
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
            # Verifica se o diretório de amostras foi definido
            if not self.samples_dir:
                print("Diretório de amostras não definido. Pulando carregamento de amostras existentes.")
                return
                
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
            # Habilita botões ML se método ML estiver selecionado
            if self.use_ml:
                self.btn_train_ml.config(state=NORMAL)
    
    def on_method_change(self):
        """Callback quando o método de treinamento é alterado."""
        self.use_ml = self.training_method_var.get() == "ml"
        
        if self.use_ml:
            # Mostra botões ML
            self.btn_train_ml.pack(side=LEFT, padx=(0, 10), before=self.btn_cancel)
            self.btn_save_ml.pack(side=LEFT, padx=(0, 10), before=self.btn_cancel)
            # Atualiza texto do botão principal
            self.btn_apply_training.config(text="🚀 Aplicar Treinamento (Tradicional)")
        else:
            # Oculta botões ML
            self.btn_train_ml.pack_forget()
            self.btn_save_ml.pack_forget()
            # Restaura texto do botão principal
            self.btn_apply_training.config(text="🚀 Aplicar Treinamento")
        
        # Atualiza estado dos botões
        self.update_info_label()
    
    def train_ml_model(self):
        """Treina o modelo de machine learning com as amostras coletadas."""
        try:
            if len(self.training_samples) < 4:
                messagebox.showwarning("Aviso", "São necessárias pelo menos 4 amostras (2 OK + 2 NG) para treinamento de ML.")
                return
            elif len(self.training_samples) < 10:
                messagebox.showinfo("Informação", f"Treinando com {len(self.training_samples)} amostras.\nPara melhor precisão, recomenda-se 10+ amostras.")
            
            # Verifica se há amostras OK e NG
            ok_samples = [s for s in self.training_samples if s['label'] == 'OK']
            ng_samples = [s for s in self.training_samples if s['label'] == 'NG']
            
            if not ok_samples or not ng_samples:
                messagebox.showwarning("Aviso", "É necessário ter amostras tanto OK quanto NG para treinamento de ML.")
                return
            
            # Mostra progresso
            progress_window = Toplevel(self)
            progress_window.title("Treinando Modelo ML")
            progress_window.geometry("400x150")
            progress_window.transient(self)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="Treinando modelo de machine learning...")
            progress_label.pack(pady=20)
            
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=10, padx=20, fill=X)
            progress_bar.start()
            
            # Força atualização da interface
            progress_window.update()
            
            # Treina o modelo
            metrics = self.ml_classifier.train(self.training_samples)
            
            # Para a barra de progresso
            progress_bar.stop()
            progress_window.destroy()
            
            # Mostra resultados
            accuracy = metrics.get('accuracy', 0)
            cv_mean = metrics.get('cv_mean', 0)
            cv_std = metrics.get('cv_std', 0)
            n_samples = metrics.get('n_samples', 0)
            
            result_msg = (
                f"🤖 Modelo ML treinado com sucesso!\n\n"
                f"📊 Métricas de Performance:\n"
                f"• Acurácia: {accuracy:.1%}\n"
                f"• Validação Cruzada: {cv_mean:.1%} (±{cv_std:.1%})\n"
                f"• Amostras utilizadas: {n_samples}\n"
                f"• Amostras OK: {metrics.get('n_ok', 0)}\n"
                f"• Amostras NG: {metrics.get('n_ng', 0)}\n\n"
                f"O modelo está pronto para uso!"
            )
            
            messagebox.showinfo("Sucesso", result_msg)
            
            # Habilita botão de salvar
            self.btn_save_ml.config(state=NORMAL)
            
            # Atualiza slot com flag ML
            self.slot_data['use_ml'] = True
            self.slot_data['ml_trained'] = True
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar modelo ML: {str(e)}")
    
    def save_ml_model(self):
        """Salva o modelo ML treinado."""
        try:
            if not self.ml_classifier.is_trained:
                messagebox.showwarning("Aviso", "Nenhum modelo ML foi treinado ainda.")
                return
            
            # Define caminho para salvar o modelo
            template_path = self.slot_data.get('template_path')
            if template_path:
                model_dir = os.path.dirname(template_path)
            else:
                model_dir = get_template_dir()
            
            model_filename = f"ml_model_slot_{self.slot_data['id']}.joblib"
            model_path = os.path.join(model_dir, model_filename)
            
            # Salva o modelo
            if self.ml_classifier.save_model(model_path):
                # Atualiza dados do slot
                self.slot_data['ml_model_path'] = model_path
                self.slot_data['use_ml'] = True
                
                # Salva no banco de dados se possível
                try:
                    if hasattr(self.montagem_instance, 'db_manager') and self.montagem_instance.db_manager:
                        # Atualiza slot no banco (modelo_id primeiro, depois os dados do slot)
                        self.montagem_instance.db_manager.update_slot(
                            self.montagem_instance.current_model_id,
                            self.slot_data
                        )
                except Exception as db_error:
                    print(f"Aviso: Não foi possível salvar no banco de dados: {db_error}")
                
                messagebox.showinfo("Sucesso", f"Modelo ML salvo com sucesso!\n\nCaminho: {model_path}")
            else:
                messagebox.showerror("Erro", "Falha ao salvar o modelo ML.")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar modelo ML: {str(e)}")
            
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
            
            if self.use_ml:
                # Modo Machine Learning
                if not self.ml_classifier.is_trained:
                    messagebox.showwarning("Aviso", "Modelo ML não foi treinado ainda. Treine o modelo primeiro.")
                    return
                
                # Testa o modelo com as amostras atuais
                correct_predictions = 0
                total_predictions = 0
                
                for sample in self.training_samples:
                    prediction = self.ml_classifier.predict(sample['roi'])
                    expected = 1 if sample['label'] == 'OK' else 0
                    if prediction == expected:
                        correct_predictions += 1
                    total_predictions += 1
                
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                # Atualiza configurações do slot para usar ML
                self.slot_data['use_ml'] = True
                self.slot_data['ml_trained'] = True
                self.slot_data['ml_accuracy'] = accuracy
                
                # Salva template melhorado se há amostras OK
                if ok_samples:
                    self.update_template_with_best_sample(ok_samples)
                
                # Atualiza o slot na instância principal
                self.montagem_instance.update_slot_data(self.slot_data)
                
                # Marca modelo como modificado
                self.montagem_instance.mark_model_modified()
                
                result_msg = (
                    f"🤖 Treinamento ML aplicado!\n\n"
                    f"📊 Acurácia nas amostras: {accuracy:.1%}\n"
                    f"✅ Amostras utilizadas: {len(self.training_samples)}\n\n"
                    f"O slot agora usará Machine Learning para classificação!"
                )
                
                messagebox.showinfo("Sucesso", result_msg)
                self.destroy()
                
            else:
                # Modo tradicional (threshold)
                new_threshold = self.calculate_optimal_threshold(ok_samples, ng_samples)
                
                if new_threshold is not None:
                    # Atualiza o slot com o novo limiar
                    old_threshold = self.slot_data.get('correlation_threshold', self.slot_data.get('detection_threshold', 0.8))
                    self.slot_data['correlation_threshold'] = new_threshold
                    self.slot_data['use_ml'] = False
                    
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
            # Validações iniciais
            if not ok_samples:
                print("Erro: Nenhuma amostra OK fornecida")
                return None
                
            # Carrega template atual ou cria um temporário
            template_path = self.slot_data.get('template_path')
            template = None
            
            if template_path and Path(template_path).exists():
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                print(f"Template carregado de: {template_path}")
            
            # Se não há template ou falhou ao carregar, usa a primeira amostra OK como template
            if template is None:
                if not ok_samples:
                    print("Erro: Não há template nem amostras OK para usar como referência")
                    return None
                    
                print("Template não encontrado. Usando primeira amostra OK como referência.")
                first_sample = ok_samples[0]
                template = cv2.cvtColor(first_sample, cv2.COLOR_BGR2GRAY)
                
                # Salva template temporário se possível
                if template_path:
                    try:
                        # Cria diretório se não existir
                        Path(template_path).parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(template_path, template)
                        print(f"Template temporário salvo em: {template_path}")
                    except Exception as e:
                        print(f"Aviso: Não foi possível salvar template temporário: {e}")
                
            # Calcula correlações para amostras OK
            ok_correlations = []
            for i, roi in enumerate(ok_samples):
                try:
                    if roi is None or roi.size == 0:
                        print(f"Aviso: Amostra OK {i} é inválida")
                        continue
                        
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    if roi_gray.size == 0:
                        print(f"Aviso: Amostra OK {i} resultou em imagem vazia")
                        continue
                    
                    # Redimensiona template se necessário
                    if roi_gray.shape != template.shape:
                        template_resized = cv2.resize(template, (roi_gray.shape[1], roi_gray.shape[0]))
                    else:
                        template_resized = template
                        
                    # Template matching
                    result = cv2.matchTemplate(roi_gray, template_resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    # Valida o resultado
                    if not np.isnan(max_val) and not np.isinf(max_val):
                        ok_correlations.append(max_val)
                    else:
                        print(f"Aviso: Correlação inválida para amostra OK {i}: {max_val}")
                        
                except Exception as e:
                    print(f"Erro ao processar amostra OK {i}: {e}")
                    continue
                
            # Calcula correlações para amostras NG (se existirem)
            ng_correlations = []
            for i, roi in enumerate(ng_samples):
                try:
                    if roi is None or roi.size == 0:
                        print(f"Aviso: Amostra NG {i} é inválida")
                        continue
                        
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    if roi_gray.size == 0:
                        print(f"Aviso: Amostra NG {i} resultou em imagem vazia")
                        continue
                    
                    if roi_gray.shape != template.shape:
                        template_resized = cv2.resize(template, (roi_gray.shape[1], roi_gray.shape[0]))
                    else:
                        template_resized = template
                        
                    result = cv2.matchTemplate(roi_gray, template_resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    # Valida o resultado
                    if not np.isnan(max_val) and not np.isinf(max_val):
                        ng_correlations.append(max_val)
                    else:
                        print(f"Aviso: Correlação inválida para amostra NG {i}: {max_val}")
                        
                except Exception as e:
                    print(f"Erro ao processar amostra NG {i}: {e}")
                    continue
                
            # Verifica se temos correlações válidas
            if not ok_correlations:
                print("Erro: Nenhuma correlação válida foi calculada para amostras OK")
                return None
                
            print(f"Correlações OK calculadas: {len(ok_correlations)} amostras")
            print(f"Correlações NG calculadas: {len(ng_correlations)} amostras")
            
            # Calcula limiar ótimo
            min_ok = min(ok_correlations)
            max_ok = max(ok_correlations)
            avg_ok = sum(ok_correlations) / len(ok_correlations)
            
            print(f"Estatísticas OK - Min: {min_ok:.3f}, Max: {max_ok:.3f}, Média: {avg_ok:.3f}")
            
            if ng_correlations:
                min_ng = min(ng_correlations)
                max_ng = max(ng_correlations)
                avg_ng = sum(ng_correlations) / len(ng_correlations)
                
                print(f"Estatísticas NG - Min: {min_ng:.3f}, Max: {max_ng:.3f}, Média: {avg_ng:.3f}")
                
                # Verifica se há separação clara entre OK e NG
                if min_ok > max_ng:
                    # Caso ideal: há separação clara
                    new_threshold = (min_ok + max_ng) / 2
                    print(f"Separação clara detectada. Limiar calculado: {new_threshold:.3f}")
                else:
                    # Caso problemático: sobreposição entre OK e NG
                    # Usa a média das amostras OK menos uma margem de segurança
                    new_threshold = avg_ok * 0.85
                    print(f"Sobreposição detectada. Usando limiar conservador: {new_threshold:.3f}")
                
                # Garante que está dentro de limites razoáveis
                new_threshold = max(0.3, min(0.95, new_threshold))
            else:
                # Se não há amostras NG, usa um valor conservador baseado na média OK
                new_threshold = max(0.5, min(0.9, avg_ok * 0.9))
                print(f"Apenas amostras OK. Limiar conservador: {new_threshold:.3f}")
                
            # Validação final
            if np.isnan(new_threshold) or np.isinf(new_threshold):
                print(f"Erro: Limiar calculado é inválido: {new_threshold}")
                return None
                
            print(f"Limiar final calculado: {new_threshold:.3f}")
            return new_threshold
                
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

