import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import LEFT, BOTH, DISABLED, NORMAL, X
from tkinter import Canvas, filedialog, messagebox, Toplevel, StringVar, colorchooser
from PIL import Image, ImageTk

try:
    from modulos.utils import load_style_config, get_color, get_font, save_style_config, apply_style_config
    from modulos.ml_classifier import MLSlotClassifier
    from modulos.camera_manager import capture_image_from_camera
    from modulos.paths import get_template_dir, get_model_template_dir, get_project_root
    from modulos.inspection import find_image_transform
    from modulos.image_utils import cv2_to_tk
except Exception:
    try:
        from utils import load_style_config, get_color, get_font, save_style_config, apply_style_config
        from ml_classifier import MLSlotClassifier
        from camera_manager import capture_image_from_camera
        from paths import get_template_dir, get_model_template_dir, get_project_root
        from inspection import find_image_transform
        from image_utils import cv2_to_tk
    except Exception:
        # Fallback for when running as standalone
        pass


class EditSlotDialog(Toplevel):
    def __init__(self, parent, slot_data, malha_frame_instance):
        super().__init__(parent)
        if not parent or not slot_data or not malha_frame_instance:
            raise ValueError("Parâmetros inválidos para EditSlotDialog")

        basic_keys = ['id', 'x', 'y', 'w', 'h', 'tipo']
        required_keys = basic_keys.copy()
        if slot_data.get('tipo') == 'clip':
            required_keys.extend(['cor', 'detection_threshold'])
        missing_keys = [key for key in required_keys if key not in slot_data]
        if missing_keys:
            raise ValueError(f"Dados do slot incompletos. Chaves ausentes: {missing_keys}")

        self.slot_data = slot_data.copy()
        self.malha_frame = malha_frame_instance
        self.result = None
        self._is_destroyed = False

        if 'style_config' not in self.slot_data:
            current_style_config = load_style_config()
            self.slot_data['style_config'] = {
                'bg_color': get_color('colors.canvas_colors.canvas_bg', current_style_config),
                'text_color': get_color('colors.text_color', current_style_config),
                'ok_color': get_color('colors.ok_color', current_style_config),
                'ng_color': get_color('colors.ng_color', current_style_config),
                'selection_color': get_color('colors.selection_color', current_style_config),
                'ok_font': 'Arial 12 bold',
                'ng_font': 'Arial 12 bold'
            }

        self.title(f"Editando Slot {slot_data['id']}")
        self.geometry("400x650")
        self.resizable(False, False)

        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.setup_ui()
        self.load_slot_data()
        self.center_window()
        self.apply_modal_grab()

    def apply_modal_grab(self):
        try:
            self.focus_set()
        except Exception:
            pass

    def center_window(self):
        try:
            self.update_idletasks()
            width = 500
            height = 400
            x = (self.winfo_screenwidth() // 2) - (width // 2)
            y = (self.winfo_screenheight() // 2) - (height // 2)
            self.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            pass

    def setup_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        info_frame = ttk.LabelFrame(main_frame, text="Informações do Slot")
        info_frame.pack(fill=X, pady=(0, 10))
        slot_info = f"ID: {self.slot_data['id']} | Tipo: {self.slot_data['tipo']}"
        ttk.Label(info_frame, text=slot_info).pack(anchor="w", padx=5, pady=5)

        mesh_frame = ttk.LabelFrame(main_frame, text="Posição e Dimensões")
        mesh_frame.pack(fill=X, pady=(0, 10))

        self.x_var = StringVar()
        self.y_var = StringVar()
        self.w_var = StringVar()
        self.h_var = StringVar()
        self.correlation_threshold_var = StringVar()
        # Novo: controle de alinhamento por slot (autoajuste vs fixo)
        self.use_alignment_var = StringVar(value="1")

        mesh_grid = ttk.Frame(mesh_frame)
        mesh_grid.pack(fill=X, padx=10, pady=10)
        mesh_grid.columnconfigure(0, weight=1)
        mesh_grid.columnconfigure(1, weight=1)

        ttk.Label(mesh_grid, text="Posição X:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(mesh_grid, textvariable=self.x_var, width=8).grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ttk.Label(mesh_grid, text="Posição Y:").grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Entry(mesh_grid, textvariable=self.y_var, width=8).grid(row=0, column=1, sticky="e", padx=5, pady=5)

        ttk.Label(mesh_grid, text="Largura:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(mesh_grid, textvariable=self.w_var, width=8).grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ttk.Label(mesh_grid, text="Altura:").grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Entry(mesh_grid, textvariable=self.h_var, width=8).grid(row=1, column=1, sticky="e", padx=5, pady=5)

        config_frame = ttk.LabelFrame(main_frame, text="Configurações")
        config_frame.pack(fill=X, pady=(0, 10))
        threshold_frame = ttk.Frame(config_frame)
        threshold_frame.pack(fill=X, padx=5, pady=5)
        ttk.Label(threshold_frame, text="Limiar de Correlação (0.0–1.0):").pack(side=LEFT)
        ttk.Entry(threshold_frame, textvariable=self.correlation_threshold_var, width=10).pack(side=LEFT, padx=(5, 0))

        # Toggle: alinhamento por slot
        alignment_frame = ttk.Frame(config_frame)
        alignment_frame.pack(fill=X, padx=5, pady=5)
        ttk.Label(alignment_frame, text="Alinhamento por Slot:").pack(side=LEFT)
        self.alignment_combo = ttk.Combobox(
            alignment_frame,
            values=["Autoajuste (alinha)", "Fixo (sem alinhar)"],
            state="readonly",
            width=22
        )
        self.alignment_combo.pack(side=LEFT, padx=(5, 0))
        def _sync_alignment_from_combo(*args):
            self.use_alignment_var.set("1" if self.alignment_combo.get().startswith("Auto") else "0")
        self.alignment_combo.bind("<<ComboboxSelected>>", _sync_alignment_from_combo)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=X, pady=(10, 0))
        ttk.Button(button_frame, text="Salvar", command=self.save_changes).pack(side=LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancelar", command=self.cancel).pack(side=LEFT)

    def load_slot_data(self):
        if self.slot_data:
            self.x_var.set(str(self.slot_data.get('x', 0)))
            self.y_var.set(str(self.slot_data.get('y', 0)))
            self.w_var.set(str(self.slot_data.get('w', 100)))
            self.h_var.set(str(self.slot_data.get('h', 100)))

            # Carrega o limiar de correlação direto em 0–1
            corr_thr = self.slot_data.get('correlation_threshold', self.slot_data.get('detection_threshold', 0.5))
            try:
                self.correlation_threshold_var.set(str(float(corr_thr)))
            except Exception:
                self.correlation_threshold_var.set(str(corr_thr))
            # Sincroniza alinhamento
            try:
                use_alignment = bool(self.slot_data.get('use_alignment', True))
                self.use_alignment_var.set("1" if use_alignment else "0")
                if use_alignment:
                    self.alignment_combo.set("Autoajuste (alinha)")
                else:
                    self.alignment_combo.set("Fixo (sem alinhar)")
            except Exception:
                pass

    def save_changes(self):
        try:
            x_val = int(self.x_var.get().strip())
            y_val = int(self.y_var.get().strip())
            w_val = int(self.w_var.get().strip())
            h_val = int(self.h_var.get().strip())
            corr_val = float(self.correlation_threshold_var.get().strip())
            if w_val <= 0 or h_val <= 0:
                raise ValueError("Largura e altura devem ser maiores que zero")
            if corr_val < 0.0 or corr_val > 1.0:
                raise ValueError("Limiar de correlação deve estar entre 0.0 e 1.0")

            self.slot_data.update({
                'x': x_val,
                'y': y_val,
                'w': w_val,
                'h': h_val,
                'correlation_threshold': corr_val,
                'use_alignment': (self.use_alignment_var.get() not in ("0", "False", "false", "no", "nao"))
            })
            self.malha_frame.update_slot_data(self.slot_data)
            self.destroy()
        except ValueError as ve:
            messagebox.showerror("Erro de Validação", f"Valores inválidos: {str(ve)}")

    def cancel(self):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


class SlotTrainingDialog(Toplevel):
    def __init__(self, parent, slot_data, montagem_instance):
        super().__init__(parent)
        self.slot_data = slot_data
        self.montagem_instance = montagem_instance
        self.training_samples = []
        self.ml_classifier = MLSlotClassifier(slot_id=str(slot_data['id']))
        self.use_ml = False

        template_path = self.slot_data.get('template_path')
        if template_path:
            template_dir = os.path.dirname(template_path)
            self.samples_dir = os.path.join(template_dir, f"slot_{slot_data['id']}_samples")
        else:
            model_name = None
            model_id = None
            if hasattr(self.montagem_instance, 'current_model') and self.montagem_instance.current_model:
                model = self.montagem_instance.current_model
                model_name = model.get('nome') or model.get('name')
            if hasattr(self.montagem_instance, 'current_model_id') and self.montagem_instance.current_model_id:
                model_id = self.montagem_instance.current_model_id
            if model_name and model_id is not None:
                model_templates_dir = get_model_template_dir(model_name, model_id)
                self.samples_dir = os.path.join(str(model_templates_dir), f"slot_{slot_data['id']}_samples")
            else:
                project_dir = Path(__file__).parent.parent
                base = project_dir / "modelos" / "_samples"
                suffix = f"model_{model_id}" if model_id is not None else "model_unknown"
                self.samples_dir = str(base / suffix / f"slot_{slot_data['id']}_samples")

        os.makedirs(os.path.join(self.samples_dir, "ok"), exist_ok=True)
        os.makedirs(os.path.join(self.samples_dir, "ng"), exist_ok=True)

        self.title(f"Treinamento - Slot {slot_data['id']}")
        self.geometry("1200x800")
        self.resizable(True, True)
        self.minsize(1000, 700)

        self.current_image = None
        self.current_roi = None

        self.setup_ui()
        self.center_window()
        self.apply_modal_grab()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def apply_modal_grab(self):
        self.transient(self.master)
        self.grab_set()

    def center_window(self):
        self.update_idletasks()
        width = max(self.winfo_width(), 1200)
        height = max(self.winfo_height(), 800)
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2 - 30)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def on_closing(self):
        try:
            try:
                self.grab_release()
            except Exception:
                pass
            self.destroy()
        except Exception:
            try:
                self.destroy()
            except Exception:
                pass

    def setup_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        title_label = ttk.Label(main_frame, text=f"🎯 Treinamento do Slot {self.slot_data['id']}", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 15))

        controls_frame = ttk.LabelFrame(main_frame, text="📷 Controles de Captura")
        controls_frame.pack(fill=X, pady=(0, 10))

        method_frame = ttk.Frame(controls_frame)
        method_frame.pack(fill=X, padx=10, pady=(10, 5))
        ttk.Label(method_frame, text="🤖 Método de Treinamento:", font=("Arial", 10, "bold")).pack(side=LEFT)
        self.training_method_var = StringVar(value="traditional")
        ttk.Radiobutton(method_frame, text="Tradicional (Threshold)", variable=self.training_method_var, value="traditional", command=self.on_method_change).pack(side=LEFT, padx=(10, 5))
        ttk.Radiobutton(method_frame, text="Machine Learning (Scikit-Learn)", variable=self.training_method_var, value="ml", command=self.on_method_change).pack(side=LEFT, padx=(5, 0))

        capture_frame = ttk.Frame(controls_frame)
        capture_frame.pack(fill=X, padx=10, pady=10)
        ttk.Button(capture_frame, text="📷 Capturar da Webcam", command=self.capture_from_webcam, width=20).pack(side=LEFT, padx=(0, 10))
        ttk.Button(capture_frame, text="📁 Carregar Imagem", command=self.load_image_file, width=20).pack(side=LEFT, padx=(0, 10))
        self.btn_clear_history = ttk.Button(capture_frame, text="🗑️ Limpar Histórico", command=self.clear_training_history, width=20)
        self.btn_clear_history.pack(side='right')

        central_frame = ttk.Frame(main_frame)
        central_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        central_frame.grid_columnconfigure(0, weight=2)
        central_frame.grid_columnconfigure(1, weight=1)
        central_frame.grid_rowconfigure(0, weight=1)

        left_frame = ttk.LabelFrame(central_frame, text="🖼️ Visualização Atual")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        self.canvas = Canvas(canvas_frame, bg=get_color('colors.canvas_colors.canvas_bg'))
        v_scrollbar_canvas = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        h_scrollbar_canvas = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar_canvas.set, xscrollcommand=h_scrollbar_canvas.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar_canvas.grid(row=0, column=1, sticky="ns")
        h_scrollbar_canvas.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        feedback_buttons_frame = ttk.Frame(left_frame)
        feedback_buttons_frame.pack(fill=X, padx=10, pady=(0, 10))
        self.btn_mark_ok = ttk.Button(feedback_buttons_frame, text="✅ Marcar como OK", command=self.mark_as_ok, state=DISABLED, width=15)
        self.btn_mark_ok.pack(side=LEFT, padx=(0, 10))
        self.btn_mark_ng = ttk.Button(feedback_buttons_frame, text="❌ Marcar como NG", command=self.mark_as_ng, state=DISABLED, width=15)
        self.btn_mark_ng.pack(side=LEFT)

        right_frame = ttk.LabelFrame(central_frame, text="📊 Histórico de Treinamento")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.history_notebook = ttk.Notebook(right_frame)
        self.history_notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)
        self.ok_frame = ttk.Frame(self.history_notebook)
        self.history_notebook.add(self.ok_frame, text="✅ Amostras OK (0)")
        self.ok_canvas = Canvas(self.ok_frame, bg=get_color('colors.special_colors.ok_canvas_bg'))
        self.ok_scrollbar = ttk.Scrollbar(self.ok_frame, orient="vertical", command=self.ok_canvas.yview)
        self.ok_scrollable_frame = ttk.Frame(self.ok_canvas)
        self.ok_scrollable_frame.bind("<Configure>", lambda e: self.ok_canvas.configure(scrollregion=self.ok_canvas.bbox("all")))
        self.ok_canvas.create_window((0, 0), window=self.ok_scrollable_frame, anchor="nw")
        self.ok_canvas.configure(yscrollcommand=self.ok_scrollbar.set)
        self.ok_canvas.pack(side="left", fill="both", expand=True)
        self.ok_scrollbar.pack(side="right", fill="y")
        self.ok_canvas.bind("<MouseWheel>", lambda e: self.ok_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self.ng_frame = ttk.Frame(self.history_notebook)
        self.history_notebook.add(self.ng_frame, text="❌ Amostras NG (0)")
        self.ng_canvas = Canvas(self.ng_frame, bg=get_color('colors.special_colors.ng_canvas_bg'))
        self.ng_scrollbar = ttk.Scrollbar(self.ng_frame, orient="vertical", command=self.ng_canvas.yview)
        self.ng_scrollable_frame = ttk.Frame(self.ng_canvas)
        self.ng_scrollable_frame.bind("<Configure>", lambda e: self.ng_canvas.configure(scrollregion=self.ng_canvas.bbox("all")))
        self.ng_canvas.create_window((0, 0), window=self.ng_scrollable_frame, anchor="nw")
        self.ng_canvas.configure(yscrollcommand=self.ng_scrollbar.set)
        self.ng_canvas.pack(side="left", fill="both", expand=True)
        self.ng_scrollbar.pack(side="right", fill="y")
        self.ng_canvas.bind("<MouseWheel>", lambda e: self.ng_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=X, pady=(10, 0))
        info_frame = ttk.LabelFrame(bottom_frame, text="📈 Estatísticas")
        info_frame.pack(fill=X, pady=(0, 10))
        stats_frame = ttk.Frame(info_frame)
        stats_frame.pack(fill=X, padx=10, pady=10)
        self.info_label = ttk.Label(stats_frame, text="Amostras coletadas: 0 OK, 0 NG", font=("Arial", 10, "bold"))
        self.info_label.pack(side=LEFT)
        self.threshold_label = ttk.Label(stats_frame, text="Threshold atual: N/A", font=("Arial", 10))
        self.threshold_label.pack(side='right')

        action_frame = ttk.Frame(bottom_frame)
        action_frame.pack(fill=X)
        self.btn_apply_training = ttk.Button(action_frame, text="🚀 Aplicar Treinamento", command=self.apply_training, state=DISABLED, width=20)
        self.btn_apply_training.pack(side=LEFT, padx=(0, 10))
        self.btn_train_ml = ttk.Button(action_frame, text="🤖 Treinar ML", command=self.train_ml_model, state=DISABLED, width=15)
        self.btn_train_ml.pack_forget()
        self.btn_save_ml = ttk.Button(action_frame, text="💾 Salvar Modelo ML", command=self.save_ml_model, state=DISABLED, width=18)
        self.btn_save_ml.pack_forget()
        self.btn_cancel = ttk.Button(action_frame, text="❌ Cancelar", command=self.cancel, width=15)
        self.btn_cancel.pack(side='right')

        current_threshold = self.slot_data.get('correlation_threshold', self.slot_data.get('detection_threshold', 'N/A'))
        if current_threshold != 'N/A':
            self.threshold_label.config(text=f"Threshold atual: {current_threshold:.3f}")
        self.load_existing_samples()

    def capture_from_webcam(self):
        try:
            if (hasattr(self.montagem_instance, 'live_capture') and self.montagem_instance.live_capture and hasattr(self.montagem_instance, 'latest_frame') and self.montagem_instance.latest_frame is not None):
                captured_image = self.montagem_instance.latest_frame.copy()
            else:
                camera_index = 0
                if hasattr(self.montagem_instance, 'camera_combo') and self.montagem_instance.camera_combo.get():
                    camera_index = int(self.montagem_instance.camera_combo.get())
                captured_image = capture_image_from_camera(camera_index, use_cache=True)
            if captured_image is not None:
                self.process_captured_image(captured_image)
            else:
                messagebox.showerror("Erro", "Falha ao capturar imagem da webcam.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao capturar da webcam: {str(e)}")

    def load_image_file(self):
        file_path = filedialog.askopenfilename(title="Selecionar Imagem", filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.process_captured_image(image)
            else:
                messagebox.showerror("Erro", "Falha ao carregar a imagem.")

    def process_captured_image(self, image):
        try:
            self.current_image = image.copy()
            if not hasattr(self.montagem_instance, 'img_original') or self.montagem_instance.img_original is None:
                messagebox.showerror("Erro", "Imagem de referência não carregada.")
                return
            M, inliers_count, error_msg = find_image_transform(self.montagem_instance.img_original, image)
            # Respeita a configuração de alinhamento do slot (autoajuste vs fixo)
            try:
                if not bool(self.slot_data.get('use_alignment', True)):
                    M = None
            except Exception:
                pass
            if M is None:
                x, y, w, h = self.slot_data['x'], self.slot_data['y'], self.slot_data['w'], self.slot_data['h']
            else:
                original_corners = np.array([[
                    [self.slot_data['x'], self.slot_data['y']],
                    [self.slot_data['x'] + self.slot_data['w'], self.slot_data['y']],
                    [self.slot_data['x'] + self.slot_data['w'], self.slot_data['y'] + self.slot_data['h']],
                    [self.slot_data['x'], self.slot_data['y'] + self.slot_data['h']]
                ]], dtype=np.float32)
                transformed_corners = cv2.perspectiveTransform(original_corners, M)[0]
                x = int(min(corner[0] for corner in transformed_corners))
                y = int(min(corner[1] for corner in transformed_corners))
                w = int(max(corner[0] for corner in transformed_corners) - x)
                h = int(max(corner[1] for corner in transformed_corners) - y)
            x = max(0, x); y = max(0, y)
            w = min(w, image.shape[1] - x); h = min(h, image.shape[0] - y)
            if w <= 0 or h <= 0:
                messagebox.showerror("Erro", "ROI inválida detectada.")
                return
            self.current_roi = image[y:y+h, x:x+w].copy()
            self.display_image_with_roi(image, x, y, w, h)
            self.btn_mark_ok.config(state=NORMAL)
            self.btn_mark_ng.config(state=NORMAL)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar imagem: {str(e)}")

    def display_image_with_roi(self, image, roi_x, roi_y, roi_w, roi_h):
        display_image = image.copy()
        cv2.rectangle(display_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 3)
        try:
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width <= 1 or canvas_height <= 1:
                window_width = self.winfo_width()
                window_height = self.winfo_height()
                canvas_width = max(int(window_width * 0.6), 800)
                canvas_height = max(int(window_height * 0.5), 400)
        except Exception:
            canvas_width = 800
            canvas_height = 400
        tk_image, _ = cv2_to_tk(display_image, max_w=canvas_width, max_h=canvas_height)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, image=tk_image, anchor="center")
        self.canvas.image = tk_image

    def mark_as_ok(self):
        if self.current_roi is not None:
            timestamp = datetime.now()
            self.training_samples.append({'roi': self.current_roi.copy(), 'label': 'OK', 'timestamp': timestamp})
            if self.samples_dir:
                ok_dir = os.path.join(self.samples_dir, "ok"); os.makedirs(ok_dir, exist_ok=True)
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(ok_dir, f"ok_sample_{timestamp_str}.png"), self.current_roi.copy())
            self.add_sample_to_history(self.current_roi.copy(), "OK", timestamp)
            self.update_info_label(); self.update_tab_titles(); self.reset_capture_state()
            messagebox.showinfo("Sucesso", "Amostra marcada como OK!")

    def mark_as_ng(self):
        if self.current_roi is not None:
            timestamp = datetime.now()
            self.training_samples.append({'roi': self.current_roi.copy(), 'label': 'NG', 'timestamp': timestamp})
            if self.samples_dir:
                ng_dir = os.path.join(self.samples_dir, "ng"); os.makedirs(ng_dir, exist_ok=True)
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(ng_dir, f"ng_sample_{timestamp_str}.png"), self.current_roi.copy())
            self.add_sample_to_history(self.current_roi.copy(), "NG", timestamp)
            self.update_info_label(); self.update_tab_titles(); self.reset_capture_state()
            messagebox.showinfo("Sucesso", "Amostra marcada como NG!")

    def reset_capture_state(self):
        self.current_image = None
        self.current_roi = None
        self.btn_mark_ok.config(state=DISABLED)
        self.btn_mark_ng.config(state=DISABLED)
        self.canvas.delete("all")

    def add_sample_to_history(self, roi_image, label, timestamp):
        thumbnail_size = (100, 100)
        roi_resized = cv2.resize(roi_image, thumbnail_size)
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)
        roi_tk = ImageTk.PhotoImage(roi_pil)
        parent_frame = self.ok_scrollable_frame if label == "OK" else self.ng_scrollable_frame
        sample_frame = ttk.Frame(parent_frame); sample_frame.pack(fill=X, padx=5, pady=2)
        inner_frame = ttk.Frame(sample_frame, relief="solid", borderwidth=1); inner_frame.pack(fill=X, padx=2, pady=2)
        content_frame = ttk.Frame(inner_frame); content_frame.pack(fill=X, padx=5, pady=5)
        img_label = ttk.Label(content_frame, image=roi_tk); img_label.image = roi_tk; img_label.pack(side='left', padx=(0, 10))
        info_frame = ttk.Frame(content_frame); info_frame.pack(side='left', fill=BOTH, expand=True)
        time_str = timestamp.strftime("%H:%M:%S"); date_str = timestamp.strftime("%d/%m/%Y")
        ttk.Label(info_frame, text=f"🕒 {time_str}", font=("Arial", 9)).pack(anchor="w")
        ttk.Label(info_frame, text=f"📅 {date_str}", font=("Arial", 8)).pack(anchor="w")
        ttk.Label(info_frame, text=f"📏 {roi_image.shape[1]}x{roi_image.shape[0]}", font=("Arial", 8)).pack(anchor="w")

    def load_existing_samples(self):
        if not self.samples_dir:
            return
        ok_samples_dir = os.path.join(self.samples_dir, "ok")
        if os.path.exists(ok_samples_dir):
            for filename in sorted(os.listdir(ok_samples_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_path = os.path.join(ok_samples_dir, filename)
                    roi_image = cv2.imread(sample_path)
                    if roi_image is not None:
                        try:
                            date_part, time_part = filename.split('_')[2:4]
                            timestamp = datetime.strptime(f"{date_part}_{time_part.split('.')[0]}", "%Y%m%d_%H%M%S")
                        except Exception:
                            timestamp = datetime.now()
                        self.training_samples.append({'roi': roi_image, 'label': 'OK', 'timestamp': timestamp})
                        self.add_sample_to_history(roi_image, "OK", timestamp)
        ng_samples_dir = os.path.join(self.samples_dir, "ng")
        if os.path.exists(ng_samples_dir):
            for filename in sorted(os.listdir(ng_samples_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_path = os.path.join(ng_samples_dir, filename)
                    roi_image = cv2.imread(sample_path)
                    if roi_image is not None:
                        try:
                            date_part, time_part = filename.split('_')[2:4]
                            timestamp = datetime.strptime(f"{date_part}_{time_part.split('.')[0]}", "%Y%m%d_%H%M%S")
                        except Exception:
                            timestamp = datetime.now()
                        self.training_samples.append({'roi': roi_image, 'label': 'NG', 'timestamp': timestamp})
                        self.add_sample_to_history(roi_image, "NG", timestamp)
        self.update_info_label(); self.update_tab_titles()

    def update_info_label(self):
        ok_count = sum(1 for sample in self.training_samples if sample['label'] == 'OK')
        ng_count = sum(1 for sample in self.training_samples if sample['label'] == 'NG')
        self.info_label.config(text=f"Amostras coletadas: {ok_count} OK, {ng_count} NG")
        if len(self.training_samples) >= 2:
            self.btn_apply_training.config(state=NORMAL)
            if self.use_ml:
                self.btn_train_ml.config(state=NORMAL)

    def on_method_change(self):
        self.use_ml = self.training_method_var.get() == "ml"
        if self.use_ml:
            self.btn_train_ml.pack(side=LEFT, padx=(0, 10), before=self.btn_cancel)
            self.btn_save_ml.pack(side=LEFT, padx=(0, 10), before=self.btn_cancel)
            self.btn_apply_training.config(text="🚀 Aplicar Treinamento (Tradicional)")
        else:
            self.btn_train_ml.pack_forget(); self.btn_save_ml.pack_forget()
            self.btn_apply_training.config(text="🚀 Aplicar Treinamento")
        self.update_info_label()

    def train_ml_model(self):
        if len(self.training_samples) < 4:
            messagebox.showwarning("Aviso", "São necessárias pelo menos 4 amostras (2 OK + 2 NG) para treinamento de ML.")
            return
        metrics = self.ml_classifier.train(self.training_samples)
        messagebox.showinfo("Sucesso", f"Modelo ML treinado! Acurácia: {metrics.get('accuracy', 0):.1%}")
        self.btn_save_ml.config(state=NORMAL)
        self.slot_data['use_ml'] = True
        self.slot_data['ml_trained'] = True

    def save_ml_model(self):
        if not self.ml_classifier.is_trained:
            messagebox.showwarning("Aviso", "Nenhum modelo ML foi treinado ainda.")
            return
        template_path = self.slot_data.get('template_path')
        model_dir = os.path.dirname(template_path) if template_path else get_template_dir()
        model_filename = f"ml_model_slot_{self.slot_data['id']}.joblib"
        model_path = os.path.join(model_dir, model_filename)
        if self.ml_classifier.save_model(model_path):
            self.slot_data['ml_model_path'] = model_path
            self.slot_data['use_ml'] = True
            try:
                if hasattr(self.montagem_instance, 'db_manager') and self.montagem_instance.db_manager:
                    self.montagem_instance.db_manager.update_slot(self.montagem_instance.current_model_id, self.slot_data)
            except Exception:
                pass
            messagebox.showinfo("Sucesso", f"Modelo ML salvo em {model_path}")
        else:
            messagebox.showerror("Erro", "Falha ao salvar o modelo ML.")

    def apply_training(self):
        if len(self.training_samples) < 2:
            messagebox.showwarning("Aviso", "São necessárias pelo menos 2 amostras para treinamento.")
            return
        ok_samples = [s['roi'] for s in self.training_samples if s['label'] == 'OK']
        ng_samples = [s['roi'] for s in self.training_samples if s['label'] == 'NG']
        if not ok_samples:
            messagebox.showwarning("Aviso", "É necessária pelo menos uma amostra OK.")
            return
        new_threshold = self.calculate_optimal_threshold(ok_samples, ng_samples)
        if new_threshold is not None:
            old_threshold = self.slot_data.get('correlation_threshold', self.slot_data.get('detection_threshold', 0.8))
            self.slot_data['correlation_threshold'] = new_threshold
            self.slot_data['use_ml'] = False
            if ok_samples:
                self.update_template_with_best_sample(ok_samples)
            self.montagem_instance.update_slot_data(self.slot_data)
            self.montagem_instance.mark_model_modified()
            messagebox.showinfo("Sucesso", f"Treinamento aplicado!\nLimiar anterior: {old_threshold:.3f}\nNovo limiar: {new_threshold:.3f}")
            self.destroy()
        else:
            messagebox.showerror("Erro", "Não foi possível calcular novo limiar.")

    def calculate_optimal_threshold(self, ok_samples, ng_samples):
        if not ok_samples:
            return None
        template_path = self.slot_data.get('template_path')
        template = None
        if template_path and Path(template_path).exists():
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            template = cv2.cvtColor(ok_samples[0], cv2.COLOR_BGR2GRAY)
        ok_correlations = []
        for roi in ok_samples:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            tpl = cv2.resize(template, (roi_gray.shape[1], roi_gray.shape[0])) if roi_gray.shape != template.shape else template
            result = cv2.matchTemplate(roi_gray, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if not (np.isnan(max_val) or np.isinf(max_val)):
                ok_correlations.append(max_val)
        ng_correlations = []
        for roi in ng_samples:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            tpl = cv2.resize(template, (roi_gray.shape[1], roi_gray.shape[0])) if roi_gray.shape != template.shape else template
            result = cv2.matchTemplate(roi_gray, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if not (np.isnan(max_val) or np.isinf(max_val)):
                ng_correlations.append(max_val)
        if not ok_correlations:
            return None
        min_ok = min(ok_correlations)
        if ng_correlations:
            max_ng = max(ng_correlations)
            new_threshold = (min_ok + max_ng) / 2 if min_ok > max_ng else (sum(ok_correlations) / len(ok_correlations)) * 0.85
            return max(0.3, min(0.95, new_threshold))
        return max(0.5, min(0.9, (sum(ok_correlations) / len(ok_correlations)) * 0.9))

    def update_template_with_best_sample(self, ok_samples):
        template_path = self.slot_data.get('template_path')
        if not template_path:
            return
        current_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if current_template is None:
            return
        best_sample = None
        best_correlation = -1
        for roi in ok_samples:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (current_template.shape[1], current_template.shape[0])) if roi_gray.shape != current_template.shape else roi_gray
            result = cv2.matchTemplate(roi_resized, current_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_correlation:
                best_correlation = max_val
                best_sample = roi_gray
        if best_sample is not None:
            if best_sample.shape != current_template.shape:
                best_sample = cv2.resize(best_sample, (current_template.shape[1], current_template.shape[0]))
            cv2.imwrite(template_path, best_sample)


class SystemConfigDialog(Toplevel):
    def __init__(self, parent, ORB_FEATURES, ORB_SCALE_FACTOR, ORB_N_LEVELS, PREVIEW_W, PREVIEW_H, THR_CORR, MIN_PX, on_save_callback):
        super().__init__(parent)
        self.parent = parent
        self.title("⚙️ Configurações do Sistema")
        self.geometry("550x750")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()
        self.style_config = load_style_config()
        self.result = False
        self.center_window()
        self.setup_ui(ORB_FEATURES, ORB_SCALE_FACTOR, ORB_N_LEVELS, PREVIEW_W, PREVIEW_H, THR_CORR, MIN_PX)
        self.on_save_callback = on_save_callback

    def center_window(self):
        self.update_idletasks()
        # Aumentar altura para acomodar todo o conteúdo
        width = 600
        height = 800
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        
        # Configurar tamanho mínimo
        self.minsize(500, 600)

    def setup_ui(self, ORB_FEATURES, ORB_SCALE_FACTOR, ORB_N_LEVELS, PREVIEW_W, PREVIEW_H, THR_CORR, MIN_PX):
        # Container principal com scroll
        container = ttk.Frame(self)
        container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Canvas com scrollbar
        canvas = Canvas(container, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        # Frame scrollável
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Configurar canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Empacotar elementos
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Configurar scroll com mouse
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _on_enter(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _on_leave(event):
            canvas.unbind_all("<MouseWheel>")
        
        scrollable_frame.bind("<Enter>", _on_enter)
        scrollable_frame.bind("<Leave>", _on_leave)
        
        # Frame principal dentro do scroll
        main_frame = ttk.Frame(scrollable_frame)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        # ORB configs
        orb_frame = ttk.LabelFrame(main_frame, text="Configurações ORB (Alinhamento de Imagem)")
        orb_frame.pack(fill=X, pady=(0, 15))
        ttk.Label(orb_frame, text="Número de Features:").pack(anchor="w", padx=5, pady=2)
        self.orb_features_var = ttk.IntVar(value=ORB_FEATURES)
        features_frame = ttk.Frame(orb_frame); features_frame.pack(fill=X, padx=5, pady=5)
        self.features_scale = ttk.Scale(features_frame, from_=1000, to=10000, variable=self.orb_features_var, orient='horizontal'); self.features_scale.pack(side=LEFT, fill=X, expand=True)
        self.features_label = ttk.Label(features_frame, text=f"{self.orb_features_var.get()}", width=8); self.features_label.pack(side='right', padx=(5, 0))
        self.features_scale.config(command=lambda v: self.features_label.config(text=f"{int(float(v))}"))
        ttk.Label(orb_frame, text="Fator de Escala:").pack(anchor="w", padx=5, pady=(10, 2))
        self.scale_factor_var = ttk.DoubleVar(value=ORB_SCALE_FACTOR)
        scale_frame = ttk.Frame(orb_frame); scale_frame.pack(fill=X, padx=5, pady=5)
        self.scale_scale = ttk.Scale(scale_frame, from_=1.1, to=2.0, variable=self.scale_factor_var, orient='horizontal'); self.scale_scale.pack(side=LEFT, fill=X, expand=True)
        self.scale_label = ttk.Label(scale_frame, text=f"{self.scale_factor_var.get():.2f}", width=8); self.scale_label.pack(side='right', padx=(5, 0))
        self.scale_scale.config(command=lambda v: self.scale_label.config(text=f"{float(v):.2f}"))
        ttk.Label(orb_frame, text="Número de Níveis:").pack(anchor="w", padx=5, pady=(10, 2))
        self.n_levels_var = ttk.IntVar(value=ORB_N_LEVELS)
        levels_spin = ttk.Spinbox(orb_frame, from_=4, to=16, textvariable=self.n_levels_var, width=10); levels_spin.pack(anchor='w', padx=5, pady=5)

        # Visualização
        canvas_frame = ttk.LabelFrame(main_frame, text="Configurações de Visualização")
        canvas_frame.pack(fill=X, pady=(0, 15))
        ttk.Label(canvas_frame, text="Largura Máxima do Preview:").pack(anchor='w', padx=5, pady=2)
        self.preview_w_var = ttk.IntVar(value=PREVIEW_W); w_spin = ttk.Spinbox(canvas_frame, from_=400, to=1600, increment=100, textvariable=self.preview_w_var, width=10); w_spin.pack(anchor='w', padx=5, pady=5)
        ttk.Label(canvas_frame, text="Altura Máxima do Preview:").pack(anchor='w', padx=5, pady=(10, 2))
        self.preview_h_var = ttk.IntVar(value=PREVIEW_H); h_spin = ttk.Spinbox(canvas_frame, from_=300, to=1200, increment=100, textvariable=self.preview_h_var, width=10); h_spin.pack(anchor='w', padx=5, pady=5)

        # Detecção
        detection_frame = ttk.LabelFrame(main_frame, text="Configurações Padrão de Detecção")
        detection_frame.pack(fill=X, pady=(0, 15))
        ttk.Label(detection_frame, text="Limiar de Correlação Padrão (Clips):").pack(anchor='w', padx=5, pady=2)
        self.thr_corr_var = ttk.DoubleVar(value=THR_CORR)
        corr_frame = ttk.Frame(detection_frame); corr_frame.pack(fill=X, padx=5, pady=5)
        self.corr_scale = ttk.Scale(corr_frame, from_=0.1, to=1.0, variable=self.thr_corr_var, orient='horizontal'); self.corr_scale.pack(side=LEFT, fill=X, expand=True)
        self.corr_label = ttk.Label(corr_frame, text=f"{self.thr_corr_var.get():.2f}", width=8); self.corr_label.pack(side='right', padx=(5, 0))
        self.corr_scale.config(command=lambda v: self.corr_label.config(text=f"{float(v):.2f}"))
        ttk.Label(detection_frame, text="Pixels Mínimos Padrão (Template Matching):").pack(anchor='w', padx=5, pady=(10, 2))
        self.min_px_var = ttk.IntVar(value=MIN_PX)
        px_spin = ttk.Spinbox(detection_frame, from_=1, to=1000, textvariable=self.min_px_var, width=10); px_spin.pack(anchor='w', padx=5, pady=5)

        # Câmera Padrão
        camera_frame = ttk.LabelFrame(main_frame, text="Configurações de Câmera (Padrão)")
        camera_frame.pack(fill=X, pady=(0, 15))
        # Backend
        ttk.Label(camera_frame, text="Backend de Captura:").pack(anchor='w', padx=5, pady=2)
        self.camera_backend_var = StringVar(value=self.style_config.get('system', {}).get('camera_backend', 'AUTO'))
        ttk.Combobox(camera_frame, textvariable=self.camera_backend_var, state='readonly',
                     values=["AUTO","DIRECTSHOW","MSMF","V4L2","GSTREAMER"]).pack(anchor='w', padx=5, pady=5)
        # Resolução/FPS
        cam_grid = ttk.Frame(camera_frame); cam_grid.pack(fill=X, padx=5, pady=5)
        ttk.Label(cam_grid, text="Largura:").grid(row=0, column=0, sticky='w', padx=(0,6))
        self.camera_w_var = ttk.IntVar(value=int(self.style_config.get('system', {}).get('camera_width', 1280)))
        ttk.Spinbox(cam_grid, from_=320, to=3840, increment=160, textvariable=self.camera_w_var, width=8).grid(row=0, column=1, sticky='w')
        ttk.Label(cam_grid, text="Altura:").grid(row=0, column=2, sticky='w', padx=(12,6))
        self.camera_h_var = ttk.IntVar(value=int(self.style_config.get('system', {}).get('camera_height', 720)))
        ttk.Spinbox(cam_grid, from_=240, to=2160, increment=120, textvariable=self.camera_h_var, width=8).grid(row=0, column=3, sticky='w')
        ttk.Label(cam_grid, text="FPS:").grid(row=0, column=4, sticky='w', padx=(12,6))
        self.camera_fps_var = ttk.IntVar(value=int(self.style_config.get('system', {}).get('camera_fps', 30)))
        ttk.Spinbox(cam_grid, from_=5, to=120, textvariable=self.camera_fps_var, width=6).grid(row=0, column=5, sticky='w')
        # Auto features
        toggles_frame = ttk.Frame(camera_frame); toggles_frame.pack(fill=X, padx=5, pady=5)
        self.auto_exposure_var = ttk.BooleanVar(value=bool(self.style_config.get('system', {}).get('auto_exposure', True)))
        self.auto_wb_var = ttk.BooleanVar(value=bool(self.style_config.get('system', {}).get('auto_wb', True)))
        ttk.Checkbutton(toggles_frame, text="Auto-Exposure", variable=self.auto_exposure_var).pack(side=LEFT, padx=(0,12))
        ttk.Checkbutton(toggles_frame, text="Auto White Balance", variable=self.auto_wb_var).pack(side=LEFT)

        # Performance
        perf_frame = ttk.LabelFrame(main_frame, text="Performance")
        perf_frame.pack(fill=X, pady=(0, 15))
        perf_grid = ttk.Frame(perf_frame); perf_grid.pack(fill=X, padx=5, pady=5)
        ttk.Label(perf_grid, text="Frame Pump FPS:").grid(row=0, column=0, sticky='w')
        self.frame_pump_fps_var = ttk.IntVar(value=int(self.style_config.get('system', {}).get('frame_pump_fps', 30)))
        ttk.Spinbox(perf_grid, from_=5, to=120, textvariable=self.frame_pump_fps_var, width=8).grid(row=0, column=1, sticky='w', padx=(6,12))
        ttk.Label(perf_grid, text="Buffer por Câmera:").grid(row=0, column=2, sticky='w')
        self.buffer_size_var = ttk.IntVar(value=int(self.style_config.get('system', {}).get('buffer_size', 1)))
        ttk.Spinbox(perf_grid, from_=1, to=10, textvariable=self.buffer_size_var, width=6).grid(row=0, column=3, sticky='w', padx=(6,12))
        ttk.Label(perf_grid, text="OpenCV Threads:").grid(row=0, column=4, sticky='w')
        self.cv_threads_var = ttk.IntVar(value=int(self.style_config.get('system', {}).get('cv_num_threads', 0)))
        ttk.Spinbox(perf_grid, from_=0, to=16, textvariable=self.cv_threads_var, width=6).grid(row=0, column=5, sticky='w', padx=(6,0))
        self.preview_gray_var = ttk.BooleanVar(value=bool(self.style_config.get('system', {}).get('preview_grayscale', False)))
        ttk.Checkbutton(perf_frame, text="Pré-visualização em tons de cinza (economiza CPU)", variable=self.preview_gray_var).pack(anchor='w', padx=5, pady=(6,0))

        # Logs
        logs_frame = ttk.LabelFrame(main_frame, text="Logs e Diagnóstico")
        logs_frame.pack(fill=X, pady=(0, 15))
        log_grid = ttk.Frame(logs_frame); log_grid.pack(fill=X, padx=5, pady=5)
        ttk.Label(log_grid, text="Nível de Log:").grid(row=0, column=0, sticky='w')
        self.log_level_var = StringVar(value=self.style_config.get('system', {}).get('log_level', 'INFO'))
        ttk.Combobox(log_grid, textvariable=self.log_level_var, state='readonly',
                     values=["DEBUG","INFO","WARNING","ERROR"]).grid(row=0, column=1, sticky='w', padx=(6,12))
        self.log_to_file_var = ttk.BooleanVar(value=bool(self.style_config.get('system', {}).get('log_to_file', False)))
        ttk.Checkbutton(logs_frame, text="Gravar logs em arquivo", variable=self.log_to_file_var).pack(anchor='w', padx=5, pady=(6,0))
        size_row = ttk.Frame(logs_frame); size_row.pack(fill=X, padx=5, pady=5)
        ttk.Label(size_row, text="Tamanho máx. do arquivo de log (MB):").pack(side=LEFT)
        self.log_max_mb_var = ttk.IntVar(value=int(self.style_config.get('system', {}).get('log_max_mb', 10)))
        ttk.Spinbox(size_row, from_=1, to=1024, textvariable=self.log_max_mb_var, width=8).pack(side=LEFT, padx=(6,0))

        # Histórico
        hist_frame = ttk.LabelFrame(main_frame, text="Histórico de Imagens")
        hist_frame.pack(fill=X, pady=(0, 15))
        toggles_hist = ttk.Frame(hist_frame); toggles_hist.pack(fill=X, padx=5, pady=5)
        self.hist_ok_var = ttk.BooleanVar(value=bool(self.style_config.get('system', {}).get('history_save_ok', True)))
        self.hist_ng_var = ttk.BooleanVar(value=bool(self.style_config.get('system', {}).get('history_save_ng', True)))
        ttk.Checkbutton(toggles_hist, text="Salvar imagens OK", variable=self.hist_ok_var).pack(side=LEFT, padx=(0,12))
        ttk.Checkbutton(toggles_hist, text="Salvar imagens NG", variable=self.hist_ng_var).pack(side=LEFT)
        qual_row = ttk.Frame(hist_frame); qual_row.pack(fill=X, padx=5, pady=5)
        ttk.Label(qual_row, text="Qualidade JPEG (histórico):").pack(side=LEFT)
        self.hist_jpeg_q_var = ttk.IntVar(value=int(self.style_config.get('system', {}).get('history_jpeg_quality', 85)))
        ttk.Spinbox(qual_row, from_=10, to=100, textvariable=self.hist_jpeg_q_var, width=8).pack(side=LEFT, padx=(6,12))
        path_row = ttk.Frame(hist_frame); path_row.pack(fill=X, padx=5, pady=5)
        ttk.Label(path_row, text="Pasta do histórico:").pack(side=LEFT)
        self.hist_dir_var = StringVar(value=self.style_config.get('system', {}).get('history_dir', str((get_project_root() / 'modelos' / 'historico_fotos').resolve())))
        entry = ttk.Entry(path_row, textvariable=self.hist_dir_var, width=40); entry.pack(side=LEFT, padx=(6,6))
        ttk.Button(path_row, text="Procurar", command=self.choose_history_dir).pack(side=LEFT)

        # Cores e fontes
        appearance_frame = ttk.LabelFrame(main_frame, text="Configurações de Aparência do Sistema")
        appearance_frame.pack(fill=X, pady=(0, 15))
        
        # Configurações de cores
        colors_frame = ttk.Frame(appearance_frame)
        colors_frame.pack(fill=X, padx=10, pady=10)
        
        # Cor de fundo principal
        ttk.Label(colors_frame, text="Cor de Fundo Principal:").pack(anchor="w", pady=(5, 2))
        bg_frame = ttk.Frame(colors_frame)
        bg_frame.pack(fill=X, pady=2)
        self.bg_color_var = StringVar(value="#2b2b2b")
        bg_color_entry = ttk.Entry(bg_frame, textvariable=self.bg_color_var, width=12)
        bg_color_entry.pack(side=LEFT, padx=(0, 10))
        ttk.Button(bg_frame, text="Escolher", command=self.choose_bg_color).pack(side=LEFT)
        
        # Cor do texto principal
        ttk.Label(colors_frame, text="Cor do Texto Principal:").pack(anchor="w", pady=(15, 2))
        text_frame = ttk.Frame(colors_frame)
        text_frame.pack(fill=X, pady=2)
        self.text_color_var = StringVar(value="#ffffff")
        text_color_entry = ttk.Entry(text_frame, textvariable=self.text_color_var, width=12)
        text_color_entry.pack(side=LEFT, padx=(0, 10))
        ttk.Button(text_frame, text="Escolher", command=self.choose_text_color).pack(side=LEFT)
        
        # Cor de destaque
        ttk.Label(colors_frame, text="Cor de Destaque:").pack(anchor="w", pady=(15, 2))
        accent_frame = ttk.Frame(colors_frame)
        accent_frame.pack(fill=X, pady=2)
        self.accent_color_var = StringVar(value="#007acc")
        accent_color_entry = ttk.Entry(accent_frame, textvariable=self.accent_color_var, width=12)
        accent_color_entry.pack(side=LEFT, padx=(0, 10))
        ttk.Button(accent_frame, text="Escolher", command=self.choose_accent_color).pack(side=LEFT)
        
        # Configurações de fonte
        font_frame = ttk.Frame(appearance_frame)
        font_frame.pack(fill=X, padx=10, pady=15)
        ttk.Label(font_frame, text="Configurações de Fonte:").pack(anchor="w", pady=(5, 10))
        
        # Tamanho da fonte principal
        size_frame = ttk.Frame(font_frame)
        size_frame.pack(fill=X, pady=5)
        ttk.Label(size_frame, text="Tamanho da Fonte Principal:").pack(side=LEFT, padx=(0, 15))
        self.font_size_var = ttk.IntVar(value=10)
        font_size_spin = ttk.Spinbox(size_frame, from_=8, to=20, textvariable=self.font_size_var, width=10)
        font_size_spin.pack(side=LEFT)
        
        # Família da fonte
        family_frame = ttk.Frame(font_frame)
        family_frame.pack(fill=X, pady=5)
        ttk.Label(family_frame, text="Família da Fonte:").pack(side=LEFT, padx=(0, 15))
        self.font_family_var = StringVar(value="Segoe UI")
        font_family_combo = ttk.Combobox(family_frame, textvariable=self.font_family_var, 
                                        values=["Segoe UI", "Arial", "Helvetica", "Consolas", "Courier New"], 
                                        state="readonly", width=18)
        font_family_combo.pack(side=LEFT)

        # Separador antes dos botões
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.pack(fill=X, pady=(20, 15))
        
        # Frame dos botões
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=X, pady=(0, 15), padx=10)
        
        # Botões com melhor espaçamento
        ttk.Button(button_frame, text="Salvar", command=self.save_config, style="Accent.TButton").pack(side=LEFT, padx=(0, 10), pady=10, expand=True, fill=X)
        ttk.Button(button_frame, text="Restaurar Padrões", command=self.restore_defaults).pack(side=LEFT, padx=(0, 10), pady=10, expand=True, fill=X)
        ttk.Button(button_frame, text="Cancelar", command=self.cancel).pack(side=LEFT, pady=10, expand=True, fill=X)

    def choose_history_dir(self):
        try:
            from tkinter import filedialog
            d = filedialog.askdirectory(title="Selecionar pasta para histórico de imagens", initialdir=self.hist_dir_var.get())
            if d:
                self.hist_dir_var.set(d)
        except Exception as e:
            print(f"Erro ao escolher pasta: {e}")

    def choose_bg_color(self):
        """Abre seletor de cor para fundo principal."""
        try:
            from tkinter import colorchooser
            color = colorchooser.askcolor(title="Escolher Cor de Fundo Principal", color=self.bg_color_var.get())
            if color[1]:
                self.bg_color_var.set(color[1])
        except Exception as e:
            print(f"Erro ao escolher cor: {e}")
    
    def choose_text_color(self):
        """Abre seletor de cor para texto principal."""
        try:
            from tkinter import colorchooser
            color = colorchooser.askcolor(title="Escolher Cor do Texto Principal", color=self.text_color_var.get())
            if color[1]:
                self.text_color_var.set(color[1])
        except Exception as e:
            print(f"Erro ao escolher cor: {e}")
    
    def choose_accent_color(self):
        """Abre seletor de cor para destaque."""
        try:
            from tkinter import colorchooser
            color = colorchooser.askcolor(title="Escolher Cor de Destaque", color=self.accent_color_var.get())
            if color[1]:
                self.accent_color_var.set(color[1])
        except Exception as e:
            print(f"Erro ao escolher cor: {e}")
    
    def save_config(self):
        try:
            cfg = {
                'ORB_FEATURES': int(self.orb_features_var.get()),
                'ORB_SCALE_FACTOR': float(self.scale_factor_var.get()),
                'ORB_N_LEVELS': int(self.n_levels_var.get()),
                'PREVIEW_W': int(self.preview_w_var.get()),
                'PREVIEW_H': int(self.preview_h_var.get()),
                'THR_CORR': float(self.thr_corr_var.get()),
                'MIN_PX': int(self.min_px_var.get()),
                # Configurações de aparência
                'BG_COLOR': self.bg_color_var.get(),
                'TEXT_COLOR': self.text_color_var.get(),
                'ACCENT_COLOR': self.accent_color_var.get(),
                'FONT_SIZE': int(self.font_size_var.get()),
                'FONT_FAMILY': self.font_family_var.get(),
                # Sistema (novos)
                'camera_backend': self.camera_backend_var.get(),
                'camera_width': int(self.camera_w_var.get()),
                'camera_height': int(self.camera_h_var.get()),
                'camera_fps': int(self.camera_fps_var.get()),
                'auto_exposure': bool(self.auto_exposure_var.get()),
                'auto_wb': bool(self.auto_wb_var.get()),
                'frame_pump_fps': int(self.frame_pump_fps_var.get()),
                'buffer_size': int(self.buffer_size_var.get()),
                'cv_num_threads': int(self.cv_threads_var.get()),
                'preview_grayscale': bool(self.preview_gray_var.get()),
                'log_level': self.log_level_var.get(),
                'log_to_file': bool(self.log_to_file_var.get()),
                'log_max_mb': int(self.log_max_mb_var.get()),
                'history_save_ok': bool(self.hist_ok_var.get()),
                'history_save_ng': bool(self.hist_ng_var.get()),
                'history_jpeg_quality': int(self.hist_jpeg_q_var.get()),
                'history_dir': self.hist_dir_var.get(),
            }
            if callable(self.on_save_callback):
                self.on_save_callback(cfg)
            self.result = True
            messagebox.showinfo("Sucesso", "Configurações salvas com sucesso!")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar configurações: {str(e)}")

    def restore_defaults(self):
        """Restaura as configurações padrão."""
        try:
            # Restaurar valores padrão
            self.orb_features_var.set(5000)
            self.scale_factor_var.set(1.2)
            self.n_levels_var.set(8)
            self.preview_w_var.set(800)
            self.preview_h_var.set(600)
            self.thr_corr_var.set(0.1)
            self.min_px_var.set(10)
            
            # Restaurar configurações de aparência padrão
            self.bg_color_var.set("#2b2b2b")
            self.text_color_var.set("#ffffff")
            self.accent_color_var.set("#007acc")
            self.font_size_var.set(10)
            self.font_family_var.set("Segoe UI")
            
            # Atualizar labels
            self.corr_label.config(text="0.10")
            self.features_label.config(text="5000")
            self.scale_label.config(text="1.20")
            
            messagebox.showinfo("Sucesso", "Configurações padrão restauradas!")
        except Exception as e:
            print(f"Erro ao restaurar padrões: {e}")
            messagebox.showerror("Erro", f"Erro ao restaurar padrões: {e}")

    def cancel(self):
        self.destroy()


__all__ = [
    'EditSlotDialog',
    'SlotTrainingDialog',
    'SystemConfigDialog',
]





