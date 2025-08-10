import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import LEFT, BOTH, DISABLED, NORMAL, X
from tkinter import Canvas, filedialog, messagebox, Toplevel, StringVar
from PIL import Image, ImageTk

try:
    from utils import load_style_config, get_color, get_font, save_style_config, apply_style_config
    from ml_classifier import MLSlotClassifier
    from camera_manager import capture_image_from_camera
    from paths import get_template_dir, get_model_template_dir
    from inspection import find_image_transform
    from image_utils import cv2_to_tk
except Exception:
    from utils import load_style_config, get_color, get_font, save_style_config, apply_style_config
    from ml_classifier import MLSlotClassifier
    from camera_manager import capture_image_from_camera
    from paths import get_template_dir, get_model_template_dir
    from inspection import find_image_transform
    from image_utils import cv2_to_tk


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
        self.detection_threshold_var = StringVar()

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
        ttk.Label(threshold_frame, text="Limiar de Detecção (%):").pack(side=LEFT)
        ttk.Entry(threshold_frame, textvariable=self.detection_threshold_var, width=10).pack(side=LEFT, padx=(5, 0))

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
            self.detection_threshold_var.set(str(self.slot_data.get('detection_threshold', 50)))

    def save_changes(self):
        try:
            x_val = int(self.x_var.get().strip())
            y_val = int(self.y_var.get().strip())
            w_val = int(self.w_var.get().strip())
            h_val = int(self.h_var.get().strip())
            threshold_val = float(self.detection_threshold_var.get().strip())
            if w_val <= 0 or h_val <= 0:
                raise ValueError("Largura e altura devem ser maiores que zero")
            if threshold_val < 0 or threshold_val > 100:
                raise ValueError("Limiar deve estar entre 0 e 100")

            self.slot_data.update({'x': x_val, 'y': y_val, 'w': w_val, 'h': h_val, 'detection_threshold': threshold_val})
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
        width = 550; height = 750
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def setup_ui(self, ORB_FEATURES, ORB_SCALE_FACTOR, ORB_N_LEVELS, PREVIEW_W, PREVIEW_H, THR_CORR, MIN_PX):
        container = ttk.Frame(self); container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        canvas = Canvas(container, width=530, height=700)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, width=520)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main_frame = ttk.Frame(scrollable_frame); main_frame.pack(fill=BOTH, expand=True)
        # ORB configs
        orb_frame = ttk.LabelFrame(main_frame, text="Configurações ORB (Alinhamento de Imagem)"); orb_frame.pack(fill=X, pady=(0, 10))
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
        canvas_frame = ttk.LabelFrame(main_frame, text="Configurações de Visualização"); canvas_frame.pack(fill=X, pady=(0, 10))
        ttk.Label(canvas_frame, text="Largura Máxima do Preview:").pack(anchor='w', padx=5, pady=2)
        self.preview_w_var = ttk.IntVar(value=PREVIEW_W); w_spin = ttk.Spinbox(canvas_frame, from_=400, to=1600, increment=100, textvariable=self.preview_w_var, width=10); w_spin.pack(anchor='w', padx=5, pady=5)
        ttk.Label(canvas_frame, text="Altura Máxima do Preview:").pack(anchor='w', padx=5, pady=(10, 2))
        self.preview_h_var = ttk.IntVar(value=PREVIEW_H); h_spin = ttk.Spinbox(canvas_frame, from_=300, to=1200, increment=100, textvariable=self.preview_h_var, width=10); h_spin.pack(anchor='w', padx=5, pady=5)

        # Detecção
        detection_frame = ttk.LabelFrame(main_frame, text="Configurações Padrão de Detecção"); detection_frame.pack(fill=X, pady=(0, 10))
        ttk.Label(detection_frame, text="Limiar de Correlação Padrão (Clips):").pack(anchor='w', padx=5, pady=2)
        self.thr_corr_var = ttk.DoubleVar(value=THR_CORR)
        corr_frame = ttk.Frame(detection_frame); corr_frame.pack(fill=X, padx=5, pady=5)
        self.corr_scale = ttk.Scale(corr_frame, from_=0.1, to=1.0, variable=self.thr_corr_var, orient='horizontal'); self.corr_scale.pack(side=LEFT, fill=X, expand=True)
        self.corr_label = ttk.Label(corr_frame, text=f"{self.thr_corr_var.get():.2f}", width=8); self.corr_label.pack(side='right', padx=(5, 0))
        self.corr_scale.config(command=lambda v: self.corr_label.config(text=f"{float(v):.2f}"))
        ttk.Label(detection_frame, text="Pixels Mínimos Padrão (Template Matching):").pack(anchor='w', padx=5, pady=(10, 2))
        self.min_px_var = ttk.IntVar(value=MIN_PX)
        px_spin = ttk.Spinbox(detection_frame, from_=1, to=1000, textvariable=self.min_px_var, width=10); px_spin.pack(anchor='w', padx=5, pady=5)

        # Cores e fontes
        appearance_frame = ttk.LabelFrame(main_frame, text="Configurações de Aparência por Local"); appearance_frame.pack(fill=X, pady=(0, 10))
        # Exemplo mínimo: apenas salvar sem UI detalhada para fontes/cores

        button_frame = ttk.Frame(main_frame); button_frame.pack(fill=X, pady=(20, 10), padx=10)
        ttk.Button(button_frame, text="Salvar", command=self.save_config).pack(side=LEFT, padx=5, pady=5, expand=True, fill=X)
        ttk.Button(button_frame, text="Restaurar Padrões", command=self.restore_defaults).pack(side=LEFT, padx=5, pady=5, expand=True, fill=X)
        ttk.Button(button_frame, text="Cancelar", command=self.cancel).pack(side=LEFT, padx=5, pady=5, expand=True, fill=X)

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
            }
            if callable(self.on_save_callback):
                self.on_save_callback(cfg)
            self.result = True
            messagebox.showinfo("Sucesso", "Configurações salvas com sucesso!")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar configurações: {str(e)}")

    def restore_defaults(self):
        self.orb_features_var.set(5000)
        self.scale_factor_var.set(1.2)
        self.n_levels_var.set(8)
        self.preview_w_var.set(800)
        self.preview_h_var.set(600)
        self.thr_corr_var.set(0.1)
        self.min_px_var.set(10)
        self.corr_label.config(text="0.10")
        self.features_label.config(text="5000")
        self.scale_label.config(text="1.20")

    def cancel(self):
        self.destroy()


__all__ = [
    'EditSlotDialog',
    'SlotTrainingDialog',
    'SystemConfigDialog',
]





