import cv2
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import LEFT, BOTH, DISABLED, NORMAL, X, Y
from tkinter import Canvas, messagebox

try:
    from utils import load_style_config, get_color, get_font
except Exception:
    from utils import load_style_config, get_color, get_font


class InspecaoWindow(ttk.Frame):
    def __init__(self, master, montagem_instance):
        super().__init__(master)
        self.master = master
        self.montagem_instance = montagem_instance
        self.style_config = load_style_config()
        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self)
        frame.pack(fill=BOTH, expand=True)
        # Placeholder simples, pois a lógica detalhada ainda está em montagem.py
        ttk.Label(frame, text="Inspeção", font=get_font('header_font')).pack(pady=10)


__all__ = ['InspecaoWindow']





