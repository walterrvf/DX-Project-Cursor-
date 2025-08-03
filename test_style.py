import json
import os
import tkinter as tk
from tkinter import ttk

def load_style_config():
    config_path = os.path.join('config', 'style_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    # Carrega as configurações de estilo
    style_config = load_style_config()
    print('Style Config:', style_config)
    
    # Cria a janela de teste
    root = tk.Tk()
    root.title('Teste de Estilo')
    
    # Configura o estilo
    style = ttk.Style()
    style.configure('TFrame', background=style_config['background_color'])
    style.configure('TLabel', font=style_config['ok_font'], foreground=style_config['text_color'])
    style.configure('TButton', font=style_config['ok_font'])
    
    # Cria os widgets
    frame = ttk.Frame(root, padding=20)
    frame.pack(padx=20, pady=20)
    
    ttk.Label(frame, text='Teste de Fonte OK').pack(pady=5)
    
    # Botão OK
    ok_button = ttk.Button(frame, text='OK')
    ok_button.pack(pady=5)
    
    # Botão NG
    ng_button = ttk.Button(frame, text='NG')
    ng_button.pack(pady=5)
    
    # Inicia o loop principal
    root.mainloop()

if __name__ == '__main__':
    main()