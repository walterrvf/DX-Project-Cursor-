import ttkbootstrap as ttk
from ttkbootstrap.constants import BOTH, LEFT, RIGHT, Y, CENTER, VERTICAL, X, SUCCESS, DANGER, PRIMARY, DISABLED, NORMAL, END, BOTTOM
from tkinter import messagebox, simpledialog
from datetime import datetime
from pathlib import Path
try:
    # Quando importado como módulo
    from database_manager import DatabaseManager
    from utils import load_style_config, get_font
except ImportError:
    # Quando executado diretamente
    try:
        from database_manager import DatabaseManager
        from utils import load_style_config, get_font
    except ImportError:
        # Quando executado a partir do diretório raiz
        from modulos.database_manager import DatabaseManager
        from modulos.utils import load_style_config, get_font

class ModelSelectorDialog:
    """Diálogo para seleção, criação e gerenciamento de modelos."""
    
    def __init__(self, parent, db_manager: DatabaseManager):
        self.parent = parent
        self.db_manager = db_manager
        self.selected_model = None
        self.result = None
        
        self.dialog = ttk.Toplevel(parent)
        self.dialog.title("Gerenciar Modelos")
        # Dimensões responsivas com base na tela
        try:
            screen_w = self.dialog.winfo_screenwidth()
            screen_h = self.dialog.winfo_screenheight()
            margin_w, margin_h = 60, 60
            # Tamanho alvo base (FullHD): 900x650, escala para telas menores/maiores
            base_w, base_h = 900, 650
            scale_w = max(0.75, min(1.1, screen_w / 1920))
            scale_h = max(0.75, min(1.1, screen_h / 1080))
            target_w = int(base_w * scale_w)
            target_h = int(base_h * scale_h)
            # Garante que não ultrapasse a área visível
            win_w = max(720, min(target_w, screen_w - margin_w))
            win_h = max(520, min(target_h, screen_h - margin_h))
            self.dialog.geometry(f"{win_w}x{win_h}")
            # Permite redimensionar
            self.dialog.resizable(True, True)
        except Exception:
            self.dialog.geometry("900x650")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Centraliza o diálogo
        self.dialog.update_idletasks()
        try:
            cur_w = self.dialog.winfo_width()
            cur_h = self.dialog.winfo_height()
            x = (self.dialog.winfo_screenwidth() // 2) - (cur_w // 2)
            y = (self.dialog.winfo_screenheight() // 2) - (cur_h // 2)
            self.dialog.geometry(f"{cur_w}x{cur_h}+{x}+{y}")
        except Exception:
            pass
        
        self.setup_ui()
        self.refresh_models()
        
        # Configura fechamento
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
    
    def setup_ui(self):
        """Configura a interface do diálogo."""
        main_frame = ttk.Frame(self.dialog, padding=15)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Título com fonte maior
        title_label = ttk.Label(main_frame, text="Gerenciar Modelos de Inspeção", 
                               font=get_font('title_font')) 
        title_label.pack(pady=(0, 10))
        
        # Frame para lista de modelos
        list_frame = ttk.LabelFrame(main_frame, text="Modelos Disponíveis", padding=10)
        list_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # Configurar label do LabelFrame com fonte maior
        style = ttk.Style()
        style.configure('Big.TLabelframe.Label', font=get_font('header_font'))
        list_frame.configure(style='Big.TLabelframe')
        
        # Configuração do estilo do Treeview com fontes maiores alinhado ao tema escuro
        style.configure('Models.Treeview',
                        font=get_font('medium_font'),
                        rowheight=30)
        style.configure('Models.Treeview.Heading',
                        font=get_font('header_font'))
        style.map('Models.Treeview',
                  background=[('selected', '#93C5FD')],
                  foreground=[('selected', '#0B1220')])

        # Treeview para modelos com altura aumentada
        columns = ('nome', 'slots', 'criado', 'atualizado')
        # Altura será ajustada dinamicamente pelo tamanho do frame; inicializamos com um valor neutro
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10, style='Models.Treeview')
        
        # Configurar colunas com larguras otimizadas
        self.tree.heading('nome', text='Nome do Modelo')
        self.tree.heading('slots', text='Slots')
        self.tree.heading('criado', text='Criado em')
        self.tree.heading('atualizado', text='Atualizado em')
        
        self.tree.column('nome', width=250)
        self.tree.column('slots', width=100, anchor=CENTER)
        self.tree.column('criado', width=180)
        self.tree.column('atualizado', width=180)
        
        # Scrollbar para treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Ajuste dinâmico da altura do Treeview conforme redimensionamento
        try:
            # Tenta obter rowheight real
            self._tree_rowheight = style.lookup('Models.Treeview', 'rowheight') or 30
        except Exception:
            self._tree_rowheight = 30
        
        # Bind para seleção
        self.tree.bind('<<TreeviewSelect>>', self.on_model_select)
        self.tree.bind('<Double-1>', self.on_load_model)
        
        # Frame para botões
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=BOTTOM, fill=X, pady=(8, 0))
        
        # Botões da esquerda (ações de modelo)
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=LEFT)
        
        # Criar estilo para botões com fonte maior
        style.configure('BigButton.TButton', font=get_font('small_font'))
        
        self.btn_new = ttk.Button(left_buttons, text="Novo Modelo", 
                                 command=self.on_new_model, bootstyle=SUCCESS,
                                 style='BigButton.TButton')
        self.btn_new.pack(side=LEFT, padx=(0, 10))
        
        self.btn_rename = ttk.Button(left_buttons, text="Renomear", 
                                    command=self.on_rename_model, state=DISABLED,
                                    style='BigButton.TButton')
        self.btn_rename.pack(side=LEFT, padx=10)
        
        self.btn_delete = ttk.Button(left_buttons, text="Excluir", 
                                    command=self.on_delete_model, 
                                    bootstyle=DANGER, state=DISABLED,
                                    style='BigButton.TButton')
        self.btn_delete.pack(side=LEFT, padx=10)
        
        # Botões da direita (ações do diálogo)
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side=RIGHT)
        
        self.btn_load = ttk.Button(right_buttons, text="Carregar Modelo", 
                                  command=self.on_load_model, 
                                  bootstyle=PRIMARY, state=DISABLED,
                                  style='BigButton.TButton')
        self.btn_load.pack(side=LEFT, padx=10)
        
        self.btn_cancel = ttk.Button(right_buttons, text="Cancelar", 
                                    command=self.on_cancel,
                                    style='BigButton.TButton')
        self.btn_cancel.pack(side=LEFT, padx=(10, 0))
        
        # Frame para informações do modelo selecionado
        info_frame = ttk.LabelFrame(main_frame, text="Informações do Modelo", padding=15)
        info_frame.pack(side=BOTTOM, fill=X, pady=(15, 0))
        
        # Configurar label do info_frame com fonte maior
        info_frame.configure(style='Big.TLabelframe')
        
        self.info_text = ttk.Text(info_frame, height=5, font=get_font('small_font'), state=DISABLED)
        self.info_text.pack(fill=X)

        # Evita que o frame da lista pressione os frames inferiores
        try:
            list_frame.pack_propagate(False)
        except Exception:
            pass

        # Redimensiona o Treeview para ocupar somente o espaço restante
        def _resize_tree(event=None):
            try:
                main_h = max(400, main_frame.winfo_height())
                title_h = title_label.winfo_height() or 40
                buttons_h = button_frame.winfo_height() or 64
                info_h = info_frame.winfo_height() or 120
                margins = 40  # paddings internos aproximados
                reserved = title_h + buttons_h + info_h + margins
                available_h = max(160, main_h - reserved)
                rows = max(6, min(24, available_h // int(self._tree_rowheight)))
                self.tree.configure(height=rows)
            except Exception:
                pass

        main_frame.bind('<Configure>', _resize_tree)
        # Primeiro ajuste após construção
        self.dialog.after(50, _resize_tree)
    
    def refresh_models(self):
        """Atualiza a lista de modelos."""
        # Limpa a árvore
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Carrega modelos do banco
        try:
            modelos = self.db_manager.list_modelos()
            # Debug visual no console para confirmar contagem
            print(f"ModelSelector: encontrados {len(modelos)} modelos")
            
            for modelo in modelos:
                # Formata datas
                criado = self.format_datetime(modelo['criado_em'])
                atualizado = self.format_datetime(modelo['atualizado_em'])
                
                self.tree.insert('', END, 
                               values=(modelo['nome'], modelo['num_slots'], criado, atualizado),
                               tags=(str(modelo['id']),))
            
            if not modelos:
                self.tree.insert('', END, values=('Nenhum modelo encontrado', '', '', ''))
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar modelos: {e}", parent=self.dialog)
    
    def format_datetime(self, dt_str: str) -> str:
        """Formata string de datetime para exibição."""
        try:
            dt = datetime.fromisoformat(dt_str)
            return dt.strftime("%d/%m/%Y %H:%M")
        except:
            return dt_str
    
    def on_model_select(self, event):
        """Callback para seleção de modelo."""
        selection = self.tree.selection()
        
        if selection:
            item = self.tree.item(selection[0])
            values = item['values']
            
            if values[0] != 'Nenhum modelo encontrado':
                # Habilita botões
                self.btn_load.config(state=NORMAL)
                self.btn_rename.config(state=NORMAL)
                self.btn_delete.config(state=NORMAL)
                
                # Obtém ID do modelo
                model_id = int(item['tags'][0])
                self.selected_model = model_id
                
                # Mostra informações do modelo
                self.show_model_info(model_id)
            else:
                self.selected_model = None
                self.btn_load.config(state=DISABLED)
                self.btn_rename.config(state=DISABLED)
                self.btn_delete.config(state=DISABLED)
                self.clear_model_info()
        else:
            self.selected_model = None
            self.btn_load.config(state=DISABLED)
            self.btn_rename.config(state=DISABLED)
            self.btn_delete.config(state=DISABLED)
            self.clear_model_info()
    
    def show_model_info(self, model_id: int):
        """Mostra informações detalhadas do modelo."""
        try:
            modelo = self.db_manager.load_modelo(model_id)
            
            info = f"Nome: {modelo['nome']}\n"
            info += f"Imagem: {Path(modelo['image_path']).name}\n"
            info += f"Slots: {len(modelo['slots'])}\n"
            info += f"Criado: {self.format_datetime(modelo['criado_em'])}\n"
            info += f"Atualizado: {self.format_datetime(modelo['atualizado_em'])}"
            
            self.info_text.config(state=NORMAL)
            self.info_text.delete(1.0, END)
            self.info_text.insert(1.0, info)
            self.info_text.config(state=DISABLED)
            
        except Exception as e:
            self.clear_model_info()
            print(f"Erro ao carregar informações do modelo: {e}")
    
    def clear_model_info(self):
        """Limpa as informações do modelo."""
        self.info_text.config(state=NORMAL)
        self.info_text.delete(1.0, END)
        self.info_text.config(state=DISABLED)
    
    def on_new_model(self):
        """Cria um novo modelo."""
        nome = simpledialog.askstring(
            "Novo Modelo",
            "Digite o nome do novo modelo:",
            parent=self.dialog
        )
        
        if nome:
            nome = nome.strip()
            if not nome:
                messagebox.showwarning("Aviso", "Nome não pode estar vazio.", parent=self.dialog)
                return
            
            # Retorna indicação de criar novo modelo
            self.result = {'action': 'new', 'name': nome}
            self.dialog.destroy()
    
    def on_load_model(self, event=None):
        """Carrega o modelo selecionado."""
        if self.selected_model:
            self.result = {'action': 'load', 'model_id': self.selected_model}
            self.dialog.destroy()
    
    def on_rename_model(self):
        """Renomeia o modelo selecionado."""
        if not self.selected_model:
            return
        
        try:
            modelo = self.db_manager.load_modelo(self.selected_model)
            
            novo_nome = simpledialog.askstring(
                "Renomear Modelo",
                f"Nome atual: {modelo['nome']}\n\nDigite o novo nome:",
                initialvalue=modelo['nome'],
                parent=self.dialog
            )
            
            if novo_nome and novo_nome.strip() != modelo['nome']:
                novo_nome = novo_nome.strip()
                
                if not novo_nome:
                    messagebox.showwarning("Aviso", "Nome não pode estar vazio.", parent=self.dialog)
                    return
                
                # Mantém o camera_index atual do modelo ao renomear
                try:
                    current = self.db_manager.get_model_by_id(self.selected_model)
                    cam_ix = current.get('camera_index', 0) if current else None
                except Exception:
                    cam_ix = None
                
                self.db_manager.update_modelo(self.selected_model, nome=novo_nome, camera_index=cam_ix)
                self.refresh_models()
                messagebox.showinfo("Sucesso", f"Modelo renomeado para '{novo_nome}'", parent=self.dialog)
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao renomear modelo: {e}", parent=self.dialog)
    
    def on_delete_model(self):
        """Exclui o modelo selecionado."""
        if not self.selected_model:
            return
        
        try:
            modelo = self.db_manager.load_modelo(self.selected_model)
            
            resposta = messagebox.askyesno(
                "Confirmar Exclusão",
                f"Tem certeza que deseja excluir o modelo '{modelo['nome']}'?\n\n"
                f"Esta ação não pode ser desfeita e removerá:\n"
                f"- O modelo\n"
                f"- Todos os {len(modelo['slots'])} slots\n"
                f"- Configurações associadas",
                parent=self.dialog
            )
            
            if resposta:
                self.db_manager.delete_modelo(self.selected_model)
                self.refresh_models()
                self.selected_model = None
                messagebox.showinfo("Sucesso", "Modelo excluído com sucesso.", parent=self.dialog)
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao excluir modelo: {e}", parent=self.dialog)
    
    def on_migrate_json(self):
        """Migra um arquivo JSON para o banco de dados."""
        from tkinter import filedialog
        
        json_file = filedialog.askopenfilename(
            title="Selecionar arquivo JSON para migrar",
            filetypes=[("Arquivos JSON", "*.json"), ("Todos os arquivos", "*.*")],
            parent=self.dialog
        )
        
        if json_file:
            # Pede nome do modelo
            nome_sugerido = Path(json_file).stem
            nome = simpledialog.askstring(
                "Nome do Modelo",
                f"Digite o nome para o modelo migrado:\n\n"
                f"Arquivo: {Path(json_file).name}",
                initialvalue=nome_sugerido,
                parent=self.dialog
            )
            
            if nome:
                nome = nome.strip()
                if not nome:
                    messagebox.showwarning("Aviso", "Nome não pode estar vazio.", parent=self.dialog)
                    return
                
                try:
                    model_id = self.db_manager.migrate_from_json(json_file, nome)
                    self.refresh_models()
                    messagebox.showinfo(
                        "Sucesso", 
                        f"Modelo '{nome}' migrado com sucesso!\nID: {model_id}",
                        parent=self.dialog
                    )
                except Exception as e:
                    messagebox.showerror("Erro", f"Erro ao migrar JSON: {e}", parent=self.dialog)
    
    def on_cancel(self):
        """Cancela o diálogo."""
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        """Mostra o diálogo e retorna o resultado."""
        self.dialog.wait_window()
        return self.result


class SaveModelDialog:
    """Diálogo para salvar modelos."""
    
    def __init__(self, parent, db_manager: DatabaseManager, current_model_id=None):
        self.parent = parent
        self.db_manager = db_manager
        self.current_model_id = current_model_id
        self.result = None
        
        # Carrega as configurações de estilo
        self.style_config = load_style_config()
        
        self.dialog = ttk.Toplevel(parent)
        self.dialog.title("Salvar Modelo")
        self.dialog.geometry("450x250")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Centraliza o diálogo
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (450 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (250 // 2)
        self.dialog.geometry(f"450x250+{x}+{y}")
        
        self.setup_ui()
        
        # Configura fechamento
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
    
    def setup_ui(self):
        """Configura a interface do diálogo."""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Se há modelo atual, mostra opções
        if self.current_model_id:
            try:
                modelo = self.db_manager.load_modelo(self.current_model_id)
                
                title_label = ttk.Label(main_frame, 
                                       text=f"Salvar alterações no modelo '{modelo['nome']}'?",
                                       font=get_font('header_font'))
                title_label.pack(pady=(0, 25))
                
                # Botões para modelo existente com fontes maiores
                btn_frame = ttk.Frame(main_frame)
                btn_frame.pack(fill=X, pady=15)
                
                style = ttk.Style()
                style.configure('BigSave.TButton', font=get_font('small_font'))
                
                ttk.Button(btn_frame, text="Sobrescrever", 
                          command=self.on_overwrite, 
                          bootstyle=PRIMARY,
                          style='BigSave.TButton').pack(side=LEFT, padx=(0, 15))
                
                ttk.Button(btn_frame, text="Salvar Como...", 
                          command=self.on_save_as, 
                          bootstyle=SUCCESS,
                          style='BigSave.TButton').pack(side=LEFT, padx=15)
                
                ttk.Button(btn_frame, text="Cancelar", 
                          command=self.on_cancel,
                          style='BigSave.TButton').pack(side=RIGHT)
                
            except Exception:
                # Se erro ao carregar modelo atual, trata como novo
                self.current_model_id = None
                self.setup_new_model_ui(main_frame)
        else:
            self.setup_new_model_ui(main_frame)
    
    def setup_new_model_ui(self, parent_frame):
        """Configura UI para novo modelo."""
        title_label = ttk.Label(parent_frame, text="Salvar Novo Modelo",
                               font=get_font('header_font'))
        title_label.pack(pady=(0, 25))
        
        # Campo para nome com fonte maior
        ttk.Label(parent_frame, text="Nome do modelo:", 
                 font=get_font('small_font')).pack(anchor='w')
        
        self.name_var = ttk.StringVar()
        name_entry = ttk.Entry(parent_frame, textvariable=self.name_var, 
                              width=40, font=get_font('small_font'))
        name_entry.pack(fill=X, pady=(8, 25))
        name_entry.focus()
        
        # Botões com fontes maiores
        btn_frame = ttk.Frame(parent_frame)
        btn_frame.pack(fill=X)
        
        style = ttk.Style()
        style.configure('BigSave.TButton', font=get_font('small_font'))
        
        ttk.Button(btn_frame, text="Salvar", 
                  command=self.on_save_new, 
                  bootstyle=PRIMARY,
                  style='BigSave.TButton').pack(side=LEFT)
        
        ttk.Button(btn_frame, text="Cancelar", 
                  command=self.on_cancel,
                  style='BigSave.TButton').pack(side=RIGHT)
        
        # Bind Enter para salvar
        name_entry.bind('<Return>', lambda e: self.on_save_new())

    def on_overwrite(self):
        """Sobrescreve o modelo atual."""
        self.result = {'action': 'overwrite', 'model_id': self.current_model_id}
        self.dialog.destroy()

    def on_save_as(self):
        """Salva como novo modelo."""
        nome = simpledialog.askstring(
            "Salvar Como",
            "Digite o nome do novo modelo:",
            parent=self.dialog
        )
        
        if nome:
            nome = nome.strip()
            if not nome:
                messagebox.showwarning("Aviso", "Nome não pode estar vazio.", parent=self.dialog)
                return
            
            self.result = {'action': 'save_as', 'name': nome}
            self.dialog.destroy()

    def on_save_new(self):
        """Salva novo modelo."""
        nome = self.name_var.get().strip()
        
        if not nome:
            messagebox.showwarning("Aviso", "Nome não pode estar vazio.", parent=self.dialog)
            return
        
        self.result = {'action': 'save_new', 'name': nome}
        self.dialog.destroy()

    def on_cancel(self):
        """Cancela o diálogo."""
        self.result = None
        self.dialog.destroy()

    def show(self):
        """Mostra o diálogo e retorna o resultado."""
        self.dialog.wait_window()
        return self.result