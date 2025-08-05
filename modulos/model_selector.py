import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox, simpledialog
from datetime import datetime
from pathlib import Path
try:
    # Quando importado como módulo
    from .database_manager import DatabaseManager
<<<<<<< HEAD
    from .utils import load_style_config
except ImportError:
    # Quando executado diretamente
    try:
        from database_manager import DatabaseManager
        from utils import load_style_config
    except ImportError:
        # Quando executado a partir do diretório raiz
        from modulos.database_manager import DatabaseManager
        from modulos.utils import load_style_config
=======
except ImportError:
    # Quando executado diretamente
    from database_manager import DatabaseManager
>>>>>>> d59fc9774a8914a83ec425c781248aed3f221ccd

class ModelSelectorDialog:
    """Diálogo para seleção, criação e gerenciamento de modelos."""
    
    def __init__(self, parent, db_manager: DatabaseManager):
        self.parent = parent
        self.db_manager = db_manager
        self.selected_model = None
        self.result = None
        
        self.dialog = ttk.Toplevel(parent)
        self.dialog.title("Gerenciar Modelos")
        self.dialog.geometry("800x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Centraliza o diálogo
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (600 // 2)
        self.dialog.geometry(f"800x600+{x}+{y}")
        
        self.setup_ui()
        self.refresh_models()
        
        # Configura fechamento
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
    
    def setup_ui(self):
        """Configura a interface do diálogo."""
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Título
<<<<<<< HEAD
        # Carrega as configurações de estilo
        style_config = load_style_config()
=======
        # Importar style_config se necessário
        try:
            import json
            with open('style_config.json', 'r') as f:
                style_config = json.load(f)
        except Exception as e:
            print(f"Erro ao carregar style_config.json: {e}")
            style_config = {"ok_font": ("Arial", 12, "bold")}
>>>>>>> d59fc9774a8914a83ec425c781248aed3f221ccd
            
        title_label = ttk.Label(main_frame, text="Gerenciar Modelos de Inspeção", 
                               font=style_config["ok_font"])
        title_label.pack(pady=(0, 20))
        
        # Frame para lista de modelos
        list_frame = ttk.LabelFrame(main_frame, text="Modelos Disponíveis", padding=10)
        list_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # Treeview para modelos
        columns = ('nome', 'slots', 'criado', 'atualizado')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        # Configurar colunas
        self.tree.heading('nome', text='Nome do Modelo')
        self.tree.heading('slots', text='Slots')
        self.tree.heading('criado', text='Criado em')
        self.tree.heading('atualizado', text='Atualizado em')
        
        self.tree.column('nome', width=200)
        self.tree.column('slots', width=80, anchor=CENTER)
        self.tree.column('criado', width=150)
        self.tree.column('atualizado', width=150)
        
        # Scrollbar para treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Bind para seleção
        self.tree.bind('<<TreeviewSelect>>', self.on_model_select)
        self.tree.bind('<Double-1>', self.on_load_model)
        
        # Frame para botões
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=X, pady=(10, 0))
        
        # Botões da esquerda (ações de modelo)
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=LEFT)
        
        self.btn_new = ttk.Button(left_buttons, text="Novo Modelo", 
                                 command=self.on_new_model, bootstyle=SUCCESS)
        self.btn_new.pack(side=LEFT, padx=(0, 5))
        
        self.btn_rename = ttk.Button(left_buttons, text="Renomear", 
                                    command=self.on_rename_model, state=DISABLED)
        self.btn_rename.pack(side=LEFT, padx=5)
        
        self.btn_delete = ttk.Button(left_buttons, text="Excluir", 
                                    command=self.on_delete_model, 
                                    bootstyle=DANGER, state=DISABLED)
        self.btn_delete.pack(side=LEFT, padx=5)
        
        self.btn_migrate = ttk.Button(left_buttons, text="Migrar JSON", 
                                     command=self.on_migrate_json, bootstyle=INFO)
        self.btn_migrate.pack(side=LEFT, padx=5)
        
        # Botões da direita (ações do diálogo)
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side=RIGHT)
        
        self.btn_load = ttk.Button(right_buttons, text="Carregar Modelo", 
                                  command=self.on_load_model, 
                                  bootstyle=PRIMARY, state=DISABLED)
        self.btn_load.pack(side=LEFT, padx=5)
        
        self.btn_cancel = ttk.Button(right_buttons, text="Cancelar", 
                                    command=self.on_cancel)
        self.btn_cancel.pack(side=LEFT, padx=(5, 0))
        
        # Frame para informações do modelo selecionado
        info_frame = ttk.LabelFrame(main_frame, text="Informações do Modelo", padding=10)
        info_frame.pack(fill=X, pady=(10, 0))
        
        self.info_text = ttk.Text(info_frame, height=4, state=DISABLED)
        self.info_text.pack(fill=X)
    
    def refresh_models(self):
        """Atualiza a lista de modelos."""
        # Limpa a árvore
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Carrega modelos do banco
        try:
            modelos = self.db_manager.list_modelos()
            
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
                
                self.db_manager.update_modelo(self.selected_model, nome=novo_nome)
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
        
<<<<<<< HEAD
        # Carrega as configurações de estilo
        self.style_config = load_style_config()
        
=======
>>>>>>> d59fc9774a8914a83ec425c781248aed3f221ccd
        self.dialog = ttk.Toplevel(parent)
        self.dialog.title("Salvar Modelo")
        self.dialog.geometry("400x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Centraliza o diálogo
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (200 // 2)
        self.dialog.geometry(f"400x200+{x}+{y}")
        
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
<<<<<<< HEAD
                                       font=self.style_config["ok_font"])
=======
                                       font=style_config["ok_font"])
>>>>>>> d59fc9774a8914a83ec425c781248aed3f221ccd
                title_label.pack(pady=(0, 20))
                
                # Botões para modelo existente
                btn_frame = ttk.Frame(main_frame)
                btn_frame.pack(fill=X, pady=10)
                
                ttk.Button(btn_frame, text="Sobrescrever", 
                          command=self.on_overwrite, 
                          bootstyle=PRIMARY).pack(side=LEFT, padx=(0, 10))
                
                ttk.Button(btn_frame, text="Salvar Como...", 
                          command=self.on_save_as, 
                          bootstyle=SUCCESS).pack(side=LEFT, padx=10)
                
                ttk.Button(btn_frame, text="Cancelar", 
                          command=self.on_cancel).pack(side=RIGHT)
                
<<<<<<< HEAD
            except Exception:
=======
            except Exception as e:
>>>>>>> d59fc9774a8914a83ec425c781248aed3f221ccd
                # Se erro ao carregar modelo atual, trata como novo
                self.current_model_id = None
                self.setup_new_model_ui(main_frame)
        else:
            self.setup_new_model_ui(main_frame)
    
    def setup_new_model_ui(self, parent_frame):
        """Configura UI para novo modelo."""
        title_label = ttk.Label(parent_frame, text="Salvar Novo Modelo",
<<<<<<< HEAD
                               font=self.style_config["ok_font"])
=======
                               font=style_config["ok_font"])
>>>>>>> d59fc9774a8914a83ec425c781248aed3f221ccd
        title_label.pack(pady=(0, 20))
        
        # Campo para nome
        ttk.Label(parent_frame, text="Nome do modelo:").pack(anchor=W)
        
        self.name_var = ttk.StringVar()
        name_entry = ttk.Entry(parent_frame, textvariable=self.name_var, width=40)
        name_entry.pack(fill=X, pady=(5, 20))
        name_entry.focus()
        
        # Botões
        btn_frame = ttk.Frame(parent_frame)
        btn_frame.pack(fill=X)
        
        ttk.Button(btn_frame, text="Salvar", 
                  command=self.on_save_new, 
                  bootstyle=PRIMARY).pack(side=LEFT)
        
        ttk.Button(btn_frame, text="Cancelar", 
                  command=self.on_cancel).pack(side=RIGHT)
        
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