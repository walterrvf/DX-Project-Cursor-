import sqlite3
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from pathlib import Path
import logging
try:
	from .paths import get_project_root as _get_project_root
except Exception:
	try:
		from paths import get_project_root as _get_project_root
	except Exception:
		def _get_project_root():
			return Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gerenciador do banco de dados SQLite para o sistema de inspeção visual."""
    
    def __init__(self, db_path: str = None):
        # Usa raiz do projeto compatível com modo congelado (PyInstaller)
        try:
            self.project_root = _get_project_root()
        except Exception:
            self.project_root = Path(__file__).resolve().parent.parent
        # Se nenhum caminho for fornecido, usa o caminho padrão na raiz do projeto
        if db_path is None:
            db_path = str(self.project_root / "modelos" / "models.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Inicializa o banco de dados criando as tabelas necessárias."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabela modelos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS modelos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome TEXT UNIQUE NOT NULL,
                    image_path TEXT NOT NULL,
                    camera_index INTEGER DEFAULT 0,
                    criado_em TEXT NOT NULL,
                    atualizado_em TEXT NOT NULL
                )
            """)
            
            # Tabela slots
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS slots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    modelo_id INTEGER NOT NULL,
                    slot_id INTEGER NOT NULL,
                    tipo TEXT NOT NULL,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    w INTEGER NOT NULL,
                    h INTEGER NOT NULL,
                    cor_r INTEGER DEFAULT 0,
                    cor_g INTEGER DEFAULT 0,
                    cor_b INTEGER DEFAULT 255,
                    h_tolerance INTEGER DEFAULT 10,
                    s_tolerance INTEGER DEFAULT 50,
                    v_tolerance INTEGER DEFAULT 50,
                    detection_threshold REAL DEFAULT 0.8,
                    correlation_threshold REAL DEFAULT 0.5,
                    template_method TEXT DEFAULT 'TM_CCOEFF_NORMED',
                    scale_tolerance REAL DEFAULT 0.5,
                    template_path TEXT,
                    detection_method TEXT DEFAULT 'template_matching',
                    shape TEXT DEFAULT 'rectangle',
                    rotation REAL DEFAULT 0,
                    FOREIGN KEY(modelo_id) REFERENCES modelos(id) ON DELETE CASCADE
                )
            """)
            
            # Tabela de histórico de inspeções
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspection_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    modelo_id INTEGER NOT NULL,
                    modelo_nome TEXT NOT NULL,
                    slot_id INTEGER NOT NULL,
                    result TEXT NOT NULL,  -- 'ok' ou 'ng'
                    confidence REAL NOT NULL,
                    processing_time REAL,
                    image_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(modelo_id) REFERENCES modelos(id) ON DELETE CASCADE
                )
            """)
            
            # Adiciona colunas shape, rotation e ok_threshold se não existirem (para compatibilidade)
            try:
                cursor.execute("ALTER TABLE slots ADD COLUMN shape TEXT DEFAULT 'rectangle'")
            except sqlite3.OperationalError:
                pass  # Coluna já existe
            
            try:
                cursor.execute("ALTER TABLE slots ADD COLUMN rotation REAL DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Coluna já existe
                
            try:
                cursor.execute("ALTER TABLE slots ADD COLUMN ok_threshold INTEGER DEFAULT 70")
            except sqlite3.OperationalError:
                pass  # Coluna já existe
                
            # Adiciona colunas ML se não existirem (para compatibilidade)
            try:
                cursor.execute("ALTER TABLE slots ADD COLUMN use_ml INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Coluna já existe
                
            try:
                cursor.execute("ALTER TABLE slots ADD COLUMN ml_model_path TEXT")
            except sqlite3.OperationalError:
                pass  # Coluna já existe

            # Adiciona coluna para controle de alinhamento por slot (fixo vs autoajuste)
            try:
                cursor.execute("ALTER TABLE slots ADD COLUMN use_alignment INTEGER DEFAULT 1")
            except sqlite3.OperationalError:
                pass  # Coluna já existe
            
            # Índices para melhor performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_slots_modelo_id ON slots(modelo_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_slots_slot_id ON slots(slot_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspection_history_modelo_id ON inspection_history(modelo_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspection_history_created_at ON inspection_history(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspection_history_result ON inspection_history(result)")
            
            conn.commit()
            logger.info("Banco de dados inicializado com sucesso")
            
            # Garantir coluna camera_index para compatibilidade retroativa
            try:
                cursor.execute("ALTER TABLE modelos ADD COLUMN camera_index INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            
            # Verifica e corrige caminhos absolutos no banco de dados
            try:
                self.fix_absolute_paths()
            except Exception as e:
                logger.warning(f"Não foi possível corrigir caminhos absolutos: {e}")

            # Bootstrap automático de modelos a partir de pastas existentes se a tabela estiver vazia
            try:
                cursor.execute("SELECT COUNT(1) FROM modelos")
                count = int(cursor.fetchone()[0] or 0)
                if count == 0:
                    self._bootstrap_models_from_folders(cursor)
                    conn.commit()
            except Exception as e:
                print(f"Aviso: Não foi possível inicializar modelos a partir das pastas: {e}")
    
    def save_modelo(self, nome: str, image_path: str, slots: List[Dict], camera_index: int = 0) -> int:
        """Salva um modelo completo no banco de dados.
        
        Args:
            nome: Nome único do modelo
            image_path: Caminho da imagem de referência
            slots: Lista de dicionários com dados dos slots
            camera_index: Índice padrão da câmera associado ao modelo
            
        Returns:
            ID do modelo criado
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            # Converte caminho absoluto para relativo se necessário
            rel_image_path = self._convert_to_relative_path(image_path)
            
            try:
                # Insere o modelo
                cursor.execute("""
                    INSERT INTO modelos (nome, image_path, camera_index, criado_em, atualizado_em)
                    VALUES (?, ?, ?, ?, ?)
                """, (nome, rel_image_path, int(camera_index or 0), now, now))
                
                modelo_id = cursor.lastrowid
                
                # Cria pasta específica para o modelo
                self._create_model_folder(nome, modelo_id)
                
                # Insere os slots
                for slot in slots:
                    self._insert_slot(cursor, modelo_id, slot)
                
                conn.commit()
                print(f"Modelo '{nome}' salvo com ID {modelo_id} e {len(slots)} slots")
                return modelo_id
                
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    raise ValueError(f"Já existe um modelo com o nome '{nome}'")
                raise
    
    def update_modelo(self, modelo_id: int, nome: str = None, image_path: str = None, slots: List[Dict] = None, camera_index: int = None):
        """Atualiza um modelo existente.
        
        Args:
            modelo_id: ID do modelo a ser atualizado
            nome: Novo nome (opcional)
            image_path: Novo caminho da imagem (opcional)
            slots: Nova lista de slots (opcional, substitui todos os slots existentes)
            camera_index: Atualiza índice da câmera (opcional)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Verifica se o modelo existe
            cursor.execute("SELECT id FROM modelos WHERE id = ?", (modelo_id,))
            if not cursor.fetchone():
                raise ValueError(f"Modelo com ID {modelo_id} não encontrado")
            
            now = datetime.now().isoformat()
            
            # Atualiza campos do modelo se fornecidos
            if nome is not None or image_path is not None or camera_index is not None:
                updates = []
                params = []
                
                if nome is not None:
                    updates.append("nome = ?")
                    params.append(nome)
                
                if image_path is not None:
                    # Converte caminho para relativo
                    rel_image_path = self._convert_to_relative_path(image_path)
                    updates.append("image_path = ?")
                    params.append(rel_image_path)
                
                if camera_index is not None:
                    updates.append("camera_index = ?")
                    params.append(int(camera_index))
                
                updates.append("atualizado_em = ?")
                params.append(now)
                params.append(modelo_id)
                
                cursor.execute(f"""
                    UPDATE modelos SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
            
            # Atualiza slots se fornecidos
            if slots is not None:
                # Remove todos os slots existentes
                cursor.execute("DELETE FROM slots WHERE modelo_id = ?", (modelo_id,))
                
                # Insere novos slots
                for slot in slots:
                    self._insert_slot(cursor, modelo_id, slot)
            
            conn.commit()
            print(f"Modelo {modelo_id} atualizado com sucesso")
    
    def load_modelo(self, modelo_id: int) -> Dict:
        """Carrega um modelo completo do banco de dados.
        
        Args:
            modelo_id: ID do modelo
            
        Returns:
            Dicionário com dados do modelo e slots
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Carrega dados do modelo
            cursor.execute("""
                SELECT nome, image_path, camera_index, criado_em, atualizado_em
                FROM modelos WHERE id = ?
            """, (modelo_id,))
            
            modelo_row = cursor.fetchone()
            if not modelo_row:
                raise ValueError(f"Modelo com ID {modelo_id} não encontrado")
            
            nome, image_path, camera_index, criado_em, atualizado_em = modelo_row
            
            # Converte caminho relativo para absoluto
            abs_image_path = self._convert_to_absolute_path(image_path)
            
            # Carrega slots do modelo
            cursor.execute("""
                SELECT slot_id, tipo, x, y, w, h, cor_r, cor_g, cor_b,
                       h_tolerance, s_tolerance, v_tolerance,
                       detection_threshold, correlation_threshold,
                       template_method, scale_tolerance, template_path,
                       detection_method, shape, rotation, ok_threshold,
                       use_ml, ml_model_path, use_alignment
                FROM slots WHERE modelo_id = ?
                ORDER BY slot_id
            """, (modelo_id,))
            
            slots = []
            for row in cursor.fetchall():
                # Converte caminho do template para absoluto se existir
                template_path = row[16]
                if template_path:
                    template_path = self._convert_to_absolute_path(template_path)
                
                # Converte caminho do modelo ML para absoluto se existir
                ml_model_path = row[22] if len(row) > 22 and row[22] else None
                if ml_model_path:
                    ml_model_path = self._convert_to_absolute_path(ml_model_path)
                
                slot = {
                    'id': row[0],
                    'tipo': row[1],
                    'x': row[2],
                    'y': row[3],
                    'w': row[4],
                    'h': row[5],
                    'cor': [row[6], row[7], row[8]],
                    'h_tolerance': row[9],
                    's_tolerance': row[10],
                    'v_tolerance': row[11],
                    'detection_threshold': row[12],
                    'correlation_threshold': row[13],
                    'template_method': row[14],
                    'scale_tolerance': row[15],
                    'template_path': template_path,
                    'detection_method': row[17],
                    'shape': row[18] if len(row) > 18 and row[18] else 'rectangle',
                    'rotation': row[19] if len(row) > 19 and row[19] is not None else 0,
                    'ok_threshold': row[20] if len(row) > 20 and row[20] is not None else 70,
                    'use_ml': bool(row[21]) if len(row) > 21 and row[21] is not None else False,
                    'ml_model_path': ml_model_path,
                    'use_alignment': bool(row[23]) if len(row) > 23 and row[23] is not None else True
                }
                slots.append(slot)
            
            return {
                'id': modelo_id,
                'nome': nome,
                'image_path': abs_image_path,
                'slots': slots,
                'camera_index': int(camera_index or 0),
                'criado_em': criado_em,
                'atualizado_em': atualizado_em
            }
    
    def get_model_by_id(self, modelo_id: int) -> Dict:
        """Retorna informações básicas de um modelo pelo ID.
        
        Args:
            modelo_id: ID do modelo
            
        Returns:
            Dicionário com informações básicas do modelo
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, nome, image_path, camera_index, criado_em, atualizado_em
                FROM modelos WHERE id = ?
            """, (modelo_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                'id': row[0],
                'nome': row[1],
                'image_path': row[2],
                'camera_index': int(row[3] or 0),
                'criado_em': row[4],
                'atualizado_em': row[5]
            }
    
    def load_modelo_by_name(self, nome: str) -> Dict:
        """Carrega um modelo pelo nome.
        
        Args:
            nome: Nome do modelo
            
        Returns:
            Dicionário com dados do modelo e slots
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM modelos WHERE nome = ?", (nome,))
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Modelo '{nome}' não encontrado")
            
            return self.load_modelo(row[0])
    
    def list_modelos(self) -> List[Dict]:
        """Lista todos os modelos disponíveis.
        
        Returns:
            Lista de dicionários com informações básicas dos modelos
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Verifica tabelas existentes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {name for (name,) in cursor.fetchall()}
            
            modelos: List[Dict] = []
            if 'modelos' in tables:
                cursor.execute(
                    """
                    SELECT m.id, m.nome, m.image_path, m.camera_index, m.criado_em, m.atualizado_em,
                           COUNT(s.id) as num_slots
                    FROM modelos m
                    LEFT JOIN slots s ON m.id = s.modelo_id
                    GROUP BY m.id, m.nome, m.image_path, m.camera_index, m.criado_em, m.atualizado_em
                    ORDER BY m.atualizado_em DESC
                    """
                )
                for row in cursor.fetchall():
                    modelos.append({
                        'id': row[0],
                        'nome': row[1],
                        'image_path': row[2],
                        'camera_index': int(row[3] or 0),
                        'criado_em': row[4],
                        'atualizado_em': row[5],
                        'num_slots': row[6],
                    })
                
                # Se a tabela existir mas estiver vazia e houver legado, tenta fallback
                if modelos or 'models' not in tables:
                    # Se vazio, tenta bootstrap a partir de pastas existentes
                    if not modelos:
                        try:
                            self._bootstrap_models_from_folders(cursor)
                            conn.commit()
                            cursor.execute(
                                """
                                SELECT m.id, m.nome, m.image_path, m.camera_index, m.criado_em, m.atualizado_em,
                                       COUNT(s.id) as num_slots
                                FROM modelos m
                                LEFT JOIN slots s ON m.id = s.modelo_id
                                GROUP BY m.id, m.nome, m.image_path, m.camera_index, m.criado_em, m.atualizado_em
                                ORDER BY m.atualizado_em DESC
                                """
                            )
                            modelos = [{
                                'id': r[0], 'nome': r[1], 'image_path': r[2], 'camera_index': int(r[3] or 0),
                                'criado_em': r[4], 'atualizado_em': r[5], 'num_slots': r[6]
                            } for r in cursor.fetchall()]
                        except Exception:
                            pass
                    return modelos
            
            # Fallback para esquemas legados que usam tabela 'models'
            if 'models' in tables:
                try:
                    cursor.execute(
                        """
                        SELECT m.id, m.nome, m.image_path, m.criado_em, m.atualizado_em
                        FROM models m
                        ORDER BY m.atualizado_em DESC
                        """
                    )
                    legacy_rows = cursor.fetchall()
                    # Verifica se existe tabela slots compatível
                    has_slots = 'slots' in tables
                    for row in legacy_rows:
                        model_id = row[0]
                        num_slots = 0
                        if has_slots:
                            try:
                                cursor.execute("SELECT COUNT(1) FROM slots WHERE modelo_id = ?", (model_id,))
                                num_slots = int(cursor.fetchone()[0] or 0)
                            except Exception:
                                num_slots = 0
                        modelos.append({
                            'id': model_id,
                            'nome': row[1],
                            'image_path': row[2],
                            'criado_em': row[3],
                            'atualizado_em': row[4],
                            'num_slots': num_slots,
                        })
                except Exception:
                    pass
            
            return modelos

    def _bootstrap_models_from_folders(self, cursor):
        """Cria registros na tabela 'modelos' com base nas pastas já existentes em 'modelos/'.
        Não cria novas pastas nem move arquivos; apenas referencia a imagem _reference existente.
        """
        try:
            models_dir: Path = self.db_path.parent
            if not models_dir.exists():
                return
            # Pastas candidatas: qualquer subpasta com padrão base_sufixo e contendo *_reference.(jpg|png)
            for sub in models_dir.iterdir():
                if not sub.is_dir():
                    continue
                name = sub.name
                if '_' not in name:
                    continue
                base = name.split('_')[0]
                # Pula pastas reservadas
                if base in {"_templates", "_samples", "historico_fotos"}:
                    continue
                # Já existe um modelo com este nome?
                cursor.execute("SELECT id FROM modelos WHERE nome = ?", (base,))
                if cursor.fetchone():
                    continue
                # Procura imagem de referência
                ref = None
                for ext in ("jpg", "png", "jpeg", "bmp"):
                    candidate = sub / f"{base}_reference.{ext}"
                    if candidate.exists():
                        ref = candidate
                        break
                if not ref:
                    # tenta qualquer *_reference.*
                    for p in sub.glob("*_reference.*"):
                        ref = p
                        break
                if not ref:
                    continue
                # Caminho relativo ao projeto
                rel_path = str((self.project_root / "modelos" / name / ref.name).relative_to(self.project_root)).replace('\\', '/')
                now = datetime.now().isoformat()
                try:
                    cursor.execute(
                        """
                        INSERT INTO modelos (nome, image_path, criado_em, atualizado_em)
                        VALUES (?, ?, ?, ?)
                        """,
                        (base, rel_path, now, now)
                    )
                    # não cria slots aqui
                    # não cria pastas novas aqui
                    # apenas referencia pastas existentes
                    print(f"Modelo bootstrap criado a partir da pasta: nome='{base}', ref='{rel_path}'")
                except Exception as e:
                    print(f"Falha ao bootstrap modelo da pasta {name}: {e}")
        except Exception:
            pass
    
    def delete_modelo(self, modelo_id: int):
        """Remove um modelo e todos os seus slots.
        
        Args:
            modelo_id: ID do modelo a ser removido
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Verifica se o modelo existe
            cursor.execute("SELECT nome FROM modelos WHERE id = ?", (modelo_id,))
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Modelo com ID {modelo_id} não encontrado")
            
            nome = row[0]
            
            # Remove a pasta do modelo antes de deletar do banco
            self._delete_model_folder(nome, modelo_id)
            
            # Remove o modelo (slots são removidos automaticamente por CASCADE)
            cursor.execute("DELETE FROM modelos WHERE id = ?", (modelo_id,))
            
            conn.commit()
            print(f"Modelo '{nome}' (ID {modelo_id}) removido com sucesso")
    
    def _create_model_folder(self, nome: str, modelo_id: int):
        """Cria pasta específica para o modelo.
        
        Args:
            nome: Nome do modelo
            modelo_id: ID do modelo
        """
        try:
            # Define o caminho da pasta do modelo
            models_dir = Path(self.db_path).parent
            model_folder = models_dir / f"{nome}_{modelo_id}"
            
            # Cria a pasta se não existir
            model_folder.mkdir(exist_ok=True)
            
            # Cria subpasta para templates se necessário
            templates_folder = model_folder / "templates"
            templates_folder.mkdir(exist_ok=True)
            
            print(f"Pasta criada para modelo '{nome}': {model_folder}")
            
        except Exception as e:
            print(f"Erro ao criar pasta para modelo '{nome}': {e}")
    
    def _delete_model_folder(self, nome: str, modelo_id: int):
        """Remove pasta específica do modelo.
        
        Args:
            nome: Nome do modelo
            modelo_id: ID do modelo
        """
        import time
        
        # Define o caminho da pasta do modelo
        models_dir = Path(self.db_path).parent
        model_folder = models_dir / f"{nome}_{modelo_id}"
        
        if not model_folder.exists():
            print(f"Pasta não encontrada para modelo '{nome}': {model_folder}")
            return
        
        # Estratégia de remoção robusta
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Primeiro, tenta remover atributos somente leitura
                self._remove_readonly_attributes(model_folder)
                
                # Tenta remover a pasta
                shutil.rmtree(model_folder)
                print(f"Pasta removida para modelo '{nome}': {model_folder}")
                return
                
            except PermissionError as e:
                print(f"Tentativa {attempt + 1}/{max_attempts} - Erro de permissão: {e}")
                
                if attempt < max_attempts - 1:
                    # Aguarda um pouco antes da próxima tentativa
                    time.sleep(1)
                    
                    # Tenta forçar a liberação de handles de arquivo
                    try:
                        import gc
                        gc.collect()
                    except:
                        pass
                else:
                    print(f"❌ Não foi possível remover a pasta após {max_attempts} tentativas.")
                    print(f"   Isso pode ser devido a:")
                    print(f"   - Arquivos em uso por outros programas")
                    print(f"   - Sincronização do OneDrive")
                    print(f"   - Permissões de sistema")
                    print(f"   A pasta permanecerá como 'órfã': {model_folder}")
                    
            except Exception as e:
                print(f"Erro inesperado ao remover pasta para modelo '{nome}': {e}")
                break
    
    def _convert_to_relative_path(self, path_str):
        """Converte um caminho absoluto para relativo à raiz do projeto.
        
        Args:
            path_str: Caminho absoluto ou relativo
            
        Returns:
            Caminho relativo à raiz do projeto com separadores normalizados
        """
        if not path_str:
            return path_str
            
        path = Path(path_str)
        
        # Se já é um caminho relativo, normaliza os separadores e retorna
        if not path.is_absolute():
            # Normaliza separadores para '/' (padrão esperado pelo banco)
            return str(path).replace('\\', '/')
            
        try:
            # Tenta converter para relativo à raiz do projeto
            rel_path = path.relative_to(self.project_root)
            # Normaliza separadores para '/' (padrão esperado pelo banco)
            return str(rel_path).replace('\\', '/')
        except ValueError:
            # Se não for possível, mantém o caminho original
            return path_str
    
    def _convert_to_absolute_path(self, path_str):
        """Converte um caminho relativo para absoluto baseado na raiz do projeto.
        
        Args:
            path_str: Caminho relativo ou absoluto
            
        Returns:
            Caminho absoluto
        """
        if not path_str:
            return path_str
            
        path = Path(path_str)
        
        # Se já é um caminho absoluto, retorna como está
        if path.is_absolute():
            return path_str
            
        # Converte para absoluto baseado na raiz do projeto
        abs_path = self.project_root / path
        return str(abs_path)
    
    def _remove_readonly_attributes(self, folder_path: Path):
        """Remove atributos somente leitura de arquivos e pastas.
        
        Args:
            folder_path: Caminho da pasta
        """
        import stat
        
        try:
            for item in folder_path.rglob("*"):
                try:
                    # Remove atributo somente leitura
                    item.chmod(stat.S_IWRITE | stat.S_IREAD)
                except:
                    pass
        except Exception:
            # Se não conseguir remover atributos, continua mesmo assim
            pass
    
    def get_model_folder_path(self, nome: str, modelo_id: int) -> Path:
        """Retorna o caminho da pasta específica do modelo.
        
        Args:
            nome: Nome do modelo
            modelo_id: ID do modelo
            
        Returns:
            Path da pasta do modelo
        """
        models_dir = Path(self.db_path).parent
        return models_dir / f"{nome}_{modelo_id}"
    
    def update_slot(self, modelo_id: int, slot_data: Dict):
        """Atualiza um slot específico.
        
        Args:
            modelo_id: ID do modelo
            slot_data: Dicionário com dados do slot
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            slot_id = slot_data['id']
            
            # Verifica se o slot existe
            cursor.execute("""
                SELECT id FROM slots WHERE modelo_id = ? AND slot_id = ?
            """, (modelo_id, slot_id))
            
            if cursor.fetchone():
                # Atualiza slot existente
                self._update_slot_data(cursor, modelo_id, slot_data)
            else:
                # Insere novo slot
                self._insert_slot(cursor, modelo_id, slot_data)
            
            # Atualiza timestamp do modelo
            cursor.execute("""
                UPDATE modelos SET atualizado_em = ? WHERE id = ?
            """, (datetime.now().isoformat(), modelo_id))
            
            conn.commit()
    
    def delete_slot(self, modelo_id: int, slot_id: int):
        """Remove um slot específico.
        
        Args:
            modelo_id: ID do modelo
            slot_id: ID do slot
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM slots WHERE modelo_id = ? AND slot_id = ?
            """, (modelo_id, slot_id))
            
            if cursor.rowcount == 0:
                raise ValueError(f"Slot {slot_id} não encontrado no modelo {modelo_id}")
            
            # Atualiza timestamp do modelo
            cursor.execute("""
                UPDATE modelos SET atualizado_em = ? WHERE id = ?
            """, (datetime.now().isoformat(), modelo_id))
            
            conn.commit()
            print(f"Slot {slot_id} removido do modelo {modelo_id}")
    
    def _insert_slot(self, cursor, modelo_id: int, slot_data: Dict):
        """Insere um slot no banco de dados."""
        cor = slot_data.get('cor', [0, 0, 255])
        
        # Converte caminho do template para relativo se existir
        template_path = slot_data.get('template_path')
        if template_path:
            template_path = self._convert_to_relative_path(template_path)
            
        # Converte caminho do modelo ML para relativo se existir
        ml_model_path = slot_data.get('ml_model_path')
        if ml_model_path:
            ml_model_path = self._convert_to_relative_path(ml_model_path)
        
        cursor.execute("""
            INSERT INTO slots (
                modelo_id, slot_id, tipo, x, y, w, h,
                cor_r, cor_g, cor_b,
                h_tolerance, s_tolerance, v_tolerance,
                detection_threshold, correlation_threshold,
                template_method, scale_tolerance, template_path,
                detection_method, shape, rotation, ok_threshold,
                use_ml, ml_model_path, use_alignment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            modelo_id,
            slot_data['id'],
            slot_data['tipo'],
            slot_data['x'],
            slot_data['y'],
            slot_data['w'],
            slot_data['h'],
            cor[0], cor[1], cor[2],
            slot_data.get('h_tolerance', 10),
            slot_data.get('s_tolerance', 50),
            slot_data.get('v_tolerance', 50),
            slot_data.get('detection_threshold', 0.8),
            slot_data.get('correlation_threshold', 0.5),
            slot_data.get('template_method', 'TM_CCOEFF_NORMED'),
            slot_data.get('scale_tolerance', 0.1),
            template_path,
            slot_data.get('detection_method', 'template_matching'),
            slot_data.get('shape', 'rectangle'),
            slot_data.get('rotation', 0),
            slot_data.get('ok_threshold', 70),
            1 if slot_data.get('use_ml', False) else 0,
            ml_model_path,
            1 if slot_data.get('use_alignment', True) else 0
        ))
    
    def _update_slot_data(self, cursor, modelo_id: int, slot_data: Dict):
        """Atualiza dados de um slot existente."""
        cor = slot_data.get('cor', [0, 0, 255])
        
        # Converte caminho do template para relativo se existir
        template_path = slot_data.get('template_path')
        if template_path:
            template_path = self._convert_to_relative_path(template_path)
            
        # Converte caminho do modelo ML para relativo se existir
        ml_model_path = slot_data.get('ml_model_path')
        if ml_model_path:
            ml_model_path = self._convert_to_relative_path(ml_model_path)
        
        cursor.execute("""
            UPDATE slots SET
                tipo = ?, x = ?, y = ?, w = ?, h = ?,
                cor_r = ?, cor_g = ?, cor_b = ?,
                h_tolerance = ?, s_tolerance = ?, v_tolerance = ?,
                detection_threshold = ?, correlation_threshold = ?,
                template_method = ?, scale_tolerance = ?, template_path = ?,
                detection_method = ?, shape = ?, rotation = ?, ok_threshold = ?,
                use_ml = ?, ml_model_path = ?, use_alignment = ?
            WHERE modelo_id = ? AND slot_id = ?
        """, (
            slot_data['tipo'],
            slot_data['x'],
            slot_data['y'],
            slot_data['w'],
            slot_data['h'],
            cor[0], cor[1], cor[2],
            slot_data.get('h_tolerance', 10),
            slot_data.get('s_tolerance', 50),
            slot_data.get('v_tolerance', 50),
            slot_data.get('detection_threshold', 0.8),
            slot_data.get('correlation_threshold', 0.5),
            slot_data.get('template_method', 'TM_CCOEFF_NORMED'),
            slot_data.get('scale_tolerance', 0.1),
            template_path,
            slot_data.get('detection_method', 'template_matching'),
            slot_data.get('shape', 'rectangle'),
            slot_data.get('rotation', 0),
            slot_data.get('ok_threshold', 70),
            1 if slot_data.get('use_ml', False) else 0,
            ml_model_path,
            1 if slot_data.get('use_alignment', True) else 0,
            modelo_id,
            slot_data['id']
        ))
    
    def fix_absolute_paths(self):
        """Verifica e corrige caminhos absolutos no banco de dados, convertendo-os para relativos.
        
        Esta função é útil para migrar bancos de dados de versões antigas do sistema que
        usavam caminhos absolutos para versões mais recentes que usam caminhos relativos.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Corrige caminhos de imagens na tabela modelos
            cursor.execute("SELECT id, nome, image_path FROM modelos")
            modelos = cursor.fetchall()
            
            for modelo_id, nome, image_path in modelos:
                # Verifica se o caminho é absoluto
                if image_path and Path(image_path).is_absolute():
                    try:
                        # Tenta converter para relativo
                        rel_path = self._convert_to_relative_path(image_path)
                        
                        # Se o caminho não mudou, tenta criar um novo caminho relativo
                        if rel_path == image_path:
                            # Extrai o nome do arquivo do caminho absoluto
                            filename = Path(image_path).name
                            
                            # Cria um novo caminho relativo baseado no nome do modelo e nome do arquivo
                            new_rel_path = f"modelos/{nome}_{modelo_id}/{filename}"
                            
                            # Verifica se o arquivo existe no novo caminho
                            abs_new_path = self.project_root / new_rel_path
                            if not abs_new_path.exists():
                                # Se o arquivo não existe, tenta um caminho padrão para a imagem de referência
                                new_rel_path = f"modelos/{nome}_{modelo_id}/{nome}_reference.jpg"
                                abs_new_path = self.project_root / new_rel_path
                                if not abs_new_path.exists():
                                    logger.warning(f"Não foi possível encontrar um arquivo de imagem válido para o modelo {nome} (ID {modelo_id})")
                                    continue
                            
                            rel_path = new_rel_path
                        
                        # Atualiza no banco de dados
                        cursor.execute(
                            "UPDATE modelos SET image_path = ? WHERE id = ?",
                            (rel_path, modelo_id)
                        )
                        logger.info(f"Caminho de imagem corrigido para modelo ID {modelo_id}: {image_path} -> {rel_path}")
                    except Exception as e:
                        logger.error(f"Erro ao converter caminho de imagem para modelo ID {modelo_id}: {e}")
            
            # Corrige caminhos de templates na tabela slots
            cursor.execute("SELECT id, modelo_id, slot_id, template_path FROM slots WHERE template_path IS NOT NULL")
            slots = cursor.fetchall()
            
            for slot_id, modelo_id, slot_num, template_path in slots:
                # Verifica se o caminho é absoluto
                if template_path and Path(template_path).is_absolute():
                    try:
                        # Tenta converter para relativo
                        rel_path = self._convert_to_relative_path(template_path)
                        
                        # Se o caminho não mudou, tenta criar um novo caminho relativo
                        if rel_path == template_path:
                            # Extrai o nome do arquivo do caminho absoluto
                            filename = Path(template_path).name
                            
                            # Cria um novo caminho relativo baseado no nome do arquivo
                            new_rel_path = f"modelos/_templates/{filename}"
                            
                            # Verifica se o arquivo existe no novo caminho
                            abs_new_path = self.project_root / new_rel_path
                            if not abs_new_path.exists():
                                logger.warning(f"Não foi possível encontrar um arquivo de template válido para o slot {slot_num} do modelo ID {modelo_id}")
                                continue
                            
                            rel_path = new_rel_path
                        
                        # Atualiza no banco de dados
                        cursor.execute(
                            "UPDATE slots SET template_path = ? WHERE id = ?",
                            (rel_path, slot_id)
                        )
                        logger.info(f"Caminho de template corrigido para slot {slot_num} do modelo ID {modelo_id}: {template_path} -> {rel_path}")
                    except Exception as e:
                        logger.error(f"Erro ao converter caminho de template para slot {slot_num} do modelo ID {modelo_id}: {e}")
            
            conn.commit()
            print("Verificação e correção de caminhos absolutos concluída")
    
    def migrate_from_json(self, json_file_path: str, modelo_name: str = None) -> int:
        """Migra dados de um arquivo JSON para o banco SQLite.
        
        Args:
            json_file_path: Caminho do arquivo JSON
            modelo_name: Nome do modelo (se não fornecido, usa o nome do arquivo)
            
        Returns:
            ID do modelo criado
        """
        json_path = Path(json_file_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Arquivo JSON não encontrado: {json_file_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if modelo_name is None:
            modelo_name = json_path.stem
        
        # Verifica se já existe um modelo com este nome
        try:
            existing = self.load_modelo_by_name(modelo_name)
            raise ValueError(f"Já existe um modelo com o nome '{modelo_name}' (ID {existing['id']})")
        except ValueError as e:
            if "não encontrado" not in str(e):
                raise
        
        # Converte o caminho da imagem para relativo se for absoluto
        image_path = data['image_path']
        
        # Processa os slots para converter caminhos de template
        slots = data['slots']
        for slot in slots:
            if 'template_path' in slot and slot['template_path']:
                # Não precisa converter aqui, pois o método save_modelo já fará isso
                pass
        
        return self.save_modelo(
            nome=modelo_name,
            image_path=image_path,
            slots=slots,
            camera_index=0  # Valor padrão para migração
        )
    
    def log_inspection_result(self, modelo_id: int, modelo_nome: str, slot_id: int, 
                             result: str, confidence: float, processing_time: float = None, 
                             image_path: str = None) -> int:
        """Registra um resultado de inspeção no histórico.
        
        Args:
            modelo_id: ID do modelo inspecionado
            modelo_nome: Nome do modelo inspecionado
            slot_id: ID do slot inspecionado
            result: Resultado da inspeção ('ok' ou 'ng')
            confidence: Nível de confiança da inspeção (0.0 a 1.0)
            processing_time: Tempo de processamento em segundos
            image_path: Caminho da imagem da inspeção
            
        Returns:
            ID do registro criado
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Converte caminho da imagem para relativo se existir
            rel_image_path = None
            if image_path:
                rel_image_path = self._convert_to_relative_path(image_path)
            
            cursor.execute("""
                INSERT INTO inspection_history 
                (modelo_id, modelo_nome, slot_id, result, confidence, processing_time, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (modelo_id, modelo_nome, slot_id, result, confidence, processing_time, rel_image_path))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_inspection_history(self, modelo_id: int = None, modelo_nome: str = None, 
                              result: str = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Recupera o histórico de inspeções com filtros opcionais.
        
        Args:
            modelo_id: Filtrar por ID do modelo
            modelo_nome: Filtrar por nome do modelo
            result: Filtrar por resultado ('ok' ou 'ng')
            limit: Limite de registros retornados
            offset: Offset para paginação
            
        Returns:
            Lista de dicionários com dados das inspeções
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Construir query com filtros
            query = "SELECT * FROM inspection_history WHERE 1=1"
            params = []
            
            if modelo_id is not None:
                query += " AND modelo_id = ?"
                params.append(modelo_id)
            
            if modelo_nome is not None:
                query += " AND modelo_nome = ?"
                params.append(modelo_nome)
            
            if result is not None:
                query += " AND result = ?"
                params.append(result)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                inspection = dict(zip(columns, row))
                # Converte caminho da imagem para absoluto se existir
                if inspection['image_path']:
                    inspection['image_path'] = self._convert_to_absolute_path(inspection['image_path'])
                results.append(inspection)
            
            return results
    
    def get_inspection_statistics(self, modelo_id: int = None, modelo_nome: str = None, 
                                 days: int = 30) -> Dict:
        """Recupera estatísticas das inspeções.
        
        Args:
            modelo_id: Filtrar por ID do modelo
            modelo_nome: Filtrar por nome do modelo
            days: Número de dias para considerar (padrão: 30)
            
        Returns:
            Dicionário com estatísticas das inspeções
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Construir query com filtros
            where_clause = "WHERE created_at >= datetime('now', '-{} days')".format(days)
            params = []
            
            if modelo_id is not None:
                where_clause += " AND modelo_id = ?"
                params.append(modelo_id)
            
            if modelo_nome is not None:
                where_clause += " AND modelo_nome = ?"
                params.append(modelo_nome)
            
            # Total de inspeções
            cursor.execute(f"SELECT COUNT(*) FROM inspection_history {where_clause}", params)
            total_inspections = cursor.fetchone()[0]
            
            # Total de OK
            cursor.execute(f"SELECT COUNT(*) FROM inspection_history {where_clause} AND result = 'ok'", params)
            total_ok = cursor.fetchone()[0]
            
            # Total de NG
            cursor.execute(f"SELECT COUNT(*) FROM inspection_history {where_clause} AND result = 'ng'", params)
            total_ng = cursor.fetchone()[0]
            
            # Taxa de aprovação
            approval_rate = (total_ok / total_inspections * 100) if total_inspections > 0 else 0
            
            # Tempo médio de processamento
            cursor.execute(f"SELECT AVG(processing_time) FROM inspection_history {where_clause} AND processing_time IS NOT NULL", params)
            avg_processing_time = cursor.fetchone()[0] or 0
            
            # Confiança média
            cursor.execute(f"SELECT AVG(confidence) FROM inspection_history {where_clause}", params)
            avg_confidence = cursor.fetchone()[0] or 0
            
            # Inspeções por dia (últimos 7 dias)
            daily_stats = []
            for i in range(7):
                date = f"date('now', '-{i} days')"
                cursor.execute(f"""
                    SELECT date(created_at) as date, 
                           COUNT(*) as total,
                           SUM(CASE WHEN result = 'ok' THEN 1 ELSE 0 END) as ok_count,
                           SUM(CASE WHEN result = 'ng' THEN 1 ELSE 0 END) as ng_count
                    FROM inspection_history 
                    WHERE date(created_at) = {date}
                    {(' AND modelo_id = ?' if modelo_id else '')}
                    {(' AND modelo_nome = ?' if modelo_nome else '')}
                    GROUP BY date(created_at)
                """, params)
                
                row = cursor.fetchone()
                if row:
                    daily_stats.append({
                        'date': row[0],
                        'total': row[1],
                        'ok_count': row[2],
                        'ng_count': row[3]
                    })
                else:
                    daily_stats.append({
                        'date': f"date('now', '-{i} days')",
                        'total': 0,
                        'ok_count': 0,
                        'ng_count': 0
                    })
            
            return {
                'total_inspections': total_inspections,
                'total_ok': total_ok,
                'total_ng': total_ng,
                'approval_rate': round(approval_rate, 2),
                'avg_processing_time': round(avg_processing_time, 3),
                'avg_confidence': round(avg_confidence, 3),
                'daily_stats': daily_stats,
                'period_days': days
            }
    
    def get_model_inspection_summary(self) -> List[Dict]:
        """Obtém um resumo das inspeções por modelo.
        
        Returns:
            Lista de dicionários com resumo por modelo
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    m.nome as modelo_nome,
                    COUNT(i.id) as total_inspections,
                    SUM(CASE WHEN i.resultado = 'OK' THEN 1 ELSE 0 END) as ok_count,
                    SUM(CASE WHEN i.resultado = 'NG' THEN 1 ELSE 0 END) as ng_count,
                    CASE 
                        WHEN COUNT(i.id) > 0 THEN 
                            ROUND(SUM(CASE WHEN i.resultado = 'OK' THEN 1 ELSE 0 END) * 100.0 / COUNT(i.id), 2)
                        ELSE 0 
                    END as approval_rate,
                    MAX(i.timestamp) as last_inspection
                FROM modelos m
                LEFT JOIN inspection_history i ON m.nome = i.modelo_nome
                GROUP BY m.id, m.nome
                ORDER BY m.nome
            """)
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def clear_inspection_history(self, modelo_nome: str = None):
        """Limpa o histórico de inspeções.
        
        Args:
            modelo_nome: Nome do modelo específico para limpar (se None, limpa todos)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if modelo_nome:
                cursor.execute("DELETE FROM inspection_history WHERE modelo_nome = ?", (modelo_nome,))
            else:
                cursor.execute("DELETE FROM inspection_history")
            
            conn.commit()