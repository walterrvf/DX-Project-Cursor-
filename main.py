#!/usr/bin/env python3
"""
Arquivo principal para executar o Sistema de Inspeção Visual.
Abre diretamente o módulo montagem.py com todas as funcionalidades.
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import shutil

# Determina diretórios base em execução normal vs congelada (PyInstaller)
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
    EXE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent
    EXE_DIR = BASE_DIR

# Garante que o pacote `modulos` esteja no sys.path tanto no dev quanto no frozen
sys.path.insert(0, str((BASE_DIR / "modulos").resolve()))

def _setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    try:
        # Log em arquivo ao lado do executável (ou na raiz do projeto no dev)
        log_path = (EXE_DIR / "run.log").resolve()
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(fh)
    except Exception:
        # Se não conseguir criar arquivo de log, continua apenas com stdout/console
        pass

def _bootstrap_bundle_assets(logger: logging.Logger) -> None:
    """Copia diretórios de dados do bundle (PyInstaller) para a pasta do executável, se necessário.

    - Garante que `assets`, `config` e `modelos` existam ao lado do .exe
    - Em execução não congelada, não faz nada
    """
    try:
        if not getattr(sys, 'frozen', False):
            return
        # Importa tardiamente para evitar dependência circular
        exe_root = EXE_DIR
        src_base = BASE_DIR
        # Sempre garantir diretórios básicos
        for dirname in ["assets", "config", "modelos"]:
            dst_dir = exe_root / dirname
            if not dst_dir.exists():
                src_dir = src_base / dirname
                if src_dir.exists():
                    try:
                        shutil.copytree(src_dir, dst_dir)
                        logger.info(f"Copiado '{dirname}' do bundle para '{dst_dir}'.")
                    except Exception as copy_err:
                        logger.warning(f"Não foi possível copiar '{dirname}': {copy_err}")
                else:
                    # Se não existir no bundle, apenas cria pasta vazia
                    dst_dir.mkdir(parents=True, exist_ok=True)
        # Garante existência do banco/pastas graváveis
        (exe_root / "modelos").mkdir(parents=True, exist_ok=True)
        (exe_root / "config").mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Falha ao preparar dados do bundle: {e}")

def main():
    """Executa o sistema principal."""
    # Parser de argumentos
    parser = argparse.ArgumentParser(description="Sistema de Inspeção Visual")
    parser.add_argument("--debug", action="store_true", help="Ativa modo debug (logs verbosos)")
    args, _ = parser.parse_known_args()

    _setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)

    try:
        _bootstrap_bundle_assets(logger)
        # Importar e executar o módulo montagem
        from montagem import create_main_window
        
        logger.info("🚀 Iniciando Sistema de Inspeção Visual...")
        logger.info("📝 Carregando módulo principal: montagem.py")
        
        # Criar e executar a janela principal
        root = create_main_window()
        root.mainloop()
        
    except ImportError as e:
        logger.exception(f"❌ Erro ao importar módulos: {e}")
        logger.error("Verifique se todos os módulos estão instalados: 'pip install -r requirements.txt'")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Erro ao executar aplicação: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


