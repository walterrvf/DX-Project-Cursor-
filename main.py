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

# Adicionar diretório modulos ao path
modulos_dir = Path(__file__).parent / "modulos"
sys.path.insert(0, str(modulos_dir))

def _setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def main():
    """Executa o sistema principal."""
    # Parser de argumentos
    parser = argparse.ArgumentParser(description="Sistema de Inspeção Visual")
    parser.add_argument("--debug", action="store_true", help="Ativa modo debug (logs verbosos)")
    args, _ = parser.parse_known_args()

    _setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)

    try:
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


