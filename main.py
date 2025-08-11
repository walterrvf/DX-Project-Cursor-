#!/usr/bin/env python3
"""
Arquivo principal para executar o Sistema de Inspeção Visual.
Abre diretamente o módulo montagem.py com todas as funcionalidades.
"""

import sys
import os
from pathlib import Path

# Adicionar diretório modulos ao path
modulos_dir = Path(__file__).parent / "modulos"
sys.path.insert(0, str(modulos_dir))

def main():
    """Executa o sistema principal."""
    try:
        # Importar e executar o módulo montagem
        from montagem import create_main_window
        
        print("🚀 Iniciando Sistema de Inspeção Visual...")
        print("📝 Carregando módulo principal: montagem.py")
        
        # Criar e executar a janela principal
        root = create_main_window()
        root.mainloop()
        
    except ImportError as e:
        print(f"❌ Erro ao importar módulos: {e}")
        print("Verifique se todos os módulos estão instalados:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro ao executar aplicação: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


