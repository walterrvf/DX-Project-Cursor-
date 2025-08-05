import subprocess
import sys

def check_and_install_dependencies():
    """
    Verifica e instala automaticamente as dependências do requirements.txt
    """
    try:
        # Ler as dependências do requirements.txt
        with open('requirements.txt', 'r') as f:
            dependencies = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print("Verificando dependências instaladas...")
        
        # Verificar cada dependência
        for dep in dependencies:
            try:
                # Extrair nome do pacote (removendo especificações de versão)
                package_name = dep.split('==')[0].split('>')[0].split('<')[0].split('~')[0]
                
                # Verificar se o pacote está instalado
                __import__(package_name)
                print(f"✓ {package_name} já está instalado")
            except ImportError:
                print(f"✗ {package_name} não encontrado. Instalando...")
                # Instalar a dependência exatamente como especificada
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"✓ {package_name} instalado com sucesso")
        
        print("Todas as dependências estão instaladas!")
    except FileNotFoundError:
        print("Erro: arquivo requirements.txt não encontrado.")
    except Exception as e:
        print(f"Erro durante a instalação: {str(e)}")

if __name__ == "__main__":
    check_and_install_dependencies()