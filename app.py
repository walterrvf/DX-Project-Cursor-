import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
from importlib import import_module
from pathlib import Path

class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Honda Vision System')
        # Configurar para abrir em tela cheia
        self.showMaximized()
        self.setStyleSheet('background-color: white;')
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Logo Honda
        logo_label = QLabel()
        logo_pixmap = QPixmap('assets/honda_logo.svg')
        logo_label.setPixmap(logo_pixmap.scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)
        
        # Grid de módulos
        modules_grid = QGridLayout()
        self.load_modules(modules_grid)
        layout.addLayout(modules_grid)
    
    def load_modules(self, grid):
        modules_dir = Path('modulos')
        if not modules_dir.exists():
            modules_dir.mkdir()
            
        row = 0
        col = 0
        max_cols = 2
        
        # Lista de módulos auxiliares que não devem aparecer no dashboard
        auxiliary_modules = {'database_manager', 'model_selector'}
        
        for module_file in modules_dir.glob('*.py'):
            if module_file.stem == '__init__' or module_file.stem in auxiliary_modules:
                continue
                
            module_name = module_file.stem
            button = QPushButton(module_name.replace('_', ' ').title())
            button.setStyleSheet("""
                QPushButton {
                    background-color: #f8f9fa;
                    border: 2px solid #e9ecef;
                    border-radius: 10px;
                    padding: 20px;
                    font-size: 16px;
                    color: #212529;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                    border-color: #dee2e6;
                }
            """)
            button.setMinimumSize(300, 150)
            
            button.clicked.connect(lambda checked, name=module_name:
                                  self.open_module(name))
            
            grid.addWidget(button, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    
    def open_module(self, module_name):
        try:
            module = import_module(f'modulos.{module_name}')
            if hasattr(module, 'main'):
                # Mantém uma referência da janela para evitar que seja destruída
                window = module.main()
                if hasattr(self, 'module_windows'):
                    self.module_windows.append(window)
                else:
                    self.module_windows = [window]
            else:
                print(f'Módulo {module_name} não possui função main()')
        except Exception as e:
            print(f'Erro ao carregar módulo {module_name}: {str(e)}')

def main():
    app = QApplication(sys.argv)
    window = DashboardWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()