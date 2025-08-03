from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

class ContagemWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Contagem de Peças')
        self.setGeometry(150, 150, 600, 400)
        self.setStyleSheet('background-color: white;')
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Conteúdo do módulo
        title = QLabel('Módulo de Contagem de Peças')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('font-size: 24px; color: #212529; margin: 20px;')
        layout.addWidget(title)

def main():
    window = ContagemWindow()
    window.show()
    window.raise_()  # Traz a janela para frente
    return window