"""
Módulo para otimização de imagens para histórico e armazenamento.
Reduz o tamanho dos arquivos mantendo qualidade adequada para visualização.
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import Tuple, Optional
import json

class ImageOptimizer:
    """Classe para otimização de imagens para histórico."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializa o otimizador de imagens.
        
        Args:
            config_file: Caminho para arquivo de configuração (opcional)
        """
        # Configurações padrão
        self.history_resolution = (800, 600)  # Resolução para histórico
        self.thumbnail_resolution = (300, 225)  # Resolução para thumbnails
        self.jpeg_quality = 85  # Qualidade JPEG (0-100)
        self.png_compression = 6  # Compressão PNG (0-9)
        
        # Carregar configurações se especificado
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Carrega configurações do arquivo JSON."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            self.history_resolution = tuple(config.get('history_resolution', self.history_resolution))
            self.thumbnail_resolution = tuple(config.get('thumbnail_resolution', self.thumbnail_resolution))
            self.jpeg_quality = config.get('jpeg_quality', self.jpeg_quality)
            self.png_compression = config.get('png_compression', self.png_compression)
            
        except Exception as e:
            print(f"Erro ao carregar configurações de imagem: {e}")
    
    def save_config(self, config_file: str):
        """Salva configurações no arquivo JSON."""
        try:
            config = {
                'history_resolution': self.history_resolution,
                'thumbnail_resolution': self.thumbnail_resolution,
                'jpeg_quality': self.jpeg_quality,
                'png_compression': self.png_compression
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Erro ao salvar configurações de imagem: {e}")
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    maintain_aspect: bool = True) -> np.ndarray:
        """
        Redimensiona uma imagem para o tamanho especificado.
        
        Args:
            image: Imagem OpenCV (numpy array)
            target_size: Tamanho desejado (width, height)
            maintain_aspect: Se deve manter a proporção da imagem
            
        Returns:
            Imagem redimensionada
        """
        if image is None:
            return None
            
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        if maintain_aspect:
            # Calcular escala mantendo proporção
            scale = min(target_width / width, target_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width, new_height = target_width, target_height
        
        # Redimensionar usando INTER_AREA para redução (melhor qualidade)
        if scale < 1:
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def create_thumbnail(self, image: np.ndarray) -> np.ndarray:
        """
        Cria um thumbnail da imagem para exibição rápida.
        
        Args:
            image: Imagem OpenCV
            
        Returns:
            Thumbnail da imagem
        """
        return self.resize_image(image, self.thumbnail_resolution, maintain_aspect=True)
    
    def create_history_image(self, image: np.ndarray) -> np.ndarray:
        """
        Cria uma versão otimizada da imagem para histórico.
        
        Args:
            image: Imagem OpenCV
            
        Returns:
            Imagem otimizada para histórico
        """
        return self.resize_image(image, self.history_resolution, maintain_aspect=True)
    
    def save_optimized_image(self, image: np.ndarray, file_path: str, 
                           image_type: str = 'history') -> bool:
        """
        Salva uma imagem otimizada no formato apropriado.
        
        Args:
            image: Imagem OpenCV para salvar
            file_path: Caminho do arquivo
            image_type: Tipo de imagem ('history', 'thumbnail', 'original')
            
        Returns:
            True se salvou com sucesso, False caso contrário
        """
        try:
            # Determinar formato e configurações baseado no tipo
            if image_type == 'thumbnail':
                # Thumbnails sempre em JPEG para menor tamanho
                optimized_image = self.create_thumbnail(image)
                file_path = str(file_path).replace('.png', '_thumb.jpg')
                return cv2.imwrite(file_path, optimized_image, 
                                 [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            
            elif image_type == 'history':
                # Imagens de histórico em JPEG com qualidade otimizada
                optimized_image = self.create_history_image(image)
                file_path = str(file_path).replace('.png', '_hist.jpg')
                return cv2.imwrite(file_path, optimized_image, 
                                 [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            
            else:  # original
                # Imagem original em PNG (sem perda)
                return cv2.imwrite(file_path, image)
                
        except Exception as e:
            print(f"Erro ao salvar imagem otimizada: {e}")
            return False
    
    def save_with_thumbnails(self, image: np.ndarray, file_path: str) -> dict:
        """
        Salva a imagem original e cria thumbnails otimizados.
        
        Args:
            image: Imagem OpenCV para salvar
            file_path: Caminho base do arquivo
            
        Returns:
            Dicionário com caminhos dos arquivos salvos
        """
        result = {
            'original': None,
            'history': None,
            'thumbnail': None,
            'success': False
        }
        
        try:
            # Salvar imagem original
            if cv2.imwrite(file_path, image):
                result['original'] = file_path
            else:
                return result
            
            # Criar e salvar versão para histórico
            history_path = str(file_path).replace('.png', '_hist.jpg')
            if self.save_optimized_image(image, history_path, 'history'):
                result['history'] = history_path
            
            # Criar e salvar thumbnail
            thumbnail_path = str(file_path).replace('.png', '_thumb.jpg')
            if self.save_optimized_image(image, thumbnail_path, 'thumbnail'):
                result['thumbnail'] = thumbnail_path
            
            result['success'] = True
            
        except Exception as e:
            print(f"Erro ao salvar com thumbnails: {e}")
        
        return result
    
    def get_file_size_mb(self, file_path: str) -> float:
        """
        Obtém o tamanho do arquivo em MB.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Tamanho em MB
        """
        try:
            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                return size_bytes / (1024 * 1024)
        except Exception as e:
            print(f"Erro ao obter tamanho do arquivo: {e}")
        
        return 0.0
    
    def compare_file_sizes(self, original_path: str, optimized_path: str) -> dict:
        """
        Compara tamanhos entre arquivo original e otimizado.
        
        Args:
            original_path: Caminho do arquivo original
            optimized_path: Caminho do arquivo otimizado
            
        Returns:
            Dicionário com informações de comparação
        """
        original_size = self.get_file_size_mb(original_path)
        optimized_size = self.get_file_size_mb(optimized_path)
        
        if original_size > 0:
            reduction_percent = ((original_size - optimized_size) / original_size) * 100
        else:
            reduction_percent = 0
        
        return {
            'original_size_mb': original_size,
            'optimized_size_mb': optimized_size,
            'reduction_mb': original_size - optimized_size,
            'reduction_percent': reduction_percent
        }
    
    def batch_optimize_directory(self, input_dir: str, output_dir: str, 
                               image_type: str = 'history') -> dict:
        """
        Otimiza todas as imagens em um diretório.
        
        Args:
            input_dir: Diretório de entrada
            output_dir: Diretório de saída
            image_type: Tipo de otimização
            
        Returns:
            Dicionário com estatísticas da otimização
        """
        stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_original_size': 0,
            'total_optimized_size': 0,
            'errors': []
        }
        
        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Processar arquivos de imagem
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            
            for file_path in input_path.iterdir():
                if file_path.suffix.lower() in image_extensions:
                    stats['total_files'] += 1
                    
                    try:
                        # Carregar imagem
                        image = cv2.imread(str(file_path))
                        if image is None:
                            continue
                        
                        # Calcular tamanho original
                        original_size = self.get_file_size_mb(str(file_path))
                        stats['total_original_size'] += original_size
                        
                        # Salvar versão otimizada
                        output_file = output_path / f"{file_path.stem}_opt{file_path.suffix}"
                        if self.save_optimized_image(image, str(output_file), image_type):
                            stats['processed_files'] += 1
                            stats['total_optimized_size'] += self.get_file_size_mb(str(output_file))
                        
                    except Exception as e:
                        stats['errors'].append(f"Erro ao processar {file_path.name}: {e}")
            
        except Exception as e:
            stats['errors'].append(f"Erro geral: {e}")
        
        return stats

# Instância global para uso em outros módulos
image_optimizer = ImageOptimizer()

def optimize_image_for_history(image: np.ndarray, file_path: str) -> dict:
    """
    Função de conveniência para otimizar imagem para histórico.
    
    Args:
        image: Imagem OpenCV
        file_path: Caminho do arquivo
        
    Returns:
        Resultado da otimização
    """
    return image_optimizer.save_with_thumbnails(image, file_path)

def create_thumbnail(image: np.ndarray) -> np.ndarray:
    """
    Função de conveniência para criar thumbnail.
    
    Args:
        image: Imagem OpenCV
        
    Returns:
        Thumbnail da imagem
    """
    return image_optimizer.create_thumbnail(image)

def resize_for_history(image: np.ndarray) -> np.ndarray:
    """
    Função de conveniência para redimensionar para histórico.
    
    Args:
        image: Imagem OpenCV
        
    Returns:
        Imagem redimensionada para histórico
    """
    return image_optimizer.create_history_image(image)
