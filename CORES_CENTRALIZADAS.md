# Guia de Cores Centralizadas

Todas as cores do sistema foram centralizadas no arquivo `config/style_config.json` para facilitar a manutenção e personalização.

## Estrutura das Cores

### Cores Básicas
- `colors.background_color`: Cor de fundo principal
- `colors.text_color`: Cor do texto principal
- `colors.ok_color`: Cor para status OK/aprovado
- `colors.ng_color`: Cor para status NG/reprovado
- `colors.selection_color`: Cor de seleção
- `colors.button_color`: Cor dos botões

### Cores do Canvas
- `colors.canvas_colors.canvas_bg`: Fundo do canvas principal
- `colors.canvas_colors.canvas_dark_bg`: Fundo escuro do canvas
- `colors.canvas_colors.panel_bg`: Fundo dos painéis
- `colors.canvas_colors.dark_panel_bg`: Fundo escuro dos painéis
- `colors.canvas_colors.button_bg`: Fundo dos botões
- `colors.canvas_colors.button_active`: Fundo dos botões ativos

### Cores do Editor
- `colors.editor_colors.clip_color`: Cor para clipping
- `colors.editor_colors.selected_color`: Cor para seleção
- `colors.editor_colors.drawing_color`: Cor para desenho

### Cores de Inspeção
- `colors.inspection_colors.pass_color`: Cor para aprovado
- `colors.inspection_colors.fail_color`: Cor para reprovado
- `colors.inspection_colors.align_fail_color`: Cor para falha de alinhamento

### Cores de Status
- `colors.status_colors.success_bg`: Fundo para sucesso
- `colors.status_colors.error_bg`: Fundo para erro
- `colors.status_colors.warning_bg`: Fundo para aviso
- `colors.status_colors.info_bg`: Fundo para informação
- `colors.status_colors.neutral_bg`: Fundo neutro

### Cores da Interface
- `colors.ui_colors.primary`: Cor primária
- `colors.ui_colors.secondary`: Cor secundária
- `colors.ui_colors.success`: Cor de sucesso
- `colors.ui_colors.danger`: Cor de perigo
- `colors.ui_colors.warning`: Cor de aviso
- `colors.ui_colors.info`: Cor de informação
- `colors.ui_colors.light`: Cor clara
- `colors.ui_colors.dark`: Cor escura
- `colors.ui_colors.muted`: Cor neutra

## Como Usar no Código

### Importar as Funções
```python
from utils import get_color, get_colors_group, update_color
```

### Obter uma Cor Específica
```python
# Obter cor de fundo
bg_color = get_color('colors.background_color')

# Obter cor do canvas
canvas_bg = get_color('colors.canvas_colors.canvas_bg')

# Usar em um widget
canvas = Canvas(parent, bg=get_color('colors.canvas_colors.canvas_bg'))
```

### Obter um Grupo de Cores
```python
# Obter todas as cores do editor
editor_colors = get_colors_group('editor_colors')
clip_color = editor_colors['clip_color']

# Obter todas as cores de status
status_colors = get_colors_group('status_colors')
success_bg = status_colors['success_bg']
```

### Atualizar uma Cor
```python
# Atualizar cor OK
update_color('colors.ok_color', '#00FF00')

# Atualizar cor de fundo sem salvar no arquivo
update_color('colors.background_color', '#333333', save_to_file=False)
```

## Vantagens da Centralização

1. **Manutenção Fácil**: Todas as cores em um só lugar
2. **Consistência**: Garante que as mesmas cores sejam usadas em todo o sistema
3. **Personalização**: Fácil de criar temas personalizados
4. **Fallback**: Sistema de cores padrão caso o arquivo não exista
5. **Flexibilidade**: Estrutura hierárquica organizada por contexto

## Migração do Código Antigo

### Antes (cores hardcoded)
```python
canvas = Canvas(parent, bg="#1E1E1E")
status_label.config(background="#00AA00")
```

### Depois (cores centralizadas)
```python
canvas = Canvas(parent, bg=get_color('colors.canvas_colors.canvas_bg'))
status_label.config(background=get_color('colors.status_colors.success_bg'))
```

## Personalização

Para personalizar as cores, edite o arquivo `config/style_config.json` ou use a interface de configurações do sistema.