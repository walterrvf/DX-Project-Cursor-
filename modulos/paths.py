from pathlib import Path


def get_project_root() -> Path:
    """Retorna o caminho da raiz do projeto."""
    return Path(__file__).parent.parent


def get_model_dir() -> Path:
    """Retorna o caminho para o diretório de modelos."""
    model_dir = get_project_root() / "modelos"
    model_dir.mkdir(exist_ok=True)
    return model_dir


def get_template_dir() -> Path:
    """Retorna o caminho para o diretório de templates global."""
    template_dir = get_model_dir() / "_templates"
    template_dir.mkdir(exist_ok=True)
    return template_dir


def get_model_template_dir(model_name: str, model_id: int) -> Path:
    """Retorna o caminho para o diretório de templates de um modelo específico."""
    template_dir = get_project_root() / f"modelos/{model_name}_{model_id}/templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    return template_dir


__all__ = [
    "get_project_root",
    "get_model_dir",
    "get_template_dir",
    "get_model_template_dir",
]


