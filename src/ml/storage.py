"""
storage.py — Registre de modèles dans le lakehouse, sur S3 ou en local selon LAKE_ROOT.

models/<nom>/<version>/<famille>_<clé>.txt  modèles LightGBM en texte (empty_model, bikes_q50…)
models/<nom>/<version>/metrics.json         métriques par horizon, baselines, importance des features
models/<nom>/latest.json                    version publiée (familles qui battent leurs baselines)
models/<nom>/status.json                    résultat du dernier entraînement, publié ou non
"""
import json
import os

from pyarrow import fs as pa_fs


def lake_root() -> str:
    return os.environ.get("LAKE_ROOT", "").rstrip("/")


def _filesystem(uri: str) -> tuple[pa_fs.FileSystem, str]:
    if uri.startswith("s3://"):
        return pa_fs.S3FileSystem(region=os.environ.get("AWS_REGION", "eu-north-1")), uri.removeprefix("s3://")
    return pa_fs.LocalFileSystem(), os.path.abspath(uri)


def model_uri(name: str, *parts: str) -> str:
    return "/".join([lake_root(), "models", name, *parts])


def write_text(uri: str, content: str) -> None:
    filesystem, path = _filesystem(uri)
    if not uri.startswith("s3://"):  # S3 n'a pas de dossiers : rien à créer
        filesystem.create_dir(os.path.dirname(path), recursive=True)
    with filesystem.open_output_stream(path, compression=None) as out:
        out.write(content.encode("utf-8"))


def read_text(uri: str) -> str | None:
    """Contenu du fichier, ou None s'il n'existe pas (registre encore vide)."""
    filesystem, path = _filesystem(uri)
    if filesystem.get_file_info(path).type == pa_fs.FileType.NotFound:
        return None
    with filesystem.open_input_stream(path) as stream:
        return stream.read().decode("utf-8")


def write_json(uri: str, data: dict) -> None:
    write_text(uri, json.dumps(data, ensure_ascii=False, indent=2, default=str))


def read_json(uri: str) -> dict | None:
    content = read_text(uri)
    return json.loads(content) if content else None
