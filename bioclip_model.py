from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image


DEFAULT_MODEL_ID = 'hf-hub:imageomics/bioclip-2'
FALLBACK_MODEL_ID = 'hf-hub:imageomics/bioclip'
DEFAULT_TOL_SPECIES_TXT = './data/bioclip_tol_species.txt'
DEFAULT_TOL_SPECIES_CSV = './data/bioclip_tol_taxa.csv'


_project_root = Path(__file__).resolve().parent
_default_hf_home = _project_root / 'models' / 'hf_cache'
os.environ.setdefault('HF_HOME', str(_default_hf_home))
_default_hf_home.mkdir(parents=True, exist_ok=True)

if os.getenv('BIOCLIP_OFFLINE', '0') == '1':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')

_TOL_CLASSIFIER_CACHE: dict[tuple[str, str], Any] = {}
_SPECIES_LIST_CACHE: dict[tuple[str, float, int], list[str]] = {}
_SPECIES_TAXONOMY_CACHE: dict[tuple[str, float], dict[str, dict[str, str]]] = {}

TAXONOMY_RANKS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
CONSTRAINT_RANKS = ['kingdom', 'phylum', 'class', 'order', 'family']
CONSTRAINT_DEGRADE_ORDER = ['family', 'order', 'class', 'phylum', 'kingdom']


def get_model_id() -> str:
    return os.getenv('BIOCLIP_MODEL_ID', DEFAULT_MODEL_ID)


def get_model_candidates() -> list[str]:
    raw = os.getenv('BIOCLIP_MODEL_CANDIDATES', '').strip()
    if raw:
        items = [x.strip() for x in raw.split(',') if x.strip()]
        if items:
            # keep order and deduplicate
            out: list[str] = []
            seen: set[str] = set()
            for item in items:
                if item in seen:
                    continue
                out.append(item)
                seen.add(item)
            return out
    return [DEFAULT_MODEL_ID, FALLBACK_MODEL_ID]


def get_embedding_dimension(model_id: str | None = None) -> int:
    model_key = (model_id or get_model_id()).strip().lower()
    raw_override = os.getenv('BIOCLIP_EMBEDDING_DIM', '').strip()
    if raw_override:
        try:
            value = int(raw_override)
            if value > 0:
                return value
        except ValueError:
            pass
    if 'bioclip-2' in model_key:
        return 768
    return 512


def model_display_name(model_id: str) -> str:
    key = model_id.strip().lower()
    if 'bioclip-2' in key:
        return 'BioCLIP 2 (ViT-L/14, 768d)'
    if key.endswith('/bioclip') or 'imageomics/bioclip' in key:
        return 'BioCLIP 1 (ViT-B/16, 512d)'
    return model_id


def get_tol_model_id() -> str:
    return os.getenv('BIOCLIP_TOL_MODEL_ID', get_model_id())


def get_tol_species_txt_path() -> str:
    value = os.getenv('BIOCLIP_SPECIES_LIST_PATH', DEFAULT_TOL_SPECIES_TXT).strip()
    return value or DEFAULT_TOL_SPECIES_TXT


def get_tol_species_csv_path() -> str:
    value = os.getenv('BIOCLIP_SPECIES_CSV_PATH', DEFAULT_TOL_SPECIES_CSV).strip()
    return value or DEFAULT_TOL_SPECIES_CSV


def get_gpu_free_memory() -> list[int]:
    """Return free memory in MiB for each GPU."""
    if not torch.cuda.is_available():
        return []
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
    except Exception:
        pass
    return []


def select_best_device(prefer_gpu: int | None = None) -> str:
    """Select GPU with most free memory, or specified GPU if prefer_gpu is set."""
    env_device = os.getenv('BIOCLIP_DEVICE', '').strip()
    if env_device:
        return env_device
    
    if not torch.cuda.is_available():
        return 'cpu'
    
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return 'cpu'
    if gpu_count == 1:
        return 'cuda:0'
    
    if prefer_gpu is not None and prefer_gpu < gpu_count:
        return f'cuda:{prefer_gpu}'
    
    free_mem = get_gpu_free_memory()
    if not free_mem or len(free_mem) < gpu_count:
        return 'cuda:0'
    
    best_idx = max(range(gpu_count), key=lambda i: free_mem[i] if i < len(free_mem) else 0)
    return f'cuda:{best_idx}'


def select_device() -> str:
    return select_best_device()

def load_bioclip_model(device: str | None = None, model_id: str | None = None):
    target_device = device or select_device()
    selected_model_id = model_id or get_model_id()
    model, preprocess = open_clip.create_model_from_pretrained(selected_model_id)
    model = model.to(target_device)
    model.eval()
    return model, preprocess, target_device


def encode_image(image: Image.Image, model, preprocess, device: str) -> np.ndarray:
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
    return embedding.detach().cpu().numpy().astype('float32')


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype='float32')
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def _label_prompt(label: str) -> str:
    clean = label.strip()
    if not clean:
        return clean
    if clean.isascii():
        return f'a photo of {clean}'
    return clean


def encode_text_labels(labels: list[str], model, device: str, model_id: str | None = None) -> np.ndarray:
    clean_labels = [x.strip() for x in labels if isinstance(x, str) and x.strip()]
    if not clean_labels:
        return np.zeros((0, get_embedding_dimension(model_id)), dtype='float32')

    tokenizer = open_clip.get_tokenizer(model_id or get_model_id())
    prompts = [_label_prompt(x) for x in clean_labels]
    tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_emb = model.encode_text(tokens)

    return text_emb.detach().cpu().numpy().astype('float32')


def suggest_species_from_embedding(
    image_embedding: np.ndarray,
    species_labels: list[str],
    model,
    device: str,
    model_id: str | None = None,
    top_n: int = 5,
) -> list[dict[str, float | str]]:
    labels = [x.strip() for x in species_labels if isinstance(x, str) and x.strip()]
    if not labels:
        return []

    text_embeddings = encode_text_labels(labels, model, device, model_id=model_id)
    if text_embeddings.size == 0:
        return []

    img = _normalize_rows(image_embedding)
    txt = _normalize_rows(text_embeddings)
    sims = np.matmul(img, txt.T)[0]

    ranked_idx = np.argsort(-sims)[: max(1, min(top_n, len(labels)))]
    out: list[dict[str, float | str]] = []
    for idx in ranked_idx:
        out.append({'species': labels[int(idx)], 'score': float(sims[int(idx)])})
    return out


def _load_tol_classifier(model_str: str, device: str):
    key = (model_str, device)
    if key in _TOL_CLASSIFIER_CACHE:
        return _TOL_CLASSIFIER_CACHE[key]

    from bioclip import TreeOfLifeClassifier

    clf = TreeOfLifeClassifier(model_str=model_str, device=device)
    _TOL_CLASSIFIER_CACHE[key] = clf
    return clf


def _clean_taxonomy_value(value: Any) -> str:
    out = str(value).strip()
    return '' if out.lower() in {'', 'nan', 'none'} else out


def _extract_taxonomy_fields(item: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rank in TAXONOMY_RANKS:
        out[rank] = _clean_taxonomy_value(item.get(rank, ''))
    common_name = _clean_taxonomy_value(item.get('common_name', ''))
    if common_name:
        out['common_name'] = common_name
    return out


def load_species_taxonomy_map(species_csv_path: str | None = None) -> tuple[dict[str, dict[str, str]], str | None]:
    csv_path = Path(species_csv_path or get_tol_species_csv_path())
    if not csv_path.exists():
        return {}, f'species taxonomy csv not found: {csv_path}'

    try:
        stat = csv_path.stat()
    except Exception as exc:
        return {}, f'cannot stat species taxonomy csv: {exc}'

    cache_key = (str(csv_path.resolve()), stat.st_mtime)
    if cache_key in _SPECIES_TAXONOMY_CACHE:
        return _SPECIES_TAXONOMY_CACHE[cache_key], None

    try:
        with csv_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            mapping: dict[str, dict[str, str]] = {}
            for row in reader:
                species = _clean_taxonomy_value(row.get('species', ''))
                if not species:
                    continue
                if species not in mapping:
                    mapping[species] = _extract_taxonomy_fields(row)
        _SPECIES_TAXONOMY_CACHE[cache_key] = mapping
        return mapping, None
    except UnicodeDecodeError:
        return {}, 'species taxonomy csv decode failed (utf-8)'
    except Exception as exc:
        return {}, f'read species taxonomy csv failed: {exc}'


def attach_taxonomy_to_species_suggestions(
    suggestions: list[dict[str, Any]],
    species_csv_path: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    taxonomy_map, err = load_species_taxonomy_map(species_csv_path)
    if not taxonomy_map:
        return suggestions, err

    out: list[dict[str, Any]] = []
    for item in suggestions:
        copied = dict(item)
        species = _clean_taxonomy_value(copied.get('species', ''))
        taxonomy = taxonomy_map.get(species)
        if taxonomy:
            for k, v in taxonomy.items():
                if v and not _clean_taxonomy_value(copied.get(k, '')):
                    copied[k] = v
        out.append(copied)
    return out, None


def _get_rank_enum(rank_name: str):
    from bioclip import Rank

    rank_map = {
        'kingdom': Rank.KINGDOM,
        'phylum': Rank.PHYLUM,
        'class': Rank.CLASS,
        'order': Rank.ORDER,
        'family': Rank.FAMILY,
        'genus': Rank.GENUS,
        'species': Rank.SPECIES,
    }
    return rank_map[rank_name]


def get_tol_taxonomy_constraints(
    image: Image.Image,
    threshold: float = 0.9,
    model_str: str | None = None,
    device: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    model_key = model_str or get_tol_model_id()
    target_device = device or select_device()

    try:
        clf = _load_tol_classifier(model_key, target_device)
    except Exception as exc:
        return None, f'load tol classifier failed: {exc}'

    rank_predictions: dict[str, dict[str, Any]] = {}
    for rank in CONSTRAINT_RANKS:
        try:
            rank_enum = _get_rank_enum(rank)
            preds = clf.predict(images=[image], rank=rank_enum, k=1)
        except Exception as exc:
            return None, f'predict rank {rank} failed: {exc}'

        if not preds or not isinstance(preds[0], dict):
            return None, f'predict rank {rank} returned empty result'

        top = preds[0]
        label = _clean_taxonomy_value(top.get(rank, ''))
        score = float(top.get('score', 0.0))
        taxonomy = _extract_taxonomy_fields(top)
        rank_predictions[rank] = {
            'rank': rank,
            'label': label,
            'score': score,
            'taxonomy': taxonomy,
        }

    active_rank: str | None = None
    for rank in CONSTRAINT_DEGRADE_ORDER:
        score = float(rank_predictions[rank]['score'])
        if score >= float(threshold):
            active_rank = rank
            break

    active_taxonomy: dict[str, str] = {}
    if active_rank is not None:
        active_idx = CONSTRAINT_RANKS.index(active_rank)
        for rank in CONSTRAINT_RANKS[: active_idx + 1]:
            active_taxonomy[rank] = _clean_taxonomy_value(rank_predictions[rank]['label'])

    rank_scores = {rank: float(rank_predictions[rank]['score']) for rank in CONSTRAINT_RANKS}
    degrade_chain = [{
        'rank': rank,
        'label': _clean_taxonomy_value(rank_predictions[rank]['label']),
        'score': float(rank_predictions[rank]['score']),
    } for rank in CONSTRAINT_DEGRADE_ORDER]

    return {
        'threshold': float(threshold),
        'active_rank': active_rank,
        'enabled': active_rank is not None,
        'active_taxonomy': active_taxonomy,
        'rank_scores': rank_scores,
        'degrade_chain': degrade_chain,
        'rank_predictions': [rank_predictions[r] for r in CONSTRAINT_RANKS],
    }, None


def suggest_species_with_tol_classifier(
    image: Image.Image,
    top_n: int = 5,
    model_str: str | None = None,
    device: str | None = None,
) -> tuple[list[dict[str, float | str]], str | None]:
    model_key = model_str or get_tol_model_id()
    target_device = device or select_device()

    try:
        from bioclip import Rank
    except Exception as exc:
        return [], f'pybioclip unavailable: {exc}'

    try:
        clf = _load_tol_classifier(model_key, target_device)
        preds = clf.predict(images=[image], rank=Rank.SPECIES, k=max(1, top_n))
    except Exception as exc:
        return [], f'pybioclip predict failed: {exc}'

    out: list[dict[str, float | str]] = []
    for item in preds:
        if not isinstance(item, dict):
            continue
        species = str(item.get('species', '')).strip()
        score = float(item.get('score', 0.0))
        if not species:
            continue
        row: dict[str, float | str] = {'species': species, 'score': score}
        for rank, value in _extract_taxonomy_fields(item).items():
            if value:
                row[rank] = value
        out.append(row)
    return out, None


def export_tol_species_list(
    output_csv_path: str,
    output_species_txt_path: str,
    model_str: str | None = None,
    device: str | None = None,
) -> tuple[int, str | None]:
    model_key = model_str or get_tol_model_id()
    target_device = device or select_device()

    try:
        clf = _load_tol_classifier(model_key, target_device)
        df = clf.get_label_data()
    except Exception as exc:
        return 0, f'load tol labels failed: {exc}'

    out_csv = Path(output_csv_path)
    out_txt = Path(output_species_txt_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(out_csv, index=False)
        species = sorted({str(x).strip() for x in df['species'].tolist() if str(x).strip()})
        out_txt.write_text('\n'.join(species) + ('\n' if species else ''), encoding='utf-8')
        return len(species), None
    except Exception as exc:
        return 0, f'write species list failed: {exc}'


def load_species_list_from_file(
    species_txt_path: str,
    max_labels: int = 0,
) -> tuple[list[str], str | None]:
    species_file = Path(species_txt_path)
    if not species_file.exists():
        return [], f'species list file not found: {species_file}'

    try:
        stat = species_file.stat()
    except Exception as exc:
        return [], f'cannot stat species list file: {exc}'

    max_keep = max(0, int(max_labels))
    cache_key = (str(species_file.resolve()), stat.st_mtime, max_keep)
    if cache_key in _SPECIES_LIST_CACHE:
        return _SPECIES_LIST_CACHE[cache_key], None

    try:
        content = species_file.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        content = species_file.read_text(encoding='utf-8', errors='ignore')
    except Exception as exc:
        return [], f'read species list failed: {exc}'

    labels: list[str] = []
    seen: set[str] = set()
    for line in content.splitlines():
        label = line.strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if max_keep > 0 and len(labels) >= max_keep:
            break

    _SPECIES_LIST_CACHE[cache_key] = labels
    return labels, None


def load_or_export_tol_species_list(
    species_txt_path: str | None = None,
    species_csv_path: str | None = None,
    max_labels: int = 0,
    auto_export: bool = True,
    model_str: str | None = None,
    device: str | None = None,
) -> tuple[list[str], str | None]:
    txt_path = species_txt_path or get_tol_species_txt_path()
    csv_path = species_csv_path or get_tol_species_csv_path()

    labels, read_error = load_species_list_from_file(txt_path, max_labels=max_labels)
    if labels:
        return labels, None

    if not auto_export:
        return [], read_error or 'species list unavailable and auto export disabled'

    count, export_error = export_tol_species_list(
        output_csv_path=csv_path,
        output_species_txt_path=txt_path,
        model_str=model_str,
        device=device,
    )
    if export_error:
        return [], export_error
    if count <= 0:
        return [], 'exported species list is empty'

    return load_species_list_from_file(txt_path, max_labels=max_labels)
