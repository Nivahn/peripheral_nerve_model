from __future__ import annotations

import sys
import os
from pathlib import Path


def _shadow_neuron_root() -> Path:
    # `ascent_repo/src/neuron` перекрывает настоящий пакет NEURON в PyCharm, если
    # `ascent_repo/src` попадает в sys.path раньше site-packages.
    return (Path(__file__).resolve().parent / "ascent_repo" / "src").resolve()


def _candidate_neuron_roots() -> list[Path]:
    roots: list[Path] = []

    # У типичной Windows-установки NEURON Python-модуль лежит в c:/nrn/lib/python.
    roots.append(Path("C:/nrn/lib/python"))

    for env_name in ("NEURONHOME", "NRNHOME"):
        env_value = os.environ.get(env_name)
        if not env_value:
            continue
        env_path = Path(env_value)
        roots.append(env_path)
        roots.append(env_path / "lib" / "python")

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            continue
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            unique_roots.append(resolved)
    return unique_roots


def load_neuron_h():
    shadow_root = _shadow_neuron_root()
    removed_entries: list[tuple[int, str]] = []
    added_entries: list[str] = []

    # Временно убираем только корень shadow-пакета, чтобы `import neuron` взял
    # установленный NEURON, а не `ascent_repo/src/neuron`.
    for idx in range(len(sys.path) - 1, -1, -1):
        entry = sys.path[idx]
        try:
            entry_path = Path(entry).resolve()
        except Exception:
            continue
        if entry_path == shadow_root:
            removed_entries.append((idx, entry))
            sys.path.pop(idx)

    try:
        loaded = sys.modules.get("neuron")
        if loaded is not None:
            loaded_path = getattr(loaded, "__file__", None)
            if loaded_path is not None:
                try:
                    loaded_file = Path(loaded_path).resolve()
                except Exception:
                    loaded_file = None
                if loaded_file is not None and shadow_root in loaded_file.parents:
                    del sys.modules["neuron"]

        try:
            from neuron import h
        except ModuleNotFoundError:
            # Если PyCharm/launcher затёр c:/nrn/lib/python из PYTHONPATH, возвращаем
            # типичные пути установки NEURON и повторяем импорт.
            for candidate_root in _candidate_neuron_roots():
                candidate_text = str(candidate_root)
                if candidate_text not in sys.path:
                    sys.path.insert(0, candidate_text)
                    added_entries.append(candidate_text)
            from neuron import h

        return h
    finally:
        for entry in added_entries:
            try:
                sys.path.remove(entry)
            except ValueError:
                pass
        for idx, entry in sorted(removed_entries, key=lambda item: item[0]):
            sys.path.insert(idx, entry)
