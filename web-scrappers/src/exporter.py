"""
    `moodle-course-scrapper/src/exporter.py`
    
    Implementação do exportador de dados para JSON.
    
    @author: nrdc
    @date: 2026-04-03
"""

import json
from dataclasses import asdict
from pathlib import Path
from models import Course

def export_json(courses: list[Course], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in courses], f,
                    ensure_ascii=False, indent=2)
    print(f"✓ exportado: {path}")