"""
    `moodle-course-scrapper/src/models.py`
    Arquivo de modelos para o scrapper.
    
    @author: nrdc
    @date: 2026-04-03
"""

from dataclasses import dataclass, field

@dataclass
class Material:
    name: str
    url: str
    type: str          # "pdf", "link", "video", etc.

@dataclass
class Assignment:
    name: str
    url: str
    due_date: str      # ISO 8601 ou "sem prazo"
    status: str        # "submitted", "pending", "overdue"
    description: str

@dataclass
class Course:
    name: str
    url: str
    sections:    list[str] = field(default_factory=list)
    assignments: list[Assignment] = field(default_factory=list)
    materials:   list[Material]   = field(default_factory=list)