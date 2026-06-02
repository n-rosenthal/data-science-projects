"""
    `moodle-course-scrapper/src/config.py`
    Arquivo de configuração para o scrapper.
    
    @author: nrdc
    @date: 2026-04-03
"""

from pathlib import Path

#   URLs para acesso ao Moodle
URL_MOODLE_BASE: str    = r"https://moodle.ufrgs.br"
URL_MOODLE_LOGIN: str   = r"login/login.php"
URL_MOODLE_COURSE: str  = r"course/view.php?id="

#   OUTPUT
OUTPUT_DIR: Path        = Path(__file__).parent.parent / "output"
OUTPUT_FILE: Path       = OUTPUT_DIR / "moodle.json"