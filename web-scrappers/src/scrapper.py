"""
    `moodle-course-scrapper/src/scrapper.py`
    Implementação do web-scrapper para cursos da plataforma Moodle.
    
    @author: nrdc
    @date: 2026-04-03
"""
from playwright.sync_api import Page
from models import Course, Assignment, Material

MOODLE_BASE = "https://moodle.ufrgs.br"

# ── Cursos ────────────────────────────────────────────────────────────────────
def get_courses(page: Page) -> list[Course]:
    """
    Seletor correto UFRGS: .coursebox > .panel-body > .coursebox-content > a
    Cada coursebox tem múltiplos <a>, pegamos apenas o do <h3> (nome do curso).
    """
    page.goto(f"{MOODLE_BASE}/my/")
    page.wait_for_selector(".coursebox", timeout=15000)

    courses = []
    for box in page.query_selector_all(".coursebox"):
        # Link com h3 = nome real do curso
        link = box.query_selector(".coursebox-content > a")
        if not link:
            continue
        name = link.query_selector("h3")
        if not name:
            continue
        courses.append(Course(
            name=name.inner_text().strip(),
            url=link.get_attribute("href")
        ))
    return courses

# ── Materiais ─────────────────────────────────────────────────────────────────
def get_materials(page: Page, course: Course) -> list[Material]:
    """
    Seletores corretos UFRGS:
      - .activityname a.aalink  →  link + nome via .instancename
      - data-activityname       →  nome alternativo no div pai
    Tipo inferido pela URL (/resource/, /url/, /folder/, /mod/*)
    """
    page.goto(course.url)
    page.wait_for_selector(".activity-item", timeout=15000)

    materials = []
    seen = set()

    for item in page.query_selector_all(".activity-item"):
        link = item.query_selector(".activityname a.aalink")
        if not link:
            continue
        href = link.get_attribute("href")
        if not href or href in seen:
            continue

        # Nome: pega .instancename e remove o .accesshide (tipo do arquivo)
        name_el = link.query_selector(".instancename")
        if not name_el:
            continue
        # Remove o span .accesshide do texto
        accesshide = name_el.query_selector(".accesshide")
        if accesshide:
            page.evaluate("el => el.remove()", accesshide)
        name = name_el.inner_text().strip()

        kind = _infer_type(href)
        # Filtra apenas file/link/folder (exclui forum, quiz, choicegroup etc.)
        if kind in ("file", "link", "folder"):
            seen.add(href)
            materials.append(Material(name=name, url=href, type=kind))

    return materials

# ── Assignments ───────────────────────────────────────────────────────────────
def get_assignments(page: Page, course: Course) -> list[Assignment]:
    """
    Assignments UFRGS: a[href*='mod/assign/view.php']
    Prazo e status são extraídos entrando em cada página de assignment.
    """
    page.goto(course.url)
    page.wait_for_selector(".activity-item", timeout=15000)

    # Coleta todos os links de assign da página do curso
    links = []
    seen = set()
    for a in page.query_selector_all("a[href*='mod/assign/view.php']"):
        href = a.get_attribute("href")
        if href and href not in seen:
            seen.add(href)
            links.append(href)

    assignments = []
    for href in links:
        page.goto(href)
        try:
            page.wait_for_selector(".submissionstatustable, .generalbox", timeout=10000)
        except Exception:
            pass

        name    = _extract(page, "h2, .page-header-headings h1") or href
        due     = _extract_due(page)
        status  = _extract_status(page)
        desc    = _extract(page, ".box.generalbox p, .generalbox .no-overflow") or ""

        assignments.append(Assignment(
            name=name,
            url=href,
            due_date=due,
            status=status,
            description=desc
        ))
        page.go_back()

    return assignments

# ── Helpers ───────────────────────────────────────────────────────────────────
def _infer_type(url: str) -> str:
    if "/resource/" in url: return "file"
    if "/url/"      in url: return "link"
    if "/folder/"   in url: return "folder"
    if "/assign/"   in url: return "assign"
    return "other"

def _extract(page: Page, selector: str) -> str | None:
    el = page.query_selector(selector)
    return el.inner_text().strip() if el else None

def _extract_due(page: Page) -> str:
    """
    Na página de assignment da UFRGS, o prazo fica em
    .submissionstatustable td com label 'Prazo de entrega' ou 'Data de entrega'.
    """
    rows = page.query_selector_all(".submissionstatustable tr")
    for row in rows:
        cells = row.query_selector_all("td")
        if len(cells) >= 2:
            label = cells[0].inner_text().strip().lower()
            if "prazo" in label or "entrega" in label or "due" in label:
                return cells[1].inner_text().strip()
    return "sem prazo"

def _extract_status(page: Page) -> str:
    """
    Status: 'Enviado para avaliação' / 'Sem entrega' / etc.
    """
    rows = page.query_selector_all(".submissionstatustable tr")
    for row in rows:
        cells = row.query_selector_all("td")
        if len(cells) >= 2:
            label = cells[0].inner_text().strip().lower()
            if "estado" in label or "status" in label or "situação" in label:
                return cells[1].inner_text().strip()
    return "unknown"