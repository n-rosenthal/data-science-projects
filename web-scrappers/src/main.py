"""
    `moodle-course-scrapper/src/main.py`
    Entry-point para execução do scrapper.
    
    @author: nrdc
    @date: 2026-04-03
"""

from playwright.sync_api import sync_playwright
from auth      import login
from scrapper  import get_courses, get_assignments, get_materials
from exporter  import export_json
from config import OUTPUT_FILE

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_page()
        login(page)

        courses = get_courses(page)
        for course in courses:
            course.assignments = get_assignments(page, course)
            course.materials   = get_materials(page, course)

        export_json(courses, OUTPUT_FILE)
        browser.close()

if __name__ == "__main__":
    run()