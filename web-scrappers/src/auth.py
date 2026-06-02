from playwright.sync_api import Page, Response, ElementHandle
from config import URL_MOODLE_BASE, URL_MOODLE_LOGIN
from re import findall, search
from pathlib import Path

URL_LOGIN = f"{URL_MOODLE_BASE}/{URL_MOODLE_LOGIN}"

def format_response(r: Response) -> str:
    return f"Response<status=`{r.status}` :: url=`{r.url}`>"

def format_element_handle(e: ElementHandle) -> str:
    return f"ElementHandle<selector=`{str(e)}` :: text=`{e.text_content()}`>"

def cprintresult(o) -> str:
    r: str = ""
    if o is None:
        return "None"
    elif isinstance(o, Response):
        r += format_response(o)
        if o.ok:
            r += " | success"
        else:
            r += " | failed"
        return r
    elif isinstance(o, ElementHandle):
        r += format_element_handle(o)
        return r + " | success"
    else:
        return "<" + str(o) + "> | error"

def create_authinfo(login: str, passw: str) -> None:
    """
    Cria o arquivo .authinfo com o login e senha do Moodle.

    Parameters
    ----------
    login : str
        Login do Moodle.
    passw : str
        Senha do Moodle.
    """
    Path(".authinfo").write_text(f"MOODLE_LOGIN={login}\nMOODLE_PASSW={passw}")

def get_authinfo() -> tuple[str, str]:
    """Retorna o login/senha do Moodle."""
    with open(r"../.authinfo", "r") as f:
        try:
            login, passw = findall(r"MOODLE_LOGIN=(.*)\nMOODLE_PASSW=(.*)", f.read())[0]
        except IndexError:
            raise Exception("Arquivo .authinfo não contém login/senha")
    return login, passw

def login(page: Page) -> None:
    """Autentica no Moodle com usuário/senha."""
    print("mc-scrapper: tentando autenticar...")
    r = page.goto(URL_LOGIN)
    print("\t" + cprintresult(r))

    e = page.wait_for_selector("#text1")
    print("\t" + cprintresult(e))

    login, passw = get_authinfo()
    page.fill("#text1", login)
    page.fill("#text",  passw)

    # Captura a navegação que ocorre após o submit
    with page.expect_navigation(wait_until="networkidle", timeout=20000):
        page.click("button[name='submit']")

    url_atual = page.url
    print(f"mc-scrapper: pós-login URL = {url_atual}")

    # Verifica se ainda está na página de login (credenciais erradas)
    if "login" in url_atual:
        raise Exception(f"Login falhou — ainda em: {url_atual}")

    print("mc-scrapper: autenticado com sucesso.")