# patcher.py

from bs4 import BeautifulSoup

def extrair_paginas(soup):
    """
    Extrai todas as páginas de um objeto BeautifulSoup.
    Uma "página" é um <article class='page-content'>, opcionalmente precedido
    por um <hr class="page-separator">.
    """
    paginas = {}
    todos_articles = soup.find_all('article', class_='page-content')

    for article in todos_articles:
        if not article.get('id'):
            continue  # Pula artigos que não tenham um ID de página

        page_id = article.get('id')
        html_completo_pagina = ''

        # Verifica o elemento irmão anterior ao <article>
        separador_anterior = article.find_previous_sibling()

        # Checa se o irmão anterior é de fato um <hr class="page-separator">
        if separador_anterior and separador_anterior.name == 'hr' and 'page-separator' in separador_anterior.get('class', []):
            html_completo_pagina = str(separador_anterior) + str(article)
        else:
            html_completo_pagina = str(article)

        paginas[page_id] = html_completo_pagina

    return paginas

def patch_html_files(original_content, corrections_content):
    """
    Recebe o CONTEÚDO de dois arquivos HTML, mescla-os e retorna o
    conteúdo HTML final como uma string.
    """
    soup_original = BeautifulSoup(original_content, 'html.parser')
    soup_correcoes = BeautifulSoup(corrections_content, 'html.parser')

    # Extrai as páginas de ambos os conteúdos
    paginas_originais = extrair_paginas(soup_original)
    paginas_correcoes = extrair_paginas(soup_correcoes)

    # Atualiza o dicionário original com as correções (substituindo ou adicionando)
    paginas_originais.update(paginas_correcoes)

    # Ordena as páginas pelo número no ID para garantir a ordem correta
    paginas_ordenadas = sorted(
        paginas_originais.items(),
        key=lambda item: int(item[0].split('-')[-1])
    )

    # Reconstrói o HTML no container principal (main ou body)
    container_principal = soup_original.find('main') or soup_original.body
    if container_principal:
        container_principal.clear()  # Limpa o conteúdo antigo
        for _, page_html in paginas_ordenadas:
            novas_tags = BeautifulSoup(page_html, 'html.parser')
            container_principal.extend(novas_tags.contents)
    else:
        # Fallback caso não encontre um container principal
        html_final = "".join([html for _, html in paginas_ordenadas])
        soup_original = BeautifulSoup(f"<html><body>{html_final}</body></html>", 'html.parser')

    return soup_original.prettify()