import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox, font
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import threading
import time
import re
import mimetypes
from PIL import Image # Para pegar dimensões da imagem
from google.api_core import exceptions as google_exceptions

# --- Configuração Inicial ---
API_KEY_ENV_VAR = "GOOGLE_API_KEY" # Nome da variável de ambiente para a chave
GEMINI_MODEL_NAME = 'gemini-2.5-flash-preview-04-17' # Modelo recomendado

# --- Lógica Principal (Funções separadas da GUI) ---

def pdf_to_images_local(pdf_path, output_dir, dpi=150):
    """Converte PDF em imagens localmente."""
    image_paths = []
    os.makedirs(output_dir, exist_ok=True)
    try:
        doc = fitz.open(pdf_path)
        print(f"Processing {doc.page_count} pages with DPI={dpi}...")
        for i, page in enumerate(doc):
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_format = "png"
            output_filename = os.path.join(output_dir, f"page_{i+1:03d}.{img_format}")
            pix.save(output_filename)
            image_paths.append(output_filename)
        doc.close()
        print("Image conversion complete.")
        return image_paths
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return None

def upload_to_gemini_file_api(image_paths, status_callback):
    """Faz upload das imagens para a API do Gemini."""
    uploaded_files_map = {}
    pdf_base_name = os.path.splitext(os.path.basename(image_paths[0]))[0].split('_page_')[0] if image_paths else "doc"

    status_callback(f"Starting upload of {len(image_paths)} images...")
    for image_path in image_paths:
        base_image_name = os.path.basename(image_path)
        display_name = f"{pdf_base_name} - {base_image_name}"
        status_callback(f"  Uploading: {base_image_name}...")

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            status_callback(f"  Skipping: Could not determine MIME type for {base_image_name}")
            continue

        try:
            file_to_upload = genai.upload_file(path=image_path, display_name=display_name, mime_type=mime_type)
            while file_to_upload.state.name == "PROCESSING":
                status_callback(f"  Waiting for {base_image_name} to be processed by API...")
                time.sleep(5)
                file_to_upload = genai.get_file(file_to_upload.name)
            
            if file_to_upload.state.name == "ACTIVE":
                uploaded_files_map[image_path] = file_to_upload
                status_callback(f"  Done (URI: {file_to_upload.uri})")
                time.sleep(1)
            else:
                status_callback(f"  Upload Failed for {base_image_name}. File state: {file_to_upload.state.name}")
                return None
        except google_exceptions.ResourceExhausted as e:
             status_callback(f"  Rate limit likely hit. Error: {e}. Stopping upload.")
             return None
        except Exception as e:
            status_callback(f"  Upload Failed for {base_image_name}. Error: {e}")
            return None
    status_callback(f"Finished uploading {len(uploaded_files_map)} files.")
    return uploaded_files_map

def create_html_prompt_with_desc(page_file_object, page_filename, img_dimensions, current_page_num_for_fn):
    """Cria o prompt pedindo descrição textual, com manejo de notas de rodapé."""
    prompt = f"""
Analyze the content of the provided image (filename: {page_filename}, dimensions: {img_dimensions[0]}x{img_dimensions[1]} pixels, representing page {current_page_num_for_fn} of the document). Your goal is to convert this page into an accessible HTML format suitable for screen readers, specifically targeting visually impaired STEM students reading Portuguese content. Don't change the original text or language, even if it's wrong; the main goal is fidelity to the original text.

**Instructions:**

1.  **Text Content:** Extract ALL readable text from the image exactly as it appears, preserving the original language (Portuguese). Preserve paragraph structure where possible. **Omit standalone page numbers** that typically appear at the very top or bottom of a page, unless they are part of a sentence or reference.
2.  **Mathematical Equations:**
    *   Identify ALL mathematical equations, formulas, and expressions.
    *   Convert them accurately into LaTeX format. Use `\\(...\\)` for inline math and `$$...$$` for display math. Ensure correct LaTeX syntax.
3.  **Tables:**
    *   Identify any tables present in the image.
    *   Extract the data accurately, maintaining row and column structure.
    *   Format the table using proper HTML table tags (`<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`). Include table headers (`<th>`) if identifiable.
4.  **Visual Elements (Descriptions - CRITICAL):**
    *   Identify any significant diagrams, graphs, figures, or images within the page content.
    *   **Instead of including the image itself, provide a concise textual description** in Portuguese of what the visual element shows and its relevance (e.g., "<p><i>[Descrição: Diagrama do circuito elétrico mostrando a ligação em série de...]</i></p>" or "<p><i>[Descrição: Gráfico de barras comparando...]</i></p>"). Use italics or similar indication for the description. Integrate these descriptions logically within the extracted text flow where the visual element appeared.
5.  **Footnotes (Notas de Rodapé - CRITICAL):**
    *   Identify footnote markers in the main text (e.g., superscript numbers like `¹`, `²`, or symbols like `*`, `†`).
    *   Identify the corresponding footnote text, typically found at the bottom of the page.
    *   Link the marker to the text using the following patterns:
        *   **In-text marker pattern:** `<sup><a href="#fn{current_page_num_for_fn}-{{FOOTNOTE_INDEX_ON_PAGE}}" id="fnref{current_page_num_for_fn}-{{FOOTNOTE_INDEX_ON_PAGE}}" aria-label="Nota de rodapé {{FOOTNOTE_INDEX_ON_PAGE}}">{{MARKER_SYMBOL_FROM_TEXT}}</a></sup>`
            *   Note: In this pattern, `{current_page_num_for_fn}` will be the actual page number for this image (it's already substituted for you).
            *   You need to replace `{{FOOTNOTE_INDEX_ON_PAGE}}` with the sequential index (1, 2, 3...) of the footnote you identify on this page.
            *   You need to replace `{{MARKER_SYMBOL_FROM_TEXT}}` with the actual marker symbol (e.g., 1, *, †) you find in the text.
        *   **Footnote list pattern:** At the VERY END of this page's HTML content (but before the closing ```html), create a list for all footnotes found on this page:
            ```html
            <hr class="footnotes-separator" />
            <div class="footnotes-section">
              <h4 class="sr-only">Notas de Rodapé da Página {current_page_num_for_fn}</h4>
              <ol class="footnotes-list">
                <li id="fn{current_page_num_for_fn}-{{FOOTNOTE_INDEX_ON_PAGE}}">TEXT_OF_THE_FOOTNOTE_HERE. <a href="#fnref{current_page_num_for_fn}-{{FOOTNOTE_INDEX_ON_PAGE}}" aria-label="Voltar para a referência da nota de rodapé {{FOOTNOTE_INDEX_ON_PAGE}}">&#8617;</a></li>
                <!-- Add more <li> items for other footnotes on this page, following the pattern above -->
              </ol>
            </div>
            ```
            *   Again, `{current_page_num_for_fn}` in the pattern will be the actual page number. You need to replace `{{FOOTNOTE_INDEX_ON_PAGE}}` with the correct index for each footnote. Replace `TEXT_OF_THE_FOOTNOTE_HERE` with the extracted footnote text.
    *   Ensure `id` attributes are unique. The `id` for a footnote `li` should be `fn{current_page_num_for_fn}-{{FOOTNOTE_INDEX_ON_PAGE}}` and its corresponding reference `id` should be `fnref{current_page_num_for_fn}-{{FOOTNOTE_INDEX_ON_PAGE}}` (where `{{FOOTNOTE_INDEX_ON_PAGE}}` is the sequential number like 1, 2, 3...).
6.  **HTML Structure:**
    *   Use semantic HTML tags (e.g., `<p>`, `<h1>`-`<h6>`, `<ul>`, `<ol>`, `<li>`, `<table>`).
    *   Structure the content logically based on the original layout.
    *   **Output ONLY the extracted text, LaTeX math, table HTML, textual descriptions, and footnote HTML as HTML body content enclosed within a single Markdown code block like this:**
        ```html
        <!-- HTML body content goes here -->
        <p>Texto extraído com uma nota<sup><a href="#fn{current_page_num_for_fn}-1" id="fnref{current_page_num_for_fn}-1" aria-label="Nota de rodapé 1">1</a></sup>.</p>
        <p><i>[Descrição: Diagrama...]</i></p>
        $$ LaTeX $$
        <table>...</table>
        ...
        <!-- Footnote section for this page, if any, goes here -->
        <hr class="footnotes-separator" />
        <div class="footnotes-section">
          <h4 class="sr-only">Notas de Rodapé da Página {current_page_num_for_fn}</h4>
          <ol class="footnotes-list">
            <li id="fn{current_page_num_for_fn}-1">Este é o conteúdo da primeira nota de rodapé. <a href="#fnref{current_page_num_for_fn}-1" aria-label="Voltar para a referência da nota de rodapé 1">&#8617;</a></li>
          </ol>
        </div>
        ```
**CRITICAL: Do NOT add any summary or explanation in any language other than the original Portuguese found in the image. Do NOT include `<img>` tags.** Output only the HTML code block.
"""
    return [prompt, page_file_object]

def extract_html_from_response(response_text: str) -> str | None:
    """Extrai HTML do bloco Markdown."""
    match = re.search(r"```html\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        print("  Warning: Could not find ```html block in response. Using raw response text.")
        if response_text.strip().startswith('<') and response_text.strip().endswith('>'):
             return response_text.strip()
        print("  Error: Raw response does not appear to be valid HTML body content.")
        return None

def generate_html_for_image(model, file_object, page_filename, img_path, status_callback, current_page_num_for_fn):
    """Gera HTML para uma única imagem."""
    try:
        img = Image.open(img_path)
        dimensions = img.size
        img.close()
    except Exception as e:
        status_callback(f"  Warning: Could not read image dimensions for {page_filename}: {e}")
        dimensions = ("unknown", "unknown")

    prompt_parts = create_html_prompt_with_desc(file_object, page_filename, dimensions, current_page_num_for_fn)
    try:
        status_callback(f"  Sending request to Gemini for {page_filename} (Page {current_page_num_for_fn})...")
        response = model.generate_content(prompt_parts)

        if not response.candidates:
             status_callback(f"  Error: No response candidates for {page_filename}.")
             if hasattr(response, 'prompt_feedback'):
                  status_callback(f"  Prompt Feedback: {response.prompt_feedback}")
             return None

        status_callback(f"  Received response for {page_filename}. Extracting HTML...")
        html_body = extract_html_from_response(response.text)

        if html_body is None:
             status_callback(f"  Error: Failed to extract HTML for {page_filename}.")
             status_callback(f"  Raw response: {response.text[:200]}...")
             return None

        status_callback(f"  HTML extracted for {page_filename}.")
        return html_body

    except google_exceptions.ResourceExhausted as e:
         status_callback(f"  Error: Rate limit likely hit for {page_filename}. {e}")
         raise
    except Exception as e:
        status_callback(f"  An unexpected error occurred processing {page_filename}: {e}")
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
             status_callback(f"  Prompt Feedback: {response.prompt_feedback}")
        if 'response' in locals() and hasattr(response, 'candidates') and response.candidates:
             if hasattr(response.candidates[0], 'finish_reason'):
                 status_callback(f"  Finish Reason: {response.candidates[0].finish_reason}")
             if hasattr(response.candidates[0], 'safety_ratings'):
                 status_callback(f"  Safety Ratings: {response.candidates[0].safety_ratings}")
        return None

def create_merged_html_with_accessibility(content_list, output_path, pdf_filename):
    """Cria o HTML final com controles de acessibilidade."""
    if not content_list:
        print("No content to merge.")
        return False

    accessibility_css = """
<style>
    html, body {margin: 0;padding: 0;overflow-x: hidden;}
    body {font-family: Verdana, Arial, sans-serif; line-height: 1.6; padding: 20px; background-color: #f0f0f0; color: #333;}

    #accessibility-controls {position: sticky; top: 0; z-index: 1000; padding: 10px; margin-bottom: 20px; border: 1px solid; border-radius: 5px;}
    body.normal-mode #accessibility-controls {background-color: #e0e0e0; border-color: #ccc; color: #000;}
    body.dark-mode #accessibility-controls {background-color: #1e1e1e; border-color: #444; color: #fff;}
    body.high-contrast-mode #accessibility-controls {background-color: #000; border-color: #00FF00; color: #00FF00;}

    #accessibility-controls {display: flex; flex-wrap: wrap; align-items: center; gap: 8px; box-sizing: border-box;}

    #accessibility-controls label,
    #accessibility-controls select,
    #accessibility-controls button,
    #accessibility-controls span {display: inline-flex; align-items: center; white-space: normal; min-width: 0;}
    #accessibility-toggle img {pointer-events: none;}

    .page-content {background-color: #fff; padding: 15px; margin-bottom: 20px; border: 1px solid #ddd; border-radius: 3px;}
    body.normal-mode {background-color: #f0f0f0; color: #333;}
    body.dark-mode {background-color: #121212; color: #ffffff;}
    body.high-contrast-mode {background-color: #000000;color: #00FF00;}
    body.normal-mode .page-content {background-color: #ffffff;border-color: #dddddd;}
    body.dark-mode .page-content {background-color: #1e1e1e;border-color: #444444;}
    body.high-contrast-mode .page-content {background-color: #000000;border-color: #00FF00;}

    h1, h2, h3 {border-bottom: 1px solid; padding-bottom: 0.3em;}
    body.normal-mode h1, body.normal-mode h2, body.normal-mode h3 { color: #000; border-color: #eee;}
    body.dark-mode h1, body.dark-mode h2, body.dark-mode h3 {color: #ffffff; border-color: #444;}
    body.high-contrast-mode h1, body.high-contrast-mode h2, body.high-contrast-mode h3 {color: #00FF00; border-color: #00FF00;}

    hr.page-separator {margin-top: 2em; margin-bottom: 2em; border: 1px dashed #ccc; }
    hr.footnotes-separator { margin-top: 1.5em; margin-bottom: 1em; border-style: dotted; border-width: 1px 0 0 0; }

    .footnotes-section { margin-top: 1em; padding-top: 0.5em; }
    .footnotes-list { list-style-type: decimal; padding-left: 20px; font-size: 0.9em; }
    .footnotes-list li { margin-bottom: 0.5em; }
    .footnotes-list li a { text-decoration: none; }

    .sr-only {position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border-width: 0;}

    p i, span i {color: #555; font-style: italic;}

    sup > a {text-decoration: none; color: #0066cc;}
    sup > a:hover {text-decoration: underline;}

    #accessibility-toggle {position: fixed; top: 20px; right: 20px; z-index: 1100; width: 50px; height: 50px; background-color: #007BFF; color: white; border: none; border-radius: 50%; font-size: 24px; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);}

    #accessibility-controls.collapsed {display: none;}

    #accessibility-controls.expanded {position: fixed; top: 80px; right: 20px; width: 100%; max-width: 360px; box-sizing: border-box; overflow: auto; max-height: calc(100vh - 100px);}
</style>
"""

    accessibility_js = """
<script>
    let currentFontSize = 16;
    const fonts = ['Atkinson Hyperlegible', 'Lexend', 'OpenDyslexicRegular', 'Verdana', 'Arial', 'Times New Roman', 'Courier New'];
    let currentFontIndex = 0;
    const synth = window.speechSynthesis;
    let utterance = null;
    let isPaused = false;

    function changeFontSize(delta) {
        currentFontSize += delta;
        if (currentFontSize < 10) currentFontSize = 10;
        if (currentFontSize > 40) currentFontSize = 40;
        document.body.style.fontSize = currentFontSize + 'px';
    }

    function setFontFamily(fontName) {
        const index = fonts.indexOf(fontName);
        if (index !== -1) {
            currentFontIndex = index;
            document.body.style.fontFamily = fonts[currentFontIndex] + ', sans-serif';
        }
    }

    function getTextToSpeak() {
        let selectedText = window.getSelection().toString();
        if (selectedText) {
            return selectedText;
        } else {
            const contentElement = document.querySelector('body');
            if (contentElement) {
                const controls = document.getElementById('accessibility-controls');
                let text = '';
                Array.from(contentElement.childNodes).forEach(node => {
                    if (node !== controls && node.nodeType === Node.ELEMENT_NODE && node.tagName !== 'SCRIPT' && node.tagName !== 'STYLE') {
                        text += node.textContent.trim().replace(/\s+/g, ' ') + '\n\n';
                    } else if (node.nodeType === Node.TEXT_NODE) {
                        text += node.textContent.trim().replace(/\s+/g, ' ') + '\n\n';
                    }
                });
                return text.trim();
            }
        }
        return '';
    }

    function speakText() {
        if (synth.speaking && !isPaused) {
            console.log("Already speaking.");
            return;
        }
        if (synth.paused && utterance) {
            console.log("Resuming speech.");
            synth.resume();
            isPaused = false;
            return;
        }

        const text = getTextToSpeak();
        if (text && synth) {
            stopSpeech();
            utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'pt-BR';
            utterance.onerror = function(event) {
                console.error('SpeechSynthesisUtterance.onerror', event);
                alert('Erro ao tentar falar o texto: ' + event.error);
            };
            utterance.onend = function() {
                console.log("Speech finished.");
                utterance = null;
                isPaused = false;
            };
            console.log("Starting speech...");
            synth.speak(utterance);
            isPaused = false;
        } else if (!synth) {
             alert('Seu navegador não suporta Text-to-Speech.');
        } else {
             alert('Selecione um texto ou clique em Play para ler a página toda.');
        }
    }

     function pauseSpeech() {
        if (synth.speaking && !isPaused) {
            console.log("Pausing speech.");
            synth.pause();
            isPaused = true;
        } else {
             console.log("Not speaking or already paused.");
        }
    }

    function stopSpeech() {
        if (synth.speaking || synth.paused) {
            console.log("Stopping speech.");
            if (utterance) {
                utterance.onerror = null;
            }
            synth.cancel();
            utterance = null;
            isPaused = false;
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
       setFontFamily(fonts[0]);
       document.getElementById('fontSelector').value = fonts[0];
       changeTheme('dark');
       document.getElementById('themeSelector').value = 'dark';
    });
    
    function changeTheme(mode) {
        document.body.classList.remove('normal-mode', 'dark-mode', 'high-contrast-mode');
        if (mode === 'dark') {
            document.body.classList.add('dark-mode');
        } else if (mode === 'high-contrast') {
            document.body.classList.add('high-contrast-mode');
        } else {
            document.body.classList.add('normal-mode');
        }
    }

    function toggleAccessibilityMenu() {
        const menu = document.getElementById('accessibility-controls');
        menu.classList.toggle('collapsed');
        menu.classList.toggle('expanded');
    }
</script>
"""

    mathjax_config_head_merged = """
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accessible Document: {doc_title}</title>
    <script>
    MathJax = {{
        tex: {{
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$']],
        processEscapes: true,
        processEnvironments: true
        }},
        options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
        }}
    }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/antijingoist/open-dyslexic@master/open-dyslexic-regular.css">
    <link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible&family=Lexend&display=swap" rel="stylesheet">
    {css}
    {js}
</head>
""".format(doc_title=pdf_filename, css=accessibility_css, js=accessibility_js)

    merged_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
{mathjax_config_head_merged}
<body>
    <h1>Documento Acessível: {pdf_filename}</h1>

    <div id="accessibility-controls" class="collapsed">
        <div class="control-group">
            <span>Tamanho da Fonte:</span>
            <button onclick="changeFontSize(-2)">A-</button>
            <button onclick="changeFontSize(2)">A+</button>
        </div>

        <div class="control-group">
            <label for="fontSelector">Fonte:</label>
            <select id="fontSelector" onchange="setFontFamily(this.value)" aria-label="Selecionar família da fonte">
                <option value="Atkinson Hyperlegible">Atkinson Hyperlegible</option>
                <option value="Lexend">Lexend</option>
                <option value="OpenDyslexicRegular">OpenDyslexic</option>
                <option value="Verdana">Verdana</option>
                <option value="Arial">Arial</option>
                <option value="Times New Roman">Times New Roman</option>
                <option value="Courier New">Courier New</option>
            </select>
        </div>

        <div class="control-group">
            <span>Leitura:</span>
            <button onclick="speakText()" aria-label="Ler ou continuar leitura">▶️ Ler/Continuar</button>
            <button onclick="pauseSpeech()" aria-label="Pausar leitura">⏸️ Pausar</button>
            <button onclick="stopSpeech()" aria-label="Parar leitura">⏹️ Parar</button>
        </div>

        <div class="control-group">
            <label for="themeSelector">Tema:</label>
            <select id="themeSelector" onchange="changeTheme(this.value)" aria-label="Selecionar tema visual">
                <option value="normal">Modo Claro</option>
                <option value="dark">Modo Escuro</option>
                <option value="high-contrast">Alto Contraste</option>
            </select>
        </div>
    </div>

    <button id="accessibility-toggle" onclick="toggleAccessibilityMenu()" aria-label="Abrir/Fechar Menu de Acessibilidade">
    <img src="https://cdn.userway.org/widgetapp/images/body_wh.svg" alt="" style="width: 130%; height: 130%;" />
    </button>
"""

    for content_data in content_list:
        page_num = content_data["page_num"]
        html_body = content_data["body"]

        merged_html += f"\n<hr class=\"page-separator\">\n"
        merged_html += f"<div class='page-content' id='page-{page_num}'>\n"
        merged_html += f"<h2 class=\"sr-only\">Conteúdo da Página {page_num} do PDF Original</h2>\n"
        merged_html += f"<h3>Página {page_num}</h3>\n"
        merged_html += html_body
        merged_html += "\n</div>\n"

    merged_html += """
</body>
</html>"""

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(merged_html)
        print(f"Successfully merged content into: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving the merged file: {e}")
        return False

# --- Função Principal do Processamento (para Thread) ---

def process_pdf_thread(pdf_path, output_html_path, dpi, status_callback, completion_callback):
    """Função que executa todo o processo em uma thread separada."""
    try:
        status_callback("Starting process...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_image_dir = os.path.join(script_dir, "temp_pdf_images_local")

        api_key = os.environ.get(API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(f"API Key not found. Set the '{API_KEY_ENV_VAR}' environment variable.")
        genai.configure(api_key=api_key)
        status_callback("API Key configured.")

        status_callback("Converting PDF to images...")
        image_paths = pdf_to_images_local(pdf_path, temp_image_dir, dpi)
        if not image_paths:
            raise Exception("Failed to convert PDF to images.")
        status_callback(f"Converted to {len(image_paths)} images.")

        uploaded_files_map = upload_to_gemini_file_api(image_paths, status_callback)
        if not uploaded_files_map:
            raise Exception("Failed to upload images to Gemini API.")

        status_callback("Initializing Gemini model...")
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
        )
        status_callback("Generating accessible HTML content for each page...")

        generated_content_list_thread = []
        sorted_paths = sorted(uploaded_files_map.keys())

        for i, img_path in enumerate(sorted_paths):
            page_num = i + 1
            base_image_name = os.path.basename(img_path)
            status_callback(f"  Processing Page {page_num}/{len(sorted_paths)} ({base_image_name})...")
            file_object = uploaded_files_map[img_path]

            html_body = generate_html_for_image(model, file_object, base_image_name, img_path, status_callback, page_num)

            if html_body:
                generated_content_list_thread.append({
                    "page_num": page_num,
                    "body": html_body,
                })
            else:
                status_callback(f"  Skipping page {page_num} due to generation error.")
            time.sleep(1)

        if not generated_content_list_thread:
             raise Exception("No HTML content was generated successfully.")

        status_callback("HTML generation finished.")

        status_callback("Merging generated HTML content...")
        pdf_filename_base = os.path.splitext(os.path.basename(pdf_path))[0]
        success = create_merged_html_with_accessibility(
            generated_content_list_thread,
            output_html_path,
            pdf_filename_base
        )
        if not success:
            raise Exception("Failed to create merged HTML file.")
        status_callback(f"Merged HTML saved to {output_html_path}")

        status_callback("Cleaning up temporary files...")
        try:
            for img_path in image_paths:
                if os.path.exists(img_path): os.remove(img_path)
            # Attempt to remove directory only if it's empty after removing files
            try:
                if os.path.exists(temp_image_dir) and not os.listdir(temp_image_dir):
                    os.rmdir(temp_image_dir)
            except OSError as e: # Catch error if directory is not empty or other issues
                status_callback(f"  Warning: Could not remove temp directory {temp_image_dir}: {e}")

            status_callback("  Local images cleanup attempted.")


            count_deleted = 0
            for file_obj in uploaded_files_map.values():
                 try:
                      genai.delete_file(file_obj.name)
                      count_deleted += 1
                      status_callback(f"  Deleted uploaded file: {file_obj.name}")
                      time.sleep(0.5)
                 except Exception as del_e:
                      status_callback(f"  Warning: Could not delete uploaded file {file_obj.name}: {del_e}")
            status_callback(f"  Attempted to delete {count_deleted} uploaded files from API.")
        except Exception as clean_e:
            status_callback(f"  Error during cleanup: {clean_e}")

        status_callback("Process finished successfully!")
        completion_callback(True, f"Success! Accessible HTML saved to:\n{output_html_path}")

    except google_exceptions.ResourceExhausted as e:
        error_message = f"API Rate Limit Reached: {e}. Please wait and try again later, or try with a smaller PDF/fewer pages."
        status_callback(f"Error: {error_message}")
        completion_callback(False, error_message)
    except Exception as e:
        error_message = f"An error occurred: {e}"
        status_callback(f"Error: {e}")
        completion_callback(False, error_message)


# --- Interface Gráfica (Tkinter) ---

class OldSchoolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF to Accessible HTML Converter")
        self.root.geometry("700x550")
        self.root.configure(bg='black')

        self.label_font = font.Font(family="Courier New", size=12)
        self.button_font = font.Font(family="Courier New", size=10, weight="bold")
        self.text_font = font.Font(family="Courier New", size=10)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="black", foreground="green", font=self.label_font)
        style.configure("TButton", background="#222", foreground="lime green", font=self.button_font, borderwidth=1, focusthickness=3, focuscolor='lime green')
        style.map("TButton", background=[('active', '#444')], foreground=[('active', 'white')])
        style.configure("TEntry", fieldbackground="#222", foreground="lime green", insertcolor='lime green', font=self.text_font)
        style.configure("TFrame", background="black")

        main_frame = ttk.Frame(root, padding="15 15 15 15")
        main_frame.pack(expand=True, fill=tk.BOTH)

        pdf_frame = ttk.Frame(main_frame)
        pdf_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pdf_frame, text="PDF File:").pack(side=tk.LEFT, padx=5)
        self.pdf_path_var = tk.StringVar()
        self.pdf_entry = ttk.Entry(pdf_frame, textvariable=self.pdf_path_var, width=60)
        self.pdf_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.browse_button = ttk.Button(pdf_frame, text="Browse...", command=self.select_pdf)
        self.browse_button.pack(side=tk.LEFT, padx=5)

        dpi_frame = ttk.Frame(main_frame)
        dpi_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dpi_frame, text="Image DPI:").pack(side=tk.LEFT, padx=5)
        self.dpi_var = tk.StringVar(value="150")
        self.dpi_entry = ttk.Entry(dpi_frame, textvariable=self.dpi_var, width=10)
        self.dpi_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(dpi_frame, text="(72-600, higher is slower but better OCR)").pack(side=tk.LEFT, padx=5)


        self.convert_button = ttk.Button(main_frame, text="Convert to Accessible HTML", command=self.start_conversion)
        self.convert_button.pack(pady=20)

        ttk.Label(main_frame, text="Status:").pack(anchor=tk.W, pady=(10, 0))
        self.status_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=18, bg='#111', fg='lime green', font=self.text_font, relief=tk.SUNKEN, bd=1)
        self.status_text.pack(expand=True, fill=tk.BOTH, pady=5)
        self.status_text.configure(state='disabled')

    def select_pdf(self):
        filepath = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=(("PDF files", "*.pdf"), ("All files", "*.*"))
        )
        if filepath:
            self.pdf_path_var.set(filepath)
            self.update_status(f"Selected PDF: {filepath}")

    def update_status(self, message):
        def append_message():
            self.status_text.configure(state='normal')
            self.status_text.insert(tk.END, message + "\n")
            self.status_text.configure(state='disabled')
            self.status_text.see(tk.END)
        self.root.after(0, append_message)

    def conversion_complete(self, success, message):
        def final_update():
            self.convert_button.config(state=tk.NORMAL, text="Convert to Accessible HTML")
            if success:
                messagebox.showinfo("Success", message)
                try:
                    # Extract the output file path from the success message
                    # Assumes the path is the last line of the message
                    output_path_from_message = message.strip().split("\n")[-1]
                    
                    # Check if the extracted string is a valid file path
                    if os.path.isfile(output_path_from_message):
                        output_dir = os.path.dirname(output_path_from_message)
                    elif os.path.isdir(output_path_from_message): # Or if it's already a directory
                        output_dir = output_path_from_message
                    else: # Fallback or if the message format changes
                        self.update_status(f"Could not determine output directory from message: {output_path_from_message}")
                        return

                    if os.path.isdir(output_dir): # Final check
                        if os.name == 'nt':
                            os.startfile(output_dir)
                        elif os.name == 'posix':
                            if 'darwin' in os.uname().sysname.lower(): # macOS
                                os.system(f'open "{output_dir}"')
                            else: # Linux
                                os.system(f'xdg-open "{output_dir}"')
                except Exception as e:
                    self.update_status(f"Could not open output directory: {e}")
            else:
                messagebox.showerror("Error", message)
            self.update_status("--------------------")
            self.update_status("Ready for next file.")
        self.root.after(0, final_update)

    def start_conversion(self):
        pdf_path = self.pdf_path_var.get()
        dpi_str = self.dpi_var.get()

        if not pdf_path or not os.path.exists(pdf_path):
            messagebox.showerror("Error", "Please select a valid PDF file.")
            return

        try:
            dpi = int(dpi_str)
            if dpi < 72 or dpi > 600:
                raise ValueError("DPI must be between 72 and 600.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid DPI value: {e}")
            return

        output_dir = os.path.dirname(pdf_path)
        pdf_filename_base = os.path.splitext(os.path.basename(pdf_path))[0]
        output_html_path = os.path.join(output_dir, f"{pdf_filename_base}_accessible.html")

        self.status_text.configure(state='normal')
        self.status_text.delete('1.0', tk.END)
        self.status_text.configure(state='disabled')
        self.convert_button.config(state=tk.DISABLED, text="Processing...")

        thread = threading.Thread(
            target=process_pdf_thread,
            args=(pdf_path, output_html_path, dpi, self.update_status, self.conversion_complete)
        )
        thread.daemon = True
        thread.start()

# --- Execução da Aplicação ---
if __name__ == "__main__":
    print("--------------------------------------------------------------------")
    print(f"IMPORTANT: Ensure your Google AI API Key is set as an environment")
    print(f"variable named '{API_KEY_ENV_VAR}' before running this script.")
    print("Example (Command Prompt): set GOOGLE_API_KEY=YOUR_API_KEY_HERE")
    print("Example (PowerShell):   $env:GOOGLE_API_KEY='YOUR_API_KEY_HERE'")
    print("Example (Linux/macOS):  export GOOGLE_API_KEY='YOUR_API_KEY_HERE'")
    print("--------------------------------------------------------------------")
    print(f"Using Gemini Model: {GEMINI_MODEL_NAME}")
    print("--------------------------------------------------------------------")

    if not os.environ.get(API_KEY_ENV_VAR):
         print(f"WARNING: Environment variable '{API_KEY_ENV_VAR}' not detected.")

    root = tk.Tk()
    app = OldSchoolApp(root)
    root.mainloop()
