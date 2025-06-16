# processing.py
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import threading
import time
import re
import mimetypes
from PIL import Image  # Para pegar dimensões da imagem
from google.api_core import exceptions as google_exceptions
import shutil  # Para rmtree
import concurrent.futures  # Para processamento concorrente
import base64  # For embedding images
import html  # For escaping HTML special characters in title
import tempfile # Para gerenciar arquivos e diretórios temporários
import boto3   # Importa boto3 aqui também
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# --- Configuração Inicial ---
API_KEY_ENV_VAR = "GOOGLE_API_KEY"
DEFAULT_GEMINI_MODEL = 'gemini-2.0-flash'
AVAILABLE_GEMINI_MODELS = [
    'gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite',
    'gemini-2.5-flash-preview-05-20', 'gemini-1.5-flash-8b', 'gemini-2.5-pro-preview-06-05'
]
MODELO_ESCALONAMENTO = 'gemini-2.5-flash-preview-05-20'
LIMITE_TOKENS_ESCALONAMENTO = 65536
FINISH_REASON_MAX_TOKENS = "MAX_TOKENS"  # Na API v1beta, o finish_reason é uma string

MAX_RETRIES_PER_CALL = 3
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 4  # seconds
PHASE_MAX_RETRIES = 1

DEFAULT_UPLOAD_MAX_WORKERS = 10
DEFAULT_GENERATE_MAX_WORKERS = 5

# Modo de Armazenamento -> LOCAL ou S3
MODE = "LOCAL"


class OperationCancelledError(Exception):
    pass


def parse_page_ranges(page_range_str: str, total_pages: int) -> list[int] | None:
    if not page_range_str.strip(): return list(range(total_pages))
    indices = set()
    parts = page_range_str.split(',')
    for part in parts:
        part = part.strip()
        if not part: continue
        if '-' in part:
            start_str, end_str = part.split('-', 1)
            try:
                start_idx = int(start_str) - 1
                if not end_str:
                    end_idx = total_pages - 1
                else:
                    end_idx = int(end_str) - 1
                if not (
                        0 <= start_idx < total_pages and 0 <= end_idx < total_pages and start_idx <= end_idx): return None
                indices.update(range(start_idx, end_idx + 1))
            except ValueError:
                return None
        else:
            try:
                idx = int(part) - 1
                if not (0 <= idx < total_pages): return None
                indices.add(idx)
            except ValueError:
                return None
    if not indices and page_range_str.strip(): return None
    return sorted(list(indices))


def gemini_api_call_with_retry(api_function, cancel_event, status_callback, *args, **kwargs):
    retries = 0
    backoff_time = INITIAL_BACKOFF
    retryable_exceptions = (
        google_exceptions.ServiceUnavailable,
        google_exceptions.TooManyRequests,
        google_exceptions.DeadlineExceeded,
        google_exceptions.InternalServerError
    )
    while retries <= MAX_RETRIES_PER_CALL:
        if cancel_event.is_set(): raise OperationCancelledError("Operação cancelada durante retentativa API.")
        try:
            return api_function(*args, **kwargs)
        except retryable_exceptions as e:
            retries += 1
            if retries > MAX_RETRIES_PER_CALL:
                status_callback(f"  Falha API após {MAX_RETRIES_PER_CALL} retentativas: {e}.")
                raise
            status_callback(
                f"  Erro API ({type(e).__name__}): {e}. Retentativa {retries}/{MAX_RETRIES_PER_CALL} em {backoff_time}s...")
            for _ in range(int(backoff_time)):
                if cancel_event.is_set(): raise OperationCancelledError("Cancelado durante espera retentativa.")
                time.sleep(1)
            backoff_time = min(backoff_time * 2, MAX_BACKOFF)
        except google_exceptions.ResourceExhausted as re_e:
            status_callback(f"  Erro Recurso Esgotado API: {re_e}. Verifique cotas.")
            raise
        except Exception as general_e:
            status_callback(f"  Erro inesperado chamada API: {type(general_e).__name__} - {general_e}")
            raise


def pdf_to_images_local(pdf_path, output_dir, dpi, selected_pages, cancel_event, status_callback, progress_callback):
    image_paths = []
    os.makedirs(output_dir, exist_ok=True)
    doc = None
    try:
        doc = fitz.open(pdf_path)
        if doc.is_encrypted:
            status_callback("ERRO: PDFs protegidos por senha não são suportados na versão web.")
            return None

        total_selected_pages = len(selected_pages)
        status_callback(f"Convertendo {total_selected_pages} págs PDF para imagens (DPI: {dpi})...")
        for i, page_num_0_indexed in enumerate(selected_pages):
            if cancel_event.is_set(): raise OperationCancelledError("Conversão PDF->Imagem cancelada.")
            page = doc.load_page(page_num_0_indexed)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            output_filename = os.path.join(output_dir, f"page_{page_num_0_indexed + 1:05d}.png")
            pix.save(output_filename)
            image_paths.append(output_filename)
            progress_callback(i + 1, total_selected_pages, f"Convertendo pág. PDF {page_num_0_indexed + 1}")
        status_callback("Conversão PDF->Imagens completa.")
        return image_paths
    except OperationCancelledError:
        status_callback("  Conversão PDF->Imagens cancelada.")
        raise
    except Exception as e:
        status_callback(f"Erro conversão PDF->Imagens: {e}")
        return None
    finally:
        if doc: doc.close()


def _upload_single_image_task(image_path, display_name, mime_type, cancel_event, status_callback_main_thread):
    if cancel_event.is_set(): raise OperationCancelledError("Upload de imagem individual cancelado antes de iniciar.")
    try:
        file_to_upload = gemini_api_call_with_retry(
            genai.upload_file, cancel_event, status_callback_main_thread,
            path=image_path, display_name=display_name, mime_type=mime_type
        )
        # O novo SDK pode não precisar deste loop de verificação, mas é bom mantê-lo por robustez
        processing_start_time = time.time()
        while file_to_upload.state.name == "PROCESSING":
            if cancel_event.is_set():
                try:
                    status_callback_main_thread(f"  Cancelamento. Deletando {file_to_upload.name} em processamento...")
                    genai.delete_file(file_to_upload.name)
                    status_callback_main_thread(f"  Arquivo {file_to_upload.name} deletado.")
                except Exception as del_e:
                    status_callback_main_thread(f"  Aviso: Falha ao deletar {file_to_upload.name}: {del_e}")
                raise OperationCancelledError("Upload cancelado durante processamento na API.")
            if time.time() - processing_start_time > 300:  # 5 minutes timeout
                status_callback_main_thread(f"  Timeout processando {os.path.basename(image_path)} na API.")
                try:
                    genai.delete_file(file_to_upload.name)
                    status_callback_main_thread(f"  Arquivo {file_to_upload.name} com timeout deletado.")
                except Exception:
                    pass
                return image_path, None
            time.sleep(2)  # Check status
            file_to_upload = gemini_api_call_with_retry(genai.get_file, cancel_event, status_callback_main_thread,
                                                        name=file_to_upload.name)

        if file_to_upload.state.name == "ACTIVE":
            return image_path, file_to_upload
        else:
            status_callback_main_thread(
                f"  Falha no upload para {os.path.basename(image_path)}. Estado: {file_to_upload.state.name}")
            if file_to_upload.state.name != "DELETED": # Avoid trying to delete already deleted files
                try:
                    status_callback_main_thread(
                        f"  Deletando arquivo não-ativo: {file_to_upload.name}"); genai.delete_file(file_to_upload.name)
                except Exception:
                    pass # Log if needed, but don't let this fail the entire process
            return image_path, None
    except Exception as e:
        status_callback_main_thread(
            f"  Exceção na task de upload para {os.path.basename(image_path)}: {type(e).__name__} - {e}")
        return image_path, None


def upload_to_gemini_file_api(image_paths_to_upload, num_upload_workers, cancel_event, status_callback,
                              progress_callback):
    successfully_uploaded_map = {}
    pdf_base_name = "document" # Default
    if image_paths_to_upload:
        first_image_basename = os.path.basename(image_paths_to_upload[0])
        # Robust extraction of PDF base name
        match = re.match(r"^(.*?)_page_\d+\.\w+$", first_image_basename)
        if match and match.group(1):
            pdf_base_name = match.group(1)
        else:
            # Fallback if the pattern doesn't match (e.g., name doesn't contain "_page_")
            pdf_base_name = os.path.splitext(first_image_basename)[0]

    current_image_paths = list(image_paths_to_upload)

    for attempt in range(PHASE_MAX_RETRIES + 1):
        if not current_image_paths or cancel_event.is_set(): break
        total_in_this_attempt = len(current_image_paths)
        if attempt > 0:
            status_callback(
                f"Retentativa de Upload (Fase {attempt + 1}/{PHASE_MAX_RETRIES + 1}) para {total_in_this_attempt} imagens...")
        else:
            status_callback(
                f"Iniciando upload concorrente de {total_in_this_attempt} imagens para '{pdf_base_name}' (workers: {num_upload_workers})...")

        processed_for_progress_bar_this_attempt = 0
        failed_in_this_attempt = []

        tasks_to_submit_this_round = []
        for image_path in current_image_paths:
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                status_callback(f"  Pulando (MIME): {os.path.basename(image_path)}")
                failed_in_this_attempt.append(image_path)
                processed_for_progress_bar_this_attempt += 1
                progress_callback(processed_for_progress_bar_this_attempt, total_in_this_attempt,
                                  f"Upload {processed_for_progress_bar_this_attempt}/{total_in_this_attempt} (Erro MIME)")
                continue
            tasks_to_submit_this_round.append({'path': image_path, 'mime': mime_type})

        if not tasks_to_submit_this_round and failed_in_this_attempt: # All failed due to MIME
            current_image_paths = failed_in_this_attempt
            if not current_image_paths or cancel_event.is_set(): break
            continue # Go to next phase retry if any left

        if not tasks_to_submit_this_round and not failed_in_this_attempt: # No tasks at all
            break

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_upload_workers) as executor:
            future_to_path = {
                executor.submit(_upload_single_image_task, task_info['path'],
                                f"{pdf_base_name} - {os.path.basename(task_info['path'])}",
                                task_info['mime'], cancel_event, status_callback): task_info['path']
                for task_info in tasks_to_submit_this_round
            }
            for future in concurrent.futures.as_completed(future_to_path):
                if cancel_event.is_set(): break
                original_path = future_to_path[future]
                try:
                    returned_path, file_object = future.result()
                    if file_object:
                        successfully_uploaded_map[returned_path] = file_object
                        status_callback(f"  Sucesso upload: {os.path.basename(returned_path)}")
                    else:
                        failed_in_this_attempt.append(returned_path)
                        status_callback(f"  Falha upload (task None): {os.path.basename(returned_path)}")
                except OperationCancelledError:
                    status_callback(f"  Upload cancelado para {os.path.basename(original_path)}.")
                    failed_in_this_attempt.append(original_path) # Add to failed if cancelled mid-operation
                    break
                except Exception as exc:
                    status_callback(f"  Erro no upload de {os.path.basename(original_path)}: {exc}")
                    failed_in_this_attempt.append(original_path)
                finally:
                    processed_for_progress_bar_this_attempt += 1
                    progress_callback(processed_for_progress_bar_this_attempt, total_in_this_attempt,
                                      f"Upload imagem {processed_for_progress_bar_this_attempt}/{total_in_this_attempt}")

        current_image_paths = failed_in_this_attempt
        if not current_image_paths or cancel_event.is_set(): break

    if cancel_event.is_set(): raise OperationCancelledError("Upload de imagens cancelado.")

    total_initial_images = len(image_paths_to_upload)
    if len(successfully_uploaded_map) < total_initial_images and total_initial_images > 0:
        status_callback(
            f"Aviso Final: {len(successfully_uploaded_map)}/{total_initial_images} imagens carregadas com sucesso após todas as tentativas.")
    if not successfully_uploaded_map and total_initial_images > 0:
        status_callback("Nenhuma imagem carregada com sucesso para a API após todas as tentativas.");
        return None

    status_callback(f"Upload concorrente finalizado. {len(successfully_uploaded_map)} arquivos ativos.");
    return successfully_uploaded_map


def create_html_prompt_with_desc(page_file_object, page_filename, img_dimensions, current_page_num_in_doc):
    """
    Gera o prompt para o Gemini, agora com uma instrução explícita para formatar URLs.
    """
    prompt = f"""
Analyze the content of the provided image (filename: {page_filename}, dimensions: {img_dimensions[0]}x{img_dimensions[1]} pixels, representing page {current_page_num_in_doc} of the document). Your goal is to convert this page into an accessible HTML format suitable for screen readers, specifically targeting visually impaired STEM students reading Portuguese content. Don't change the original text or language, even if it's wrong; the main goal is fidelity to the original text.
**Instructions:**
1.  **Text Content (MAXIMUM FIDELITY REQUIRED):**
    * Extract ALL readable text from the image **EXACTLY** as it appears.
    * Preserve the original language (Portuguese) and the **PRECISE WORDING AND ORDER OF WORDS**.
    * Do NOT paraphrase, reorder, or 'correct' the text. Reproduce it verbatim.
    * Preserve paragraph structure where possible.
    * **Omit standalone page numbers** that typically appear at the very top or bottom of a page, unless they are part of a sentence or reference.

2.  **Web Links (URLs):**
    * Identify all web addresses (URLs, links) in the text (e.g., starting with `http://`, `https://`, `www.`).
    * **CRITICAL: You MUST format them as clickable HTML links using the `<a>` tag.**
    * The `href` attribute must contain the full, correct URL, and the link text should also be the full URL.
    * **Example:** If the text reads `disponível em https://www.exemplo.com/doc.pdf`, the HTML output MUST be `disponível em <a href="https://www.exemplo.com/doc.pdf">https://www.exemplo.com/doc.pdf</a>`.

3.  **Mathematical Equations:**
    * Identify ALL mathematical equations, formulas, and expressions.
    * Convert them accurately into LaTeX format.
    * **CRITICAL DELIMITER USAGE:** For inline mathematics, YOU MUST USE `\\(...\\)` (e.g., `\\(x=y\\)`). For display mathematics (equations on their own line), YOU MUST USE `$$...$$` (e.g., `$$x = \\sum y_i$$`).
    * **Ensure that *all* mathematical symbols, including single-letter variables mentioned in prose (e.g., '...where v is velocity...'), are enclosed in inline LaTeX delimiters (e.g., output as '...where \\(v\\) is velocity...').** This applies to all isolated symbols.

4.  **Tables (CRITICAL FOR ACCESSIBILITY):**
    * Identify any tables.
    * **CRITICAL:** Extract the table's main title or header and place it inside a `<caption>` tag as the very first element within the `<table>`. Example: `<table><caption>Vendas Mensais por Região</caption>...</table>`. This provides essential context for screen reader users.
    * Format the table using proper HTML tags (`<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`).
    * **Pay close attention to table structure:**
        * Correctly identify header cells and use the `<th>` tag for them.
        * For column and row headers, use the `scope` attribute (e.g., `<th scope="col">Nome da Coluna</th>`, `<th scope="row">Nome da Linha</th>`).
        * Accurately detect and represent merged cells using `colspan` for horizontally merged cells and `rowspan` for vertically merged cells.
    * **Example of a complex table structure:**
        ```html
        <table>
          <caption>Exemplo de Tabela Complexa</caption>
          <thead>
            <tr>
              <th scope="col">Produto</th>
              <th scope="col" colspan="2">Detalhes de Vendas</th>
            </tr>
            <tr>
              <th scope="col"></th>
              <th scope="col">Região A</th>
              <th scope="col">Região B</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">Item 1</th>
              <td>100</td>
              <td>150</td>
            </tr>
            <tr>
              <th scope="row" rowspan="2">Itens Agrupados</th>
              <td>200</td>
              <td>210</td>
            </tr>
            <tr>
              <td>205</td>
              <td>215</td>
            </tr>
          </tbody>
        </table>
        ```

5.  **Hierarquia de Títulos (NAVIGATION CRITICAL):**
    * Identify the document's structural hierarchy within the page content (e.g., section titles, sub-section titles).
    * **CRITICAL:** DO NOT create new titles or headings that do not exist in the original text.
    * Use `<h3>`, `<h4>`, `<h5>`, and `<h6>` tags to mark this hierarchy. A main section title on the page should be `<h3>`, a subsection within it `<h4>`, and so on.
    * **YOU MUST add a unique `id` to every heading you create.** Use the format `id="h{{LEVEL}}-{current_page_num_in_doc}-{{INDEX}}"`, where `{{LEVEL}}` is the heading number (3, 4, etc.) and `{{INDEX}}` is a sequential number for that heading level on the page (1, 2, 3...).
    * Example: `<h3 id="h3-{current_page_num_in_doc}-1">Primeira Seção</h3>`, `<h4 id="h4-{current_page_num_in_doc}-1">Primeira Subseção</h4>`.

6.  **Visual Elements (Descriptions):**
    * Identify significant diagrams, graphs, figures, or images relevant to the academic content. **DO NOT** describe purely decorative elements like logos that are not relevant to the understanding of the text.
    * **Instead of including the image itself, provide a concise textual description** in Portuguese, wrapped in `<p><em>...</em></p>`. The description should explain what the visual element shows and its relevance.
    * In the case of a bar code, QR code, or similar, **ALWAYS** indicate that it is there and describe its purpose using the `<p><em>...</em></p>` tag.
    * Example: `<p><em>[Descrição: Diagrama de um circuito elétrico RLC em série, mostrando a fonte de tensão, o resistor R, o indutor L e o capacitor C.]</em></p>`.

7.  **Footnotes (Notas de Rodapé):**
    * Identify footnote markers and their corresponding text.
    * Link them using the following precise patterns:
        * **In-text marker:** `<sup><a href="#fn{current_page_num_in_doc}-{{INDEX}}" id="fnref{current_page_num_in_doc}-{{INDEX}}" aria-label="Nota de rodapé {{INDEX}}">{{MARKER}}</a></sup>`
        * **Footnote list (at the very end of the page's HTML):**
            ```html
            <hr class="footnotes-separator" />
            <div class="footnotes-section">
              <h4 class="sr-only">Notas de Rodapé da Página {current_page_num_in_doc}</h4>
              <ol class="footnotes-list">
                <li id="fn{current_page_num_in_doc}-{{INDEX}}">TEXT_OF_THE_FOOTNOTE. <a href="#fnref{current_page_num_in_doc}-{{INDEX}}" aria-label="Voltar para a referência da nota de rodapé {{INDEX}}">&#8617;</a></li>
              </ol>
            </div>
            ```

8.  **Abreviações e Acrônimos:**
    * If you identify a known abbreviation or acronym (e.g., ABNT, PIB, DNA), use the `<abbr>` tag to provide its full expansion. This helps screen readers pronounce them correctly.
    * Example: `Segundo a <abbr title="Associação Brasileira de Normas Técnicas">ABNT</abbr>, a regra é...`

9.  **Final HTML Structure:**
    * Use semantic HTML.
    * **AVOID UNNECESSARY TAGS:** Do NOT use `<bdi>` tags. They are generally not needed for Portuguese or mathematical content and can interfere with screen readers.
    * Output ONLY the extracted content as HTML body content in a single Markdown code block.
    * **AVOID UNNECESSARY TAGS:** Do NOT use `<bdi>` tags unless there is a clear, demonstrable need for bi-directional text isolation. This is generally not required for mathematical variables or simple text in Portuguese.
**CRITICAL: Do NOT add any summary/explanation beyond original Portuguese. NO `<img>` tags.** Output only HTML code block.
"""
    return [prompt, page_file_object]


def extract_html_from_response(response_text: str) -> str | None:
    match = re.search(r"```html\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # Fallback for cases where the AI might forget the markdown block
        trimmed_text = response_text.strip()
        if trimmed_text.startswith("<") and trimmed_text.endswith(">") and \
           re.search(r"<p|<div|<span|<table|<ul|<ol|<h[1-6]", trimmed_text, re.IGNORECASE): # Basic check for HTML content
            return trimmed_text
    return None


def generate_html_for_image_task(model_name, file_object_from_api, page_filename_local, local_img_path, cancel_event,
                                 status_callback_main_thread, current_page_num_in_doc, original_page_order_index):
    if cancel_event.is_set(): raise OperationCancelledError("Geração HTML (task) cancelada antes de iniciar.")

    base64_image_data = None
    try:
        with open(local_img_path, "rb") as img_file:
            base64_image_data = base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e_b64:
        status_callback_main_thread(
            f"  Aviso: Falha ao ler/codificar imagem {os.path.basename(local_img_path)} para Base64: {e_b64}")
        base64_image_data = None

    dimensions_tuple = ("desconhecida", "desconhecida")
    try:
        img = Image.open(local_img_path)
        dimensions_tuple = img.size
        img.close()
    except Exception as e:
        status_callback_main_thread(f"  Aviso: Falha ao ler dimensões de {page_filename_local}: {e}.")

    prompt_parts = create_html_prompt_with_desc(file_object_from_api, page_filename_local, dimensions_tuple,
                                                current_page_num_in_doc)

    # --- LÓGICA DE ESCALONAMENTO DE CONFIGURAÇÃO ---
    generation_config_dict = {'temperature': 0.1}
    if model_name == MODELO_ESCALONAMENTO:
        generation_config_dict['max_output_tokens'] = LIMITE_TOKENS_ESCALONAMENTO
        status_callback_main_thread(
            f"  Usando modelo de escalonamento '{model_name}' para pág. {current_page_num_in_doc} com limite de {LIMITE_TOKENS_ESCALONAMENTO} tokens.")

    generation_config = genai.types.GenerationConfig(**generation_config_dict)

    # Inicializa o modelo aqui dentro da task
    model = genai.GenerativeModel(model_name=model_name)

    response = None
    final_finish_reason = None
    html_body = None

    try:
        response = gemini_api_call_with_retry(model.generate_content, cancel_event, status_callback_main_thread,
                                              contents=prompt_parts, generation_config=generation_config)
    except Exception as e:
        status_callback_main_thread(
            f"  Exceção na API Gemini para {page_filename_local} (pág {current_page_num_in_doc}): {type(e).__name__} - {e}")
        return original_page_order_index, current_page_num_in_doc, None, base64_image_data, None  # Retorna None para motivo

    if cancel_event.is_set(): raise OperationCancelledError("Geração HTML (task) cancelada após chamada API.")

    if response and response.candidates:
        # Pega o motivo de finalização para análise posterior
        final_finish_reason = response.candidates[0].finish_reason.value

        response_text_content = response.text
        if not response_text_content and response.candidates[0].content and response.candidates[0].content.parts:
            response_text_content = ''.join(
                part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')
            )

        if response_text_content:
            html_body = extract_html_from_response(response_text_content)
            # ... (post-processing de bdi continua o mesmo)
            if html_body:
                html_body = re.sub(r'<bdi>([a-zA-Z0-9_](?:<sup>.*?</sup>)?)</bdi>', r'\1', html_body)
                html_body = re.sub(r'<bdi>(\\[a-zA-Z]+(?:\{.*?\})?(?:\s*\^\{.*?\})?(?:\s*_\{.*?\})?)</bdi>', r'\1',
                                   html_body)
                html_body = re.sub(r'<bdi>\s*</bdi>', '', html_body)

    if html_body is None:
        status_callback_main_thread(
            f"  Erro: Falha ao extrair HTML para {page_filename_local} (pág {current_page_num_in_doc}).")
        if response:
            status_callback_main_thread(f"  Texto bruto (300c): {str(response.text)[:300]}...")
            status_callback_main_thread(
                f"  Motivo: {final_finish_reason} ({response.candidates[0].finish_reason.name})")
        return original_page_order_index, current_page_num_in_doc, None, base64_image_data, final_finish_reason

    status_callback_main_thread(f"  HTML extraído para pág. PDF {current_page_num_in_doc} ({page_filename_local}).")
    return original_page_order_index, current_page_num_in_doc, html_body, base64_image_data, final_finish_reason


def create_merged_html_with_accessibility(content_list, pdf_filename_title, output_path=None, s3_client=None, s3_bucket=None, output_s3_key=None):
    if not content_list: return False

    accessibility_css = """
<style>
    html, body {margin: 0;padding: 0;overflow-x: auto;}
    body {font-family: Verdana, Arial, sans-serif; line-height: 1.6; padding: 20px; background-color: #f0f0f0; color: #333; transition: background-color 0.3s, color 0.3s;}
    #accessibility-controls {position: sticky; top: 0; z-index: 1000; padding: 10px; margin-bottom: 20px; border: 1px solid; border-radius: 5px; display: flex; flex-wrap: wrap; align-items: center; gap: 8px; box-sizing: border-box;}
    body.normal-mode #accessibility-controls:not(.expanded) {background-color: #e0e0e0; border-color: #ccc; color: #000;}
    body.dark-mode #accessibility-controls:not(.expanded) {background-color: #1e1e1e; border-color: #444; color: #fff;}
    body.high-contrast-mode #accessibility-controls:not(.expanded) {background-color: #000; border-color: #00FF00; color: #00FF00;}
    #accessibility-controls .control-group > button,
    #accessibility-controls .control-group > select,
    #accessibility-controls .control-group > label:not(:first-child),
    #accessibility-controls .control-group > span:not(:first-child) {display: inline-flex; align-items: center; justify-content: center; white-space: normal; min-width: 0; word-break: break-all; flex-grow: 0; flex-shrink: 1; flex-basis: auto; max-width: 100%; padding: 5px 10px; border-radius: 3px; box-sizing: border-box; cursor: pointer; text-align: center; margin: 0;}
    #accessibility-controls .control-group > *:first-child {width: 100%; flex-shrink: 0; box-sizing: border-box; margin-bottom: 8px; font-weight: bold; white-space: normal; word-break: break-word;}
    #accessibility-toggle img {pointer-events: none;}
    .page-content {padding: 15px; margin-bottom: 20px; border: 1px solid; border-radius: 3px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
    body.normal-mode {background-color: #f0f0f0; color: #333;}
    body.normal-mode .page-content {background-color: #ffffff;border-color: #dddddd;}
    body.normal-mode h1, body.normal-mode h2 { color: #000; border-color: #eee;}
    body.normal-mode p i, body.normal-mode span i, body.normal-mode p em, body.normal-mode span em { color: #555555 !important; }
    body.normal-mode sup > a { color: #0066cc; }
    body.normal-mode hr.page-separator { border-color: #ccc; }
    body.normal-mode hr.footnotes-separator { border-color: #ccc; }
    body.normal-mode #accessibility-controls.expanded {background-color: #f0f0f0; border-color: #ccc; color: #000;}
    body.normal-mode #accessibility-controls button, body.normal-mode #accessibility-controls select { background-color: #fff; border: 1px solid #bbb; color: #000; }
    body.dark-mode {background-color: #121212; color: #e0e0e0;}
    body.dark-mode .page-content {background-color: #1e1e1e;border-color: #444444;}
    body.dark-mode h1, body.dark-mode h2 {color: #ffffff; border-color: #444;}
    body.dark-mode p i, body.dark-mode span i, body.dark-mode p em, body.dark-mode span em { color: #AAAAAA !important; }
    body.dark-mode sup > a { color: #87CEFA; }
    body.dark-mode hr.page-separator { border-color: #555; }
    body.dark-mode hr.footnotes-separator { border-color: #555; }
    body.dark-mode #accessibility-controls.expanded {background-color: #2c2c2c; border-color: #555; color: #e0e0e0;}
    body.dark-mode #accessibility-controls button, body.dark-mode #accessibility-controls select { background-color: #333; border: 1px solid #555; color: #e0e0e0; }
    body.high-contrast-mode {background-color: #000000;color: #FFFF00;}
    body.high-contrast-mode .page-content {background-color: #000000;border: 2px solid #FFFF00;}
    body.high-contrast-mode h1, body.high-contrast-mode h2 {color: #FFFF00; border-color: #FFFF00;}
    body.high-contrast-mode p, body.high-contrast-mode span, body.high-contrast-mode li, body.high-contrast-mode td, body.high-contrast-mode th { color: #FFFF00 !important; }
    body.high-contrast-mode p i, body.high-contrast-mode span i, body.high-contrast-mode p em, body.high-contrast-mode span em {color: #01FF01 !important;}
    body.high-contrast-mode sup > a { color: #00FFFF; text-decoration: underline; }
    body.high-contrast-mode hr.page-separator { border: 2px dashed #FFFF00; }
    body.high-contrast-mode hr.footnotes-separator { border: 1px dotted #FFFF00; }
    body.high-contrast-mode #accessibility-controls.expanded {background-color: #000; border-color: #FFFF00; color: #FFFF00;}
    body.high-contrast-mode #accessibility-controls button, body.high-contrast-mode #accessibility-controls select { background-color: #111; color: #FFFF00; border: 1px solid #FFFF00;}
    h1 { font-size: 2em; border-bottom: 2px solid; padding-bottom: 0.3em; margin-top: 1em; margin-bottom: 0.5em; }
    h2 { font-size: 1.75em; border-bottom: 1px solid; padding-bottom: 0.3em; margin-top: 1em; margin-bottom: 0.5em;}
    hr.page-separator { margin-top: 2.5em; margin-bottom: 2.5em; border-width: 0; border-top: 2px dashed; }
    hr.footnotes-separator { margin-top: 1.5em; margin-bottom: 1em; border-style: dotted; border-width: 1px 0 0 0; }
    .footnotes-section { margin-top: 1em; padding-top: 0.5em; font-size: 0.9em; }
    .footnotes-list { list-style-type: decimal; padding-left: 25px; }
    .footnotes-list li { margin-bottom: 0.5em; }
    .footnotes-list li a { text-decoration: none; margin-left: 5px;}
    .footnotes-list li a:hover { text-decoration: underline; }
    .sr-only {position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border-width: 0;}
    p i, span i {font-style: italic;}
    sup > a {text-decoration: none;} sup > a:hover {text-decoration: underline;}
    table { border-collapse: collapse; width: auto; margin: 1em 0; }
    th, td { border: 1px solid; padding: 0.5em; text-align: left; }
    body.normal-mode th, body.normal-mode td { border-color: #ccc; }
    body.dark-mode th, body.dark-mode td { border-color: #555; }
    body.high-contrast-mode th, body.high-contrast-mode td { border-color: #FFFF00; }
    .MathJax_Display { margin: 1em 0 !important; } /* Ensure MathJax display math has proper margins */
    #accessibility-toggle { position: fixed; top: 20px; right: 20px; z-index: 1100; width: 50px; height: 50px; background-color: #007BFF; color: white; border: none; border-radius: 50%; font-size: 24px; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2); }
    body.dark-mode #accessibility-toggle { background-color: #4dabf7; }
    body.high-contrast-mode #accessibility-toggle { background-color: #FFFF00; color: #000; border: 1px solid #000;}
    body.high-contrast-mode #accessibility-toggle img { filter: invert(1) brightness(0.8); }
    #accessibility-controls.collapsed { display: none; }
    #accessibility-controls.expanded {position: fixed; top: 80px; right: 20px; width: 100%; max-width: 360px; box-sizing: border-box; overflow: auto; max-height: calc(100vh - 100px);}
    .control-group {display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 10px; margin-bottom: 15px; border: 1px solid; border-radius: 4px; box-sizing: border-box;}
    body.normal-mode .control-group { border-color: #bbb; }
    body.dark-mode .control-group { border-color: #555; }
    body.high-contrast-mode .control-group { border-color: #FFFF00; }

    /* Styles for embedded original page image viewer */
    details.original-page-viewer {
        margin-top: 20px;
        margin-bottom: 20px;
        border: 1px dashed #999;
        border-radius: 5px;
        padding: 10px;
        background-color: rgba(0,0,0,0.02); /* Light background for normal mode */
    }
    body.dark-mode details.original-page-viewer {
        border-color: #666;
        background-color: rgba(255,255,255,0.03); /* Slightly lighter for dark mode */
    }
    body.high-contrast-mode details.original-page-viewer {
        border-color: #FFFF00;
        background-color: #111; /* Dark for high contrast */
    }
    details.original-page-viewer summary {
        cursor: pointer;
        padding: 8px;
        font-weight: bold;
        color: #0056b3; /* Link-like color for normal mode */
        background-color: rgba(0,0,0,0.03);
        border-radius: 3px;
        margin: -10px -10px 10px -10px; /* Extend to edges of padding */
    }
    body.dark-mode details.original-page-viewer summary {
        color: #87CEFA; /* Lighter blue for dark mode */
        background-color: rgba(255,255,255,0.05);
    }
    body.high-contrast-mode details.original-page-viewer summary {
        color: #00FFFF; /* Cyan for high contrast */
        background-color: #222;
        border: 1px solid #FFFF00;
    }
    details.original-page-viewer summary:hover,
    details.original-page-viewer summary:focus {
        text-decoration: underline;
        background-color: rgba(0,0,0,0.05);
    }
    body.dark-mode details.original-page-viewer summary:hover,
    body.dark-mode details.original-page-viewer summary:focus {
        background-color: rgba(255,255,255,0.08);
    }
    body.high-contrast-mode details.original-page-viewer summary:hover,
    body.high-contrast-mode details.original-page-viewer summary:focus {
        background-color: #333;
    }
    details.original-page-viewer img {
        display: block;
        max-width: 100%;
        height: auto;
        margin-top: 10px;
        border: 1px solid #ccc; /* Border for the image */
        background-color: white; /* Background for transparent PNGs, if any */
    }
    body.dark-mode details.original-page-viewer img {
        border-color: #444;
        background-color: #333; /* Darker background for image in dark mode */
    }
    body.high-contrast-mode details.original-page-viewer img {
        border-color: #FFFF00; /* Yellow border in high contrast */
        background-color: #000;
    }
    .tts-highlight {
        background-color: yellow !important;
        color: black !important; /* Força o texto para a cor preta */
        box-shadow: 0 0 8px rgba(218, 165, 32, 0.7); /* Sombra sutil dourada/amarela */
        transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
        border-radius: 3px;
    }
        .dark-mode .tts-highlight { background-color: #58a6ff; }
        .high-contrast-mode .tts-highlight { background-color: #FFFF00; color: black !important; }
</style>
"""
    accessibility_js = r"""
<script>
// --- CONFIGURAÇÃO E ESTADO GLOBAL ---
let currentFontSize = 16;
const fonts = ['Atkinson Hyperlegible', 'Lexend', 'OpenDyslexicRegular', 'Verdana', 'Arial', 'Times New Roman', 'Courier New'];
let currentFontIndex = 0;

const synth = window.speechSynthesis;
let voices = [];
let utterance = null;
let isPaused = false;

// Estado da fila de leitura
let speechQueue = [];
let currentSegmentIndex = 0;
let currentlyHighlightedElement = null;


// --- FUNÇÕES DE SÍNTESE DE VOZ E PROCESSAMENTO DE TEXTO ---

// NOVA FUNÇÃO PARA PROCESSAR TABELAS DE FORMA INTELIGENTE
// VERSÃO APRIMORADA E MAIS ROBUSTA
function processTable(tableNode) {
    let content = [];
    const caption = tableNode.querySelector('caption');
    if (caption && caption.innerText.trim()) {
        content.push(`Iniciando tabela com título: ${caption.innerText.trim()}.`);
    } else {
        content.push("Iniciando tabela.");
    }

    const headers = [];
    // Procura por cabeçalhos (th) na primeira linha da tabela
    const firstRow = tableNode.querySelector('tr');
    if (firstRow) {
        firstRow.querySelectorAll('th').forEach(th => {
            headers.push(th.innerText.trim());
        });
    }

    const allRows = Array.from(tableNode.querySelectorAll('tr'));
    // Se encontramos cabeçalhos na primeira linha, pulamos ela no loop principal
    const bodyRows = headers.length > 0 ? allRows.slice(1) : allRows;

    bodyRows.forEach(row => {
        let rowContent = [];
        // Lida com o cabeçalho da linha (um 'th' no início da linha)
        const rowHeader = row.querySelector('th');
        if (rowHeader && rowHeader.innerText.trim()) {
            rowContent.push(`Linha: ${rowHeader.innerText.trim()}.`);
        }

        const cells = row.querySelectorAll('td');
        cells.forEach((cell, index) => {
            // Associa a célula ao cabeçalho da coluna pelo índice
            const headerText = headers[index] || `Coluna ${index + 1}`;
            const cellText = cell.innerText.trim();
            if (cellText) {
                rowContent.push(`${headerText}: ${cellText}.`);
            }
        });

        if (rowContent.length > 0) {
            content.push(rowContent.join(' '));
        }
    });

    // Salvaguarda: se a análise estruturada falhar, lê o texto bruto da tabela.
    if (content.length <= 1) { // Apenas contém "Iniciando tabela."
        const fallbackText = tableNode.innerText.trim().replace(/\s+/g, ' ');
        if (fallbackText) {
            return { text: "Tabela encontrada. Conteúdo: " + fallbackText, element: tableNode };
        } else {
            return { text: "Tabela encontrada, mas está vazia.", element: tableNode };
        }
    }

    return {
        text: content.join(' '),
        element: tableNode
    };
}

function populateVoiceList() {
    voices = synth.getVoices().sort((a, b) => a.lang.localeCompare(b.lang));
    const voiceSelector = document.getElementById('voiceSelector');
    if (!voiceSelector) return;
    voiceSelector.innerHTML = '';

    const ptVoices = voices.filter(voice => voice.lang.startsWith('pt'));
    //const otherVoices = voices.filter(voice => !voice.lang.startsWith('pt'));
    const sortedVoices = [...ptVoices];

    sortedVoices.forEach(voice => {
        const option = document.createElement('option');
        option.textContent = `${voice.name} (${voice.lang})`;
        option.setAttribute('data-lang', voice.lang);
        option.setAttribute('data-name', voice.name);
        voiceSelector.appendChild(option);
    });

    const savedVoiceName = localStorage.getItem('accessibilityVoiceName');
    if (savedVoiceName) {
        const savedOption = Array.from(voiceSelector.options).find(opt => opt.textContent.includes(savedVoiceName));
        if (savedOption) savedOption.selected = true;
    }
}


// VERSÃO APRIMORADA E NÃO RECURSIVA
function extractContentWithSemantics(rootNode) {
    const segments = [];
    // 1. Pega uma lista de todos os elementos de conteúdo relevantes na ordem do documento
    const elementsToProcess = rootNode.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, blockquote, table');

    elementsToProcess.forEach(node => {
        // 2. Pula qualquer elemento que esteja DENTRO de uma tabela (como um <p> numa célula),
        //    pois a função processTable cuidará dele.
        if (node.closest('table') && node.tagName !== 'TABLE') {
            return;
        }

        // 3. Se o elemento é uma tabela, usa nosso processador especializado
        if (node.tagName === 'TABLE') {
            const tableSegment = processTable(node);
            if (tableSegment && tableSegment.text) {
                segments.push(tableSegment);
            }
        // 4. Se for qualquer outro tipo de bloco de texto, processa seu conteúdo
        } else {
            let prefix = '';
            if (node.tagName === 'LI') {
                prefix = 'Item da lista: ';
            }
            const text = (node.innerText || node.textContent).trim();
            if (text) {
                segments.push({
                    text: prefix + text.replace(/\s+/g, ' '),
                    element: node
                });
            }
        }
    });

    return segments;
}


function speakText() {
    if (synth.speaking && !isPaused) return;

    if (synth.paused && utterance) {
        synth.resume();
        return;
    }

    // Se a fila está vazia, cria uma nova
    if (speechQueue.length === 0) {
        let rootNode;
        const selectedText = window.getSelection().toString().trim();
        if (selectedText) {
            speechQueue = [{ text: "Texto selecionado: " + selectedText, element: null }];
        } else {
            rootNode = document.getElementById('main-content') || document.body;
            speechQueue = extractContentWithSemantics(rootNode);
        }
    }

    // Define o ponto de partida e inicia a leitura em cadeia
    currentSegmentIndex = 0;
    playQueue();
}

function playQueue() {
    if (currentSegmentIndex >= speechQueue.length) {
        stopSpeech();
        return;
    }

    const segment = speechQueue[currentSegmentIndex];
    if (!segment || !segment.text) {
        currentSegmentIndex++;
        playQueue();
        return;
    }

    // Limpa destaque anterior e aplica o novo
    if (currentlyHighlightedElement) {
        currentlyHighlightedElement.classList.remove('tts-highlight');
    }
    if (segment.element) {
        currentlyHighlightedElement = segment.element;
        currentlyHighlightedElement.classList.add('tts-highlight');
        currentlyHighlightedElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    utterance = new SpeechSynthesisUtterance(segment.text);

    // Configura voz, velocidade e tom
    const voiceSelector = document.getElementById('voiceSelector');
    const selectedOption = voiceSelector.options[voiceSelector.selectedIndex];
    if (selectedOption) {
        const voiceName = selectedOption.getAttribute('data-name');
        utterance.voice = voices.find(voice => voice.name === voiceName);
    }
    utterance.rate = parseFloat(document.getElementById('rateSlider').value);
    utterance.pitch = parseFloat(document.getElementById('pitchSlider').value);

    utterance.onerror = (event) => {
        console.error('SpeechSynthesisUtterance.onerror', event);
        if (currentlyHighlightedElement) currentlyHighlightedElement.classList.remove('tts-highlight');
    };

    // Ao terminar, avança para o próximo item da fila
    utterance.onend = () => {
        currentSegmentIndex++;
        playQueue();
    };

    // Trata a retomada da pausa
    utterance.onresume = () => {
        isPaused = false;
        if (currentlyHighlightedElement) currentlyHighlightedElement.classList.add('tts-highlight');
    };

    // Trata a pausa
    utterance.onpause = () => {
        isPaused = true;
        if (currentlyHighlightedElement) currentlyHighlightedElement.classList.remove('tts-highlight');
    };

    synth.speak(utterance);
    isPaused = false;
}

function pauseSpeech() {
    if (synth.speaking && !isPaused) {
        synth.pause();
    }
}

function stopSpeech() {
    if (utterance) utterance.onend = null; // Impede que o onend seja chamado após o cancelamento
    isPaused = false;
    synth.cancel();

    if (currentlyHighlightedElement) {
        currentlyHighlightedElement.classList.remove('tts-highlight');
        currentlyHighlightedElement = null;
    }

    speechQueue = [];
    currentSegmentIndex = 0;
    utterance = null;
}

// --- FUNÇÕES DE NAVEGAÇÃO (LÓGICA REFEITA) ---

function skipToPrevious() {
    // Só funciona se houver uma fila e não estiver no primeiro item
    if (speechQueue.length === 0 || currentSegmentIndex <= 0) {
        return;
    }

    // 1. Move o índice para o item anterior
    currentSegmentIndex--;

    // 2. Para a fala atual, desanexando o 'onend' para evitar chamadas duplas
    if (utterance) utterance.onend = null;
    synth.cancel();

    // 3. Toca o novo item com um pequeno atraso para o navegador processar o cancelamento
    setTimeout(playQueue, 100);
}

function skipToNext() {
    // Só funciona se houver uma fila e não estiver no último item
    if (speechQueue.length === 0 || currentSegmentIndex >= speechQueue.length - 1) {
        stopSpeech();
        return;
    }

    // 1. Move o índice para o próximo item
    currentSegmentIndex++;

    // 2. Para a fala atual, desanexando o 'onend'
    if (utterance) utterance.onend = null;
    synth.cancel();

    // 3. Toca o novo item com um pequeno atraso
    setTimeout(playQueue, 100);
}


// --- FUNÇÕES DE ACESSIBILIDADE VISUAL E CONTROLES ---

function updateFontSizeDisplay() {
    const fontSizeValue = document.getElementById('fontSizeValue');
    if (fontSizeValue) fontSizeValue.textContent = `${currentFontSize}px`;
}

function changeFontSize(delta) {
    currentFontSize += delta;
    if (currentFontSize < 10) currentFontSize = 10;
    if (currentFontSize > 48) currentFontSize = 48;
    document.body.style.fontSize = currentFontSize + 'px';
    localStorage.setItem('accessibilityFontSize', currentFontSize);
    updateFontSizeDisplay();
}

function setFontFamily(fontName) {
    document.body.style.fontFamily = fontName + ', sans-serif';
    localStorage.setItem('accessibilityFontFamily', fontName);
}

function changeTheme(themeName) {
    document.body.className = '';
    document.body.classList.add(themeName + '-mode');
    localStorage.setItem('accessibilityTheme', themeName);
}

function updateSliderLabels() {
    const rateSlider = document.getElementById('rateSlider');
    const rateValue = document.getElementById('rateValue');
    const pitchSlider = document.getElementById('pitchSlider');
    const pitchValue = document.getElementById('pitchValue');

    if(rateValue) rateValue.textContent = `${parseFloat(rateSlider.value).toFixed(1)}x`;
    if(pitchValue) pitchValue.textContent = parseFloat(pitchSlider.value).toFixed(1);
}

function saveSpeechSettings() {
    const voiceSelector = document.getElementById('voiceSelector');
    if (voiceSelector) {
        const selectedVoice = voiceSelector.options[voiceSelector.selectedIndex];
        if (selectedVoice) {
            localStorage.setItem('accessibilityVoiceName', selectedVoice.getAttribute('data-name'));
        }
    }
    localStorage.setItem('accessibilityRate', document.getElementById('rateSlider').value);
    localStorage.setItem('accessibilityPitch', document.getElementById('pitchSlider').value);
}

function toggleAccessibilityMenu() {
    const menu = document.getElementById('accessibility-controls');
    const toggleButton = document.getElementById('accessibility-toggle');
    const isExpanded = menu.classList.toggle('expanded');
    menu.classList.toggle('collapsed', !isExpanded);
    toggleButton.setAttribute('aria-expanded', isExpanded.toString());
    toggleButton.setAttribute('aria-label', isExpanded ? 'Fechar Menu de Acessibilidade' : 'Abrir Menu de Acessibilidade');
}


// --- INICIALIZAÇÃO ---
document.addEventListener('DOMContentLoaded', () => {
    if (synth.onvoiceschanged !== undefined) {
        synth.onvoiceschanged = populateVoiceList;
    }
    populateVoiceList();

    const savedTheme = localStorage.getItem('accessibilityTheme') || 'dark';
    changeTheme(savedTheme);
    document.getElementById('themeSelector').value = savedTheme;

    const savedFontSize = parseInt(localStorage.getItem('accessibilityFontSize'), 10) || 16;
    currentFontSize = savedFontSize;
    document.body.style.fontSize = currentFontSize + 'px';
    updateFontSizeDisplay();

    const savedFontFamily = localStorage.getItem('accessibilityFontFamily') || fonts[0];
    setFontFamily(savedFontFamily);
    document.getElementById('fontSelector').value = savedFontFamily;

    const savedRate = localStorage.getItem('accessibilityRate') || '1';
    document.getElementById('rateSlider').value = savedRate;
    const savedPitch = localStorage.getItem('accessibilityPitch') || '1';
    document.getElementById('pitchSlider').value = savedPitch;

    updateSliderLabels();

    document.addEventListener('keydown', function(event) {
        if (event.key === "Escape") {
            stopSpeech();
        }
    });

    const menu = document.getElementById('accessibility-controls');
    const toggleButton = document.getElementById('accessibility-toggle');
    if (menu && !menu.classList.contains('expanded')) {
        menu.classList.add('collapsed');
    }
    if (toggleButton && menu) {
        const isExpanded = menu.classList.contains('expanded');
        toggleButton.setAttribute('aria-expanded', isExpanded.toString());
        toggleButton.setAttribute('aria-label', isExpanded ? 'Fechar Menu de Acessibilidade' : 'Abrir Menu de Acessibilidade');
    }
});
</script>

"""
    safe_title = html.escape(pdf_filename_title if pdf_filename_title else "Documento")
    mathjax_config_head_merged = f"""
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accessible Document: {safe_title}</title>
    <script>
    MathJax = {{
        tex: {{
            inlineMath: [['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$']],
            processEscapes: true,
            processEnvironments: true,
            tags: 'ams'
        }},
        options: {{
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
            ignoreHtmlClass: 'tex2jax_ignore',
            processHtmlClass: 'tex2jax_process'
        }},
        svg: {{
            fontCache: 'global'
        }}
    }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&family=Lexend:wght@100..900&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/antijingoist/open-dyslexic@master/open-dyslexic-regular.css">
    {accessibility_css}
    {accessibility_js}
</head>
"""

    safe_h1_title = html.escape(pdf_filename_title if pdf_filename_title else "Documento")
    merged_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
{mathjax_config_head_merged}
<body class="dark-mode"> <header role="banner"><h1>Documento Acessível: {safe_h1_title}</h1></header>
    <button id="accessibility-toggle" onclick="toggleAccessibilityMenu()" aria-label="Abrir Menu de Acessibilidade" aria-expanded="false">
        <img src="https://cdn.userway.org/widgetapp/images/body_wh.svg" alt="" style="width: 130%; height: 130%;"/>
    </button>

    <div id="accessibility-controls" class="collapsed" role="region" aria-labelledby="accessibility-menu-heading">
        <h2 id="accessibility-menu-heading" class="sr-only">Menu de Controles de Acessibilidade</h2>

        <div class="control-group">
            <span>Tamanho da Fonte: <span id="fontSizeValue" aria-live="polite">16px</span></span>
            <button onclick="changeFontSize(-2)" aria-label="Diminuir tamanho da fonte">A-</button>
            <button onclick="changeFontSize(2)" aria-label="Aumentar tamanho da fonte">A+</button>
        </div>

        <div class="control-group">
            <label for="fontSelector">Fonte:</label>
            <select id="fontSelector" onchange="setFontFamily(this.value)" aria-label="Selecionar família da fonte">
                <option value="Atkinson Hyperlegible">Atkinson Hyperlegible</option><option value="Lexend">Lexend</option>
                <option value="OpenDyslexicRegular">OpenDyslexic</option><option value="Verdana">Verdana</option>
                <option value="Arial">Arial</option><option value="Times New Roman">Times New Roman</option>
                <option value="Courier New">Courier New</option>
            </select>
        </div>

        <div class="control-group">
            <label for="themeSelector">Tema Visual:</label>
            <select id="themeSelector" onchange="changeTheme(this.value)" aria-label="Selecionar tema visual">
                <option value="normal">Modo Claro</option>
                <option value="dark">Modo Escuro</option>
                <option value="high-contrast">Alto Contraste</option>
            </select>
        </div>
        <div class="control-group">
            <span>Leitura em Voz Alta:</span>
            <button onclick="speakText()" aria-label="Ler ou continuar leitura">▶️ Ler/Continuar</button>
            <button onclick="pauseSpeech()" aria-label="Pausar leitura">⏸️ Pausar</button>
            <button onclick="stopSpeech()" aria-label="Parar leitura (Tecla Esc)">⏹️ Parar (Esc)</button>
        </div>

        <div class="control-group">
            <span>Navegar no Texto:</span>
            <button onclick="skipToPrevious()" aria-label="Ler segmento anterior">⏪ Anterior</button>
            <button onclick="skipToNext()" aria-label="Ler próximo segmento">Próximo ⏩</button>
        </div>

        <div class="control-group">
            <label for="voiceSelector">Voz:</label>
            <select id="voiceSelector" aria-label="Selecionar voz"></select>
        </div>

        <div class="control-group">
            <label for="rateSlider">Velocidade:</label>
            <input type="range" id="rateSlider" min="0.5" max="2" step="0.1" value="1" oninput="updateSliderLabels(); saveSpeechSettings();">
            <span id="rateValue" aria-live="polite">1x</span>
        </div>

        <div class="control-group">
            <label for="pitchSlider">Tom:</label>
            <input type="range" id="pitchSlider" min="0" max="2" step="0.1" value="1" oninput="updateSliderLabels(); saveSpeechSettings();">
            <span id="pitchValue" aria-live="polite">1</span>
        </div>

    </div>
    <main id="main-content" role="main">
"""
    for i, content_data in enumerate(content_list):
        page_num_in_doc = content_data["page_num_in_doc"]
        html_body = content_data.get("body", "") # Default to empty string if not present
        base64_image = content_data.get("base64_image") # Get base64 image data

        if i > 0: merged_html += f"\n<hr class=\"page-separator\" aria-hidden=\"true\">\n"

        merged_html += f"<article class='page-content' id='page-{page_num_in_doc}' aria-labelledby='page-heading-{page_num_in_doc}'>\n"
        merged_html += f"<h2 id='page-heading-{page_num_in_doc}'>Página {page_num_in_doc}</h2>\n"

        merged_html += html_body if html_body else f"<p><i>[Conteúdo não pôde ser extraído para a página {page_num_in_doc}.]</i></p>"

        # Add the <details> section for the original image if HTML body exists, base64 image exists,
        # and the AI was instructed to provide a description (heuristic: "[Descrição:" is present).
        if html_body and base64_image and "[Descrição:" in html_body:
            safe_alt_text = html.escape(f"Imagem original da página {page_num_in_doc}")
            merged_html += f"""
                <details class="original-page-viewer">
                    <summary>Ver Imagem da Página Original {page_num_in_doc}</summary>
                    <div style="text-align: center; padding: 10px;">
                        <img src="data:image/png;base64,{base64_image}" alt="{safe_alt_text}" style="max-width: 100%; height: auto;" aria-hidden="true">
                    </div>
                </details>
            """
        merged_html += "\n</article>\n"
    merged_html += "\n    </main>\n</body>\n</html>"

    if MODE == "LOCAL":
        try:
            print(output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(merged_html)
            return True
        except Exception as e:
            raise IOError(f"Falha ao escrever arquivo HTML: {e}")
    elif MODE == "S3":
        try:
            # Converte a string HTML para bytes
            html_bytes = merged_html.encode('utf-8')
            # Faz o upload dos bytes diretamente para o S3
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=output_s3_key,
                Body=html_bytes,
                ContentType='text/html',
                ContentEncoding='utf-8'
            )
            return True
        except ClientError as e:
            # Lança um erro que será capturado pela função principal
            raise IOError(f"Falha ao fazer upload do arquivo HTML para o S3: {e}")
    return None


def process_pdf_web(
        dpi, page_range_str, selected_model_name,
        num_upload_workers, num_generate_workers,
        cancel_event, status_callback, completion_callback, progress_callback,
        s3_bucket=None, s3_pdf_object_name=None, output_s3_key=None, pdf_path=None, initial_output_html_path_base=None
):
    if MODE == "LOCAL":
        image_paths = []
        temp_image_dir = ""
    elif MODE == "S3":
        s3_client = boto3.client('s3', region_name=os.environ.get('S3_REGION'))

    uploaded_files_map = {}

    def phase_progress_callback(step, total, text_prefix):
        phase_map = {
            "Convertendo": (0, 30), "Upload": (30, 60), "Gerando": (60, 90),
            "Mesclando": (90, 95), "Limpando": (95, 100)
        }
        phase_start, phase_end = next(((s, e) for k, (s, e) in phase_map.items() if k in text_prefix), (0, 100))

        phase_duration = phase_end - phase_start
        phase_progress = (step / total) * phase_duration if total > 0 else 0
        overall_progress = int(phase_start + phase_progress)

        progress_callback(overall_progress, 100, f"{text_prefix} ({step}/{total})")

    try:
        status_callback("Iniciando processo...")
        progress_callback(0, 100, "Inicializando")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf_file:
            if MODE == "LOCAL":
                pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
            elif MODE == "S3":
                status_callback(f"Baixando PDF do armazenamento: {s3_pdf_object_name}")
                s3_client.download_fileobj(s3_bucket, s3_pdf_object_name, tmp_pdf_file)
                pdf_path = tmp_pdf_file.name  # Agora temos um caminho local temporário
                pdf_basename = os.path.splitext(os.path.basename(s3_pdf_object_name))[0]

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_image_dir = os.path.join('temp_processing', f"{pdf_basename}_{timestamp}")
        os.makedirs(temp_image_dir, exist_ok=True)
        status_callback(f"Diretório temporário: {temp_image_dir}")

        api_key = os.environ.get(API_KEY_ENV_VAR)
        if not api_key: raise ValueError(f"Chave API '{API_KEY_ENV_VAR}' não encontrada.")
        genai.configure(api_key=api_key)
        status_callback("Chave API configurada.")

        with fitz.open(pdf_path) as temp_doc:
            total_doc_pages = temp_doc.page_count

        selected_pages_0_indexed = parse_page_ranges(page_range_str, total_doc_pages)
        if selected_pages_0_indexed is None: raise ValueError(f"Intervalo de páginas inválido: '{page_range_str}'.")

        image_paths = pdf_to_images_local(pdf_path, temp_image_dir, dpi, selected_pages_0_indexed, cancel_event,
                                          status_callback, phase_progress_callback)
        if not image_paths: raise Exception("Falha ao converter PDF para imagens.")
        if cancel_event.is_set(): raise OperationCancelledError("Cancelado após conversão.")

        uploaded_files_map = upload_to_gemini_file_api(image_paths, num_upload_workers, cancel_event, status_callback,
                                                       phase_progress_callback)
        if not uploaded_files_map: raise Exception("Falha ao fazer upload de imagens para API Gemini.")
        if cancel_event.is_set(): raise OperationCancelledError("Cancelado após upload.")

        # --- Início da Lógica de Geração de HTML (Substituindo a simulação) ---
        num_files_successfully_uploaded = len(uploaded_files_map)
        status_callback(
            f"Gerando HTML para {num_files_successfully_uploaded} imagem(ns) usando o modelo base '{selected_model_name}'...")

        local_path_to_original_page_num = {
            img_path: int(m.group(1)) if (m := re.search(r"page_(\d+)\.\w+$", os.path.basename(img_path))) else -1
            for img_path in uploaded_files_map.keys()
        }

        tasks_for_html_generation = []
        sorted_uploaded_image_paths = sorted(
            uploaded_files_map.keys(),
            key=lambda x: local_path_to_original_page_num.get(x, float('inf'))
        )

        for original_order_idx, local_img_path in enumerate(sorted_uploaded_image_paths):
            page_num_in_doc = local_path_to_original_page_num.get(local_img_path, -1)
            file_api_object = uploaded_files_map.get(local_img_path)
            if page_num_in_doc == -1 or not file_api_object:
                status_callback(f"  Pulando {os.path.basename(local_img_path)} para geração HTML (dados inválidos).")
                continue

            task_args = (file_api_object, os.path.basename(local_img_path),
                         local_img_path, cancel_event, status_callback, page_num_in_doc, original_order_idx)
            tasks_for_html_generation.append({
                'args': task_args,
                'original_idx': original_order_idx,
                'model_to_use': selected_model_name
            })

        temp_generated_html_results = [None] * len(tasks_for_html_generation)
        current_html_tasks_for_retry_phase = list(tasks_for_html_generation)

        for attempt in range(PHASE_MAX_RETRIES + 1):
            if not current_html_tasks_for_retry_phase or cancel_event.is_set(): break
            total_in_this_attempt = len(current_html_tasks_for_retry_phase)
            if attempt > 0:
                status_callback(
                    f"Retentativa de Geração HTML (Fase {attempt + 1}) para {total_in_this_attempt} páginas...")

            completed_in_this_attempt = 0
            failed_tasks_for_next_round = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_generate_workers) as executor:
                future_to_task_info = {
                    executor.submit(generate_html_for_image_task, task_info['model_to_use'],
                                    *task_info['args']): task_info
                    for task_info in current_html_tasks_for_retry_phase
                }
                for future in concurrent.futures.as_completed(future_to_task_info):
                    if cancel_event.is_set(): break
                    original_task_info = future_to_task_info[future]
                    idx_for_assignment = original_task_info['original_idx']

                    try:
                        returned_idx, page_num, html_body, base64_data, finish_reason = future.result()
                        if html_body:
                            temp_generated_html_results[idx_for_assignment] = {
                                "page_num_in_doc": page_num, "body": html_body, "base64_image": base64_data
                            }
                        else:
                            if finish_reason == FINISH_REASON_MAX_TOKENS:
                                status_callback(
                                    f"  Falha por MAX_TOKENS na pág. {page_num}. Escalando modelo para retentativa...")
                                original_task_info['model_to_use'] = MODELO_ESCALONAMENTO
                            else:
                                status_callback(
                                    f"  HTML não gerado para pág. {page_num}. Motivo: {finish_reason}. Agendando retentativa.")
                            failed_tasks_for_next_round.append(original_task_info)
                    except OperationCancelledError:
                        failed_tasks_for_next_round.append(original_task_info)
                        break
                    except Exception as e:
                        status_callback(
                            f"  Erro ao processar resultado HTML para pág. (índice {idx_for_assignment}): {e}")
                        failed_tasks_for_next_round.append(original_task_info)
                    finally:
                        completed_in_this_attempt += 1
                        phase_progress_callback(completed_in_this_attempt, total_in_this_attempt, "Gerando HTML")

            current_html_tasks_for_retry_phase = failed_tasks_for_next_round
            if not current_html_tasks_for_retry_phase or cancel_event.is_set(): break

        if cancel_event.is_set(): raise OperationCancelledError("Cancelado durante geração HTML.")

        generated_content_list_thread = [res for res in temp_generated_html_results if res is not None]

        num_successfully_generated = len(generated_content_list_thread)
        total_attempted = len(tasks_for_html_generation)
        if num_successfully_generated < total_attempted:
            status_callback(
                f"AVISO: {num_successfully_generated}/{total_attempted} páginas foram processadas com sucesso.")

        if not generated_content_list_thread:
            raise Exception("Nenhum conteúdo HTML foi gerado com sucesso após todas as tentativas.")

        # --- Fim da Lógica de Geração de HTML ---

        status_callback("Mesclando HTML...")
        phase_progress_callback(0, 1, "Mesclando")

        if MODE == "LOCAL":
            success_merge = create_merged_html_with_accessibility(generated_content_list_thread, pdf_basename, output_path=initial_output_html_path_base)
        elif MODE == "S3":
            success_merge = create_merged_html_with_accessibility(generated_content_list_thread, pdf_basename, s3_client=s3_client,s3_bucket=s3_bucket, output_s3_key=output_s3_key)
        if not success_merge:
            raise Exception("Falha ao criar arquivo HTML mesclado.")

        if MODE == "LOCAL":
            completion_callback(True, initial_output_html_path_base)
        elif MODE == "S3":
            completion_callback(True, output_s3_key)

    except Exception as e:
        import traceback
        error_details = f"Erro: {type(e).__name__} - {e}\n{traceback.format_exc()}"
        status_callback(error_details)
        completion_callback(False, str(e))
    finally:
        status_callback("Iniciando limpeza final de recursos...")
        phase_progress_callback(0, 1, "Limpando")
        if temp_image_dir and os.path.exists(temp_image_dir):
            try:
                shutil.rmtree(temp_image_dir)
                status_callback(f"  Diretório temporário removido.")
            except Exception as e_rm:
                status_callback(f"  Aviso: Falha ao remover diretório temporário: {e_rm}")

        if uploaded_files_map:
            status_callback(f"  Limpando {len(uploaded_files_map)} arquivos da API...")
            for file_obj in uploaded_files_map.values():
                try:
                    # Adicionado um pequeno delay para não sobrecarregar a API de delete
                    time.sleep(0.1)
                    genai.delete_file(file_obj.name)
                except Exception as e_del:
                    status_callback(f"  Aviso: Falha ao deletar arquivo da API {file_obj.name}: {e_del}")

        phase_progress_callback(1, 1, "Limpando")
        status_callback("Limpeza finalizada.")