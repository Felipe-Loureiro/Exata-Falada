import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox, font, simpledialog
import fitz  # PyMuPDF
import google.generativeai as genai
import os
import threading
import time
import re
import mimetypes
from PIL import Image  # Para pegar dimensões da imagem
from google.api_core import exceptions as google_exceptions
import queue  # Para comunicação de senha
import shutil  # Para rmtree
import concurrent.futures  # Para processamento concorrente
import base64 # For embedding images
import html # For escaping HTML special characters in title

# --- Configuração Inicial ---
API_KEY_ENV_VAR = "GOOGLE_API_KEY"
DEFAULT_GEMINI_MODEL = 'gemini-2.0-flash'
AVAILABLE_GEMINI_MODELS = [
    'gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite',
    'gemini-2.5-flash-preview-05-20', 'gemini-1.5-flash-8b',
]

MAX_RETRIES_PER_CALL = 2
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 8  # seconds
PHASE_MAX_RETRIES = 1

DEFAULT_UPLOAD_MAX_WORKERS = 5
DEFAULT_GENERATE_MAX_WORKERS = 3


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
    retries = 0;
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
                status_callback(f"  Falha API após {MAX_RETRIES_PER_CALL} retentativas: {e}.");
                raise
            status_callback(
                f"  Erro API ({type(e).__name__}): {e}. Retentativa {retries}/{MAX_RETRIES_PER_CALL} em {backoff_time}s...")
            for _ in range(int(backoff_time)):
                if cancel_event.is_set(): raise OperationCancelledError("Cancelado durante espera retentativa.")
                time.sleep(1)
            backoff_time = min(backoff_time * 2, MAX_BACKOFF)
        except google_exceptions.ResourceExhausted as re_e:
            status_callback(f"  Erro Recurso Esgotado API: {re_e}. Verifique cotas.");
            raise
        except Exception as general_e:
            status_callback(f"  Erro inesperado chamada API: {type(general_e).__name__} - {general_e}");
            raise


def pdf_to_images_local(pdf_path, output_dir, dpi, selected_pages, cancel_event, status_callback,
                        password_request_callback, progress_callback):
    image_paths = []
    os.makedirs(output_dir, exist_ok=True)
    doc = None
    try:
        doc = fitz.open(pdf_path)
        if doc.is_encrypted:
            status_callback("  PDF protegido. Solicitando senha...")
            password = password_request_callback()
            if password is None: status_callback("  Entrada de senha cancelada."); return None
            if not doc.authenticate(password): status_callback("  Senha incorreta."); return None
            status_callback("  PDF descriptografado.")
        total_selected_pages = len(selected_pages)
        status_callback(f"Convertendo {total_selected_pages} págs PDF para imagens (DPI: {dpi})...")
        for i, page_num_0_indexed in enumerate(selected_pages):
            if cancel_event.is_set(): raise OperationCancelledError("Conversão PDF->Imagem cancelada.")
            page = doc.load_page(page_num_0_indexed)
            zoom = dpi / 72.0;
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            output_filename = os.path.join(output_dir, f"page_{page_num_0_indexed + 1:05d}.png")
            pix.save(output_filename);
            image_paths.append(output_filename)
            progress_callback(i + 1, total_selected_pages, f"Convertendo pág. PDF {page_num_0_indexed + 1}")
        status_callback("Conversão PDF->Imagens completa.");
        return image_paths
    except OperationCancelledError:
        status_callback("  Conversão PDF->Imagens cancelada."); raise
    except Exception as e:
        status_callback(f"Erro conversão PDF->Imagens: {e}"); return None
    finally:
        if doc: doc.close()


def _upload_single_image_task(image_path, display_name, mime_type, cancel_event, status_callback_main_thread):
    if cancel_event.is_set(): raise OperationCancelledError("Upload de imagem individual cancelado antes de iniciar.")
    try:
        file_to_upload = gemini_api_call_with_retry(
            genai.upload_file, cancel_event, status_callback_main_thread,
            path=image_path, display_name=display_name, mime_type=mime_type
        )
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
            if time.time() - processing_start_time > 300: # 5 minutes timeout for processing
                status_callback_main_thread(f"  Timeout processando {os.path.basename(image_path)} na API.")
                try:
                    genai.delete_file(file_to_upload.name); status_callback_main_thread(
                        f"  Arquivo {file_to_upload.name} com timeout deletado.")
                except Exception:
                    pass
                return image_path, None
            time.sleep(5) # Check status every 5 seconds
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
    prompt = f"""
Analyze the content of the provided image (filename: {page_filename}, dimensions: {img_dimensions[0]}x{img_dimensions[1]} pixels, representing page {current_page_num_in_doc} of the document). Your goal is to convert this page into an accessible HTML format suitable for screen readers, specifically targeting visually impaired STEM students reading Portuguese content. Don't change the original text or language, even if it's wrong; the main goal is fidelity to the original text.
**Instructions:**
1.  **Text Content (MAXIMUM FIDELITY REQUIRED):**
    * Extract ALL readable text from the image **EXACTLY** as it appears.
    * Preserve the original language (Portuguese) and the **PRECISE WORDING AND ORDER OF WORDS**.
    * Do NOT paraphrase, reorder, or 'correct' the text, even if it seems stylistically better or contains errors in the original. Reproduce it verbatim.
    * Preserve paragraph structure where possible.
    * **Omit standalone page numbers** that typically appear at the very top or bottom of a page, unless they are part of a sentence or reference.
2.  **Mathematical Equations:**
    * Identify ALL mathematical equations, formulas, and expressions.
    * Convert them accurately into LaTeX format.
    * **CRITICAL DELIMITER USAGE:** For inline mathematics, YOU MUST USE `\\(...\\)` (e.g., `\\(x=y\\)`). DO NOT use single `$` signs for inline math (e.g., AVOID `$x=y$`). Single `$` signs might not be rendered correctly by the system.
    * For display mathematics (equations on their own line), YOU MUST USE `$$...$$` (e.g., `$$x = \\sum y_i$$`).
    * **Ensure that *all* mathematical symbols, including single-letter variables mentioned in descriptive text or prose (e.g., if the original text says '...where v is velocity...'), are enclosed in inline LaTeX delimiters (e.g., output as '...where \\(v\\) is velocity...').** This applies even to isolated symbols or variables.
    * Ensure correct LaTeX syntax within these delimiters.
3.  **Tables:**
    * Identify any tables present in the image.
    * Extract the data accurately, maintaining row and column structure.
    * Format the table using proper HTML table tags (`<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`). Include table headers (`<th>`) if identifiable.
4.  **Visual Elements (Descriptions - CRITICAL):**
    * Identify any significant diagrams, graphs, figures, or images within the page content. Only do this if the image is relevant to the text, avoid doing this for logos in headers and other similar elements that are not relevant for the understanding of the page.
    * **Instead of including the image itself, provide a concise textual description** in Portuguese of what the visual element shows and its relevance (e.g., "<p><i>[Descrição: Diagrama do circuito elétrico mostrando a ligação em série de...]</i></p>" or "<p><i>[Descrição: Gráfico de barras comparando...]</i></p>"). Use italics or similar indication for the description. Integrate these descriptions logically within the extracted text flow where the visual element appeared.
5.  **Footnotes (Notas de Rodapé - CRITICAL):**
    * Identify footnote markers in the main text (e.g., superscript numbers like `¹`, `²`, or symbols like `*`, `†`).
    * Identify the corresponding footnote text, typically found at the bottom of the page.
    * Link the marker to the text using the following patterns:
        * **In-text marker pattern:** `<sup><a href="#fn{current_page_num_in_doc}-{{FOOTNOTE_INDEX_ON_PAGE}}" id="fnref{current_page_num_in_doc}-{{FOOTNOTE_INDEX_ON_PAGE}}" aria-label="Nota de rodapé {{FOOTNOTE_INDEX_ON_PAGE}}">{{MARKER_SYMBOL_FROM_TEXT}}</a></sup>`
            * `{current_page_num_in_doc}` is the actual page number. Replace `{{FOOTNOTE_INDEX_ON_PAGE}}` with sequential index (1, 2, 3...) on this page. Replace `{{MARKER_SYMBOL_FROM_TEXT}}` with actual marker.
        * **Footnote list pattern:** At VERY END of this page's HTML (before closing ```html):
            ```html
            <hr class="footnotes-separator" />
            <div class="footnotes-section">
              <h4 class="sr-only">Notas de Rodapé da Página {current_page_num_in_doc}</h4>
              <ol class="footnotes-list">
                <li id="fn{current_page_num_in_doc}-{{FOOTNOTE_INDEX_ON_PAGE}}">TEXT_OF_THE_FOOTNOTE_HERE. <a href="#fnref{current_page_num_in_doc}-{{FOOTNOTE_INDEX_ON_PAGE}}" aria-label="Voltar para a referência da nota de rodapé {{FOOTNOTE_INDEX_ON_PAGE}}">&#8617;</a></li>
              </ol>
            </div>
            ```
            * Replace `{{FOOTNOTE_INDEX_ON_PAGE}}` and `TEXT_OF_THE_FOOTNOTE_HERE`.
    * Ensure unique `id` attributes: `fn{current_page_num_in_doc}-{{INDEX}}` and `fnref{current_page_num_in_doc}-{{INDEX}}`.
6.  **HTML Structure:**
    * Use semantic HTML. Output ONLY extracted text, LaTeX, table HTML, descriptions, and footnote HTML as HTML body content in a single Markdown code block:
        ```html
        <p>Texto com nota<sup><a href="#fn{current_page_num_in_doc}-1" id="fnref{current_page_num_in_doc}-1" aria-label="Nota de rodapé 1">1</a></sup>.</p>
        <p><i>[Descrição: Diagrama...]</i></p>
        $$ LaTeX $$
        <hr class="footnotes-separator" /><div class="footnotes-section"><h4 class="sr-only">Notas de Rodapé da Página {current_page_num_in_doc}</h4><ol class="footnotes-list"><li id="fn{current_page_num_in_doc}-1">Nota aqui. <a href="#fnref{current_page_num_in_doc}-1" aria-label="Voltar para a referência 1">&#8617;</a></li></ol></div>
        ```
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


def generate_html_for_image_task(model, file_object_from_api, page_filename_local, local_img_path, cancel_event,
                                 status_callback_main_thread, current_page_num_in_doc, original_page_order_index):
    if cancel_event.is_set(): raise OperationCancelledError("Geração HTML (task) cancelada antes de iniciar.")

    base64_image_data = None
    try:
        with open(local_img_path, "rb") as img_file:
            base64_image_data = base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e_b64:
        status_callback_main_thread(f"  Aviso: Falha ao ler/codificar imagem {os.path.basename(local_img_path)} para Base64: {e_b64}")
        base64_image_data = None # Ensure it's None if encoding fails

    dimensions_tuple = ("desconhecida", "desconhecida")
    try:
        img = Image.open(local_img_path)
        dimensions_tuple = img.size
        img.close()
    except Exception as e:
        status_callback_main_thread(f"  Aviso: Falha ao ler dimensões de {page_filename_local}: {e}.")

    prompt_parts = create_html_prompt_with_desc(file_object_from_api, page_filename_local, dimensions_tuple,
                                                current_page_num_in_doc)
    generation_config = genai.types.GenerationConfig(temperature=0.1)

    try:
        response = gemini_api_call_with_retry(model.generate_content, cancel_event, status_callback_main_thread,
                                              contents=prompt_parts, generation_config=generation_config)
    except Exception as e:
        status_callback_main_thread(
            f"  Exceção na API Gemini para {page_filename_local} (pág {current_page_num_in_doc}): {type(e).__name__} - {e}")
        return original_page_order_index, current_page_num_in_doc, None, base64_image_data

    if cancel_event.is_set(): raise OperationCancelledError("Geração HTML (task) cancelada após chamada API.")

    if not response.candidates:
        status_callback_main_thread(
            f"  Erro: Sem candidatos de resposta para {page_filename_local} (pág {current_page_num_in_doc}).")
        if hasattr(response, 'prompt_feedback'): status_callback_main_thread(f"  Feedback: {response.prompt_feedback}")
        return original_page_order_index, current_page_num_in_doc, None, base64_image_data

    # Try to get text, prioritizing response.text, then parts
    response_text_content = response.text # Preferred way
    if not response_text_content and response.candidates[0].content and response.candidates[0].content.parts:
        response_text_content = ''.join(
            part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')
        )

    if not response_text_content:
        status_callback_main_thread(
            f"  Erro: Conteúdo da resposta vazio para {page_filename_local} (pág {current_page_num_in_doc}).")
        return original_page_order_index, current_page_num_in_doc, None, base64_image_data

    html_body = extract_html_from_response(response_text_content)

    # Post-processing to remove potentially problematic <bdi> tags if Gemini adds them
    if html_body:
        # Remove <bdi> around simple variables or LaTeX commands
        html_body = re.sub(r'<bdi>([a-zA-Z0-9_](?:<sup>.*?</sup>)?)</bdi>', r'\1', html_body)
        html_body = re.sub(r'<bdi>(\\[a-zA-Z]+(?:\{.*?\})?(?:\s*\^\{.*?\})?(?:\s*_\{.*?\})?)</bdi>', r'\1', html_body)
        # Remove empty <bdi> tags
        html_body = re.sub(r'<bdi>\s*</bdi>', '', html_body)


    if html_body is None:
        status_callback_main_thread(
            f"  Erro: Falha ao extrair HTML para {page_filename_local} (pág {current_page_num_in_doc}).")
        status_callback_main_thread(f"  Texto bruto (300c): {response_text_content[:300]}...")
        if hasattr(response.candidates[0], 'finish_reason'): status_callback_main_thread(
            f"  Motivo: {response.candidates[0].finish_reason}")
        return original_page_order_index, current_page_num_in_doc, None, base64_image_data

    status_callback_main_thread(f"  HTML extraído para pág. PDF {current_page_num_in_doc} ({page_filename_local}).")
    return original_page_order_index, current_page_num_in_doc, html_body, base64_image_data


def create_merged_html_with_accessibility(content_list, output_path, pdf_filename_title):
    if not content_list: return False

    accessibility_css = """
<style>
    html, body {margin: 0;padding: 0;overflow-x: hidden;}
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
    body.normal-mode p i, body.normal-mode span i { color: #555; }
    body.normal-mode sup > a { color: #0066cc; }
    body.normal-mode hr.page-separator { border-color: #ccc; }
    body.normal-mode hr.footnotes-separator { border-color: #ccc; }
    body.normal-mode #accessibility-controls.expanded {background-color: #f0f0f0; border-color: #ccc; color: #000;}
    body.normal-mode #accessibility-controls button, body.normal-mode #accessibility-controls select { background-color: #fff; border: 1px solid #bbb; color: #000; }
    body.dark-mode {background-color: #121212; color: #e0e0e0;}
    body.dark-mode .page-content {background-color: #1e1e1e;border-color: #444444;}
    body.dark-mode h1, body.dark-mode h2 {color: #ffffff; border-color: #444;}
    body.dark-mode p i, body.dark-mode span i { color: #aaa; }
    body.dark-mode sup > a { color: #87CEFA; }
    body.dark-mode hr.page-separator { border-color: #555; }
    body.dark-mode hr.footnotes-separator { border-color: #555; }
    body.dark-mode #accessibility-controls.expanded {background-color: #2c2c2c; border-color: #555; color: #e0e0e0;}
    body.dark-mode #accessibility-controls button, body.dark-mode #accessibility-controls select { background-color: #333; border: 1px solid #555; color: #e0e0e0; }
    body.high-contrast-mode {background-color: #000000;color: #FFFF00;}
    body.high-contrast-mode .page-content {background-color: #000000;border: 2px solid #FFFF00;}
    body.high-contrast-mode h1, body.high-contrast-mode h2 {color: #FFFF00; border-color: #FFFF00;}
    body.high-contrast-mode p, body.high-contrast-mode span, body.high-contrast-mode li, body.high-contrast-mode td, body.high-contrast-mode th { color: #FFFF00 !important; }
    body.high-contrast-mode p i, body.high-contrast-mode span i { color: #00FF00; }
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
</style>
"""
    accessibility_js = r"""
<script>
    let currentFontSize = 16;
    const fonts = ['Atkinson Hyperlegible', 'Lexend', 'OpenDyslexicRegular', 'Verdana', 'Arial', 'Times New Roman', 'Courier New'];
    let currentFontIndex = 0; const synth = window.speechSynthesis;
    let utterance = null; let isPaused = false; let voices = [];
    function populateVoiceList() {
        voices = synth.getVoices().filter(voice => voice.lang.startsWith('pt'));
        if (voices.length === 0) { voices = synth.getVoices(); } // Fallback to all voices if no Portuguese found
    }
    if (synth.onvoiceschanged !== undefined) { synth.onvoiceschanged = populateVoiceList; }
    populateVoiceList(); // Initial call
    function changeFontSize(delta) {
        currentFontSize += delta; if (currentFontSize < 10) currentFontSize = 10; if (currentFontSize > 48) currentFontSize = 48;
        document.body.style.fontSize = currentFontSize + 'px'; localStorage.setItem('accessibilityFontSize', currentFontSize);
    }
    function setFontFamily(fontName) {
        const index = fonts.indexOf(fontName);
        if (index !== -1) { currentFontIndex = index; document.body.style.fontFamily = fonts[currentFontIndex] + ', sans-serif'; localStorage.setItem('accessibilityFontFamily', fontName); }
    }
    function getTextToSpeak() {
        let selectedText = window.getSelection().toString().trim();
        if (selectedText) { return selectedText; } else {
            const mainContentElement = document.getElementById('main-content');
            if (mainContentElement) {
                let text = '';
                // Iterate through direct children of main-content to respect semantic structure better
                Array.from(mainContentElement.childNodes).forEach(node => {
                    if (node.nodeType === Node.ELEMENT_NODE && node.tagName !== 'SCRIPT' && node.tagName !== 'STYLE' && !node.classList.contains('sr-only') && node.tagName !== 'DETAILS') {
                        // Get text content, replace multiple whitespaces/newlines with a single space, then add two newlines for paragraph breaks
                        text += node.textContent.trim().replace(/\s+/g, ' ') + '\n\n';
                    } else if (node.nodeType === Node.TEXT_NODE) {
                        let nodeText = node.textContent.trim().replace(/\s+/g, ' ');
                        if (nodeText) text += nodeText + '\n\n';
                    }
                }); return text.trim();
            }
        } return '';
    }
    function speakText() {
        if (synth.speaking && !isPaused) return; if (synth.paused && utterance) { synth.resume(); isPaused = false; return; }
        const textToSay = getTextToSpeak();
        if (textToSay && synth) {
            stopSpeech(); utterance = new SpeechSynthesisUtterance(textToSay);
            const portugueseVoice = voices.find(voice => voice.lang === 'pt-BR') || voices.find(voice => voice.lang === 'pt-PT');
            if (portugueseVoice) { utterance.voice = portugueseVoice; utterance.lang = portugueseVoice.lang; }
            else if (voices.length > 0) { utterance.voice = voices[0]; utterance.lang = voices[0].lang; } // Fallback to first available voice
            else { utterance.lang = 'pt-BR'; } // Default lang if no voices
            utterance.onerror = function(event) { console.error('SpeechSynthesisUtterance.onerror', event); };
            utterance.onend = function() { utterance = null; isPaused = false; };
            synth.speak(utterance); isPaused = false;
        } else if (!synth) { console.warn('Text-to-Speech not supported.'); }
        else { console.info('No text selected or available to speak.'); }
    }
    function pauseSpeech() { if (synth.speaking && !isPaused) { synth.pause(); isPaused = true; } }
    function stopSpeech() {
        if (synth.speaking || synth.paused) { if (utterance) { utterance.onerror = null; } synth.cancel(); }
        utterance = null; isPaused = false;
    }
    function changeTheme(themeName) {
        document.body.classList.remove('normal-mode', 'dark-mode', 'high-contrast-mode');
        document.body.classList.add(themeName + '-mode'); localStorage.setItem('accessibilityTheme', themeName);
    }
    function toggleAccessibilityMenu() {
        const menu = document.getElementById('accessibility-controls');
        const toggleButton = document.getElementById('accessibility-toggle');

        if (menu.classList.contains('expanded')) {
            menu.classList.remove('expanded');
            menu.classList.add('collapsed');
        } else {
            menu.classList.remove('collapsed');
            menu.classList.add('expanded');
        }

        const isNowExpanded = menu.classList.contains('expanded');
        toggleButton.setAttribute('aria-expanded', isNowExpanded.toString());
        toggleButton.setAttribute('aria-label', isNowExpanded ? 'Fechar Menu de Acessibilidade' : 'Abrir Menu de Acessibilidade');
    }
    document.addEventListener('DOMContentLoaded', () => {
        populateVoiceList(); // Populate voices on DOM load
        const savedTheme = localStorage.getItem('accessibilityTheme') || 'dark'; // Default to dark
        changeTheme(savedTheme); document.getElementById('themeSelector').value = savedTheme;
        const savedFontSize = parseInt(localStorage.getItem('accessibilityFontSize'), 10) || 16;
        currentFontSize = savedFontSize; document.body.style.fontSize = currentFontSize + 'px';
        const defaultFont = fonts.includes('Atkinson Hyperlegible') ? 'Atkinson Hyperlegible' : fonts[0];
        const savedFontFamily = localStorage.getItem('accessibilityFontFamily') || defaultFont;
        setFontFamily(savedFontFamily); document.getElementById('fontSelector').value = savedFontFamily;
        document.addEventListener('keydown', function(event) { if (event.key === "Escape") { stopSpeech(); } });
        const menu = document.getElementById('accessibility-controls'); const toggleButton = document.getElementById('accessibility-toggle');
        if (menu && !menu.classList.contains('expanded')) { menu.classList.add('collapsed'); }
        if (toggleButton && menu) { // Set initial aria attributes for the toggle button
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
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
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
            <span>Tamanho da Fonte:</span>
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
            <span>Leitura em Voz Alta:</span>
            <button onclick="speakText()" aria-label="Ler ou continuar leitura">▶️ Ler/Continuar</button>
            <button onclick="pauseSpeech()" aria-label="Pausar leitura">⏸️ Pausar</button>
            <button onclick="stopSpeech()" aria-label="Parar leitura (Tecla Esc)">⏹️ Parar (Esc)</button>
        </div>
        <div class="control-group">
            <label for="themeSelector">Tema Visual:</label>
            <select id="themeSelector" onchange="changeTheme(this.value)" aria-label="Selecionar tema visual">
                <option value="normal">Modo Claro</option><option value="dark">Modo Escuro</option>
                <option value="high-contrast">Alto Contraste</option>
            </select>
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
                    <img src="data:image/png;base64,{base64_image}" alt="{safe_alt_text}" style="max-width: 100%; height: auto;">
                </div>
            </details>
            """
        merged_html += "\n</article>\n"
    merged_html += "\n    </main>\n</body>\n</html>"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(merged_html)
        return True
    except Exception:
        raise


def process_pdf_thread(
        pdf_path, initial_output_html_path_base, dpi, page_range_str, selected_model_name,
        num_upload_workers, num_generate_workers,
        cancel_event, status_callback, completion_callback,
        password_request_callback_gui, progress_callback_gui
):
    image_paths = [];
    uploaded_files_map = {};
    temp_image_dir = ""
    generated_content_list_thread = []; # Stores dicts: {"page_num_in_doc": ..., "body": ..., "base64_image": ...}

    # Enhanced progress callback to give more fine-grained updates
    def phase_progress_callback(step, total_in_phase, text_prefix):
        overall_progress = 0;
        overall_total = 100 # Assuming 100 as max for overall progress bar
        # Define phases and their weight in overall progress
        if "Convertendo" in text_prefix: # Phase 1: PDF to Images (e.g., 0-30%)
            overall_progress = int((step / total_in_phase) * 30) if total_in_phase > 0 else 0
        elif "Upload" in text_prefix: # Phase 2: Uploading (e.g., 30-60%)
            overall_progress = 30 + int((step / total_in_phase) * 30) if total_in_phase > 0 else 30
        elif "HTML" in text_prefix and "Gerando" in text_prefix: # Phase 3: Generating HTML (e.g., 60-90%)
            overall_progress = 60 + int((step / total_in_phase) * 30) if total_in_phase > 0 else 60
        elif "Mesclando" in text_prefix or "Limpando" in text_prefix or "Concluíd" in text_prefix : # Phase 4: Merging/Cleaning (e.g., 90-100%)
            overall_progress = 90 + int((step / total_in_phase) * 10) if total_in_phase > 0 else 90

        progress_callback_gui(overall_progress, overall_total, f"{text_prefix} ({step}/{total_in_phase})")


    try:
        status_callback("Iniciando processo...");
        phase_progress_callback(0, 1, "Inicializando") # Initial progress update
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        pdf_basename_for_dir = re.sub(r'[^\w\-_\. ]', '_', os.path.splitext(os.path.basename(pdf_path))[0])
        temp_image_dir = os.path.join(script_dir, f"temp_{pdf_basename_for_dir}_{timestamp}")
        status_callback(f"Diretório temporário: {temp_image_dir}")

        api_key = os.environ.get(API_KEY_ENV_VAR)
        if not api_key: raise ValueError(f"Chave API '{API_KEY_ENV_VAR}' não encontrada nas variáveis de ambiente.")
        genai.configure(api_key=api_key);
        status_callback("Chave API Google AI configurada.")

        # Get total pages for range parsing and initial encryption check
        total_doc_pages_for_range_parsing = 0
        try:
            temp_doc = fitz.open(pdf_path) # Open once for metadata
            total_doc_pages_for_range_parsing = temp_doc.page_count
            is_encrypted_initial_check = temp_doc.is_encrypted # Check encryption status
            temp_doc.close()
            if is_encrypted_initial_check: status_callback("  PDF parece protegido. Senha será solicitada se necessário.")
        except Exception as e:
            raise Exception(f"Falha ao ler metadados do PDF: {e}")

        selected_pages_0_indexed = parse_page_ranges(page_range_str, total_doc_pages_for_range_parsing)
        if selected_pages_0_indexed is None: raise ValueError(f"Intervalo de páginas inválido: '{page_range_str}'. Verifique o formato (ex: 1-3,5,7-).")

        num_pages_initially_selected = len(selected_pages_0_indexed)
        if num_pages_initially_selected == 0:
            if total_doc_pages_for_range_parsing == 0: raise ValueError("PDF está vazio ou não pôde ser lido. Nenhuma página para processar.")
            if page_range_str.strip(): raise ValueError( # Only raise if user actually provided a range that yielded nothing
                f"O intervalo de páginas '{page_range_str}' não resultou em páginas válidas para um PDF com {total_doc_pages_for_range_parsing} páginas.")
            # If no range provided and PDF has pages, this shouldn't happen due to parse_page_ranges behavior.
            # But as a safeguard:
            raise ValueError("Nenhuma página selecionada para processar (verifique o PDF e o intervalo de páginas).")

        status_callback(
            f"Páginas para converter (0-idx): {', '.join(map(str, [p+1 for p in selected_pages_0_indexed]))} ({num_pages_initially_selected} pág.)")

        image_paths = pdf_to_images_local(pdf_path, temp_image_dir, dpi, selected_pages_0_indexed, cancel_event,
                                          status_callback, password_request_callback_gui, phase_progress_callback)
        if cancel_event.is_set(): raise OperationCancelledError("Cancelado após conversão para imagem.")
        if not image_paths: raise Exception("Falha crítica ao converter PDF para imagens. Nenhuma imagem foi gerada.")
        # `image_paths` now contains paths to successfully converted images

        uploaded_files_map = upload_to_gemini_file_api(image_paths, num_upload_workers, cancel_event, status_callback,
                                                       phase_progress_callback)
        if cancel_event.is_set(): raise OperationCancelledError("Cancelado após upload.")
        if not uploaded_files_map: raise Exception("Falha crítica ao fazer upload de imagens para API Gemini. Nenhum arquivo foi carregado.")
        # `uploaded_files_map` maps local image paths to their File API objects

        num_files_successfully_uploaded = len(uploaded_files_map)
        status_callback(f"Inicializando modelo Gemini: {selected_model_name}...")
        model = genai.GenerativeModel(model_name=selected_model_name,
                                      safety_settings={'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE', # Adjust as needed
                                                       'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                                                       'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                                                       'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'})
        status_callback(f"Modelo Gemini pronto. Gerando HTML para {num_files_successfully_uploaded} imagem(ns)...")

        # Determine original page number from filename (e.g., "page_001.png" -> 1)
        local_path_to_original_page_num = {
            img_path: int(m.group(1)) if (m := re.search(r"page_(\d+)\.\w+$", os.path.basename(img_path))) else -1
            for img_path in uploaded_files_map.keys() # Only consider successfully uploaded images
        }
        # Verify page number extraction
        for img_path, page_num in local_path_to_original_page_num.items():
            if page_num == -1: status_callback(f"Aviso: Falha ao determinar pág. original de {os.path.basename(img_path)}")


        # Prepare tasks for HTML generation, ensuring we process in original page order
        tasks_for_html_generation = []
        # Sort by the extracted original page number to maintain document order
        sorted_uploaded_image_paths = sorted(
            uploaded_files_map.keys(), # These are local paths
            key=lambda x: local_path_to_original_page_num.get(x, float('inf')) # Sort by page number
        )

        for original_order_idx, local_img_path in enumerate(sorted_uploaded_image_paths):
            page_num_in_doc = local_path_to_original_page_num.get(local_img_path)
            if page_num_in_doc is None or page_num_in_doc == -1:
                status_callback(f"  Pulando {os.path.basename(local_img_path)} para geração HTML (pág. original não determinada).")
                continue

            file_api_object = uploaded_files_map.get(local_img_path) # Get the File API object
            if not file_api_object: # Should not happen if logic is correct
                status_callback(f"  Pulando {os.path.basename(local_img_path)} (objeto da API não encontrado no mapa).")
                continue

            task_args = (model, file_api_object, os.path.basename(local_img_path), # Pass local filename
                         local_img_path, # Pass full local path for reading dimensions/base64
                         cancel_event, status_callback, page_num_in_doc, original_order_idx)
            tasks_for_html_generation.append({'args': task_args, 'original_idx': original_order_idx})

        temp_generated_html_results = [None] * len(tasks_for_html_generation) # Pre-allocate list for ordered results
        current_html_tasks_for_retry_phase = list(tasks_for_html_generation)

        for attempt in range(PHASE_MAX_RETRIES + 1):
            if not current_html_tasks_for_retry_phase or cancel_event.is_set(): break
            total_in_this_attempt = len(current_html_tasks_for_retry_phase)
            if attempt > 0: status_callback(
                f"Retentativa de Geração HTML (Fase {attempt + 1}/{PHASE_MAX_RETRIES + 1}) para {total_in_this_attempt} páginas...")

            completed_in_this_attempt = 0
            failed_tasks_for_next_round = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_generate_workers) as executor:
                future_to_task_info_dict = { # Map future to original task info for tracking
                    executor.submit(generate_html_for_image_task, *task_info['args']): task_info
                    for task_info in current_html_tasks_for_retry_phase
                }
                for future in concurrent.futures.as_completed(future_to_task_info_dict):
                    if cancel_event.is_set(): break
                    original_task_info = future_to_task_info_dict[future]
                    idx_for_assignment = original_task_info['original_idx'] # Use original index for result placement

                    try:
                        # Now expects (original_idx, page_num, html_body, base64_data)
                        returned_task_idx, page_num_doc, html_body, base64_data = future.result()

                        if returned_task_idx != idx_for_assignment: # Sanity check
                            status_callback(
                                f"ALERTA DE ÍNDICE: Mapeado: {idx_for_assignment}, Retornado pela Task: {returned_task_idx} para pág {page_num_doc}. USANDO ÍNDICE MAPEADO ({idx_for_assignment}).")

                        if html_body: # If HTML was successfully generated
                            if 0 <= idx_for_assignment < len(temp_generated_html_results): # Bounds check
                                temp_generated_html_results[idx_for_assignment] = {
                                    "page_num_in_doc": page_num_doc,
                                    "body": html_body,
                                    "base64_image": base64_data # Store base64 image data
                                }
                            else: # Should ideally not happen if original_order_idx is managed well
                                status_callback(f"!! ERRO GRAVE DE ÍNDICE !! Tentativa de usar índice {idx_for_assignment}, mas tamanho da lista é {len(temp_generated_html_results)}. Pág: {page_num_doc}. Marcando task como falha.")
                                failed_tasks_for_next_round.append(original_task_info)
                        else: # HTML generation failed for this page
                            status_callback(
                                f"  HTML não gerado para pág. PDF {page_num_doc} (Índice Original: {idx_for_assignment}). Adicionando para retentativa de fase.")
                            failed_tasks_for_next_round.append(original_task_info)
                    except OperationCancelledError:
                        status_callback(
                            "  Geração HTML cancelada para uma página."); failed_tasks_for_next_round.append(
                            original_task_info); break # Break from this loop on cancellation
                    except Exception as e:
                        status_callback(
                            f"  Erro ao processar resultado HTML para pág. (índice mapeado {idx_for_assignment}), len(lista)={len(temp_generated_html_results)}: {type(e).__name__} - {e}")
                        failed_tasks_for_next_round.append(original_task_info)
                    finally:
                        completed_in_this_attempt += 1
                        phase_progress_callback(completed_in_this_attempt, total_in_this_attempt, f"Gerando HTML")

            current_html_tasks_for_retry_phase = failed_tasks_for_next_round
            if not current_html_tasks_for_retry_phase or cancel_event.is_set(): break


        if cancel_event.is_set(): raise OperationCancelledError("Cancelado após geração HTML.")
        # Filter out None results (for pages that might have consistently failed or were skipped)
        generated_content_list_thread = [res for res in temp_generated_html_results if res is not None]

        num_successfully_generated_html_pages = len(generated_content_list_thread)
        total_pages_html_generation_attempted = len(tasks_for_html_generation) # Number of pages we tried to get HTML for
        is_partial_generation = num_successfully_generated_html_pages > 0 and num_successfully_generated_html_pages < total_pages_html_generation_attempted

        if not generated_content_list_thread and total_pages_html_generation_attempted > 0 : # If we attempted but got nothing
            status_callback("AVISO: Nenhum conteúdo HTML foi gerado com sucesso após todas as tentativas.")
        status_callback(
            f"Geração HTML finalizada. {num_successfully_generated_html_pages}/{total_pages_html_generation_attempted} págs tentadas foram processadas com sucesso.")


        output_html_path_final = initial_output_html_path_base
        pdf_filename_for_html_title = os.path.splitext(os.path.basename(pdf_path))[0]

        if is_partial_generation:
            status_callback(
                f"AVISO: HTML gerado parcialmente ({num_successfully_generated_html_pages}/{total_pages_html_generation_attempted} das páginas tentadas).")
            path_dir, base_ext_name = os.path.split(initial_output_html_path_base)
            base_name, ext = os.path.splitext(base_ext_name)
            if base_name.endswith("_accessible"): base_name = base_name[:-len("_accessible")]
            output_html_path_final = os.path.join(path_dir, f"{base_name}_parcial_accessible{ext}")
            pdf_filename_for_html_title += " (Parcial)"


        status_callback("Mesclando HTML...");
        phase_progress_callback(0, 1, "Mesclando HTML")
        success_merge = create_merged_html_with_accessibility(generated_content_list_thread, output_html_path_final,
                                                              pdf_filename_for_html_title)

        if not success_merge and generated_content_list_thread: # If merge failed but there was content
            raise Exception("Falha ao criar arquivo HTML mesclado (havia conteúdo para mesclar).")
        elif not generated_content_list_thread and not success_merge: # No content, so merge "failing" (not creating file) is expected
            status_callback("Nenhum HTML para mesclar, arquivo de saída pode não ter sido criado ou pode estar vazio.")

        status_callback(f"HTML salvo em: {output_html_path_final}");
        phase_progress_callback(1, 1, "HTML Mesclado")


        final_message = f"Sucesso! HTML acessível salvo em:\n{output_html_path_final}"
        if is_partial_generation:
            final_message += f"\n\nAVISO: Geração parcial ({num_successfully_generated_html_pages}/{total_pages_html_generation_attempted} das páginas para as quais a geração de HTML foi tentada foram processadas com sucesso)."
        elif num_successfully_generated_html_pages == 0 and total_pages_html_generation_attempted > 0:
            final_message = f"Processo concluído, mas NENHUMA página foi gerada com sucesso de {total_pages_html_generation_attempted} páginas tentadas.\nVerifique o log. Arquivo de saída (pode estar vazio): {output_html_path_final}"

        completion_callback(True, final_message)

    except ValueError as ve: # Configuration or input errors
        status_callback(f"Erro de Entrada/Configuração: {ve}"); completion_callback(False, f"Erro de Entrada/Configuração: {ve}")
    except OperationCancelledError:
        status_callback("Processo cancelado pelo usuário."); completion_callback(False, "Processo cancelado pelo usuário.")
    except google_exceptions.ResourceExhausted as e: # API quota issues
        status_callback(f"Erro API (Limite de Taxa/Recurso Esgotado): {e}."); completion_callback(False, f"Erro da API (Limite de Taxa/Recurso Esgotado): {e}. Tente novamente mais tarde ou verifique suas cotas.")
    except Exception as e: # Catch-all for other unexpected errors
        import traceback
        error_details = f"Erro Inesperado: {type(e).__name__} - {e}\n{traceback.format_exc()}"
        status_callback(error_details);
        completion_callback(False, f"Erro Inesperado durante o processamento: {e}")
    finally:
        status_callback("Iniciando limpeza final...");
        phase_progress_callback(0, 1, "Limpando arquivos")
        # Clean local images
        if image_paths: # Check if image_paths was initialized and populated
            cleaned_local = 0
            for img_path in image_paths:
                try:
                    if os.path.exists(img_path): os.remove(img_path); cleaned_local += 1
                except Exception as e_rem:
                    status_callback(f"  Aviso: Falha ao remover imagem local {os.path.basename(img_path)}: {e_rem}")
            status_callback(f"  {cleaned_local} arquivos de imagem locais limpos.")

        # Clean temporary directory
        if temp_image_dir and os.path.exists(temp_image_dir):
            try:
                shutil.rmtree(temp_image_dir); status_callback(f"  Diretório temporário removido: {temp_image_dir}")
            except OSError as e_rmdir: # More specific exception for rmtree issues
                status_callback(f"  Aviso: Falha ao remover diretório temporário {temp_image_dir}: {e_rmdir}")

        # Clean uploaded files from Gemini API
        if uploaded_files_map: # Check if map exists and is not empty
            deleted_api = 0
            status_callback(f"  Limpando {len(uploaded_files_map)} arquivos da API Gemini...")
            for file_obj in uploaded_files_map.values():
                if file_obj and hasattr(file_obj, 'name'): # Ensure file_obj is valid
                    try:
                        genai.delete_file(file_obj.name); deleted_api += 1; time.sleep(0.2) # Small delay
                    except Exception as e_del_api:
                         status_callback(f"  Aviso: Falha ao deletar arquivo da API {file_obj.name}: {e_del_api}")
            status_callback(f"  {deleted_api} arquivos da API Gemini limpos.")
        else:
            status_callback("  Nenhum arquivo para limpar da API Gemini (mapa de upload vazio ou não inicializado).")

        status_callback("Limpeza finalizada.");
        phase_progress_callback(1, 1, "Limpeza Concluída") # Final progress update


class OldSchoolApp:
    def __init__(self, root_window):
        self.root = root_window;
        self.root.title("PDF -> HTML Acessível (Gemini AI)");
        self.root.geometry("850x720");
        self.root.configure(bg='black') # Base background for the window
        self.cancel_event = threading.Event();
        self.processing_thread = None
        self.password_queue = queue.Queue() # For password communication

        # Font setup (try Consolas, then Courier New, then generic monospace)
        font_family = "Consolas"
        if font_family not in font.families():
            font_family = "Courier New"
            if font_family not in font.families():
                font_family = "monospace" # Generic fallback

        self.label_font = font.Font(family=font_family, size=11)
        self.button_font = font.Font(family=font_family, size=10, weight="bold")
        self.text_font = font.Font(family=font_family, size=9)

        # --- Style Configuration ---
        s = ttk.Style();
        s.theme_use('clam'); # 'clam' is often good for dark themes and customization

        # Define colors
        bg = 'black'
        fg = 'lime green'
        entry_bg = '#181818' # Dark grey for entry backgrounds
        btn_bg = '#222222'   # Slightly lighter grey for button backgrounds
        btn_active_bg = '#333333' # For active/pressed buttons
        disabled_fg = '#555555' # Muted foreground for disabled state
        disabled_bg = '#444444' # Muted background for disabled state


        s.configure("TLabel", background=bg, foreground=fg, font=self.label_font)
        s.configure("TButton", background=btn_bg, foreground=fg, font=self.button_font, borderwidth=1, relief=tk.RAISED)
        s.map("TButton",
              background=[('active', btn_active_bg), ('pressed', btn_active_bg), ('disabled', disabled_bg)],
              foreground=[('disabled', disabled_fg)])

        s.configure("TEntry", fieldbackground=entry_bg, foreground=fg, insertcolor=fg, font=self.text_font,
                    relief=tk.SUNKEN, borderwidth=1) # Sunken relief for entries
        s.configure("TFrame", background=bg) # Ensure frames also have the dark background
        s.configure("Horizontal.TProgressbar", troughcolor=entry_bg, background=fg, thickness=20)

        # Combobox listbox styling (important for dark themes)
        self.root.option_add('*TCombobox*Listbox*Background', entry_bg);
        self.root.option_add('*TCombobox*Listbox*Foreground', fg)
        self.root.option_add('*TCombobox*Listbox*selectBackground', btn_active_bg); # Selection color
        self.root.option_add('*TCombobox*Listbox*selectForeground', fg)

        s.configure("TCombobox", background=btn_bg, fieldbackground=entry_bg, foreground=fg,
                    selectbackground=entry_bg, # Background of the selected item in entry part
                    selectforeground=fg,       # Foreground of the selected item in entry part
                    arrowcolor=fg, font=self.text_font, relief=tk.SUNKEN, borderwidth=1, padding=3)
        s.map('TCombobox',
              fieldbackground=[('readonly', entry_bg), ('focus', entry_bg)], # Keep bg same on focus
              foreground=[('readonly', fg)],
              selectbackground=[('readonly', entry_bg)], # Keep consistent for readonly state
              selectforeground=[('readonly', fg)],
              relief=[('readonly', tk.SUNKEN), ('focus', tk.SUNKEN)])


        # --- Main Layout ---
        mf = ttk.Frame(self.root, padding="15"); # Main frame with padding
        mf.pack(expand=True, fill=tk.BOTH)

        # PDF Path Entry and Browse Button
        tcf = ttk.Frame(mf); # Top control frame
        tcf.pack(fill=tk.X, pady=(0, 10)) # Add some bottom padding
        ttk.Label(tcf, text="PDF:").pack(side=tk.LEFT, padx=(0, 5));
        self.pdf_path_var = tk.StringVar()
        self.pdf_entry = ttk.Entry(tcf, textvariable=self.pdf_path_var, width=70); # Generous width
        self.pdf_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5)) # Fill available space
        self.browse_btn = ttk.Button(tcf, text="Procurar...", command=self.select_pdf);
        self.browse_btn.pack(side=tk.LEFT, padx=(5,0))

        # Settings Area (DPI, Page Range, Workers, Model) - organized into columns
        main_settings_frame = ttk.Frame(mf);
        main_settings_frame.pack(fill=tk.X, pady=5)

        settings_col1 = ttk.Frame(main_settings_frame);
        settings_col1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10)) # First column
        settings_col2 = ttk.Frame(main_settings_frame);
        settings_col2.pack(side=tk.LEFT, fill=tk.X, expand=True) # Second column

        # DPI
        dpi_frame = ttk.Frame(settings_col1);
        dpi_frame.pack(fill=tk.X, pady=2, anchor='w') # Anchor to west
        ttk.Label(dpi_frame, text="DPI Imagem:").pack(side=tk.LEFT, anchor='w');
        self.dpi_var = tk.StringVar(value="150")
        self.dpi_entry = ttk.Entry(dpi_frame, textvariable=self.dpi_var, width=5);
        self.dpi_entry.pack(side=tk.LEFT, padx=(5, 0), anchor='w')

        # Page Range
        pagerange_frame = ttk.Frame(settings_col1);
        pagerange_frame.pack(fill=tk.X, pady=2, anchor='w')
        ttk.Label(pagerange_frame, text="Págs (ex:1-3,5):").pack(side=tk.LEFT, anchor='w');
        self.page_range_var = tk.StringVar() # Default empty for all pages
        self.page_entry = ttk.Entry(pagerange_frame, textvariable=self.page_range_var, width=15);
        self.page_entry.pack(side=tk.LEFT, padx=(5, 0), anchor='w')

        # Upload Workers
        upload_workers_frame = ttk.Frame(settings_col1);
        upload_workers_frame.pack(fill=tk.X, pady=2, anchor='w')
        ttk.Label(upload_workers_frame, text="Upload Workers:").pack(side=tk.LEFT, anchor='w');
        self.upload_workers_var = tk.StringVar(value=str(DEFAULT_UPLOAD_MAX_WORKERS))
        self.upload_workers_entry = ttk.Entry(upload_workers_frame, textvariable=self.upload_workers_var, width=4);
        self.upload_workers_entry.pack(side=tk.LEFT, padx=(5,0), anchor='w')


        # Gemini Model
        model_frame = ttk.Frame(settings_col2);
        model_frame.pack(fill=tk.X, pady=2, anchor='w')
        ttk.Label(model_frame, text="Modelo Gemini:").pack(side=tk.LEFT, anchor='w');
        self.model_var = tk.StringVar(value=DEFAULT_GEMINI_MODEL)
        self.model_cb = ttk.Combobox(model_frame, textvariable=self.model_var, values=AVAILABLE_GEMINI_MODELS,
                                     state="readonly", width=25) # Readonly for selection only
        self.model_cb.pack(side=tk.LEFT, padx=(5, 0), anchor='w', fill=tk.X, expand=True)

        # Generate Workers
        generate_workers_frame = ttk.Frame(settings_col2);
        generate_workers_frame.pack(fill=tk.X, pady=2, anchor='w')
        ttk.Label(generate_workers_frame, text="Generate Workers:").pack(side=tk.LEFT, anchor='w');
        self.generate_workers_var = tk.StringVar(value=str(DEFAULT_GENERATE_MAX_WORKERS))
        self.generate_workers_entry = ttk.Entry(generate_workers_frame, textvariable=self.generate_workers_var,
                                                width=4);
        self.generate_workers_entry.pack(side=tk.LEFT, padx=(5,0), anchor='w')


        # Action Buttons (Convert, Cancel, Clear Log)
        abf = ttk.Frame(mf); # Action buttons frame
        abf.pack(fill=tk.X, pady=10)

        self.conv_btn = ttk.Button(abf, text="Converter para HTML Acessível", command=self.start_conversion, style="Accent.TButton")
        s.configure("Accent.TButton", font=font.Font(family=font_family, size=11, weight="bold"), padding=8); # Make main button prominent
        self.conv_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5)) # Main action takes more space

        self.cancel_btn = ttk.Button(abf, text="Cancelar", command=self.request_cancellation, state=tk.DISABLED); # Initially disabled
        self.cancel_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(abf, text="Limpar Log", command=self.clear_log);
        self.clear_btn.pack(side=tk.LEFT, padx=(5, 0))


        # Progress Bar and Label
        pf = ttk.Frame(mf); # Progress frame
        pf.pack(fill=tk.X, pady=(5,0)); # Pad top
        self.progress_var = tk.DoubleVar();
        self.progress_lbl_var = tk.StringVar(value="Pronto.") # Initial status
        self.prog_lbl = ttk.Label(pf, textvariable=self.progress_lbl_var, anchor=tk.W); # West anchored label
        self.prog_lbl.pack(fill=tk.X)
        self.prog_bar = ttk.Progressbar(pf, variable=self.progress_var, maximum=100, mode='determinate',
                                        style="Horizontal.TProgressbar");
        self.prog_bar.pack(fill=tk.X, pady=(2,10)) # Padding around progress bar


        # Status Log Area
        ttk.Label(mf, text="Log de Processamento:").pack(anchor=tk.W, pady=(5,2))
        self.status_txt = scrolledtext.ScrolledText(mf, wrap=tk.WORD, height=15, bg='#0A0A0A', fg=fg, # Slightly darker log bg
                                                    font=self.text_font, relief=tk.SUNKEN, bd=1, insertbackground=fg)
        self.status_txt.pack(expand=True, fill=tk.BOTH, pady=5);
        self.status_txt.configure(state='disabled') # Read-only


    def select_pdf(self):
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Em Progresso", "Uma conversão já está em andamento. Aguarde ou cancele.", parent=self.root)
            return
        fp = filedialog.askopenfilename(title="Selecionar Arquivo PDF", filetypes=(("PDF files", "*.pdf"), ("All files", "*.*")))
        if fp:
            self.pdf_path_var.set(fp);
            self.update_status(f"Arquivo PDF selecionado: {fp}")

    def update_status(self, msg):
        # Schedules the GUI update on the main thread
        def append():
            if not self.root or not hasattr(self.status_txt, 'insert'): return # Check if GUI still exists
            try:
                self.status_txt.configure(state='normal'); # Enable editing
                ts = time.strftime("%H:%M:%S"); # Timestamp
                self.status_txt.insert(tk.END, f"[{ts}] {msg}\n"); # Append message
                self.status_txt.configure(state='disabled'); # Disable editing
                self.status_txt.see(tk.END) # Scroll to the end
            except tk.TclError:
                pass # Ignore Tcl errors if widget is destroyed

        if self.root: # Ensure root window exists
             self.root.after(0, append) # Schedule non-blocking update

    def update_progress_bar(self, val, total, phase_text):
        # Schedules the progress bar update on the main thread
        def update():
            if not self.root: return # Check if GUI still exists
            try:
                self.progress_var.set(val);
                self.prog_bar['maximum'] = total if total > 0 else 100; # Avoid division by zero or zero max
                self.progress_lbl_var.set(f"{phase_text}")
            except tk.TclError:
                pass # Ignore Tcl errors
        if self.root:
            self.root.after(0, update) # Schedule non-blocking update

    def request_password_from_gui(self):
        # This function is called from the worker thread, so GUI updates need to be scheduled
        def ask_password_dialog():
            # This dialog runs in the main GUI thread
            pwd = simpledialog.askstring("PDF Protegido por Senha", "Digite a senha do PDF:",
                                         parent=self.root, show='*') # Show '*' for password
            self.password_queue.put(pwd) # Put result (or None if cancelled) into the queue

        self.root.after(0, ask_password_dialog) # Schedule the dialog display
        try:
            # Wait for the password from the queue (with a timeout)
            password = self.password_queue.get(timeout=300) # 5 minutes timeout
            return password
        except queue.Empty: # Timeout occurred
            self.update_status("Timeout: Nenhuma senha fornecida para o PDF protegido.");
            # Notify the main processing thread that cancellation might be appropriate
            # or that it should handle the None password gracefully.
            # Here, we rely on the pdf_to_images_local to handle a None password.
            return None

    def clear_log(self):
        if self.status_txt:
            try:
                self.status_txt.configure(state='normal');
                self.status_txt.delete('1.0', tk.END); # Delete all content
                self.status_txt.configure(state='disabled');
                self.update_status("Log limpo pelo usuário.")
            except tk.TclError:
                pass

    def request_cancellation(self):
        if self.processing_thread and self.processing_thread.is_alive():
            if messagebox.askyesno("Confirmar Cancelamento", "Tem certeza que deseja cancelar a operação atual?", parent=self.root):
                self.cancel_event.set(); # Set the event to signal cancellation
                self.update_status("Cancelamento solicitado pelo usuário...");
                self.cancel_btn.config(state=tk.DISABLED, text="Cancelando...") # Update button state
        else:
            self.update_status("Nenhuma operação em andamento para cancelar.")

    def set_controls_state(self, new_state):
        # Helper to enable/disable GUI controls during processing
        button_state = tk.NORMAL if new_state == 'normal' else tk.DISABLED;
        entry_state = 'normal' if new_state == 'normal' else 'disabled'; # For ttk.Entry, 'disabled' is used
        combobox_state = 'readonly' if new_state == 'normal' else 'disabled'; # Combobox is 'readonly' or 'disabled'

        # Buttons
        self.conv_btn.config(state=button_state);
        self.browse_btn.config(state=button_state);
        # self.clear_btn.config(state=button_state) # Clear log can usually remain active

        # Entries
        self.dpi_entry.config(state=entry_state)
        self.page_entry.config(state=entry_state);
        self.model_cb.config(state=combobox_state) # Combobox
        self.upload_workers_entry.config(state=entry_state);
        self.generate_workers_entry.config(state=entry_state)

        # Cancel button state is inverse of others during processing
        self.cancel_btn.config(state=tk.NORMAL if new_state == 'disabled' else tk.DISABLED,
                               text="Cancelar" if new_state == 'disabled' else "Cancelar")


    def conversion_complete(self, success, msg):
        # Called from the worker thread when processing is done
        def final_actions():
            if not self.root: return # Check if GUI still exists

            self.set_controls_state('normal') # Re-enable controls
            self.progress_var.set(100 if success else self.progress_var.get()) # Update progress to 100% or current if failed
            self.progress_lbl_var.set("Concluído com Sucesso!" if success else "Falha na Operação.")

            if success:
                messagebox.showinfo("Sucesso", msg, parent=self.root)
                # Try to open the output directory
                try:
                    path_lines = msg.strip().split("\n"); # Message might be multi-line
                    out_path = ""
                    # Find the line with the output path
                    for line in reversed(path_lines): # Search from bottom up
                        if "HTML acessível salvo em:" in line:
                            out_path = line.split("HTML acessível salvo em:")[-1].strip();
                            break
                    # Fallback if specific phrase not found but last line might be a path
                    if not out_path and len(path_lines) > 0 and os.path.exists(path_lines[-1].strip()):
                         out_path = path_lines[-1].strip()

                    out_dir = ""
                    if out_path and os.path.isfile(out_path): # If it's a file path
                        out_dir = os.path.dirname(out_path)
                    elif out_path and os.path.isdir(out_path): # If it's already a directory path
                        out_dir = out_path

                    if out_dir and os.path.isdir(out_dir):
                        self.update_status(f"Tentando abrir diretório de saída: {out_dir}")
                        if os.name == 'nt': # Windows
                            os.startfile(out_dir)
                        elif os.name == 'posix': # macOS, Linux
                            opener = 'open' if 'darwin' in os.uname().sysname.lower() else 'xdg-open'
                            os.system(f"{opener} \"{out_dir}\"") # Use appropriate opener
                    else:
                        self.update_status(f"Diretório de saída não encontrado ou inválido: '{out_path}'")
                except Exception as e_open:
                    self.update_status(f"Falha ao tentar abrir diretório de saída: {e_open}")
            else: # If not successful
                messagebox.showerror("Erro na Conversão", msg, parent=self.root)

            self.update_status("-------------------- Fim da Operação --------------------");
            self.update_status("Pronto para nova conversão.")
            self.cancel_event.clear(); # Reset cancel event for next run
            self.processing_thread = None # Clear thread reference

        if self.root:
            self.root.after(0, final_actions) # Schedule final actions in main GUI thread

    def start_conversion(self):
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Em Progresso", "Uma conversão já está em andamento. Aguarde ou cancele.", parent=self.root)
            return

        # Get all input values
        pdf = self.pdf_path_var.get();
        dpi_s = self.dpi_var.get();
        pages_s = self.page_range_var.get(); # Can be empty for all pages
        model_s = self.model_var.get()
        upload_w_s = self.upload_workers_var.get();
        generate_w_s = self.generate_workers_var.get()

        # --- Input Validation ---
        if not pdf or not os.path.exists(pdf):
            messagebox.showerror("Erro de Entrada", "Por favor, selecione um arquivo PDF válido.", parent=self.root); return

        try:
            dpi_i = int(dpi_s);
            upload_w = int(upload_w_s);
            generate_w = int(generate_w_s)
            # Ensure workers are at least 1
            if upload_w < 1: upload_w = 1; self.upload_workers_var.set("1")
            if generate_w < 1: generate_w = 1; self.generate_workers_var.set("1")
        except ValueError:
            messagebox.showerror("Erro de Entrada", "Valores de DPI e Workers devem ser números inteiros.", parent=self.root); return

        if not (72 <= dpi_i <= 600): # DPI range check
            messagebox.showerror("Erro de Entrada", "O valor de DPI deve estar entre 72 e 600.", parent=self.root); return
        if not model_s: # Model selection check
            messagebox.showerror("Erro de Entrada", "Por favor, selecione um modelo Gemini.", parent=self.root); return

        # Construct output path
        base_out = os.path.join(os.path.dirname(pdf), f"{os.path.splitext(os.path.basename(pdf))[0]}_accessible.html")

        # Clear log if it wasn't just cleared
        log_now = self.status_txt.get('1.0', tk.END).strip()
        if "Log limpo" not in log_now and log_now: # Avoid clearing if it's already "Log limpo"
             self.status_txt.configure(state='normal');
             self.status_txt.delete('1.0', tk.END);
             self.status_txt.configure(state='disabled')


        self.update_status(f"Iniciando conversão para: {os.path.basename(pdf)}");
        self.update_status(
            f"Configurações -> Páginas: '{pages_s if pages_s else 'Todas'}', DPI: {dpi_i}, Modelo: {model_s}, Upload Workers: {upload_w}, Generate Workers: {generate_w}")

        self.set_controls_state('disabled'); # Disable GUI controls
        self.cancel_event.clear() # Ensure cancel event is cleared
        self.progress_var.set(0); # Reset progress bar
        self.progress_lbl_var.set("Iniciando...")

        # Start the processing in a new thread
        self.processing_thread = threading.Thread(target=process_pdf_thread,
                                                  args=(pdf, base_out, dpi_i, pages_s, model_s, upload_w, generate_w,
                                                        self.cancel_event, self.update_status, self.conversion_complete,
                                                        self.request_password_from_gui, self.update_progress_bar))
        self.processing_thread.daemon = True; # Allow main program to exit even if thread is running
        self.processing_thread.start()


if __name__ == "__main__":
    print("--------------------------------------------------------------------")
    print(f"PDF -> HTML Acessível (Google Gemini AI)")
    print(f"Este script requer uma chave da API Google AI configurada na variável de ambiente: '{API_KEY_ENV_VAR}'")
    print(f"Modelo Padrão: {DEFAULT_GEMINI_MODEL}. Outros modelos disponíveis no app.")
    print("--------------------------------------------------------------------")

    if not os.environ.get(API_KEY_ENV_VAR):
        # This print will go to console, GUI will still start but API calls will fail.
        # A GUI-based warning might be better if Tkinter is already being initialized.
        print(f"AVISO URGENTE: A variável de ambiente '{API_KEY_ENV_VAR}' não foi detectada.")
        print("O aplicativo pode não funcionar corretamente sem ela. Configure a chave da API e reinicie.")
        # Optionally, could show a messagebox here if root is created, or prevent app start.

    root = tk.Tk();
    app = OldSchoolApp(root);
    root.mainloop()
