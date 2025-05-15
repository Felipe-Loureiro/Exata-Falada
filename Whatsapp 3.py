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

def pdf_to_images_local(pdf_path, output_dir, dpi=100):
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
            img_format = "png" # Usar PNG para melhor qualidade no OCR
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
            response = genai.upload_file(path=image_path, display_name=display_name, mime_type=mime_type)
            uploaded_files_map[image_path] = response
            status_callback(f"  Done (URI: {response.uri})")
            time.sleep(2) # Evitar rate limit
        except google_exceptions.ResourceExhausted as e:
             status_callback(f"  Rate limit likely hit. Error: {e}. Stopping upload.")
             return None # Parar se der rate limit
        except Exception as e:
            status_callback(f"  Upload Failed for {base_image_name}. Error: {e}")
            # Continuar ou parar? Vamos parar por enquanto.
            return None
    status_callback(f"Finished uploading {len(uploaded_files_map)} files.")
    return uploaded_files_map

def create_html_prompt_with_desc(page_file_object, page_filename, img_dimensions):
    """Cria o prompt pedindo descrição textual."""
    # Prompt ajustado para pedir descrição e manter português
    prompt = f"""
Analyze the content of the provided image (filename: {page_filename}, dimensions: {img_dimensions[0]}x{img_dimensions[1]} pixels). Your goal is to convert this page into an accessible HTML format suitable for screen readers, specifically targeting visually impaired STEM students reading Portuguese content. Don't change the original text or language, even if it's wrong, the main goal is fidellity to the original text.

**Instructions:**

1.  **Text Content:** Extract ALL readable text from the image exactly as it appears, preserving the original language (Portuguese). Preserve paragraph structure where possible.
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
5.  **HTML Structure:**
    *   Use semantic HTML tags (e.g., `<p>`, `<h1>`-`<h6>`, `<ul>`, `<ol>`, `<li>`, `<table>`).
    *   Structure the content logically based on the original layout.
    *   **Output ONLY the extracted text, LaTeX math, table HTML, and textual descriptions as HTML body content enclosed within a single Markdown code block like this:**
        ```html
        <!-- HTML body content goes here -->
        <p>Texto extraído...</p>
        <p><i>[Descrição: Diagrama...]</i></p>
        $$ LaTeX $$
        <table>...</table>
        ...
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

def generate_html_for_image(model, file_object, page_filename, img_path, status_callback):
    """Gera HTML para uma única imagem."""
    try:
        img = Image.open(img_path)
        dimensions = img.size
        img.close()
    except Exception as e:
        status_callback(f"  Warning: Could not read image dimensions for {page_filename}: {e}")
        dimensions = ("unknown", "unknown")

    prompt_parts = create_html_prompt_with_desc(file_object, page_filename, dimensions)
    try:
        status_callback(f"  Sending request to Gemini for {page_filename}...")
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
             status_callback(f"  Raw response: {response.text[:200]}...") # Log part of raw response
             return None

        status_callback(f"  HTML extracted for {page_filename}.")
        return html_body

    except google_exceptions.ResourceExhausted as e:
         status_callback(f"  Error: Rate limit likely hit for {page_filename}. {e}")
         raise # Re-raise to stop processing in the main thread function
    except Exception as e:
        status_callback(f"  An unexpected error occurred processing {page_filename}: {e}")
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
             status_callback(f"  Prompt Feedback: {response.prompt_feedback}")
        if 'response' in locals() and hasattr(response, 'candidates') and response.candidates:
             status_callback(f"  Finish Reason: {response.candidates.finish_reason}")
             status_callback(f"  Safety Ratings: {response.candidates.safety_ratings}")
        return None # Return None on other errors for this page

def create_merged_html_with_accessibility(content_list, output_path, pdf_filename):
    """Cria o HTML final com controles de acessibilidade."""
    if not content_list:
        print("No content to merge.")
        return False

    # <<< ADICIONADO: CSS e JavaScript para acessibilidade >>>
    accessibility_css = """
<style>
    body {
      font-family: Verdana, Arial, sans-serif;
      line-height: 1.6;
      padding: 20px;
      background-color: #f0f0f0;
      color: #333;
    }

    /* menu agora sticky */
    #accessibility-controls {
      position: sticky;
      top: 0;
      z-index: 1000;
      background-color: #e0e0e0;
      padding: 10px;
      margin-bottom: 20px;
      border: 1px solid #ccc;
      border-radius: 5px;
    }

    #accessibility-controls button,
    #accessibility-controls select {
      margin: 0 5px;
      padding: 5px 10px;
      cursor: pointer;
    }

    .page-content {
      background-color: #fff;
      padding: 15px;
      margin-bottom: 20px;
      border: 1px solid #ddd;
      border-radius: 3px;
    }

    h1, h2, h3 {
      border-bottom: 1px solid #eee;
      padding-bottom: 0.3em;
      color: #000;
    }

    hr.page-separator {
      margin-top: 2em;
      margin-bottom: 2em;
      border: 1px dashed #ccc;
    }

    p i, span i {
      color: #555;
      font-style: italic;
    }
</style>
"""

    accessibility_js = """
<script>
    let currentFontSize = 16; // Tamanho inicial em pixels
    const fonts = ['Verdana', 'Arial', 'Times New Roman', 'Courier New']; // Fontes disponíveis
    let currentFontIndex = 0;
    const synth = window.speechSynthesis;
    let utterance = null;
    let isPaused = false;

    function changeFontSize(delta) {
        currentFontSize += delta;
        if (currentFontSize < 10) currentFontSize = 10; // Mínimo
        if (currentFontSize > 40) currentFontSize = 40; // Máximo
        document.body.style.fontSize = currentFontSize + 'px';
    }

    function changeFontFamily() {
        currentFontIndex = (currentFontIndex + 1) % fonts.length;
        document.body.style.fontFamily = fonts[currentFontIndex] + ', sans-serif';
        document.getElementById('fontSelector').value = fonts[currentFontIndex];
    }

    function setFontFamily(fontName) {
        const index = fonts.indexOf(fontName);
        if (index !== -1) {
            currentFontIndex = index;
            document.body.style.fontFamily = fonts[currentFontIndex] + ', sans-serif';
        }
    }

    function getTextToSpeak() {
        // Tenta pegar o texto selecionado, senão pega o corpo todo
        let selectedText = window.getSelection().toString();
        if (selectedText) {
            return selectedText;
        } else {
            // Pega o conteúdo principal, excluindo os controles
            const contentElement = document.querySelector('body'); // Ou um ID mais específico se tiver
            if (contentElement) {
                const controls = document.getElementById('accessibility-controls');
                let text = '';
                contentElement.childNodes.forEach(node => {
                    if (node !== controls && node.textContent) {
                         // Tenta remover scripts e styles, embora seja melhor estruturar o HTML para isso
                         if (node.tagName !== 'SCRIPT' && node.tagName !== 'STYLE') {
                            text += node.textContent.trim() + '\\n\\n';
                         }
                    }
                });
                return text;
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
            stopSpeech(); // Para qualquer fala anterior
            utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'pt-BR'; // Define o idioma
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
            synth.cancel();
            utterance = null;
            isPaused = false;
        }
    }

    // Inicializa a fonte ao carregar
    document.addEventListener('DOMContentLoaded', () => {
       setFontFamily(fonts[currentFontIndex]);
       document.getElementById('fontSelector').value = fonts[currentFontIndex];
    });

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
    {css}
    {js}
</head>
""".format(doc_title=pdf_filename, css=accessibility_css, js=accessibility_js)

    # Monta o HTML final
    merged_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
{mathjax_config_head_merged}
<body>
    <h1>Documento Acessível: {pdf_filename}</h1>

    <div id="accessibility-controls">
        <span>Tamanho da Fonte:</span>
        <button onclick="changeFontSize(-2)">A-</button>
        <button onclick="changeFontSize(2)">A+</button>
        <span>Fonte:</span>
        <select id="fontSelector" onchange="setFontFamily(this.value)">
            <option value="Verdana">Verdana</option>
            <option value="Arial">Arial</option>
            <option value="Times New Roman">Times New Roman</option>
            <option value="Courier New">Courier New</option>
        </select>
        <span>Leitura:</span>
        <button onclick="speakText()">▶️ Ler/Continuar</button>
        <button onclick="pauseSpeech()">⏸️ Pausar</button>
        <button onclick="stopSpeech()">⏹️ Parar</button>
    </div>
"""

    for content_data in content_list:
        page_num = content_data["page_num"]
        html_body = content_data["body"]

        merged_html += f"\n<hr class=\"page-separator\">\n"
        merged_html += f"<div class='page-content' id='page-{page_num}'>\n" # Div para cada página
        merged_html += f"<h2>Página {page_num}</h2>\n"
        merged_html += html_body
        merged_html += "\n</div>\n" # Fecha a div da página

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
        temp_image_dir = "/Exata_Falada/temp_pdf_images_local" # Diretório temporário

        # 1. Configurar API Key (Lendo da variável de ambiente)
        api_key = os.environ.get(API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(f"API Key not found. Set the '{API_KEY_ENV_VAR}' environment variable.")
        genai.configure(api_key=api_key)
        status_callback("API Key configured.")

        # 2. Converter PDF para Imagens
        status_callback("Converting PDF to images...")
        image_paths = pdf_to_images_local(pdf_path, temp_image_dir, dpi)
        if not image_paths:
            raise Exception("Failed to convert PDF to images.")
        status_callback(f"Converted to {len(image_paths)} images.")

        # 3. Fazer Upload para Gemini File API
        uploaded_files_map = upload_to_gemini_file_api(image_paths, status_callback)
        if not uploaded_files_map:
            raise Exception("Failed to upload images to Gemini API.")

        # 4. Gerar HTML para cada imagem via Gemini
        status_callback("Initializing Gemini model...")
        model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
        status_callback("Generating accessible HTML content for each page...")

        generated_content_list_thread = []
        sorted_paths = sorted(uploaded_files_map.keys())

        for i, img_path in enumerate(sorted_paths):
            page_num = i + 1
            base_image_name = os.path.basename(img_path)
            status_callback(f"  Processing Page {page_num}/{len(sorted_paths)} ({base_image_name})...")
            file_object = uploaded_files_map[img_path]

            html_body = generate_html_for_image(model, file_object, base_image_name, img_path, status_callback)

            if html_body:
                generated_content_list_thread.append({
                    "page_num": page_num,
                    "body": html_body,
                })
            else:
                status_callback(f"  Skipping page {page_num} due to generation error.")
            time.sleep(2) # Delay

        if not generated_content_list_thread:
             raise Exception("No HTML content was generated successfully.")

        status_callback("HTML generation finished.")

        # 5. Juntar HTMLs
        status_callback("Merging generated HTML content...")
        pdf_filename_base = os.path.splitext(os.path.basename(pdf_path))
        success = create_merged_html_with_accessibility(
            generated_content_list_thread,
            output_html_path,
            pdf_filename_base
        )
        if not success:
            raise Exception("Failed to create merged HTML file.")
        status_callback(f"Merged HTML saved to {output_html_path}")

        # 6. (Opcional) Limpar arquivos temporários (imagens e uploads)
        status_callback("Cleaning up temporary files...")
        try:
             # Para abrir o primeiro arquivo no Windows Explorer (opcional)
            for img_path in image_paths:
                os.remove(img_path)
            os.rmdir(temp_image_dir)
            status_callback("  Local images removed.")
            # Deletar arquivos da API do Gemini
            count_deleted = 0
            for file_obj in uploaded_files_map.values():
                 try:
                      genai.delete_file(file_obj.name)
                      count_deleted += 1
                      time.sleep(1) # Delay para delete API
                 except Exception as del_e:
                      status_callback(f"  Warning: Could not delete uploaded file {file_obj.name}: {del_e}")
            status_callback(f"  Attempted to delete {count_deleted} uploaded files from API.")
        except Exception as clean_e:
            status_callback(f"  Error during cleanup: {clean_e}")

        status_callback("Process finished successfully!")
        completion_callback(True, f"Success! Accessible HTML saved to:\n{output_html_path}")

    except Exception as e:
        error_message = f"An error occurred: {e}"
        status_callback(f"Error: {e}")
        completion_callback(False, error_message)


# --- Interface Gráfica (Tkinter) ---

class OldSchoolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF to Accessible HTML Converter")
        self.root.geometry("600x450")
        self.root.configure(bg='black')

        # Fontes
        self.label_font = font.Font(family="Courier New", size=12)
        self.button_font = font.Font(family="Courier New", size=10, weight="bold")
        self.text_font = font.Font(family="Courier New", size=10)

        # Estilo
        style = ttk.Style()
        style.theme_use('clam') # Tema base que permite mais customização
        style.configure("TLabel", background="black", foreground="green", font=self.label_font)
        style.configure("TButton", background="#222", foreground="lime green", font=self.button_font, borderwidth=1, focusthickness=3, focuscolor='lime green')
        style.map("TButton", background=[('active', '#444')], foreground=[('active', 'white')])
        style.configure("TEntry", fieldbackground="#222", foreground="lime green", insertcolor='lime green', font=self.text_font)
        style.configure("TFrame", background="black")

        # Frame Principal
        main_frame = ttk.Frame(root, padding="10 10 10 10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Seleção de PDF
        pdf_frame = ttk.Frame(main_frame)
        pdf_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pdf_frame, text="PDF File:").pack(side=tk.LEFT, padx=5)
        self.pdf_path_var = tk.StringVar()
        self.pdf_entry = ttk.Entry(pdf_frame, textvariable=self.pdf_path_var, width=50)
        self.pdf_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.browse_button = ttk.Button(pdf_frame, text="Browse...", command=self.select_pdf)
        self.browse_button.pack(side=tk.LEFT, padx=5)

        # Opção de DPI
        dpi_frame = ttk.Frame(main_frame)
        dpi_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dpi_frame, text="Image DPI:").pack(side=tk.LEFT, padx=5)
        self.dpi_var = tk.StringVar(value="150") # Valor padrão
        self.dpi_entry = ttk.Entry(dpi_frame, textvariable=self.dpi_var, width=10)
        self.dpi_entry.pack(side=tk.LEFT, padx=5)

        # Botão de Conversão
        self.convert_button = ttk.Button(main_frame, text="Convert to Accessible HTML", command=self.start_conversion)
        self.convert_button.pack(pady=15)

        # Área de Status/Log
        ttk.Label(main_frame, text="Status:").pack(anchor=tk.W, pady=(10, 0))
        self.status_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=15, bg='#111', fg='lime green', font=self.text_font, relief=tk.SUNKEN, bd=1)
        self.status_text.pack(expand=True, fill=tk.BOTH, pady=5)
        self.status_text.configure(state='disabled') # Começa desabilitado para escrita

    def select_pdf(self):
        filepath = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=(("PDF files", "*.pdf"), ("All files", "*.*"))
        )
        if filepath:
            self.pdf_path_var.set(filepath)
            self.update_status(f"Selected PDF: {filepath}")

    def update_status(self, message):
        """Atualiza a área de status de forma segura para threads."""
        def append_message():
            self.status_text.configure(state='normal')
            self.status_text.insert(tk.END, message + "\n")
            self.status_text.configure(state='disabled')
            self.status_text.see(tk.END) # Rola para o final
        # Usa root.after para garantir que a atualização ocorra na thread principal da GUI
        self.root.after(0, append_message)

    def conversion_complete(self, success, message):
        """Chamado quando a thread de conversão termina."""
        def final_update():
            self.convert_button.config(state=tk.NORMAL, text="Convert to Accessible HTML")
            if success:
                messagebox.showinfo("Success", message)
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

        # Define o caminho de saída (mesmo diretório do PDF, com novo nome)
        output_dir = os.path.dirname(pdf_path)
        pdf_filename_base = os.path.splitext(os.path.basename(pdf_path))
        output_html_path = os.path.join(output_dir, f"{pdf_filename_base}_accessible.html")

        # Limpa status e desabilita botão
        self.status_text.configure(state='normal')
        self.status_text.delete('1.0', tk.END)
        self.status_text.configure(state='disabled')
        self.convert_button.config(state=tk.DISABLED, text="Processing...")

        # Inicia o processamento em uma nova thread
        thread = threading.Thread(
            target=process_pdf_thread,
            args=(pdf_path, output_html_path, dpi, self.update_status, self.conversion_complete)
        )
        thread.daemon = True # Permite fechar a GUI mesmo se a thread estiver rodando
        thread.start()

# --- Execução da Aplicação ---
if __name__ == "__main__":
    # Instrução para a chave API (IMPORTANTE)
    print("--------------------------------------------------------------------")
    print(f"IMPORTANT: Ensure your Google AI API Key is set as an environment")
    print(f"variable named '{API_KEY_ENV_VAR}' before running this script.")
    print("Example (Command Prompt): set GOOGLE_API_KEY=YOUR_API_KEY_HERE")
    print("Example (PowerShell):   $env:GOOGLE_API_KEY='YOUR_API_KEY_HERE'")
    print("Example (Linux/macOS):  export GOOGLE_API_KEY='YOUR_API_KEY_HERE'")
    print("--------------------------------------------------------------------")

    # Verifica se a chave API está definida (apenas um aviso inicial)
    if not os.environ.get(API_KEY_ENV_VAR):
         print(f"WARNING: Environment variable '{API_KEY_ENV_VAR}' not detected.")
         # Poderia mostrar um erro e sair, mas vamos deixar o process_pdf_thread falhar
         # messagebox.showerror("API Key Error", f"Environment variable '{API_KEY_ENV_VAR}' not set.")
         # sys.exit(1) # Descomente para sair se a chave não estiver definida

    root = tk.Tk()
    app = OldSchoolApp(root)
    root.mainloop()
