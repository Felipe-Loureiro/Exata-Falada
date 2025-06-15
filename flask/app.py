import os
import uuid
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from processing import process_pdf_web, AVAILABLE_GEMINI_MODELS, DEFAULT_GEMINI_MODEL, API_KEY_ENV_VAR

# --- Construção de Caminhos Absolutos ---
# Pega o diretório onde o arquivo app.py está localizado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Configuração do Flask ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'pdf'}
app = Flask(__name__)
# Usa os caminhos absolutos para definir as pastas
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'temp_outputs')

# Garante que as pastas existam no servidor
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limite de 50 MB para upload

# --- Armazenamento de Tarefas em Memória ---
# ATENÇÃO: Este dicionário será resetado se o servidor for reiniciado.
# Para produção persistente, seria necessário um banco de dados (SQLite, etc.)
TASKS = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Rotas da Aplicação ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. Validar a requisição
        if 'pdf_file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        file = request.files['pdf_file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Arquivo inválido ou extensão não permitida'}), 400

        # 2. Salvar o arquivo no servidor
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)

        # 3. Obter parâmetros do formulário
        dpi = int(request.form.get('dpi', 100))
        page_range = request.form.get('page_range', '')
        model = request.form.get('model', DEFAULT_GEMINI_MODEL)
        upload_workers = int(request.form.get('upload_workers', 10))
        generate_workers = int(request.form.get('generate_workers', 5))

        # 4. Criar e registrar a tarefa
        task_id = str(uuid.uuid4())
        output_filename = f"{os.path.splitext(filename)[0]}_{task_id[:8]}.html"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        cancel_event = threading.Event()
        TASKS[task_id] = {
            'status': 'Iniciando...',
            'progress': 0,
            'total': 100,
            'log': [],
            'is_complete': False,
            'success': None,
            'output_path': None,
            'output_filename': None,
            'cancel_event': cancel_event
        }

        # 5. Definir callbacks para atualizar o estado da tarefa
        def status_callback(msg):
            TASKS[task_id]['log'].append(msg)

        def progress_callback(val, total, phase_text):
            TASKS[task_id]['progress'] = val
            TASKS[task_id]['total'] = total
            TASKS[task_id]['status'] = phase_text

        def completion_callback(success, result_msg):
            TASKS[task_id]['is_complete'] = True
            TASKS[task_id]['success'] = success
            if success:
                TASKS[task_id]['output_path'] = output_path
                TASKS[task_id]['output_filename'] = output_filename
                TASKS[task_id]['status'] = "Concluído com Sucesso!"
            else:
                TASKS[task_id]['status'] = f"Falha: {result_msg}"

        # 6. Iniciar a thread de processamento
        thread = threading.Thread(
            target=process_pdf_web,
            args=(
                pdf_path, output_path, dpi, page_range, model,
                upload_workers, generate_workers, cancel_event,
                status_callback, completion_callback, progress_callback
            )
        )
        thread.daemon = True
        thread.start()

        return jsonify({'task_id': task_id})

    # Para requisições GET
    api_key_set = bool(os.environ.get(API_KEY_ENV_VAR))
    return render_template('index.html', models=AVAILABLE_GEMINI_MODELS, default_model=DEFAULT_GEMINI_MODEL,
                           api_key_set=api_key_set)


@app.route('/status/<task_id>')
def task_status(task_id):
    task = TASKS.get(task_id)
    if not task:
        return jsonify({'error': 'Tarefa não encontrada'}), 404

    # Retorna o estado atual da tarefa
    return jsonify({
        'status': task['status'],
        'progress': task['progress'],
        'total': task['total'],
        'log': task['log'],
        'is_complete': task['is_complete'],
        'success': task['success'],
        'output_filename': task['output_filename']
    })


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)


@app.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    task = TASKS.get(task_id)
    if not task:
        return jsonify({'error': 'Tarefa não encontrada'}), 404

    if not task['is_complete']:
        task['cancel_event'].set()
        task['status'] = 'Cancelamento solicitado...'
        return jsonify({'message': 'Cancelamento solicitado'})

    return jsonify({'message': 'A tarefa já foi concluída'})


if __name__ == '__main__':
    # Cria as pastas necessárias se não existirem
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, 'temp_processing']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    app.run(debug=True, host='0.0.0.0')

""" Para um deploy em produção, você usaria um servidor WSGI como o Gunicorn:
gunicorn --workers 3 --threads 4 --bind 0.0.0.0:8000 app:app
O uso de --threads é importante aqui, pois permite que o servidor lide com as requisições de status enquanto as threads de trabalho estão ocupadas processando os PDFs. """