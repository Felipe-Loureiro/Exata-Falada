import os
import uuid
import threading
import boto3
from botocore.exceptions import ClientError
from flask import Flask, render_template, request, jsonify, redirect, send_from_directory
from werkzeug.utils import secure_filename
from processing import process_pdf_web, AVAILABLE_GEMINI_MODELS, DEFAULT_GEMINI_MODEL, API_KEY_ENV_VAR
from config import MODE
from dotenv import load_dotenv

load_dotenv()

# --- Configuração do Flask ---
ALLOWED_EXTENSIONS = {'pdf'}
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limite de 50 MB para upload


if MODE == "LOCAL":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'outputs'
    # Usa os caminhos absolutos para definir as pastas
    app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, UPLOAD_FOLDER)
    app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, OUTPUT_FOLDER)

    # Garante que as pastas existam no servidor
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
elif MODE == "S3":
    # Carrega as configurações do S3 a partir de variáveis de ambiente
    S3_BUCKET = os.environ.get('S3_BUCKET')
    S3_REGION = os.environ.get('S3_REGION')

    # Validação para garantir que as variáveis de ambiente foram configuradas
    if not S3_BUCKET or not S3_REGION:
        raise ValueError("As variáveis de ambiente S3_BUCKET e S3_REGION devem ser definidas.")

    # Instancia o cliente S3 uma vez para reutilização
    s3_client = boto3.client('s3', region_name=S3_REGION)

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

        if MODE == "LOCAL":
            # 2. Salvar o arquivo no servidor
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(pdf_path)
        elif MODE == "S3":
            # --- LÓGICA DE UPLOAD PARA O S3 ---
            original_filename = secure_filename(file.filename)
            # Cria um nome de objeto único no S3 para evitar colisões
            s3_pdf_object_name = f"uploads/{original_filename}"

            try:
                # Faz o upload do arquivo diretamente do stream da requisição para o S3
                s3_client.upload_fileobj(file, S3_BUCKET, s3_pdf_object_name)
            except ClientError as e:
                app.logger.error(f"Erro no upload para o S3: {e}")
                return jsonify({'error': 'Falha ao salvar o arquivo no armazenamento externo.'}), 500

        # 3. Obter parâmetros do formulário
        dpi = int(request.form.get('dpi', 100))
        page_range = request.form.get('page_range', '')
        model = request.form.get('model', DEFAULT_GEMINI_MODEL)
        upload_workers = int(request.form.get('upload_workers', 10))
        generate_workers = int(request.form.get('generate_workers', 5))

        # 4. Criar e registrar a tarefa
        task_id = str(uuid.uuid4())
        if MODE == "LOCAL":
            output_filename = f"{os.path.splitext(filename)[0]}_{task_id[:8]}.html"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        elif MODE == "S3":
            # O nome do arquivo de saída agora é apenas uma referência, será a chave no S3
            output_s3_key = f"outputs/{os.path.splitext(original_filename)[0]}_{task_id[:8]}.html"

        cancel_event = threading.Event()
        if MODE == "LOCAL":
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
        elif MODE == "S3":
            TASKS[task_id] = {
                'status': 'Iniciando...', 'progress': 0, 'total': 100, 'log': [],
                'is_complete': False, 'success': None,
                'output_s3_key': None,  # Armazena a chave do S3 em vez do caminho local
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
                if MODE == "LOCAL":
                    TASKS[task_id]['output_path'] = output_path
                    TASKS[task_id]['output_filename'] = output_filename
                elif MODE == "S3":
                    TASKS[task_id]['output_s3_key'] = output_s3_key
                TASKS[task_id]['status'] = "Concluído com Sucesso!"
            else:
                TASKS[task_id]['status'] = f"Falha: {result_msg}"

        # 6. Iniciar a thread de processamento
        if MODE == "LOCAL":
            thread = threading.Thread(
                target=process_pdf_web,
                args=(
                    dpi, page_range, model,
                    upload_workers, generate_workers, cancel_event,
                    status_callback, completion_callback, progress_callback,
                    None, None, None, pdf_path, output_path
                )
            )
        elif MODE == "S3":
            thread = threading.Thread(
                target=process_pdf_web,
                args=(
                    dpi, page_range, model, upload_workers, generate_workers, cancel_event,
                    status_callback, completion_callback, progress_callback,
                    S3_BUCKET, s3_pdf_object_name, output_s3_key, None, None
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
    if MODE == "LOCAL":
        return jsonify({
            'status': task['status'],
            'progress': task['progress'],
            'total': task['total'],
            'log': task['log'],
            'is_complete': task['is_complete'],
            'success': task['success'],
            'output_filename': task['output_filename']
        })
    elif MODE == "S3":
        return jsonify({
            'status': task['status'], 'progress': task['progress'], 'total': task['total'],
            'log': task['log'], 'is_complete': task['is_complete'], 'success': task['success'],
            'output_s3_key': task['output_s3_key']  # Envia a chave S3 para o front-end
        })
    return None


@app.route('/download/<task_id>')
def download_file(task_id):
    task = TASKS.get(task_id)
    if not task:
        return jsonify({'error': 'Tarefa não encontrada'}), 404
    if MODE == "LOCAL":
        filename = task['output_filename']
        return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)
    elif MODE == "S3":
        task = TASKS.get(task_id)
        if not task or not task.get('is_complete') or not task.get('success'):
            return "Tarefa não encontrada ou não concluída.", 404

        output_s3_key = task.get('output_s3_key')
        if not output_s3_key:
            return "Arquivo de saída não encontrado.", 404

        try:
            # Pega o nome do arquivo a partir da chave S3 para usar no download
            # Ex: de 'outputs/meu_arquivo.html' pega 'meu_arquivo.html'
            download_filename = os.path.basename(output_s3_key)

            # Gera uma URL de download temporária e segura (válida por 5 minutos)
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': output_s3_key, 'ResponseContentDisposition': f'attachment; filename="{download_filename}"'},
                ExpiresIn=300
            )
            # Redireciona o navegador do usuário para a URL do S3
            return redirect(url)
        except ClientError as e:
            app.logger.error(f"Erro ao gerar URL presignada do S3: {e}")
            return "Não foi possível gerar o link para download.", 500
    return None


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
    if MODE == "LOCAL":
        # Cria as pastas necessárias se não existirem
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, 'temp_processing']:
            if not os.path.exists(folder):
                os.makedirs(folder)
    elif MODE == "S3":
        # Garante que a chave da AWS está configurada, se não, não inicia
        if not os.environ.get('AWS_ACCESS_KEY_ID') or not os.environ.get('AWS_SECRET_ACCESS_KEY'):
            raise ValueError("Credenciais da AWS (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) não definidas.")
    app.run(debug=True, host='0.0.0.0')

""" Para um deploy em produção, você usaria um servidor WSGI como o Gunicorn:
gunicorn --workers 3 --threads 4 --bind 0.0.0.0:8000 app:app
O uso de --threads é importante aqui, pois permite que o servidor lide com as requisições de status enquanto as threads de trabalho estão ocupadas processando os PDFs. """