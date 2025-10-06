import os
import uuid
import threading
import boto3
import json
from botocore.exceptions import ClientError
from botocore.client import Config
from flask import Flask, render_template, request, jsonify, redirect, send_from_directory, flash, send_file
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
from processing import process_pdf_web, AVAILABLE_GEMINI_MODELS, DEFAULT_GEMINI_MODEL, API_KEY_ENV_VAR
from config import MODE
from dotenv import load_dotenv
from io import BytesIO  # Importe BytesIO para manipulação em memória
from patcher import patch_html_files  # Importe a nova função
from flask_cors import CORS

# --- IMPORTAÇÕES DO BANCO DE DADOS ---
import database

load_dotenv()

# --- Configuração do Flask ---
ALLOWED_EXTENSIONS = {'pdf'}
app = Flask(__name__)
# Substitua "http://localhost:5173" pelo domínio em produção.
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5000"]}})
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limite de 50 MB para upload
app.secret_key = os.urandom(24)  # NECESSÁRIO para usar o sistema de mensagens 'flash'

# --- INICIALIZAÇÃO DO BANCO DE DADOS ---
# Garante que o arquivo do banco de dados e a tabela sejam criados na inicialização
database.init_db()


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
elif MODE == "BUCKET":
    # Carrega as configurações do S3 a partir de variáveis de ambiente
    S3_BUCKET = os.environ.get('S3_BUCKET')
    S3_REGION = os.environ.get('S3_REGION')
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

    OCI_BUCKET = os.environ.get('OCI_BUCKET')
    OCI_ENDPOINT_URL = os.environ.get('OCI_ENDPOINT_URL')
    OCI_ACCESS_KEY_ID = os.environ.get('OCI_ACCESS_KEY_ID')
    OCI_SECRET_ACCESS_KEY = os.environ.get('OCI_SECRET_ACCESS_KEY')
    OCI_REGION = os.environ.get('OCI_REGION')

    # Validação para garantir que as variáveis de ambiente foram configuradas
    if not all([S3_BUCKET, S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        if not all([OCI_BUCKET, OCI_ENDPOINT_URL, OCI_ACCESS_KEY_ID, OCI_SECRET_ACCESS_KEY, OCI_REGION]):
            raise ValueError("As variáveis de ambiente AWS ou OCI devem ser definidas.")

    if S3_BUCKET and OCI_BUCKET:
        raise ValueError("Ambas as variáveis de ambiente AWS e OCI foram definidas.")

    # Instancia o cliente S3 uma vez para reutilização
    if S3_BUCKET:
        s3_client = boto3.client('s3', region_name=S3_REGION)
    elif OCI_BUCKET:
        config = Config(
            request_checksum_calculation="WHEN_REQUIRED",
            response_checksum_validation="WHEN_REQUIRED"
        )
        s3_client = boto3.client(
            's3',
            region_name=OCI_REGION,
            endpoint_url=OCI_ENDPOINT_URL,
            aws_access_key_id=OCI_ACCESS_KEY_ID,
            aws_secret_access_key=OCI_SECRET_ACCESS_KEY,
            config = config
        )


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_html_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'html', 'htm'}


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

        pdf_path, s3_pdf_object_name, original_filename = None, None, None
        if MODE == "LOCAL":
            # 2. Salvar o arquivo no servidor
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(pdf_path)
        elif MODE == "BUCKET":
            # --- LÓGICA DE UPLOAD PARA O S3 ---
            original_filename = secure_filename(file.filename)
            # Cria um nome de objeto único no S3 para evitar colisões
            s3_pdf_object_name = f"uploads/{original_filename}"

            try:
                # Faz o upload do arquivo diretamente do stream da requisição para o S3
                if S3_BUCKET:
                    s3_client.upload_fileobj(file, S3_BUCKET, s3_pdf_object_name)
                elif OCI_BUCKET:
                    s3_client.upload_fileobj(file, OCI_BUCKET, s3_pdf_object_name)
            except ClientError as e:
                app.logger.error(f"Erro no upload para o Bucket: {e}")
                return jsonify({'error': 'Falha ao salvar o arquivo no armazenamento externo.'}), 500

        # 3. Obter parâmetros do formulário
        dpi = int(request.form.get('dpi', 100))
        page_range = request.form.get('page_range', '')
        model = request.form.get('model', DEFAULT_GEMINI_MODEL)
        upload_workers = int(request.form.get('upload_workers', 10))
        generate_workers = int(request.form.get('generate_workers', 5))

        # 4. Criar e registrar a tarefa
        task_id = str(uuid.uuid4())
        database.create_task(task_id)

        output_path, output_s3_key = None, None
        if MODE == "LOCAL":
            output_filename = f"{os.path.splitext(filename)[0]}_{task_id[:8]}.html"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        elif MODE == "BUCKET":
            # O nome do arquivo de saída agora é apenas uma referência, será a chave no S3
            output_s3_key = f"outputs/{os.path.splitext(original_filename)[0]}_{task_id[:8]}.html"

        # 5. Definir callbacks para atualizar o estado da tarefa
        def status_callback(msg):
            database.append_to_log(task_id, msg)

        def progress_callback(val, total, phase_text):
            database.update_task_progress(task_id, val, total, phase_text)

        def completion_callback(success, result_msg):
            output_info = {}
            if success:
                if MODE == "LOCAL":
                    output_info = {'output_path': output_path, 'output_filename': output_filename}
                elif MODE == "BUCKET":
                    output_info = {'output_s3_key': output_s3_key}

            database.update_task_completion(task_id, success, result_msg, output_info)

        # 6. Iniciar a thread de processamento
        thread_args_dict = {
            "dpi": dpi, "page_range_str": page_range, "selected_model_name": model,
            "num_upload_workers": upload_workers, "num_generate_workers": generate_workers,
            "task_id": task_id,
            "status_callback": status_callback, "completion_callback": completion_callback,
            "progress_callback": progress_callback
        }

        if MODE == "LOCAL":
            thread_args_dict.update({"pdf_path": pdf_path, "initial_output_html_path_base": output_path})
        elif MODE == "BUCKET":
            bucket_name = S3_BUCKET if S3_BUCKET else OCI_BUCKET
            thread_args_dict.update(
                {"s3_bucket": bucket_name, "s3_pdf_object_name": s3_pdf_object_name, "output_s3_key": output_s3_key})

        thread = threading.Thread(target=process_pdf_web, kwargs=thread_args_dict)
        thread.daemon = True
        thread.start()

        return jsonify({'task_id': task_id})

    # Para requisições GET
    api_key_set = bool(os.environ.get(API_KEY_ENV_VAR))
    return render_template('index.html', models=AVAILABLE_GEMINI_MODELS, default_model=DEFAULT_GEMINI_MODEL,
                           api_key_set=api_key_set)


@app.route('/status/<task_id>')
def task_status(task_id):
    task_row = database.get_task(task_id)
    if not task_row:
        return jsonify({'error': 'Tarefa não encontrada'}), 404

    # Converte a linha do banco de dados em um dicionário e decodifica o log
    task_dict = dict(task_row)
    task_dict['log'] = json.loads(task_dict['log'])

    # Retorna o estado atual da tarefa
    response_data = {
        'status': task_dict['status'],
        'progress': task_dict['progress'],
        'total': task_dict['total'],
        'log': task_dict['log'],
        'is_complete': bool(task_dict['is_complete']),
        'success': bool(task_dict['success']) if task_dict['success'] is not None else None,
    }
    if MODE == "LOCAL":
        response_data['output_filename'] = task_dict['output_filename']
    elif MODE == "BUCKET":
        response_data['output_s3_key'] = task_dict['output_s3_key']

    return jsonify(response_data)


@app.route('/download/<task_id>')
def download_file(task_id):
    task = database.get_task(task_id)
    if not task or not task['is_complete'] or not task['success']:
        return "Tarefa não encontrada, não concluída ou falhou.", 404

    if MODE == "LOCAL":
        filename = task['output_filename']
        return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)
    elif MODE == "BUCKET":
        output_s3_key = task['output_s3_key']
        if not output_s3_key:
            return "Arquivo de saída não encontrado.", 404

        try:
            # Pega o nome do arquivo a partir da chave S3 para usar no download
            # Ex: de 'outputs/meu_arquivo.html' pega 'meu_arquivo.html'
            download_filename = os.path.basename(output_s3_key)

            # Gera uma URL de download temporária e segura (válida por 5 minutos)
            bucket_name = S3_BUCKET if S3_BUCKET else OCI_BUCKET
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': output_s3_key,
                        'ResponseContentDisposition': f'attachment; filename="{download_filename}"'},
                ExpiresIn=300
            )
            return redirect(url)
        except ClientError as e:
            app.logger.error(f"Erro ao gerar URL presignada: {e}")
            return "Não foi possível gerar o link para download.", 500
    return None


@app.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    task = database.get_task(task_id)
    if not task:
        return jsonify({'error': 'Tarefa não encontrada'}), 404

    if not task['is_complete']:
        database.request_cancel(task_id)
        database.update_task_progress(task_id, task['progress'], task['total'], 'Cancelamento solicitado...')
        return jsonify({'message': 'Cancelamento solicitado'})

    return jsonify({'message': 'A tarefa já foi concluída'})

# ==========================================================
# ===   NOVA ROTA PARA VALIDAÇÃO DA SENHA DE DEV         ===
# ==========================================================
@app.route('/unlock-dev', methods=['POST'])
def unlock_dev_mode():
    """
    Recebe uma senha via JSON e a valida contra a variável de ambiente.
    """
    data = request.get_json()
    if not data or 'password' not in data:
        return jsonify({'success': False, 'error': 'Senha não fornecida'}), 400

    submitted_password = data['password']
    correct_password = os.environ.get('DEV_PASSWORD')

    # É crucial que a variável DEV_PASSWORD esteja configurada no ambiente
    if not correct_password:
        app.logger.error("A variável de ambiente DEV_PASSWORD não foi configurada no servidor.")
        # Retorna erro genérico para não expor detalhes da configuração
        return jsonify({'success': False, 'error': 'Erro de configuração no servidor'}), 500

    if submitted_password == correct_password:
        return jsonify({'success': True}), 200
    else:
        # Retorna 401 Unauthorized para senhas incorretas
        return jsonify({'success': False, 'error': 'Senha incorreta'}), 401


# ==========================================================
# ===   NOVA ROTA PARA CORREÇÃO/MERGE DE ARQUIVOS HTML   ===
# ==========================================================
@app.route('/patch', methods=['GET', 'POST'])
def patch_html():
    if request.method == 'POST':
        # 1. Validar se os arquivos foram enviados
        if 'original_file' not in request.files or 'corrections_file' not in request.files:
            flash('Ambos os arquivos (original e correções) são necessários.', 'danger')
            return redirect(request.url)

        original_file = request.files['original_file']
        corrections_file = request.files['corrections_file']

        # 2. Validar nomes e extensões dos arquivos
        if original_file.filename == '' or corrections_file.filename == '':
            flash('Por favor, selecione os dois arquivos.', 'danger')
            return redirect(request.url)

        if not allowed_html_file(original_file.filename) or not allowed_html_file(corrections_file.filename):
            flash('Apenas arquivos .html ou .htm são permitidos.', 'danger')
            return redirect(request.url)

        try:
            # 3. Ler o conteúdo dos arquivos em memória (evita salvar no disco)
            original_content = original_file.read().decode('utf-8')
            corrections_content = corrections_file.read().decode('utf-8')

            # 4. Chamar a função de processamento do patcher.py
            final_html_content = patch_html_files(original_content, corrections_content)

            # 5. Preparar o arquivo final para download, também em memória
            buffer = BytesIO()
            buffer.write(final_html_content.encode('utf-8'))
            buffer.seek(0)  # "Rebobina" o buffer para o início

            # Gera um nome para o arquivo de saída
            original_basename = os.path.splitext(secure_filename(original_file.filename))[0]
            output_filename = f"{original_basename}_corrigido.html"

            # Envia o arquivo do buffer diretamente para o usuário
            return send_file(
                buffer,
                as_attachment=True,
                download_name=output_filename,
                mimetype='text/html'
            )

        except Exception as e:
            app.logger.error(f"Erro ao processar o patch de HTML: {e}")
            flash(f'Ocorreu um erro inesperado durante o processamento: {e}', 'danger')
            return redirect(request.url)

    # Para requisições GET, apenas renderiza a página com o formulário
    return render_template('patch.html')


# ==========================================================
# ===   NOVAS ROTAS /api/ PARA O FRONTEND REACT          ===
# ==========================================================

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'pdf_file' not in request.files:
        return jsonify({'success': False, 'error': 'Nenhum arquivo enviado'}), 400
    file = request.files['pdf_file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Arquivo inválido ou extensão não permitida'}), 400

    try:
        # A lógica de processamento é a mesma da rota original, mas sem flash/redirect
        pdf_path, s3_pdf_object_name, original_filename = None, None, None
        if MODE == "LOCAL":
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(pdf_path)
        elif MODE == "BUCKET":
            original_filename = secure_filename(file.filename)
            s3_pdf_object_name = f"uploads/{original_filename}"
            # ... (sua lógica de upload para S3/OCI)

        dpi = int(request.form.get('dpi', 100))
        page_range = request.form.get('page_range', '')
        model = request.form.get('model', DEFAULT_GEMINI_MODEL)
        upload_workers = int(request.form.get('upload_workers', 10))
        generate_workers = int(request.form.get('generate_workers', 5))

        task_id = str(uuid.uuid4())
        database.create_task(task_id)

        output_path, output_s3_key = None, None
        if MODE == "LOCAL":
            output_filename = f"{os.path.splitext(filename)[0]}_{task_id[:8]}.html"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        elif MODE == "BUCKET":
            # O nome do arquivo de saída agora é apenas uma referência, será a chave no S3
            output_s3_key = f"outputs/{os.path.splitext(original_filename)[0]}_{task_id[:8]}.html"

        # 5. Definir callbacks para atualizar o estado da tarefa
        def status_callback(msg):
            database.append_to_log(task_id, msg)

        def progress_callback(val, total, phase_text):
            database.update_task_progress(task_id, val, total, phase_text)

        def completion_callback(success, result_msg):
            output_info = {}
            if success:
                if MODE == "LOCAL":
                    output_info = {'output_path': output_path, 'output_filename': output_filename}
                elif MODE == "BUCKET":
                    output_info = {'output_s3_key': output_s3_key}

            database.update_task_completion(task_id, success, result_msg, output_info)

        # 6. Iniciar a thread de processamento
        thread_args_dict = {
            "dpi": dpi, "page_range_str": page_range, "selected_model_name": model,
            "num_upload_workers": upload_workers, "num_generate_workers": generate_workers,
            "task_id": task_id,
            "status_callback": status_callback, "completion_callback": completion_callback,
            "progress_callback": progress_callback
        }

        if MODE == "LOCAL":
            thread_args_dict.update({"pdf_path": pdf_path, "initial_output_html_path_base": output_path})
        elif MODE == "BUCKET":
            bucket_name = S3_BUCKET if S3_BUCKET else OCI_BUCKET
            thread_args_dict.update(
                {"s3_bucket": bucket_name, "s3_pdf_object_name": s3_pdf_object_name, "output_s3_key": output_s3_key})

        thread = threading.Thread(target=process_pdf_web, kwargs=thread_args_dict)
        thread.daemon = True
        thread.start()

        return jsonify({'success': True, 'task_id': task_id}), 202
    except Exception as e:
        app.logger.error(f"Erro no endpoint /api/upload: {e}")
        return jsonify({'success': False, 'error': 'Ocorreu um erro interno no servidor.'}), 500


@app.route('/api/status/<task_id>', methods=['GET'])
def api_task_status(task_id):
    return task_status(task_id)


@app.route('/api/download/<task_id>', methods=['GET'])
def api_download_file(task_id):
    return download_file(task_id)


@app.route('/api/cancel/<task_id>', methods=['POST'])
def api_cancel_task(task_id):
    return cancel_task(task_id)


@app.route('/api/unlock-dev', methods=['POST'])
def api_unlock_dev_mode():
    return unlock_dev_mode()

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'available_models': AVAILABLE_GEMINI_MODELS,
        'default_model': DEFAULT_GEMINI_MODEL
    })


@app.route('/api/patch', methods=['POST'])
def api_patch_html():
    if 'original_file' not in request.files or 'corrections_file' not in request.files:
        return jsonify({'success': False, 'error': 'Ambos os arquivos (original e correções) são necessários.'}), 400

    original_file = request.files['original_file']
    corrections_file = request.files['corrections_file']

    if original_file.filename == '' or corrections_file.filename == '':
        return jsonify({'success': False, 'error': 'Por favor, selecione os dois arquivos.'}), 400

    if not allowed_html_file(original_file.filename) or not allowed_html_file(corrections_file.filename):
        return jsonify({'success': False, 'error': 'Apenas arquivos .html ou .htm são permitidos.'}), 400

    try:
        original_content = original_file.read().decode('utf-8')
        corrections_content = corrections_file.read().decode('utf-8')
        final_html_content = patch_html_files(original_content, corrections_content)

        buffer = BytesIO()
        buffer.write(final_html_content.encode('utf-8'))
        buffer.seek(0)

        original_basename = os.path.splitext(secure_filename(original_file.filename))[0]
        output_filename = f"{original_basename}_corrigido.html"

        return send_file(
            buffer,
            as_attachment=True,
            download_name=output_filename,
            mimetype='text/html'
        )
    except Exception as e:
        app.logger.error(f"Erro ao processar o patch de HTML na API: {e}")
        return jsonify({'success': False, 'error': f'Ocorreu um erro inesperado durante o processamento: {e}'}), 500

if __name__ == '__main__':
    if MODE == "LOCAL":
        # Cria as pastas necessárias se não existirem
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, 'temp_processing']:
            if not os.path.exists(folder):
                os.makedirs(folder)
    app.run(debug=True, host='0.0.0.0')

""" Para um deploy em produção, você usaria um servidor WSGI como o Gunicorn:
gunicorn --workers 3 --threads 4 --timeout 300 --bind 0.0.0.0:5000 app:app
O uso de --threads é importante aqui, pois permite que o servidor lide com as requisições de status enquanto as threads de trabalho estão ocupadas processando os PDFs. """