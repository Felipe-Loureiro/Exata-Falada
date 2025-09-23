# database.py
import sqlite3
import json
import threading
from config import MODE

DATABASE_FILE = 'tasks.db'

# Usamos um lock para evitar problemas de concorrência ao escrever no log,
# que é um processo de "leia-modifique-escreva".
db_lock = threading.Lock()


def get_db_connection():
    """Cria uma conexão com o banco de dados."""
    conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
    # Retorna as linhas como dicionários para facilitar o acesso por nome de coluna
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Inicializa o banco de dados e cria a tabela de tarefas se ela não existir."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL,
                total INTEGER NOT NULL,
                log TEXT,
                is_complete INTEGER NOT NULL,
                success INTEGER,
                output_path TEXT,
                output_filename TEXT,
                output_s3_key TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    #print("Banco de dados SQLite inicializado.")


def create_task(task_id):
    """Insere uma nova tarefa no banco de dados com valores iniciais."""
    with get_db_connection() as conn:
        conn.execute(
            '''INSERT INTO tasks (id, status, progress, total, log, is_complete)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (task_id, 'Iniciando...', 0, 100, json.dumps([]), 0)
        )
        conn.commit()


def get_task(task_id):
    """Busca uma tarefa específica pelo seu ID."""
    with get_db_connection() as conn:
        task = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
        return task


def append_to_log(task_id, message):
    """Adiciona uma mensagem de log a uma tarefa existente."""
    with db_lock:
        with get_db_connection() as conn:
            # 1. Pega o log atual
            task = conn.execute('SELECT log FROM tasks WHERE id = ?', (task_id,)).fetchone()
            if task:
                log_list = json.loads(task['log'])
                log_list.append(message)
                # 2. Atualiza com o novo log
                conn.execute('UPDATE tasks SET log = ? WHERE id = ?', (json.dumps(log_list), task_id))
                conn.commit()


def update_task_progress(task_id, progress, total, status_text):
    """Atualiza o progresso e o status de uma tarefa."""
    with get_db_connection() as conn:
        conn.execute(
            'UPDATE tasks SET progress = ?, total = ?, status = ? WHERE id = ?',
            (progress, total, status_text, task_id)
        )
        conn.commit()


def update_task_completion(task_id, success, result_msg, output_info):
    """Marca uma tarefa como concluída e salva o resultado final."""
    final_status = "Concluído com Sucesso!" if success else f"Falha: {result_msg}"

    with get_db_connection() as conn:
        # Dependendo do modo (LOCAL ou BUCKET), preenche os campos corretos
        if MODE == "LOCAL":
            conn.execute(
                '''UPDATE tasks
                   SET is_complete     = 1,
                       success         = ?,
                       status          = ?,
                       output_path     = ?,
                       output_filename = ?
                   WHERE id = ?''',
                (int(success), final_status, output_info['output_path'], output_info['output_filename'], task_id)
            )
        elif MODE == "BUCKET":
            conn.execute(
                '''UPDATE tasks
                   SET is_complete   = 1,
                       success       = ?,
                       status        = ?,
                       output_s3_key = ?
                   WHERE id = ?''',
                (int(success), final_status, output_info['output_s3_key'], task_id)
            )
        conn.commit()


# Chame esta função uma vez ao iniciar a aplicação para garantir que a tabela exista.
# Você pode rodar `python database.py` diretamente uma vez para criar o arquivo .db
if __name__ == '__main__':
    init_db()