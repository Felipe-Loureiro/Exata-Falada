document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('conversion-form');
    const submitBtn = document.getElementById('submit-btn');
    const cancelBtn = document.getElementById('cancel-btn');
    const progressSection = document.getElementById('progress-section');
    const resultSection = document.getElementById('result-section');
    const statusBar = document.getElementById('progress-bar');
    const statusText = document.getElementById('status-text');
    const logOutput = document.getElementById('log-output');
    const fileInput = document.getElementById('pdf_file');
    const devBtn = document.getElementById('dev-unlock-btn');

    let currentTaskId = null;
    let pollingInterval = null;

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // --- INÍCIO DA CORREÇÃO ---
        // 1. Verificar se um arquivo foi selecionado no lado do cliente
        if (fileInput.files.length === 0) {
            showResult(false, 'Por favor, selecione um arquivo PDF primeiro.');
            return;
        }

        // 2. Verificar o tamanho do arquivo (limite de 10MB, por exemplo)
        const file = fileInput.files[0];
        const maxSize = 50 * 1024 * 1024; // 50MB em bytes
        if (file.size > maxSize) {
            showResult(false, 'O arquivo selecionado é muito grande. O tamanho máximo permitido é 50MB.');
            return;
        }

        resetUI();
        setControls(false); // Desabilita controles

        // 3. Construir o FormData explicitamente
        // Esta é a parte mais importante da correção.
        const formData = new FormData();
        formData.append('pdf_file', fileInput.files[0]);
        formData.append('dpi', document.getElementById('dpi').value);
        formData.append('page_range', document.getElementById('page_range').value);
        formData.append('model', document.getElementById('model').value);
        formData.append('upload_workers', document.getElementById('upload_workers').value);
        formData.append('generate_workers', document.getElementById('generate_workers').value);
        // --- FIM DA CORREÇÃO ---

        try {
            // O cabeçalho 'Content-Type' NÃO deve ser definido manualmente.
            // O navegador o definirá como 'multipart/form-data' com o boundary correto.
            const response = await fetch('', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                showResult(false, `Erro na requisição: ${errorData.error || 'Erro desconhecido'}`);
                setControls(true);
                return;
            }

            const data = await response.json();
            currentTaskId = data.task_id;

            progressSection.style.display = 'block';
            pollingInterval = setInterval(pollStatus, 2000); // Consulta a cada 2 segundos

        } catch (error) {
            showResult(false, `Erro de conexão: ${error.message}`);
            setControls(true);
        }
    });

    cancelBtn.addEventListener('click', async function() {
        if (!currentTaskId) return;

        try {
            await fetch(`cancel/${currentTaskId}`, { method: 'POST' });
            cancelBtn.innerText = 'Cancelando...';
            cancelBtn.disabled = true;
        } catch (error) {
            console.error("Erro ao cancelar:", error);
        }
    });

    function pollStatus() {
        if (!currentTaskId) return;

        fetch(`status/${currentTaskId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showResult(false, `Erro na tarefa: ${data.error}`);
                    stopPolling();
                    return;
                }

                statusText.textContent = data.status;
                const progress = data.total > 0 ? (data.progress / data.total * 100).toFixed(1) : 0;
                statusBar.style.width = `${progress}%`;
                statusBar.textContent = `${progress}%`;

                logOutput.innerHTML = data.log.join('\n');
                logOutput.scrollTop = logOutput.scrollHeight;

                if (data.is_complete) {
                    stopPolling();
                    if (data.success) {
                        const downloadLink = `<a href="download/${currentTaskId}" target="_blank" class="download-link">Clique aqui para baixar o HTML</a>`;
                        showResult(true, `Conversão concluída! ${downloadLink}`);
                    } else {
                        showResult(false, `Falha na conversão. Verifique o log para detalhes.`);
                    }
                }
            })
            .catch(err => {
                console.error("Polling error:", err);
                showResult(false, 'Erro de comunicação com o servidor.');
                stopPolling();
            });
    }

    function stopPolling() {
        clearInterval(pollingInterval);
        pollingInterval = null;
        setControls(true);
    }

    function resetUI() {
        progressSection.style.display = 'none';
        resultSection.style.display = 'none';
        resultSection.className = 'alert';
        statusBar.style.width = '0%';
        statusBar.textContent = '0%';
        logOutput.innerHTML = '';
        currentTaskId = null;
    }

    function setControls(enabled) {
        submitBtn.disabled = !enabled;

        if (enabled) {
            form.querySelectorAll('input:not([data-role="dev"]), select:not([data-role="dev"])').forEach(el => el.disabled = !enabled);
        } else {
            form.querySelectorAll('input, select').forEach(el => el.disabled = !enabled);
        }

        if (enabled) {
            submitBtn.style.display = 'inline-block';
            cancelBtn.style.display = 'none';
            cancelBtn.innerText = 'Cancelar';
            cancelBtn.disabled = false;
        } else {
            submitBtn.style.display = 'none';
            cancelBtn.style.display = 'inline-block';
        }
    }

    function showResult(success, message) {
        resultSection.style.display = 'block';
        resultSection.className = success ? 'alert alert-success' : 'alert alert-danger';
        resultSection.innerHTML = message;
    }

    devBtn.addEventListener('click', async function () {
        const senha = prompt('Digite a senha de desenvolvedor:');

        // Se o usuário cancelar o prompt, a senha será null.
        if (senha === null) {
            return;
        }

        try {
            const response = await fetch('unlock-dev', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ password: senha })
            });

            // O servidor respondeu com sucesso (status 200-299)
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    document.querySelectorAll('[data-role="dev"]').forEach(el => {
                        el.disabled = false;
                    });
                }
            } else {
                // O servidor respondeu com um erro (ex: 401 Senha incorreta)
                const errorData = await response.json();
                alert(`Falha na autenticação: ${errorData.error || 'Tente novamente.'}`);
            }

        } catch (error) {
            console.error('Erro ao contatar o servidor para validação:', error);
            alert('Não foi possível conectar ao servidor para validar a senha.');
        }
    });
});