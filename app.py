import os
import sys
import logging
import json
import signal
import re
import gc
import torch
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from whisper import load_model as whisper_load_model
from openai import OpenAI
from werkzeug.utils import secure_filename

# Configuração de logging verbose
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/labubu/speech-analyzer/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# CORS com permissão para Authorization header
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "allow_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST", "OPTIONS"]
    }
})

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac'}
MODEL_PATH = '/home/labubu/speech-analyzer/models/base.pt'

_model_cache = {}

# Carrega configurações persistentes se existirem (DEVE ESTAR ANTES DE USAR)
def load_lmstudio_config_from_file():
    """Carrega configurações do arquivo de persistência"""
    config_path = '/home/labubu/speech-analyzer/.lmstudio_config.json'
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            logger.info(f'Configurações carregadas do arquivo: {saved_config}')
            return saved_config
    except Exception as e:
        logger.error(f'Erro ao carregar configurações: {e}')
    return {}

# Carrega config salva ou cria nova
def save_lmstudio_config_to_file(config):
    """Salva configuração no arquivo de persistência"""
    config_path = '/home/labubu/speech-analyzer/.lmstudio_config.json'
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f'Configuração salva no arquivo')
        return True
    except Exception as e:
        logger.error(f'Erro ao salvar configurações: {e}')
        return False

# Configuração LM Studio (carregada via endpoint /api/config-lmstudio)
lmstudio_config = {
    'url': os.environ.get('LMSTUDIO_URL', 'http://localhost:1234'),
    'model': os.environ.get('LMSTUDIO_MODEL', 'llama3.2:3b'),
    'token': os.environ.get('LMSTUDIO_TOKEN', ''),  # Opcional
    'prompt_type': os.environ.get('LMSTUDIO_PROMPT_TYPE', 'auto_research'),
    'temperature': float(os.environ.get('LMSTUDIO_TEMPERATURE', '0.7')),
    'max_tokens': int(os.environ.get('LMSTUDIO_MAX_TOKENS', '2000'))
}

# Carrega configurações persistidas se existirem (ANTES de criar cliente)
saved_config = load_lmstudio_config_from_file()
if saved_config:
    lmstudio_config.update(saved_config)
    logger.info(f'Configurações carregadas do arquivo: {saved_config}')

# Inicializa cliente OpenAI para LM Studio com a config carregada
lm_client = OpenAI(
    base_url=f"{lmstudio_config['url']}/v1",
    api_key=f"{lmstudio_config['token']}" if lmstudio_config['token'] else "not-provided"
)
logger.info(f'LM Studio configurado: {lmstudio_config["url"]}, modelo: {lmstudio_config["model"]}, token: {"***" if lmstudio_config["token"] else "não fornecido"}')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/load-model', methods=['GET'])
def load_model():
    model_name = request.args.get('model', 'base')
    
    # SEMPRE limpa cache anterior para liberar VRAM antes de carregar novo
    # Isso garante que apenas UM modelo esteja na VRAM por vez
    if _model_cache:
        for cached_model in list(_model_cache.values()):
            del cached_model
        _model_cache.clear()
        logger.info('Cache de modelos limpo - VRAM liberada antes de carregar novo modelo')
        # Força liberação imediata da VRAM do PyTorch
        torch.cuda.empty_cache()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_path = '/home/labubu/speech-analyzer/models'
    logger.info(f'Carregando modelo Whisper {model_name} em {device}...')
    
    try:
        # Carrega o modelo especificado
        model = whisper_load_model(
            model_name, 
            device=device,
            download_root=cache_path
        )
        _model_cache[model_name] = model
        logger.info(f'Modelo {model_name} carregado com sucesso!')
    except Exception as e:
        logger.error(f'Erro ao carregar {model_name}: {str(e)}')
        return jsonify({'status': 'error', 'model': model_name, 'error': str(e)}), 500
    
    return jsonify({'status': 'ok', 'model': model_name})


@app.route('/upload', methods=['POST'])
def upload():
    logger.info('Recebendo requisição de upload...')
    
    if 'audio' not in request.files:
        logger.warning('Sem arquivo no body da requisição')
        return jsonify({'error': 'Nenhum arquivo de áudio fornecido'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        logger.warning('Nome do arquivo vazio')
        return jsonify({'error': 'Nome do arquivo vazio'}), 400
    
    # SEPARAÇÃO: Transcrição usa Whisper local, análise usa LM Studio remoto
    # O model_name aqui é apenas para Whisper (tiny, base, small, medium, large-v3, turbo)
    whisper_model = request.form.get('model', 'base')
    logger.info(f'Modelo Whisper selecionado: {whisper_model}')
    
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            logger.info(f'Arquivo salvo: {filepath}')
            
            # Carrega modelo Whisper local para transcrição
            model = _model_cache.get(whisper_model)
            if model is None:
                # SEMPRE limpa cache anterior para liberar VRAM antes de carregar novo
                if _model_cache:
                    for cached_model in list(_model_cache.values()):
                        del cached_model
                    _model_cache.clear()
                    logger.info('Cache limpo em /upload antes de carregar novo modelo')
                    # Força liberação imediata da VRAM do PyTorch
                    torch.cuda.empty_cache()
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                cache_path = '/home/labubu/speech-analyzer/models'
                logger.info(f'Carregando modelo Whisper {whisper_model} em {device}...')
                
                model = whisper_load_model(
                    whisper_model, 
                    device=device,
                    download_root=cache_path
                )
                _model_cache[whisper_model] = model
                logger.info(f'Modelo Whisper {whisper_model} carregado!')
            
            result = model.transcribe(filepath)
            
            os.remove(filepath)
            logger.info('Arquivo removido após transcrição')
            
            # LIBERA VRAM IMEDIATAMENTE APÓS TRANSCRIÇÃO
            # Isso garante espaço em vídeo para LM Studio carregar seus modelos pesados
            _model_cache.clear()
            del model
            torch.cuda.empty_cache()
            gc.collect()
            logger.info('Modelo Whisper liberado da VRAM - espaço disponível para LM Studio')
            
            return jsonify({
                'text': result['text'],
                'language': result.get('language', 'unknown'),
                'duration': result.get('duration', 0),
                'model': whisper_model
            })
        except Exception as e:
            logger.error(f'Erro na transcrição: {str(e)}')
            return jsonify({'error': str(e)}), 500
    
    logger.warning('Tipo de arquivo não suportado')
    return jsonify({'error': 'Tipo de arquivo não suportado'}), 400


@app.route('/health')
def health():
    model = _model_cache.get('base')
    
    if model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cache_path = '/home/labubu/speech-analyzer/models'
        logger.info(f'Carregando modelo base para health check em {device}...')
        
        # Limpa cache anterior antes de carregar
        if _model_cache:
            for cached_model in list(_model_cache.values()):
                del cached_model
            _model_cache.clear()
            logger.info('Cache limpo antes de health check')
            # Força liberação imediata da VRAM do PyTorch
            torch.cuda.empty_cache()
        
        model = whisper_load_model(
            'base', 
            device=device,
            download_root=cache_path
        )
        _model_cache['base'] = model
        logger.info('Modelo base carregado para health!')
    
    return jsonify({
        'status': 'ok',
        'model': type(model).__name__,
        'device': str(model.device)
    })


@app.route('/api/ask-with-prompt', methods=['POST'])
def ask_with_prompt():
    """
    Endpoint para enviar pergunta customizada com system prompt
    Recebe: {system_prompt, question, temperature, max_tokens}
    Retorna: {status, response}
    """
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        question = data.get('question', '')
        temperature = float(data.get('temperature', lmstudio_config['temperature']))
        max_tokens = int(data.get('max_tokens', lmstudio_config['max_tokens']))
        
        if not system_prompt or not question:
            return jsonify({
                'status': 'error',
                'message': 'System prompt e pergunta são obrigatórios'
            }), 400
        
        logger.info(f'Pergunta customizada recebida')
        logger.info(f'  System prompt: {system_prompt[:100]}...')
        logger.info(f'  Question: {question[:100]}...')
        
        # Envia para LM Studio
        response = lm_client.chat.completions.create(
            model=lmstudio_config['model'],
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': question}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        ai_response = response.choices[0].message.content if response.choices else ""
        
        if not ai_response:
            logger.error('Resposta vazia do LM Studio')
            return jsonify({
                'status': 'error',
                'message': 'LM Studio retornou resposta vazia'
            }), 500
        
        logger.info(f'Resposta recebida: {len(ai_response)} caracteres')
        
        return jsonify({
            'status': 'ok',
            'response': ai_response,
            'model': lmstudio_config['model']
        })
    
    except Exception as e:
        logger.error(f'Erro ao processar pergunta customizada: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """
    Endpoint para analisar APENAS texto já transcrito (SEM recarregar Whisper)
    Recebe: {text}
    Retorna: {status, analysis}
    
    IMPORTANTE: Este endpoint NÃO recarrega o Whisper!
    Use após já ter feito a transcrição para análise rápida sem duplicar processamento de VRAM.
    """
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'status': 'error',
                'message': 'Texto para análise é obrigatório'
            }), 400
        
        logger.info(f'Analisando texto (SEM recarregar Whisper): {len(text)} caracteres')
        
        system_prompt = """Você é um especialista em análise de texto. Forneça uma análise estruturada em JSON:
{
  "summary": "resumo conciso em 2-3 frases",
  "key_points": ["ponto1", "ponto2", "ponto3"],
  "entities": ["pessoa/organização mencionada"],
  "topics": ["tema1", "tema2"],
  "sentiment": "positivo|negativo|neutro",
  "confidence": 0.0-1.0
}

Responda APENAS com o JSON, sem markdown."""
        
        response = lm_client.chat.completions.create(
            model=lmstudio_config['model'],
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'Analise este texto:\n\n{text[:3000]}'}
            ],
            temperature=lmstudio_config['temperature'],
            max_tokens=2000
        )
        
        ai_response = response.choices[0].message.content if response.choices else ""
        
        if not ai_response:
            return jsonify({
                'status': 'error',
                'message': 'LM Studio retornou resposta vazia'
            }), 500
        
        logger.info(f'Análise concluída: {len(ai_response)} caracteres')
        
        return jsonify({
            'status': 'ok',
            'analysis': ai_response,
            'model': lmstudio_config['model'],
            'text_length': len(text)
        })
    
    except Exception as e:
        logger.error(f'Erro ao analisar texto: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/test-lmstudio', methods=['POST'])
def test_lmstudio():
    """
    Endpoint para testar conexão com LM Studio
    Recebe: {url, token, model}
    Retorna: {status, message}
    """
    try:
        data = request.json
        url = data.get('url', lmstudio_config['url'])
        token = data.get('token', lmstudio_config['token'])
        model = data.get('model', lmstudio_config['model'])
        
        logger.info(f'Testando LM Studio: url={url}, model={model}, token={token[:20] if token else "none"}...')
        
        # Cria cliente temporário para teste
        test_client = OpenAI(
            base_url=f"{url}/v1",
            api_key=token if token else "not-provided"
        )
        
        # Tenta fazer uma requisição simples
        response = test_client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'user', 'content': 'Test'}
            ],
            max_tokens=10
        )
        
        logger.info(f'Teste bem-sucedido! Response: {response}')
        
        return jsonify({
            'status': 'ok',
            'message': f'Conexão bem-sucedida com {model}',
            'response_sample': str(response)[:200]
        })
    
    except Exception as e:
        logger.error(f'Erro ao testar LM Studio: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/config-lmstudio', methods=['POST'])
def config_lmstudio():
    """
    Endpoint para configurar LM Studio via WebUI
    Recebe: {url, model, token, prompt_type, temperature, max_tokens}
    Salva configurações no objeto global E NO ARQUIVO DE PERSISTÊNCIA
    """
    global lm_client
    try:
        data = request.json
        
        # Valida se modelo foi fornecido
        novo_modelo = data.get('model')
        if not novo_modelo:
            return jsonify({
                'status': 'error',
                'message': 'Modelo é obrigatório'
            }), 400
        
        logger.info(f'Recebido POST /api/config-lmstudio:')
        logger.info(f'  Modelo novo: {novo_modelo}')
        logger.info(f'  Modelo antigo: {lmstudio_config["model"]}')
        
        # Atualiza configuração com novos valores
        lmstudio_config['url'] = data.get('url', lmstudio_config['url'])
        lmstudio_config['model'] = novo_modelo  # Força atualizar modelo SEMPRE
        lmstudio_config['token'] = data.get('token', '')
        lmstudio_config['prompt_type'] = data.get('prompt_type', lmstudio_config['prompt_type'])
        lmstudio_config['temperature'] = float(data.get('temperature', lmstudio_config['temperature']))
        lmstudio_config['max_tokens'] = int(data.get('max_tokens', lmstudio_config['max_tokens']))
        
        logger.info(f'Config em memória atualizada: {lmstudio_config}')
        
        # Salva no arquivo de persistência
        save_lmstudio_config_to_file(lmstudio_config)
        logger.info(f'Config salva no arquivo: {lmstudio_config}')
        
        # Recria cliente OpenAI com nova configuração (ATUALIZA GLOBAL)
        lm_client = OpenAI(
            base_url=f"{lmstudio_config['url']}/v1",
            api_key=lmstudio_config['token'] if lmstudio_config['token'] else "not-provided"
        )
        logger.info(f'Cliente OpenAI recriado com novo modelo: {lmstudio_config["model"]}')
        
        return jsonify({
            'status': 'ok',
            'message': f'Configuração salva para modelo: {lmstudio_config["model"]}',
            'config': {
                'url': lmstudio_config['url'],
                'model': lmstudio_config['model'],
                'prompt_type': lmstudio_config['prompt_type'],
                'temperature': lmstudio_config['temperature'],
                'max_tokens': lmstudio_config['max_tokens']
            }
        })
    
    except Exception as e:
        logger.error(f'Erro ao configurar LM Studio: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/check-current-model', methods=['GET'])
def check_current_model():
    """
    Endpoint para verificar qual modelo está REALMENTE sendo usado
    """
    return jsonify({
        'status': 'ok',
        'current_model': lmstudio_config['model'],
        'lmstudio_url': lmstudio_config['url'],
        'full_config': lmstudio_config
    })


@app.route('/api/load-lmstudio-config', methods=['GET'])
def load_lmstudio_config():
    """
    Endpoint para carregar configurações salvas do arquivo
    Útil para reiniciar o servidor e manter configurações
    """
    try:
        saved_config = load_lmstudio_config_from_file()
        if saved_config:
            return jsonify({
                'status': 'ok',
                'config': saved_config
            })
        else:
            return jsonify({
                'status': 'no_saved_config',
                'message': 'Nenhuma configuração salva encontrada'
            })
    except Exception as e:
        logger.error(f'Erro ao carregar configurações: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/transcribe-and-analyze', methods=['POST'])
def transcribe_and_analyze():
    """
    Endpoint completo: Transcreve áudio + envia para LM Studio para análise
    Recebe: {audio, model}
    Retorna: {status, transcription, analysis, summary, entities, topics, search_results}
    
    FLUXO CORRETO:
    1. Transcrever áudio com Whisper (usa modelo do cache)
    2. Enviar texto para LM Studio com prompt de análise + pesquisa + verificação
       (usa o MODEL CONFIGURADO EM /api/config-lmstudio, NÃO o model_name do formulário!)
    3. Retornar todos os resultados estruturados
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Nenhum arquivo de áudio fornecido'}), 400
        
        file = request.files['audio']
        model_name = request.form.get('model', 'base')  # Só usado para log, NÃO para envio ao LM Studio
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Tipo de arquivo não suportado'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f'Arquivo salvo para transcrição: {filepath}')
        
        # Carrega modelo Whisper LOCAL (tiny, base, small, medium, large-v3, turbo)
        whisper_model = model_name
        model = _model_cache.get(whisper_model)
        if model is None:
            if _model_cache:
                for cached_model in list(_model_cache.values()):
                    del cached_model
                _model_cache.clear()
                torch.cuda.empty_cache()
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cache_path = '/home/labubu/speech-analyzer/models'
            logger.info(f'Carregando modelo Whisper {whisper_model} em {device}...')
            
            model = whisper_load_model(whisper_model, device=device, download_root=cache_path)
            _model_cache[whisper_model] = model
        
        # Transcreve
        result = model.transcribe(filepath)
        transcription_text = result['text']
        logger.info(f'Transcrição completa: {len(transcription_text)} caracteres')
        
        os.remove(filepath)
        logger.info('Arquivo removido após transcrição')
        
        # LIBERA WHISPER DA VRAM IMEDIATAMENTE APÓS TRANSCRIÇÃO
        _model_cache.clear()
        del model
        torch.cuda.empty_cache()
        gc.collect()
        logger.info('Modelo Whisper liberado da VRAM - espaço disponível para LM Studio')
        
        # Prompt para LM Studio com análise + pesquisa + verificação de fatos
        system_prompt = """Você é um especialista em análise, pesquisa e verificação de informações. Sua tarefa é processar uma transcrição de áudio e fornecer uma análise completa.

SUA TAREFA:
1. **RESUMO EXECUTIVO**: Crie um resumo conciso do conteúdo (2-3 parágrafos)
2. **ANÁLISE DETALHADA**: Identifique pontos principais, argumentos, insights relevantes
3. **ENTIDADES MENCIONADAS**: Liste pessoas, organizações, lugares, produtos citados
4. **TEMAS PRINCIPAIS**: Extraia os temas centrais discutidos
5. **CONCLUSÕES/RECOMENDAÇÕES**: Sintetize as conclusões principais
6. **VERIFICAÇÃO DE FATOS**: Identifique afirmações factuais e verifique sua plausibilidade
7. **PESQUISA NA INTERNET (SE NECESSÁRIO)**: Se o conteúdo mencionar eventos recentes, dados estatísticos, ou fatos que podem ser verificados online, pesquise para confirmar informações.

FORMATO DE RESPOSTA (JSON APENAS):
{
  "summary": "Resumo executivo conciso",
  "detailed_analysis": "Análise detalhada com pontos principais e insights",
  "entities": ["entidade1", "entidade2", ...],
  "topics": ["tema1", "tema2", ...],
  "conclusions": "Conclusões e recomendações principais",
  "fact_checks": [
    {"claim": "afirmação verificável", "verified": true/false, "source": "fonte ou 'não verificado'"}
  ],
  "search_queries_needed": ["query1", "query2", ...],
  "key_insights": "Insights mais importantes extraídos"
}

IMPORTANTE: Responda APENAS com o JSON, sem markdown, sem texto adicional.
Se precisar pesquisar, identifique as queries necessárias e responda que precisa pesquisar.
"""
        
        user_prompt = f"""Transcrição de áudio para análise:\n\n{transcription_text[:5000]}\n\nPor favor, processe conforme as instruções acima e retorne o JSON estruturado."""
        
        logger.info(f'Enviando transcrição para LM Studio (modelo configurado: {lmstudio_config["model"]})...')
        
        # Envia para LM Studio - usa o modelo CONFIGURADO em /api/config-lmstudio, NÃO model_name!
        lm_model = lmstudio_config['model']  # Modelo do LM Studio (ex: llama3.2:3b, mistral:7b, etc.)
        
        logger.info(f'Enviando para LM Studio:')
        logger.info(f'  URL: {lmstudio_config["url"]}')
        logger.info(f'  Modelo: {lm_model}')
        logger.info(f'  Token: {"***" if lmstudio_config["token"] else "não fornecido"}')
        
        response = lm_client.chat.completions.create(
            model=lm_model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            temperature=lmstudio_config['temperature'],
            max_tokens=lmstudio_config['max_tokens']
        )
        
        # DEBUG: Loga resposta completa + detecta MCP tool calls
        logger.info(f'Response object type: {type(response)}')
        mcp_tool_calls = []
        mcp_detected_in_content = False
        
        if response.choices:
            choice = response.choices[0]
            
            # Check se há tool_calls estruturados (padrão OpenAI)
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    mcp_tool_calls.append(tool_call.function.name)
                logger.warning(f'MCP Tool Calls Estruturados Detectadas: {mcp_tool_calls}')
            
            # Check se há menção de MCP no conteúdo (formato LM Studio)
            content = choice.message.content or ""
            if 'mcp-server' in content.lower() or '"name":' in content and 'arguments' in content:
                mcp_detected_in_content = True
                logger.warning(f'MCP detectado no conteúdo (resposta contém tentativa de MCP)')
                # Tenta extrair nome da ferramenta
                tool_matches = re.findall(r'"name":\s*"([^"]+)"', content)
                if tool_matches:
                    mcp_tool_calls.extend(tool_matches)
            
            logger.info(f'First choice content length: {len(content) if content else 0}')
        
        ai_response = response.choices[0].message.content if response.choices and response.choices[0].message else ""
        logger.info(f'Resposta da IA recebida: {len(ai_response)} caracteres | MCP tool calls: {len(mcp_tool_calls)} | MCP em conteúdo: {mcp_detected_in_content}')
        
        if not ai_response:
            logger.error(f'Resposta vazia do LM Studio! Response: {response}')
            return jsonify({
                'status': 'error',
                'message': 'LM Studio retornou resposta vazia. Verifique se o token é válido e se o modelo está selecionado.'
            }), 500
        
        # Tenta parsear como JSON, ou retorna texto se não for JSON válido
        try:
            analysis_result = json.loads(ai_response.strip())
        except json.JSONDecodeError:
            logger.warning('Resposta não é JSON puro, tentando extrair...')
            
            # Remove whitespace e procura pelo primeiro {
            stripped = ai_response.strip()
            start_idx = stripped.find('{')
            if start_idx == -1:
                logger.warning('Nenhum JSON encontrado na resposta')
                return jsonify({
                    'status': 'ok',
                    'result': ai_response,
                    'transcription': transcription_text[:1000],
                    'whisper_model_used': whisper_model,
                    'lm_model_used': lm_model
                })
            
            # Tenta parsear começando do { até o final, removendo caracteres do fim se necessário
            json_str = stripped[start_idx:]
            
            # Tenta parsear do final para o início (remove caracteres inválidos)
            for i in range(len(json_str), 0, -1):
                try:
                    analysis_result = json.loads(json_str[:i])
                    logger.info(f'JSON extraído com sucesso (caracteres 0:{i})')
                    break
                except json.JSONDecodeError:
                    continue
            else:
                # Se não conseguir parsear nenhuma substring, retorna como texto
                logger.warning('Não foi possível extrair JSON válido da resposta')
                return jsonify({
                    'status': 'ok',
                    'result': ai_response,
                    'transcription': transcription_text[:1000],
                    'whisper_model_used': whisper_model,
                    'lm_model_used': lm_model
                })
        
        logger.info('Análise completa concluída')
        
        return jsonify({
            'status': 'ok',
            'analysis': analysis_result,
            'transcription': transcription_text[:2000],
            'whisper_model_used': whisper_model,
            'lm_model_used': lm_model
        })
    
    except Exception as e:
        logger.error(f'Erro ao transcrever e analisar: {str(e)}')
        error_msg = str(e)
        if 'connection' in error_msg.lower() or 'timeout' in error_msg.lower():
            error_msg = 'LM Studio não está respondendo. Verifique se o servidor está rodando.'
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500


@app.route('/api/process-with-ai', methods=['POST'])
def process_with_ai():
    """
    Endpoint para processar texto transcrito com IA do LM Studio
    Recebe: {text, model}
    Retorna: {status, result, model_used}
    
    Prompt padrão (auto_research):
    - Faz resumo do conteúdo
    - Analisa pontos principais  
    - Pesquisa na internet se necessário (via MCP SearXNG)
    """
    try:
        data = request.json
        text = data.get('text', '')
        model = data.get('model', lmstudio_config['model'])
        prompt_type = lmstudio_config['prompt_type']
        
        if not text:
            return jsonify({
                'status': 'error',
                'message': 'Texto transcrito é obrigatório'
            }), 400
        
        # Prompt base adaptável conforme prompt_type
        system_prompt = """Você é um assistente de IA especializado em análise e pesquisa. Sua tarefa é:

1. **RESUMO EXECUTIVO**: Crie um resumo conciso do texto transcrito.
2. **ANÁLISE DETALHADA**: Identifique os pontos principais, argumentos e insights relevantes.
3. **PESSOAS/ENTIDADES**: Liste pessoas, organizações ou entidades mencionadas.
4. **TEMAS PRINCIPAIS**: Extraia os temas centrais discutidos.
5. **CONCLUSÕES**: Sintetize as conclusões principais.

Comportamento de Pesquisa:
- Se o usuário especificar "research_first", pesquise na internet antes de responder.
- Se for "auto_research" (padrão), decida internamente se precisa pesquisar com base nos temas.
- Se for "summary_only", faça apenas resumo e análise sem pesquisa.

Para pesquisas, use as ferramentas MCP disponíveis. Formate sua resposta final em JSON:
{
  "type": "summary" | "research_result" | "search_needed",
  "summary": "resumo executivo",
  "analysis": "análise detalhada com pontos principais",
  "entities": ["lista", "de", "entidades"],
  "topics": ["tema1", "tema2"],
  "conclusions": "conclusões",
  "search_query": "query de pesquisa se necessário (se type=search_needed)",
  "search_results": "resultados da pesquisa (se aplicável)"
}

Responda apenas com o JSON, sem markdown ou texto adicional."""
        
        user_prompt = f"""Texto transcrito para análise:\n\n{text[:4000]}\n\nPor favor, processe conforme as instruções acima. Se precisar pesquisar, use a ferramenta de pesquisa apropriada."""
        
        logger.info(f'Processando texto com IA (modelo: {model}, tipo: {prompt_type})...')
        logger.info(f'  URL: {lmstudio_config["url"]}')
        logger.info(f'  Token: {"***" if lmstudio_config["token"] else "não fornecido"}')
        
        # Chamada ao LM Studio via OpenAI API
        response = lm_client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            temperature=lmstudio_config['temperature'],
            max_tokens=lmstudio_config['max_tokens']
        )
        
        logger.info(f'Resposta completa: {response}')
        
        ai_response = response.choices[0].message.content if response.choices and response.choices[0].message else ""
        logger.info(f'Resposta da IA recebida: {len(ai_response)} caracteres')
        
        if not ai_response:
            logger.error(f'Resposta vazia do LM Studio!')
            return jsonify({
                'status': 'error',
                'message': 'LM Studio retornou resposta vazia.'
            }), 500
        
        # Retorna resultado processado
        return jsonify({
            'status': 'ok',
            'result': ai_response,
            'model_used': model,
            'prompt_type': prompt_type
        })
    
    except Exception as e:
        logger.error(f'Erro ao processar com IA: {str(e)}')
        error_msg = str(e)
        
        # Tenta capturar erro de conexão com LM Studio
        if 'connection' in error_msg.lower() or 'timeout' in error_msg.lower():
            error_msg = 'LM Studio não está respondendo. Verifique se o servidor está rodando na porta 1234 (ou URL configurada).'
        
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500


@app.route('/api/test-mcp-search', methods=['POST'])
def test_mcp_search():
    try:
        data = request.json or {}
        query = data.get('query', 'Ultimas noticias tecnologia')
        max_tokens = int(data.get('max_tokens', 6000))
        
        logger.info(f'Teste MCP Search: {query}')
        
        system_prompt = """Você é um assistente que DEVE usar ferramentas MCP para responder.

FERRAMENTAS DISPONÍVEIS:
- searxng-web-search: Pesquisa web usando SearXNG
- web-fetch: Busca conteúdo de URLs
- mcp-server-commands: Executa comandos de shell
- ssh-server-access: Acesso SSH remoto
- sequential-thinking: Processamento sequencial

REGRA IMPORTANTE: Para QUALQUER pergunta sobre data, hora, eventos atuais, notícias ou informações que mudem:
1. SEMPRE use searxng-web-search ou web-fetch
2. Formate a chamada como JSON estruturado
3. Nunca responda sem pesquisar

FORMATO DE CHAMADA (use EXATAMENTE assim):
{
  "tool": "searxng-web-search",
  "query": "sua query aqui"
}

Pesquise AGORA e depois responda com as fontes."""

        response = lm_client.chat.completions.create(
            model=lmstudio_config['model'],
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'{query}'}
            ],
            temperature=lmstudio_config['temperature'],
            max_tokens=max_tokens
        )
        
        ai_response = response.choices[0].message.content if response.choices else ""
        logger.info(f'Resposta: {len(ai_response)} caracteres')
        
        tool_calls_detected = False
        tool_calls_list = []
        search_queries = []
        mcp_attempted = False
        
        if '"tool"' in ai_response and 'searxng-web-search' in ai_response:
            mcp_attempted = True
            tool_calls_detected = True
            logger.warning('MCP searxng-web-search detectado no conteúdo')
            
            tool_pattern = re.findall(r'"tool":\s*"([^"]+)"', ai_response)
            query_pattern = re.findall(r'"query":\s*"([^"]+)"', ai_response)
            
            for tool_name in tool_pattern:
                tool_calls_list.append({'function': tool_name, 'type': 'searxng_json'})
            
            for q in query_pattern:
                if q not in search_queries:
                    search_queries.append(q)
        
        elif 'mcp-server' in ai_response.lower() or ('"name"' in ai_response and 'arguments' in ai_response):
            tool_calls_detected = True
            tool_matches = re.findall(r'"name":\s*"([^"]+)"', ai_response)
            args_matches = re.findall(r'"arguments":\s*({[^}]+})', ai_response)
            
            for tool_name in tool_matches:
                tool_calls_list.append({'function': tool_name, 'type': 'mcp_standard'})
                logger.warning(f'MCP detectado: {tool_name}')
            
            for args_str in args_matches:
                try:
                    args_json = json.loads(args_str)
                    if 'query' in args_json:
                        search_queries.append(args_json['query'])
                except:
                    pass
        
        if '"tool":' in ai_response and tool_calls_detected == False:
            mcp_attempted = True
            logger.warning('MCP tentado (formato JSON, nao reconhecido)')
            tool_pattern = re.findall(r'"tool":\s*"([^"]+)"', ai_response)
            query_pattern = re.findall(r'"query":\s*"([^"]+)"', ai_response)
            
            for tool_name in tool_pattern:
                if tool_name not in [t.get('function') for t in tool_calls_list]:
                    tool_calls_list.append({'function': tool_name, 'type': 'attempted_json'})
            
            for q in query_pattern:
                if q not in search_queries:
                    search_queries.append(q)
            
            if tool_pattern or query_pattern:
                tool_calls_detected = True
        
        if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            tool_calls_detected = True
            for tc in response.choices[0].message.tool_calls:
                tool_calls_list.append({'function': tc.function.name, 'type': 'structured'})
                logger.warning(f'MCP estruturado: {tc.function.name}')
        
        return jsonify({
            'status': 'ok',
            'query': query,
            'mcp_detected': tool_calls_detected,
            'mcp_attempted': mcp_attempted,
            'tool_calls': tool_calls_list,
            'search_queries': search_queries,
            'response': ai_response[:2000],
            'token_usage': {
                'total': response.usage.total_tokens,
                'completion': response.usage.completion_tokens,
                'prompt': response.usage.prompt_tokens
            }
        })
    
    except Exception as e:
        logger.error(f'Erro MCP: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500


def signal_handler(signum, frame):
    """Limpador cache e encerra graciosamente"""
    logger.info(f'Recebido sinal {signum} - Encerrando...')
    
    # Limpa cache de modelos
    if _model_cache:
        for cached_model in list(_model_cache.values()):
            del cached_model
        _model_cache.clear()
        logger.info('Cache de modelos limpo')
        torch.cuda.empty_cache()
    
    # Fecha cliente OpenAI se necessário
    try:
        import gc
        gc.collect()
    except:
        pass
    
    exit(0)

# Registra handlers para sinais comuns de encerramento
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # pkill

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logger.info('Inicializando Whisper Web UI com tratamento de sinais...')
    app.run(
        host=os.environ.get('FLASK_HOST', '0.0.0.0'),
        port=int(os.environ.get('FLASK_PORT', 5000)),
        debug=True,
        use_reloader=False,
        threaded=True
    )
    logger.info('Servidor encerrado limpo!')
