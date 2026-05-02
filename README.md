# Whisper Web UI - Transcrição e Análise de Áudio

## Aviso Importante

**Este projeto foi inteiramente gerado por inteligência artificial.** Pode conter bugs, vulnerabilidades de segurança ou comportamentos inesperados. Use por sua conta e risco em ambientes de produção. Recomenda-se revisão completa do código antes de qualquer implantação crítica.

---

## Descrição

Aplicação web para transcrever arquivos de áudio em texto usando o modelo Whisper e analisar o conteúdo com um servidor LM Studio. O projeto combina processamento de áudio local com análise de linguagem natural, permitindo extrair insights, resumos e verificação de fatos a partir de conteúdo falado.

## Instalação e Execução

### Requisitos

- Python 3.10 ou superior
- PyTorch (com suporte CUDA opcional, CPU também funciona)
- Mínimo 6GB de RAM para o modelo base.pt
- LM Studio instalado e rodando na máquina (para análise de IA)

### Setup

```bash
./start.sh
```

Acesse a aplicação em `http://localhost:5000` ou `seu ip local` (para acesso na rede local).

## Funcionalidades

### Transcrição de Áudio

- Suporte para MP3, WAV, OGG, M4A, FLAC (até 100MB)
- Interface web com arrastar-e-soltar
- Gravação em tempo real via microfone do navegador
- Seleção de modelos Whisper (tiny, base, small, medium, large-v3, turbo)
- Cache automático de modelos para carregamento mais rápido

### Análise com IA

- Resumo do conteúdo transcrito
- Identificação de temas principais e entidades
- Extração de pontos-chave e conclusões
- Verificação de fatos quando configurado
- Pesquisa na internet para validação (quando necessário)

### Configuração Persistente

- Painel de configuração para LM Studio
- Salva URL do servidor, modelo, token e tipo de prompt
- Configurações persistem após reinicialização do servidor

## Endpoints da API

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Interface web |
| `/upload` | POST | Transcrição simples (multipart/form-data) |
| `/api/transcribe-and-analyze` | POST | Transcrição completa com análise |
| `/health` | GET | Verificação de saúde da aplicação |
| `/api/config-lmstudio` | POST | Configurar LM Studio |
| `/api/load-lmstudio-config` | GET | Carregar configurações salvas |

## Notas de Operação

- O arquivo `.lmstudio_config.json` armazena as configurações do LM Studio (inclui token). Não inclua este arquivo em repositórios públicos.
- Aufers muito longos (acima de 10MB) podem ser truncados pelo Whisper.
- A análise com IA depende de um servidor LM Studio rodando localmente na máquina.
- CORS está habilitado para permitir requisições de qualquer origem, incluindo headers Authorization.

## Variáveis de Ambiente

Opcionalmente, configure via variáveis de ambiente:

```
LMSTUDIO_URL=http://localhost:1234
LMSTUDIO_MODEL=llama3.2:3b
LMSTUDIO_TOKEN=seu_token_aqui
LMSTUDIO_PROMPT_TYPE=auto_research
LMSTUDIO_TEMPERATURE=0.7
LMSTUDIO_MAX_TOKENS=2000
```

## Estrutura do Projeto

```
.
├── app.py                      # Aplicação Flask principal
├── templates/index.html        # Interface web
├── models/                     # Modelos Whisper em cache
├── uploads/                    # Arquivos enviados (temporário)
├── requirements.txt            # Dependências Python
└── start.sh                    # Script de inicialização
```
