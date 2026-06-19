# FinGuard AI 🛡️⚙️

**Sistema Híbrido de Detecção de Fraude e Inadimplência com Machine Learning Clássico e GenAI.**

O **FinGuard AI** é um pipeline avançado de engenharia de dados focado na resolução de um dos maiores desafios da adoção de Inteligência Artificial Generativa no setor financeiro: o **trade-off entre custo computacional (tokens/latência) e capacidade analítica.**

Em vez de processar todas as transações em Large Language Models (LLMs) pesados e custosos, este projeto propõe uma **arquitetura de roteamento dinâmico**. Ele utiliza um modelo estatístico rápido como primeira linha de defesa e escala apenas os casos complexos para o ecossistema Google Gemini.

---

## 📐 A Arquitetura do Funil Híbrido

O sistema varre bases de dados reais (utilizando o *German Credit Risk Data* como prova de conceito) e roteia os clientes em camadas:

```text
[Base de Dados Real: Credit.csv] 
       │
       ▼
[Camada 0: XGBoost Classifier] ─── Avaliação Tabular de Risco
       │
       ├── Risco < 30%  ─────────► 🟢 Aprovação Automática (Custo Zero GenAI)
       │
       ├── Risco 30% a 75% ──────► 🟡 [Camada 1 e 2: Gemini Flash-Lite & Flash]
       │                             Triagem de logs e Parecer Contextual Rápido
       │
       └── Risco > 75%  ─────────► 🔴 [Camada 3: Gemini Pro / Flash Advanced]
                                     Auditoria Forense Profunda (Chain-of-Thought)
```
## 🛠️ Desafios de Engenharia e Soluções Aplicadas

Durante o desenvolvimento deste pipeline, enfrentamos e resolvemos gargalos reais de infraestrutura de IA:

### 1. Limites de Taxa e Quedas de Servidor (Erros 429 e 503)
* **Problema:** O processamento em lote (looping) disparou o limite de RPM (Requisições Por Minuto) da camada gratuita da API do Google, gerando os erros `429 RESOURCE_EXHAUSTED` e `503 UNAVAILABLE`.
* **Solução:** Implementação de uma arquitetura resiliente de rede com blocos `try/except`, aplicando *Rate Limiting* (pausas de 5 segundos) e *Exponential Backoff* (espera de 15 segundos em caso de falha antes de realizar o *retry*).

### 2. Modelos Obsoletos ou Sem Cota (Erro 404 e Quota 0)
* **Problema:** O uso inicial da família `Gemini 1.5` ou versões `Pro` no Free Tier resultou em falta de acesso temporário e erros de rota.
* **Solução:** Centralização da parametrização dos modelos na classe `__init__`, permitindo a migração rápida e global do sistema para o estável modelo `gemini-2.5-flash`, garantindo alta disponibilidade e contornando tetos operacionais restritivos.

### 3. Conflito de Versões entre SHAP e XGBoost (ValueError)
* **Problema:** As versões recentes do XGBoost (>=2.1.0) alteraram a formatação interna do `base_score`, causando falha fatal (crash) no `TreeExplainer` do SHAP.
* **Solução:** Realizado o *downgrade* do XGBoost para a versão de estabilidade garantida `2.0.3` no `requirements.txt`, assegurando total compatibilidade da extração matemática.

### 4. Falhas de Extração de JSON (Expecting value: line 1)
* **Problema:** Mesmo utilizando `response_schema`, o LLM ocasionalmente inseria lixo em texto ou marcações markdown (```` ```json ````) antes do código, quebrando a função `json.loads()` do Python.
* **Solução:** Construído um **Extrator Robusto de JSON** na Camada 3 (`clean_json = json_audit[inicio:fim+1]`), somado a um mecanismo de segurança (Fallback) que redireciona automaticamente o cliente para "Revisão Manual" caso a IA sofra alucinações na formatação, impedindo que o servidor da aplicação caia.

---
## 🛠️ Stack Tecnológico 
Linguagem: Python 3.10+

Machine Learning (Estruturado): XGBoost, Scikit-Learn, Pandas

Inteligência Artificial Generativa: Google GenAI SDK (Família Gemini 2.5)

Gerenciamento de Ambiente: python-dotenv

## 📂 Estrutura do Projeto
```text
finguard-ai/
├── src/
│   ├── __init__.py
│   ├── classical_ml.py     # Pipeline de ML tradicional com XGBoost
│   └── gemini_pipeline.py  # Orquestração e prompts estruturados do Gemini
├── dados/
│   └── Credit.csv          # Base de dados real de comportamento financeiro
├── main.py                 # Ponto de entrada e lógica de roteamento dinâmico
├── requirements.txt        # Dependências
└── .env.example            # Template de variáveis de ambiente
```
## 🚀 Como Executar Localmente
1. Clone o repositório
git clone 
cd FINDGUARD_IA

2. Instale as dependências
pip install -r requirements.txt

3. Configure suas credenciais
Crie um arquivo .env na raiz do projeto e insira sua chave da API do Google AI Studio:
GEMINI_API_KEY=sua_chave_aqui

4. Execute o orquestrador
python main.py

O sistema irá treinar o modelo na base real, pescar automaticamente clientes de diferentes faixas de risco (Baixo, Moderado e Crítico) e executar o funil híbrido, mostrando as decisões de IA no terminal.

## 💡 Mapa completo das limitações vivenciadas

### 1. Limite de Requisições por Minuto (RPM) - Erro 429
O que vivenciamos: Quando colocamos o projeto para testar os 3 cenários em um loop (for cliente in lista_de_testes:), o Python enviou as requisições para a Google em questão de milissegundos. A API bloqueou o acesso imediatamente com o aviso RESOURCE_EXHAUSTED.

A Regra do Free Tier: A conta gratuita tem um limite rígido de chamadas simultâneas ou por minuto (geralmente 15 RPM).

Como resolvemos: Implementamos um "respiro" no código adicionando time.sleep(5) para cadenciar as chamadas.

### 2. Teto de Requisições Diárias (RPD) - Erro 429
O que vivenciamos: Ao testarmos o modelo recém-lançado gemini-2.5-flash-lite repetidas vezes para ajustar os prompts, recebemos novamente o erro RESOURCE_EXHAUSTED, mas dessa vez o log dizia: limit: 20.

A Regra do Free Tier: Modelos muito novos ou experimentais costumam ter travas agressivas na conta gratuita para evitar abusos. Nesse caso, havia um teto de apenas 20 requisições por dia.

Como resolvemos: Parametrizamos as variáveis do pipeline para fazer o downgrade temporário para a família gemini-1.5, que possui um limite diário massivo (cerca de 1.500 requisições diárias gratuitas).

### 3. Restrição de Acesso a Modelos Pesados (Quota Zero)
O que vivenciamos: Logo no início, quando tentamos acionar a Camada 3 de auditoria profunda com o modelo gemini-2.5-pro, recebemos um erro dizendo que a cota foi excedida, mas o limite apontado era limit: 0.

A Regra do Free Tier: O Google reserva o poder computacional massivo dos modelos "Pro" para clientes pagantes. Dependendo da região ou demanda, a cota gratuita para esses modelos é zerada intencionalmente.

Como resolvemos: Refatoramos o código usando variáveis de classe (self.AUDIT_MODEL) para deixar a arquitetura pronta para o modelo Pro no futuro, mas o substituímos pelo modelo Flash durante a homologação para evitar a quebra do pipeline.

### 4. Indisponibilidade por Tráfego Global - Erro 503
O que vivenciamos: No nosso teste final com a base real (CSV), o código estava perfeito e não tínhamos estourado nenhuma cota, mas recebemos o erro 503 UNAVAILABLE: This model is currently experiencing high demand.

A Regra do Free Tier: Usuários gratuitos não têm SLA (Acordo de Nível de Serviço) nem prioridade de roteamento. Se houver um pico global de uso nos servidores do Google, as requisições gratuitas são as primeiras a serem enfileiradas ou derrubadas.

Como resolvemos: A solução em nível de arquitetura para isso foi discutida no final: Podemos implementar um bloco try/except com lógica de Retry (tentar novamente após alguns segundos) e Exponential Backoff (aumentar o tempo de espera a cada tentativa).
## 👨‍💻 Autor
Tiago Rodrigues Plum Ferreira
Engenheiro de Computação Apaixonado por Data Science,Machine Learning, Visão Computacional e Generative AI.
Conecte-se comigo no LinkedIn para trocarmos ideias sobre arquitetura de software e inteligência artificial!
