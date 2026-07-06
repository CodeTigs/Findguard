# FinGuard AI 🛡️⚙️ | Agentic Pipeline para Risco de Crédito

Sistema Híbrido de Detecção de Fraude e Inadimplência baseado em **Inteligência Artificial Agêntica**, combinando Machine Learning Clássico (XGBoost + SHAP) e GenAI (Google Gemini).

O FinGuard AI é um pipeline avançado de engenharia de dados focado na resolução de um dos maiores desafios da adoção de Inteligência Artificial Generativa no setor financeiro: o *trade-off* entre custo computacional (tokens/latência) e capacidade analítica. Em vez de processar todas as transações em Large Language Models (LLMs) pesados, este projeto propõe uma arquitetura de roteamento dinâmico e explicável.

---

## 📐 A Arquitetura do Funil Híbrido

O sistema varre bases de dados reais (utilizando o *German Credit Risk Data* como prova de conceito) e roteia os clientes em camadas autônomas:

```text
[Base de Dados Real: Credit.csv] 
       │
       ▼
[Camada 0: XGBoost + SHAP] ────── Avaliação Tabular & Explainable AI
       │
       ├── Risco < 30%  ─────────► 🟢 Aprovação Automática (Custo Zero GenAI)
       │
       ├── Risco 30% a 75% ──────► 🟡 [Camada 1 e 2: Gemini Flash]
       │                             Triagem de logs e Parecer Contextual Rápido
       │
       └── Risco > 75%  ─────────► 🔴 [Camada 3: Auditoria Forense]
                                     Chain-of-Thought + JSON Estruturado
```

## 🚀 Destaques Técnicos

* **Explainable AI (XAI):** O pipeline não confia em "caixas pretas". A biblioteca SHAP faz a engenharia reversa do XGBoost, enviando para a GenAI a justificativa matemática exata do risco.
* **Saídas Estruturadas (Pydantic):** Integração nativa com a API da Google GenAI para garantir contratos de dados estritos, forçando o modelo a devolver um JSON rigoroso e tipado para leitura sistêmica.
* **Resiliência (Fallback):** Tratamento avançado de erros de parseamento JSON (limpeza de markdown) e mecanismo de fallback automático para revisão humana em caso de falha sistêmica da IA.

## 🛠️ Desafios de Engenharia e Soluções Aplicadas

Durante o desenvolvimento deste pipeline, resolvemos gargalos reais de infraestrutura de IA:

**Limites de Taxa e Quedas de Servidor (Erros 429 e 503)**
* **Problema:** O processamento em lote disparou o limite de RPM (Requisições Por Minuto) da API do Google, gerando erros RESOURCE_EXHAUSTED e UNAVAILABLE.
* **Solução:** Implementação de uma arquitetura resiliente com blocos `try/except`, aplicando Rate Limiting (pausas estratégicas) e Exponential Backoff (espera cadenciada em caso de falha antes do retry).

**Conflito de Versões de Motor Matemático (ValueError)**
* **Problema:** As versões mais recentes do XGBoost (>=2.1.0) alteraram a formatação interna do `base_score`, causando falha fatal no TreeExplainer do SHAP.
* **Solução:** Realizado o isolamento e downgrade seguro do XGBoost para a versão de estabilidade garantida 2.0.3, assegurando total compatibilidade da extração matemática.

**Falhas de Extração de JSON ("Expecting value: line 1")**
* **Problema:** Mesmo utilizando `response_schema`, o LLM ocasionalmente inseria lixo em texto ou marcações markdown (```json) antes do código.
* **Solução:** Construído um Extrator Robusto de JSON na Camada 3 (`json_audit[inicio:fim+1]`), somado a um mecanismo de segurança (Fallback) que redireciona automaticamente o cliente para "Revisão Manual" caso a IA sofra alucinações de formatação.

## ⚙️ Stack Tecnológico

* **Linguagem:** Python 3.10+
* **Machine Learning:** XGBoost (v2.0.3), Scikit-Learn, Pandas
* **Explainable AI:** SHAP
* **Generative AI:** Google GenAI SDK (Família Gemini 2.5 Flash)
* **Engenharia de Dados:** Pydantic (Schema Validation)
* **Gerenciamento:** python-dotenv

##  🚀 Como Executar Localmente
### 1 Clone o repositório:
git clone https://github.com/CodeTigs/Findguard.git
cd Findguard

### 2 Instale as dependências:
Bash
pip install -r requirements.txt

### 3 Configure suas credenciais:
Crie um arquivo .env na raiz do projeto e insira sua chave da API do Google AI Studio:
Snippet de código
GEMINI_API_KEY=sua_chave_aqui

### 4 Execute o orquestrador:
python main.py

### Sucesso ✅✅✅
O sistema irá treinar o modelo na base real, pescar automaticamente clientes de diferentes faixas de risco e executar o funil híbrido, exibindo as auditorias e decisões da IA no terminal.

## 👨‍💻 Autor
### Tiago Rodrigues Plum Ferreira
### Engenheiro de Computação (INATEL) | profissional em Data Science, Machine Learning, Visão Computacional e Generative AI.
### 🔌 Conecte-se comigo no LinkedIn para trocarmos ideias sobre arquitetura de software, IA agêntica e o futuro da engenharia de dados!
