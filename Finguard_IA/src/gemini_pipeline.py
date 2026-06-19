import os
from google import genai
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

class GeminiFraudPipeline:
    def __init__(self):
        # Inicializa o cliente oficial da Google GenAI SDK
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # PARAMETRIZAÇÃO DOS MODELOS
        # Utilizando a família 2.5 para contornar o limite diário e cota da versão 2.5 no Free Tier
        self.TRIAGE_MODEL = 'gemini-2.5-flash-lite' # Camada 1: Triagem rápida
        self.CONTEXT_MODEL = 'gemini-2.5-flash'   # Camada 2: Parecer contextual
        self.AUDIT_MODEL = 'gemini-2.5-flash'     # Camada 3: Auditoria (Substituído temporariamente devido à cota do Free Tier podendo ser 'gemini-pro' em produção)

    def analyze_with_flash_lite(self, history_text: str) -> str:
        """Camada 1: Triagem rápida e barata de logs/textos históricos"""
        prompt = f"""
        Analise o seguinte histórico de suporte e transações do cliente.
        Extraia de forma ultra-resumida se há menção a cartões perdidos ou contestações.
        Histórico: {history_text}
        """
        response = self.client.models.generate_content(
            model=self.TRIAGE_MODEL,
            contents=prompt
        )
        return response.text

    def analyze_with_flash(self, structured_summary: str, metadata: dict, shap_explanation: str) -> str:
        """Camada 2: Análise contextual combinando dados estruturados, notas e SHAP"""
        prompt = f"""
        Atue como um analista de risco antifraude sénior. 
        O nosso modelo matemático estatístico sinalizou risco neste cliente pelos seguintes motivos:
        {shap_explanation}
        
        Combine a explicação algorítmica acima com os metadados do cliente e o resumo de triagem.
        Gere um parecer de risco médio conciso e inteligente.
        
        Metadados: {metadata}
        Resumo de Triagem: {structured_summary}
        """
        response = self.client.models.generate_content(
            model=self.CONTEXT_MODEL,
            contents=prompt
        )
        return response.text

    def audit_with_pro(self, contextual_analysis: str, shap_explanation: str) -> str:
        """Camada 3: Auditoria profunda cruzando a matemática (SHAP) com o contexto (Gemini)"""
        prompt = f"""
        [ALERTA CRÍTICO DE FRAUDE]
        Execute uma análise forense profunda (Chain of Thought).
        O modelo XGBoost bloqueou inicialmente o cliente com a seguinte prova matemática:
        {shap_explanation}
        
        Com base no parecer contextual abaixo e na matemática descrita acima, determine se 
        a conta deve ser bloqueada preventivamente e justifique a sua linha de raciocínio lógico.
        
        Parecer técnico (Camada Anterior): {contextual_analysis}
        """
        response = self.client.models.generate_content(
            model=self.AUDIT_MODEL,
            contents=prompt
        )
        return response.text