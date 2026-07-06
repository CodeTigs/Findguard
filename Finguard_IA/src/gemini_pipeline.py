import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum

load_dotenv()

# ==========================================
# CONTRATO DE DADOS (SCHEMA OBRIGATÓRIO)
# ==========================================
class DecisaoRisco(str, Enum):
    APROVAR = "APROVAR_COM_RESSALVAS"
    REVISAO_MANUAL = "ENCAMINHAR_MESA_DE_CREDITO"
    BLOQUEAR = "BLOQUEIO_IMEDIATO"

class RelatorioAuditoria(BaseModel):
    analise_forense: str = Field(description="Parecer forense detalhado (Chain of Thought) cruzando a matemática SHAP e o contexto.")
    score_confianca: int = Field(description="Nível de confiança da IA nesta decisão de 0 a 100.")
    decisao_final: DecisaoRisco = Field(description="Veredito final exato e restrito à lista de opções.")

class GeminiFraudPipeline:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.TRIAGE_MODEL = 'gemini-2.5-flash' 
        self.CONTEXT_MODEL = 'gemini-2.5-flash'   
        self.AUDIT_MODEL = 'gemini-2.5-flash'     

    def analyze_with_flash_lite(self, history_text: str) -> str:
        prompt = f"Analise o histórico e extraia menções a cartões perdidos ou contestações de forma ultra-resumida. Histórico: {history_text}"
        response = self.client.models.generate_content(model=self.TRIAGE_MODEL, contents=prompt)
        return response.text

    def analyze_with_flash(self, structured_summary: str, metadata: dict, shap_explanation: str) -> str:
        prompt = f"Analista de risco sénior. Dados SHAP: {shap_explanation}. Metadados: {metadata}. Resumo: {structured_summary}. Gere parecer conciso."
        response = self.client.models.generate_content(model=self.CONTEXT_MODEL, contents=prompt)
        return response.text

    def audit_with_pro(self, contextual_analysis: str, shap_explanation: str) -> str:
        """Camada 3: Auditoria profunda com retorno OBRIGATÓRIO em JSON"""
        prompt = f"Execute análise forense (Chain of Thought). Prova matemática SHAP: {shap_explanation}. Parecer: {contextual_analysis}."
        
        response = self.client.models.generate_content(
            model=self.AUDIT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RelatorioAuditoria,
                temperature=0.1
            )
        )
        return response.text
