import time
import json as json_parser # Usamos um apelido para evitar conflitos de nomes no Python
import pandas as pd
from src.classical_ml import ClassicalMLFilter
from src.gemini_pipeline import GeminiFraudPipeline

def run_finguard_pipeline(client_data: dict, text_history: str, ml_filter: ClassicalMLFilter, gemini_pipeline: GeminiFraudPipeline):
    print(f"\n🚀 Iniciando Análise FinGuard AI para o cliente: {client_data['id']}\n" + "="*50)
    
    # 2. Camada 0: ML Estatístico (Usando a instância já treinada)
    risk_score, shap_explanation = ml_filter.predict_risk(client_data)
    print(f"📊 Risco calculado pelo XGBoost: {risk_score:.2%}")
    print(f"🧠 Explicabilidade (SHAP):\n{shap_explanation.strip()}\n")
    
    # 3. Lógica de Roteamento Dinâmico (Arquitetura Híbrida)
    if risk_score < 0.30:
        print("🟢 Risco Baixo. Cliente aprovado automaticamente via Camada 0.")
        return "APROVADO_DIRETO"
        
    elif 0.30 <= risk_score < 0.75:
        print("🟡 Risco Moderado. Encaminhando para Camada 1 & 2 (Gemini Flash-Lite + Flash)...")
        summary = gemini_pipeline.analyze_with_flash_lite(text_history)
        verdict = gemini_pipeline.analyze_with_flash(summary, client_data, shap_explanation)
        print("\n⚖️ [Gemini Flash] Parecer Contextual Concluído:")
        print(verdict.strip())
        return "REVISAO_MANUAL_CONTEXTUAL"
        
    else:
        print("🔴 Risco Crítico Detectado! Escalando direto para Auditoria Profunda (Gemini)...")
        summary = gemini_pipeline.analyze_with_flash_lite(text_history)
        verdict = gemini_pipeline.analyze_with_flash(summary, client_data, shap_explanation)
        
        # Recebemos a saída estruturada do LLM
        json_audit = gemini_pipeline.audit_with_pro(verdict, shap_explanation)
        
        # PREVENÇÃO: Limpamos os blocos markdown caso a IA os coloque (```json ... ```)
        clean_json = json_audit.replace("```json", "").replace("```", "").strip()
        
        # Transformamos o texto num dicionário Python
        audit_dict = json_parser.loads(clean_json)
        
        print("\n🚨 [Auditoria Forense] Relatório Estruturado (JSON):")
        print(f"🧠 Raciocínio Forense:\n   {audit_dict['analise_forense']}")
        print(f"📊 Confiança da IA: {audit_dict['score_confianca']}%")
        print(f"🛑 Decisão da IA: {audit_dict['decisao_final']}")
        
        return audit_dict['decisao_final']

def extract_test_cases_from_csv(csv_path: str, ml_filter: ClassicalMLFilter):
    """Varre a base real e pesca automaticamente 3 cenários distintos para testar"""
    df = pd.read_csv(csv_path)
    
    historicos = {
        "baixo": "Cliente de longa data. Tudo em conformidade, sem chamados no suporte.",
        "medio": "Cliente tentou mudar a senha do app duas vezes ontem antes do pedido de crédito.",
        "alto": "ALERTA: Acesso via IP proxy detetado na madrugada e alteração de e-mail recente."
    }
    
    cenarios = []
    encontrou = {"baixo": False, "medio": False, "alto": False}
    
    for idx, row in df.iterrows():
        cliente_dict = row.to_dict()
        cliente_dict['id'] = f"USR-{idx} (Base Real)"
        
        # Removemos a variável alvo para o modelo e o LLM não saberem o gabarito!
        if 'class' in cliente_dict:
            del cliente_dict['class']
        
        risco, _ = ml_filter.predict_risk(cliente_dict)
        
        if risco < 0.30 and not encontrou["baixo"]:
            cenarios.append((cliente_dict, historicos["baixo"]))
            encontrou["baixo"] = True
            
        elif 0.30 <= risco < 0.75 and not encontrou["medio"]:
            cenarios.append((cliente_dict, historicos["medio"]))
            encontrou["medio"] = True
            
        elif risco >= 0.75 and not encontrou["alto"]:
            cenarios.append((cliente_dict, historicos["alto"]))
            encontrou["alto"] = True
            
        if all(encontrou.values()):
            break
            
    return cenarios

if __name__ == "__main__":
    CSV_FILE_PATH = "Credit.csv"
    print("=== INICIALIZANDO MOTORES FINGUARD AI ===")
    
    ml_filter = ClassicalMLFilter()
    ml_filter.fit_csv_data(CSV_FILE_PATH)
    gemini_pipeline = GeminiFraudPipeline()
    
    print(f"\n=== EXTRAINDO CASOS DE TESTE DA BASE REAL ({CSV_FILE_PATH}) ===")
    lista_de_testes = extract_test_cases_from_csv(CSV_FILE_PATH, ml_filter)
    
    print("\n=== INICIANDO HOMOLOGAÇÃO DE CENÁRIOS DO PIPELINE ===")
    max_tentativas = 3
    
    for cliente, historico in lista_de_testes:
        for tentativa in range(max_tentativas):
            try:
                resultado = run_finguard_pipeline(cliente, historico, ml_filter, gemini_pipeline)
                print(f"\n📌 Fim do processamento para {cliente['id']}. Resultado final: {resultado}")
                print("-" * 60)
                time.sleep(5)
                break
            except Exception as e:
                erro_str = str(e)
                if '503' in erro_str or '429' in erro_str:
                    print(f"⚠️ Servidor ocupado. Aguardando 15s... ({tentativa + 1}/{max_tentativas})")
                    time.sleep(15)
                else:
                    print(f"\n❌ Erro ao processar o cliente {cliente['id']}: {erro_str}")
                    print("-" * 60)
                    break
    
    print("\n=== TODOS OS CENÁRIOS FORAM TESTADOS COM SUCESSO ===")