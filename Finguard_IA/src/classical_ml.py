import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap # Importamos o motor de Explainable AI

class ClassicalMLFilter:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            random_state=42, 
            enable_categorical=True 
        )
        self.features = []
        self.explainer = None # Preparar o explicador matemático
        
    def fit_csv_data(self, csv_path: str):
        """Treina o modelo e inicializa o explicador SHAP"""
        df = pd.read_csv(csv_path)
        
        self.features = [
            'duration', 'credit_amount', 'age', 
            'checking_status', 'credit_history', 'purpose'
        ]
        
        X = df[self.features].copy()
        
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category')
            
        y = df['class'].map({'bad': 1, 'good': 0})
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # INOVAÇÃO: Inicializamos o SHAP TreeExplainer após o treino
        self.explainer = shap.TreeExplainer(self.model)
        print(f"✅ Camada 0: Modelo XGBoost treinado (com SHAP Explainer) na base real ({csv_path}).")

    def predict_risk(self, client_data: dict) -> tuple[float, str]:
        """Devolve a probabilidade de risco e a explicação matemática (SHAP)"""
        df = pd.DataFrame([client_data])
        df = df[self.features]
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
            
        proba = float(self.model.predict_proba(df)[0][1])
        
        # --- EXPLAINABLE AI (Extração dos motivos) ---
        # O SHAP calcula a contribuição exata de cada variável para o score final
        shap_values = self.explainer.shap_values(df)
        
        impactos = []
        # O XGBoost com SHAP devolve valores positivos para a classe 1 (Risco Alto)
        for i, feature_name in enumerate(self.features):
            valor_impacto = shap_values[0][i]
            if valor_impacto > 0: # Variável puxou o risco para cima
                valor_real_cliente = df[feature_name].iloc[0]
                impactos.append((feature_name, valor_impacto, valor_real_cliente))
        
        # Ordenar os fatores do mais grave para o menos grave
        impactos.sort(key=lambda x: x[1], reverse=True)
        
        explicacao = "Fatores Matemáticos Críticos (SHAP):\n"
        if impactos:
            for nome, imp, val_real in impactos[:3]: # Mostra os 3 piores motivos
                explicacao += f"- A variável '{nome}' (Valor: {val_real}) elevou drasticamente o risco.\n"
        else:
            explicacao += "- Nenhum fator isolado elevou o risco de forma anómala.\n"
            
        return proba, explicacao