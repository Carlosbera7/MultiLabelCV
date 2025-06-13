import pandas as pd
from sklearn.metrics import classification_report
import xgboost as xgb
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
# Importação correta para validação cruzada multi-rótulo
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from nltk.corpus import stopwords
import logging

nltk.download('stopwords', quiet=True)

# Configurações de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    stop_words = set(stopwords.words('portuguese'))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

def load_and_prepare_data(file_path):
    try:
        logging.info("Carregando os dados...")
        data = pd.read_csv(file_path)
        # Limpeza do texto
        data['text'] = data['text'].apply(clean_text)
        X = data['text']
        # Remove a coluna de texto para obter os rótulos
        y = data.drop(columns=['text'])
        return X, y
    except FileNotFoundError:
        logging.error(f"Arquivo {file_path} não encontrado.")
        return None, None

def filter_labels(y, min_count=10):
    label_counts = y.sum(axis=0)
    valid_labels = label_counts[label_counts >= min_count].index
    return y[valid_labels]

def cross_validate_multilabel(X, y, n_splits=10):
    """
    Executa validação cruzada para o modelo de classificação multi-rótulo,
    corrigindo o vazamento de dados e usando a estratificação correta.
    """
    # Usando MultilabelStratifiedKFold para manter a distribuição dos rótulos
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Lista para armazenar o F1-score (macro avg) de cada fold
    macro_f1_scores = []

    # O split é feito em X e y. Para o Tfidf, usaremos os índices.
    # y deve ser um array numpy para o split
    y_np = y.values
    # X precisa ser um array ou lista para ser indexado
    X_np = X.values

    for fold, (train_index, test_index) in enumerate(mskf.split(X_np, y_np)):
        logging.info(f"----- Iniciando Fold {fold + 1}/{n_splits} -----")
        
        X_train, X_test = X_np[train_index], X_np[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]

        # Adiciona a quantidade de exemplos positivos por rótulo no treino e teste
        print(f"\nFold {fold + 1} - Quantidade de exemplos positivos por rótulo:")
        for idx, label in enumerate(y.columns):
            train_count = np.sum(y_train[:, idx])
            test_count = np.sum(y_test[:, idx])
            print(f"  {label}: treino={train_count}, teste={test_count}")

        # --- Correção do Data Leakage ---
        # 1. Crie e ajuste o TfidfVectorizer APENAS com os dados de TREINO
        vectorizer = TfidfVectorizer(max_features=5000) # Stop words já removidas no clean_text
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        # 2. APENAS transforme os dados de TESTE com o vetorizador já ajustado
        X_test_tfidf = vectorizer.transform(X_test)
        # ---------------------------------
        
        predictions = np.zeros(y_test.shape)
        
        # Estratégia One-vs-Rest: treinar um modelo por rótulo
        for label_idx in range(y_train.shape[1]):
            # Checa se o rótulo tem instâncias positivas no treino
            if np.sum(y_train[:, label_idx]) < 2:
                logging.warning(f"Rótulo {y.columns[label_idx]} (idx {label_idx}) tem poucos exemplos no treino do fold {fold + 1}. Pulando.")
                continue

            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=100,
                max_depth=6
            )
            
            # Treina o modelo para o rótulo atual
            model.fit(X_train_tfidf, y_train[:, label_idx])
            
            # Faz as previsões (probabilidades)
            preds_proba = model.predict_proba(X_test_tfidf)[:, 1]
            predictions[:, label_idx] = preds_proba

        # Converte probabilidades em predições binárias (0 ou 1)
        y_pred = (predictions >= 0.5).astype(int)

        # Avaliação e armazenamento da métrica de interesse
        report = classification_report(y_test, y_pred, target_names=y.columns, zero_division=0, output_dict=True)
        
        # Print F1-score de cada rótulo para o fold atual
        print(f"\nFold {fold + 1} - F1-score por rótulo:")
        for label in y.columns:
            f1 = report[label]['f1-score']
            print(f"  {label}: {f1:.4f}")

        macro_f1 = report['macro avg']['f1-score']
        macro_f1_scores.append(macro_f1)
        
        logging.info(f"Fold {fold + 1} - Macro F1-Score: {macro_f1:.4f}")

    # Apresenta os resultados finais agregados
    logging.info("\n----- Resultados Finais da Validação Cruzada -----")
    mean_f1 = np.mean(macro_f1_scores)
    std_f1 = np.std(macro_f1_scores)
    
    print(f"Macro F1-Score Médio: {mean_f1:.4f}")
    print(f"Desvio Padrão do F1-Score: {std_f1:.4f}")
    print("Menor número de exemplos positivos em um rótulo:", y.sum(axis=0).min())

def gerar():
    X, y = load_and_prepare_data('2019-05-28_portuguese_hate_speech_hierarchical_classification.csv')
    if X is None or y is None:
        return

    y = filter_labels(y)
    logging.info(f"Rótulos mantidos para o modelo: {list(y.columns)}")

    # Chama a função de validação cruzada corrigida
    cross_validate_multilabel(X, y, n_splits=5)

if __name__ == "__main__":
    gerar()