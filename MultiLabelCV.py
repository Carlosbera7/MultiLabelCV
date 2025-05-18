import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.model_selection import iterative_train_test_split
from nltk.corpus import stopwords
import logging

nltk.download('stopwords')

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
        data['text'] = data['text'].apply(clean_text)
        X = data['text']
        y = data.drop(columns=['text'])
        return X, y
    except FileNotFoundError:
        logging.error(f"Arquivo {file_path} não encontrado.")
        return None, None

def filter_labels(y, min_count=20):
    label_counts = y.sum(axis=0)
    valid_labels = label_counts[label_counts >= min_count].index
    return y[valid_labels]

def cross_validate(X, y, n_splits=5):
    """
    Executa validação cruzada para o modelo de classificação multi-label.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=30)
    fold_metrics = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"\n----- Fold {fold + 1} -----")
        
        # Divisão dos dados em treino e teste para o fold atual
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Verificar se os dados estão vazios
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            print(f"Fold {fold + 1} ignorado devido a dados insuficientes.")
            continue

        # Configuração do modelo XGBoost
        params = {
            'max_depth': 6,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'base_score': 0.5
        }

        # Treinamento e previsão para cada rótulo
        models = {}
        predictions = np.zeros(y_test.shape)

        for label_idx in range(y_train.shape[1]):
            # Verificar se há exemplos positivos para o rótulo atual
            if y_train[:, label_idx].sum() == 0 or y_test[:, label_idx].sum() == 0:
                print(f"Rótulo {label_idx} ignorado no fold {fold + 1} devido à ausência de exemplos.")
                continue

            print(f"Treinando modelo para o rótulo: {label_idx}")
            dtrain_label = xgb.DMatrix(data=X_train, label=y_train[:, label_idx])
            model = xgb.train(params, dtrain_label, num_boost_round=100)
            models[label_idx] = model

            # Fazendo previsões para o rótulo atual
            dtest_label = xgb.DMatrix(data=X_test)
            predictions[:, label_idx] = model.predict(dtest_label)

        # Verificar se há previsões válidas antes de calcular métricas
        if np.all(predictions == 0):
            print(f"Nenhuma previsão válida no fold {fold + 1}.")
            continue

        # Avaliação das métricas para este fold
        report = classification_report(
            y_test, (predictions >= 0.5).astype(int), zero_division=0, output_dict=True
        )
        fold_metrics.append(report)

    # Agregação das métricas
    if fold_metrics:  # Garantir que há métricas válidas
        aggregated_metrics = {}
        for metric in fold_metrics[0]:
            try:
                aggregated_metrics[metric] = np.mean([
                    fold[metric]['macro avg']['f1-score']
                    for fold in fold_metrics if 'macro avg' in fold[metric]
                ])
            except KeyError:
                print(f"Métrica {metric} não calculada corretamente.")
                aggregated_metrics[metric] = float('nan')

        print("\n----- Resultados agregados -----")
        for metric, score in aggregated_metrics.items():
            print(f"{metric}: {score:.4f}")
    else:
        print("Nenhum fold válido para calcular métricas agregadas.")



def gerar():
    X, y = load_and_prepare_data('2019-05-28_portuguese_hate_speech_hierarchical_classification.csv')
    if X is None or y is None:
        return

    y = filter_labels(y)
    logging.info(f"Rótulos mantidos: {list(y.columns)}")

    portuguese_stopwords = stopwords.words('portuguese')
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=portuguese_stopwords)
    X_tfidf = vectorizer.fit_transform(X)

    #X_train, y_train, X_test, y_test = iterative_train_test_split(X_tfidf, y.values, test_size=0.3)
    #logging.info(f"Formatos: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    cross_validate(X_tfidf, y.values)

if __name__ == "__main__":
    gerar()