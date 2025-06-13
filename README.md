# Experimento: Classificação de Discurso de Ódio em Português XGBoost Multi-Label com Cross Validation de 10 Folds

Este repositório contém a implementação do experimento utilizando Xgboost para Multi-Label adapatado de https://gabrielziegler3.medium.com/multiclass-multilabel-classification-with-xgboost-66195e4d9f2d. 

## Descrição do Experimento
O experimento segue as etapas descritas no artigo:

1. **Carregamento dos Dados**:
   - O arquivo CSV 2019-05-28_portuguese_hate_speech_hierarchical_classification_reduzido.csv é carregado.
   - A coluna text é separada como as features (X), e as demais colunas são tratadas como rótulos (y).

2. **Pré-processamento dos Rótulos**:
     - Os rótulos (y) são convertidos para valores numéricos
     - Valores inválidos ou fora do intervalo [0, 1] são substituídos por 0.
     - Valores NaN são preenchidos com 0.
     - Os rótulos são convertidos para inteiros e transformados em uma matriz NumPy.   

3. **Vetorização do Texto**:
   - O texto (X) é vetorizado usando TF-IDF com um limite de 5000 features.
   - Stopwords em português são removidas utilizando a biblioteca NLTK.
      
4. **Divisão dos Dados**:
   - Os dados são divididos em conjuntos de treino e teste utilizando stratificação hierárquica com a função iterative_train_test_split da biblioteca scikit-multilearn.
   - A distribuição das classes nos conjuntos de treino e teste é verificada.
  
5. **Treinamento do Modelo**:
   - Um modelo XGBoost é treinado para cada rótulo (coluna de y).
   - O modelo utiliza a função de objetivo binary:logistic para classificação binária.
     
## Implementação
O experimento foi implementado em Python 3.6 utilizando as bibliotecas:
- pandas
- NLTK
- Scikit-learn
- XGBoost

## Divisão
![Divisao](https://github.com/user-attachments/assets/7da2dc03-7fc2-4680-8d21-094c31f174a9)

O script principal executa as seguintes etapas:
1. Carregamento das partições salvas.
2. Tokenização e padding das sequências de texto.
3. Carregamento dos embeddings GloVe.
4. Construção e treinamento do modelo LSTM.
5. Extração das representações intermediárias.
6. Treinamento e avaliação do XGBoost.
7. Busca de hiperparâmetros com validação cruzada.

## Estrutura do Repositório
- [`Scripts/ClassificadorHierarquicoValido.py`](https://github.com/Carlosbera7/ClassificadorMultiLabel/blob/main/Script/ClassificadorHierarquicoValido.py): Script principal para executar o experimento.
- [`Data/`](https://github.com/Carlosbera7/ClassificadorMultiLabel/tree/main/Data): Pasta contendo o conjunto de dados e o Embeddings GloVe pré-treinados (necessário para execução).
- [`Execução`](https://musical-space-yodel-9rpvjvw9qr39vw4.github.dev/): O código pode ser executado diretamente no ambiente virtual.

## Resultados
Os resultados incluem:

Exemplo das 5 primeiras linhas das Predições : 
```
## Resultados da Validação Cruzada por Fold

### Quantidade de Exemplos Positivos por Rótulo
| Rótulo               | Fold 1 (treino/teste) | Fold 2 (treino/teste) | Fold 3 (treino/teste) | Fold 4 (treino/teste) | Fold 5 (treino/teste) |
|----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| Hate.speech          | 982 / 246            | 982 / 246            | 982 / 246            | 983 / 245            | 983 / 245            |
| Sexism               | 538 / 134            | 537 / 135            | 538 / 134            | 539 / 133            | 536 / 136            |
| Body                 | 131 / 33             | 131 / 33             | 132 / 32             | 131 / 33             | 131 / 33             |
| Racism               | 75 / 19              | 76 / 18              | 75 / 19              | 75 / 19              | 75 / 19              |
| Ideology             | 73 / 19              | 74 / 18              | 73 / 19              | 74 / 18              | 74 / 18              |
| Homophobia           | 257 / 65             | 258 / 64             | 258 / 64             | 257 / 65             | 258 / 64             |
| Origin               | 21 / 5               | 21 / 5               | 21 / 5               | 20 / 6               | 21 / 5               |
| Religion             | 24 / 6               | 24 / 6               | 24 / 6               | 24 / 6               | 24 / 6               |
| OtherLifestyle       | 16 / 4               | 16 / 4               | 16 / 4               | 16 / 4               | 16 / 4               |
| Fat.people           | 128 / 32             | 128 / 32             | 128 / 32             | 128 / 32             | 128 / 32             |
| Left.wing.ideology   | 21 / 5               | 21 / 5               | 20 / 6               | 21 / 5               | 21 / 5               |
| Ugly.people          | 104 / 26             | 105 / 26             | 105 / 26             | 105 / 26             | 105 / 26             |
| Black.people         | 41 / 11              | 42 / 10              | 42 / 10              | 42 / 10              | 41 / 11              |
| Fat.women            | 122 / 31             | 123 / 30             | 122 / 31             | 122 / 31             | 123 / 30             |
| Feminists            | 52 / 13              | 52 / 13              | 52 / 13              | 52 / 13              | 52 / 13              |
| Gays                 | 45 / 11              | 45 / 11              | 45 / 11              | 45 / 11              | 44 / 12              |
| Immigrants           | 12 / 3               | 12 / 3               | 12 / 3               | 12 / 3               | 12 / 3               |
| Islamists            | 14 / 3               | 13 / 4               | 13 / 4               | 14 / 3               | 14 / 3               |
| Lesbians             | 199 / 49             | 200 / 48             | 196 / 52             | 200 / 48             | 197 / 51             |
| Men                  | 56 / 14              | 56 / 14              | 56 / 14              | 56 / 14              | 56 / 14              |
| Muslims              | 9 / 2                | 9 / 2                | 9 / 2                | 8 / 3                | 9 / 2                |
| Refugees             | 57 / 13              | 56 / 14              | 57 / 13              | 54 / 16              | 56 / 14              |
| Trans.women          | 21 / 5               | 20 / 6               | 21 / 5               | 21 / 5               | 21 / 5               |
| Women                | 435 / 109            | 435 / 109            | 435 / 109            | 435 / 109            | 436 / 108            |
| Transexuals          | 11 / 3               | 11 / 3               | 11 / 3               | 12 / 2               | 11 / 3               |
| Ugly.women           | 104 / 26             | 104 / 26             | 104 / 26             | 104 / 26             | 104 / 26             |
| Migrants             | 66 / 16              | 66 / 16              | 66 / 16              | 65 / 17              | 65 / 17              |
| Homossexuals         | 230 / 58             | 231 / 57             | 231 / 57             | 230 / 58             | 230 / 58             |

### F1-Score por Rótulo
| Rótulo               | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|----------------------|--------|--------|--------|--------|--------|
| Hate.speech          | 0.6484 | 0.6232 | 0.6500 | 0.6633 | 0.6715 |
| Sexism               | 0.6121 | 0.6043 | 0.6228 | 0.6612 | 0.6466 |
| Body                 | 0.8955 | 0.8136 | 0.9032 | 0.8615 | 0.8197 |
| Racism               | 0.2000 | 0.2143 | 0.2963 | 0.4000 | 0.2857 |
| Ideology             | 0.2857 | 0.0909 | 0.3571 | 0.4000 | 0.0000 |
| Homophobia           | 0.8621 | 0.8548 | 0.8889 | 0.8246 | 0.8525 |
| Origin               | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Religion             | 0.0000 | 0.4444 | 0.0000 | 0.2222 | 0.2222 |
| OtherLifestyle       | 0.0000 | 0.3333 | 0.5714 | 0.0000 | 0.0000 |
| Fat.people           | 0.9231 | 0.7931 | 0.9180 | 0.8615 | 0.8525 |
| Left.wing.ideology   | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Ugly.people          | 0.8475 | 0.7826 | 0.8679 | 0.7636 | 0.9020 |
| Black.people         | 0.3810 | 0.3750 | 0.1429 | 0.5000 | 0.2667 |
| Fat.women            | 0.8788 | 0.8214 | 0.9333 | 0.8438 | 0.8421 |
| Feminists            | 0.3636 | 0.1000 | 0.3000 | 0.4444 | 0.1176 |
| Gays                 | 0.3077 | 0.4444 | 0.3077 | 0.5556 | 0.4706 |
| Immigrants           | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Islamists            | 0.0000 | 0.3333 | 0.0000 | 0.4000 | 0.0000 |
| Lesbians             | 0.9796 | 0.9792 | 0.9714 | 0.9583 | 0.9623 |
| Men                  | 0.5185 | 0.3846 | 0.3200 | 0.4000 | 0.4762 |
| Muslims              | 0.0000 | 0.4000 | 0.0000 | 0.0000 | 0.4000 |
| Refugees             | 0.2857 | 0.1111 | 0.3333 | 0.3704 | 0.4545 |
| Trans.women          | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Women                | 0.6344 | 0.6531 | 0.5838 | 0.6429 | 0.6703 |
| Transexuals          | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Ugly.women           | 0.8276 | 0.7826 | 0.8679 | 0.7857 | 0.9020 |
| Migrants             | 0.2857 | 0.1905 | 0.2963 | 0.2963 | 0.3846 |
| Homossexuals         | 0.9074 | 0.8909 | 0.9369 | 0.8704 | 0.8870 |

### Estatísticas Gerais
- **Macro F1-Score Médio**: 0.4325
- **Desvio Padrão do F1-Score**: 0.0124
- **Menor número de exemplos positivos em um rótulo**: 11

```






