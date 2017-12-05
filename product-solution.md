# Otto Group Product Classification

Este notebook é uma proposta de solução
utilizando técnicas de data-mining e machine learn para o problema de
classificação de produtos da companhia Otto disponíveis em: [Kaggle (challenge):
Otto group product classification](https://www.kaggle.com/c/otto-group-product-
classification-challenge)

## Contexto

Retirado da descrição do problema, temos
que o grupo Otto é uma das maiores companhias de *e-commerce* do mundo, e possui
s filiais em mais de 20 paises. Vendem milhões de produtos ao redor do mundo
todos os dias com centezas de produtos sendo adicionados constantemente.

A
análise de consistência da performance dos produtos deles é crucial, entretando,
com a infraestrutura de escala global que possuem, produtos identicos são
classifidados de maneira diferenciada. Entretanto a análise da qualidade dos
produtos depende fortemente da acurácia na habilidade de agrupar produtos
semelhantes. Quanto melhor for a classificação, mais intuitivamente eles ter um
maior alcance com seus produtos.

## Dados

Foram disponibilizados 2 bases de
dados separadas. A primeira delas contém 61878 registros com rótulo da
classificação do produto e 144368 de registros sem o rótulo.

São um total de 93
características na qual não há a descrição do que significa cada uma delas.
Sendo que não há dados faltando. O range dos dados vão de 0 a 352.

```{.python .input  n=1}
# Configure to show multiples outputs from a single cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
import math
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

```

```{.python .input  n=2}
with zipfile.ZipFile('Datasets.zip') as ziped_file:
    with ziped_file.open('Datasets/train.csv') as train_file:
        df_train = pd.read_csv(train_file, header=0).set_index('id')
    with ziped_file.open('Datasets/test.csv') as test_file:
        df_test = pd.read_csv(test_file, header=0).set_index('id')
df_target = pd.DataFrame(df_train.pop('target')) # Get the target
df_target.target = pd.Categorical(df_target.target) # Transform target in Categorical type
df_target['categories'] = df_target.target.cat.codes # Add the codes in a columns
df_target.head() # Show target classes
df_train.head() # The train dataset
df_test.head() # It hasn't target
```

```{.json .output n=2}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>target</th>\n      <th>categories</th>\n    </tr>\n    <tr>\n      <th>id</th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1</th>\n      <td>Class_1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>Class_1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>Class_1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>Class_1</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>5</th>\n      <td>Class_1</td>\n      <td>0</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "     target  categories\nid                     \n1   Class_1           0\n2   Class_1           0\n3   Class_1           0\n4   Class_1           0\n5   Class_1           0"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>feat_1</th>\n      <th>feat_2</th>\n      <th>feat_3</th>\n      <th>feat_4</th>\n      <th>feat_5</th>\n      <th>feat_6</th>\n      <th>feat_7</th>\n      <th>feat_8</th>\n      <th>feat_9</th>\n      <th>feat_10</th>\n      <th>...</th>\n      <th>feat_84</th>\n      <th>feat_85</th>\n      <th>feat_86</th>\n      <th>feat_87</th>\n      <th>feat_88</th>\n      <th>feat_89</th>\n      <th>feat_90</th>\n      <th>feat_91</th>\n      <th>feat_92</th>\n      <th>feat_93</th>\n    </tr>\n    <tr>\n      <th>id</th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1</th>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>6</td>\n      <td>1</td>\n      <td>5</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>...</td>\n      <td>22</td>\n      <td>0</td>\n      <td>1</td>\n      <td>2</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>5</th>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 93 columns</p>\n</div>",
   "text/plain": "    feat_1  feat_2  feat_3  feat_4  feat_5  feat_6  feat_7  feat_8  feat_9  \\\nid                                                                           \n1        1       0       0       0       0       0       0       0       0   \n2        0       0       0       0       0       0       0       1       0   \n3        0       0       0       0       0       0       0       1       0   \n4        1       0       0       1       6       1       5       0       0   \n5        0       0       0       0       0       0       0       0       0   \n\n    feat_10   ...     feat_84  feat_85  feat_86  feat_87  feat_88  feat_89  \\\nid            ...                                                            \n1         0   ...           0        1        0        0        0        0   \n2         0   ...           0        0        0        0        0        0   \n3         0   ...           0        0        0        0        0        0   \n4         1   ...          22        0        1        2        0        0   \n5         0   ...           0        1        0        0        0        0   \n\n    feat_90  feat_91  feat_92  feat_93  \nid                                      \n1         0        0        0        0  \n2         0        0        0        0  \n3         0        0        0        0  \n4         0        0        0        0  \n5         1        0        0        0  \n\n[5 rows x 93 columns]"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>feat_1</th>\n      <th>feat_2</th>\n      <th>feat_3</th>\n      <th>feat_4</th>\n      <th>feat_5</th>\n      <th>feat_6</th>\n      <th>feat_7</th>\n      <th>feat_8</th>\n      <th>feat_9</th>\n      <th>feat_10</th>\n      <th>...</th>\n      <th>feat_84</th>\n      <th>feat_85</th>\n      <th>feat_86</th>\n      <th>feat_87</th>\n      <th>feat_88</th>\n      <th>feat_89</th>\n      <th>feat_90</th>\n      <th>feat_91</th>\n      <th>feat_92</th>\n      <th>feat_93</th>\n    </tr>\n    <tr>\n      <th>id</th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1</th>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>3</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>11</td>\n      <td>1</td>\n      <td>20</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>2</td>\n      <td>2</td>\n      <td>14</td>\n      <td>16</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>4</td>\n      <td>0</td>\n      <td>0</td>\n      <td>2</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>0</td>\n      <td>1</td>\n      <td>12</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>2</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>...</td>\n      <td>0</td>\n      <td>3</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n    <tr>\n      <th>5</th>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>0</td>\n      <td>0</td>\n      <td>1</td>\n      <td>2</td>\n      <td>0</td>\n      <td>3</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>9</td>\n      <td>0</td>\n      <td>0</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 93 columns</p>\n</div>",
   "text/plain": "    feat_1  feat_2  feat_3  feat_4  feat_5  feat_6  feat_7  feat_8  feat_9  \\\nid                                                                           \n1        0       0       0       0       0       0       0       0       0   \n2        2       2      14      16       0       0       0       0       0   \n3        0       1      12       1       0       0       0       0       0   \n4        0       0       0       1       0       0       0       0       0   \n5        1       0       0       1       0       0       1       2       0   \n\n    feat_10   ...     feat_84  feat_85  feat_86  feat_87  feat_88  feat_89  \\\nid            ...                                                            \n1         3   ...           0        0       11        1       20        0   \n2         0   ...           0        0        0        0        0        4   \n3         0   ...           0        0        0        0        2        0   \n4         0   ...           0        3        1        0        0        0   \n5         3   ...           0        0        0        0        0        0   \n\n    feat_90  feat_91  feat_92  feat_93  \nid                                      \n1         0        0        0        0  \n2         0        0        2        0  \n3         0        0        0        1  \n4         0        0        0        0  \n5         0        9        0        0  \n\n[5 rows x 93 columns]"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

# Benchmark

A variável results é um acumulador para salvar os resultados na
base de treino e teste de cada um dos modelos e compará-los ao final.

Segue a
estrutura:

`
 'modelo':
     'teste': value
     'treino': value
`

```{.python .input  n=3}
from sklearn.model_selection import train_test_split

results = {}
def add_results(model, train, test):
    results[model] = {
        'train': train*100,
        'test': test*100,
    }
```

# Cross Validation

A abordagem para a Validação Cruzada é a utilização do
método de k-partições. Neste método, o conjunto de dados é dividido em k
partições [(WITTEN e FRANK,
2000)](ftp://ftp.ingv.it/pub/manuela.sbarra/Data%20Mining%20Practical%20Machine%
20Learning%20Tools%20and%20Techniques%20-%20WEKA.pdf),
testes extensivos em diversas bases de dados, utilizando diversos algoritmos,
identificaram o valor de k para identificar a melhor margem de erro como sendo
10, também de forma randômica. Então, o conjunto de dados de treinamento é
criado com k – 1 partições, e apenas uma partição é utilizada para testes. São
realizadas k iterações, aonde cada partição é utilizada uma vez para testes
enquanto as outras são utilizadas para treinamento. Após todas as partições
terem sido utilizadas para teste, a margem de erro de cada iteração é somada e a
média das k iterações se torna a margem de erro do modelo.

![cross
val](crossval.png)
<center>Representação do método Cross Validation com k = 10.
**Fonte**: BABATUNDE et al., 2015.</center>

# Tratamento

Será realizada as
etapas de feature selection e feature
engineering.
Correlação entre features
Será realizada uma análise da correlação
entre as features. Visto que há um
total de 93 colunas que não foi
disponibilizada nenhuma informação sobre o que
são elas e o que representam e
portanto, esta análize ajudará a identificar as
relações entre as features.

## Correlação


A correlação entre duas variáveis é
quando existe algum laço
matemático que envolve o valor de duas variáveis de
alguma forma [ESTATÍSTICA II
- [CORRELAÇÃO E
REGRESSÃO](http://www.ctec.ufal.br/professor/mgn/05CorrelacaoERegressao.pdf).
Uma das maneiras mais simples de se identificar a correlação entre duas
variáveis é plotando-as em um gráfico, para tentar identificar alguma relação
entre elas, entretanto, como são um total de 93 features, dificulta visualizar a
correlação em forma gráfica.

A correlação de
[Pearson](https://pt.wikipedia.org/wiki/Coeficiente_de_correla%C3%A7%C3%A3o_de%0
A_Pearson)
mede o grau da correlação (e a direcção dessa correlação - se
positiva ou
negativa) entre duas variáveis de escala métrica (intervalar ou de
rácio/razão).
Já a correlação de
[Spearman](https://pt.wikipedia.org/wiki/Coeficiente_de_correla%C3%A7%C3%A3o_de_
postos_de_Spearman)
entre duas variáveis é igual à correlação de Pearson entre
os valores de postos
daquelas duas variáveis. Enquanto a correlação de Pearson
avalia relações
lineares, a correlação de Spearman avalia relações monótonas,
sejam elas
lineares ou não.

Visto ambos os tipos de correlação, utilizaremos a
de Pearson
para avaliar se há alguma correlação linear crescente ou decrescente
entre as
variáveis, pois esta relação nos possibilita remover uma delas sem
prejuizos aos
modelos de machine learn

```{.python .input  n=4}
shape = (df_train.shape[1], df_train.shape[1])
upper_matrix = np.tril(np.ones(shape)).astype(np.bool)
np.fill_diagonal(upper_matrix, False)
correlation = df_train.corr('pearson').abs().where(upper_matrix)
correlation
```

```{.json .output n=4}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>feat_1</th>\n      <th>feat_2</th>\n      <th>feat_3</th>\n      <th>feat_4</th>\n      <th>feat_5</th>\n      <th>feat_6</th>\n      <th>feat_7</th>\n      <th>feat_8</th>\n      <th>feat_9</th>\n      <th>feat_10</th>\n      <th>...</th>\n      <th>feat_84</th>\n      <th>feat_85</th>\n      <th>feat_86</th>\n      <th>feat_87</th>\n      <th>feat_88</th>\n      <th>feat_89</th>\n      <th>feat_90</th>\n      <th>feat_91</th>\n      <th>feat_92</th>\n      <th>feat_93</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>feat_1</th>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_2</th>\n      <td>0.031332</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_3</th>\n      <td>0.027807</td>\n      <td>0.082573</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_4</th>\n      <td>0.027529</td>\n      <td>0.134987</td>\n      <td>0.583523</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_5</th>\n      <td>0.042973</td>\n      <td>0.020926</td>\n      <td>0.010880</td>\n      <td>1.729026e-02</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_6</th>\n      <td>0.043603</td>\n      <td>0.041343</td>\n      <td>0.004288</td>\n      <td>1.405895e-02</td>\n      <td>0.145355</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_7</th>\n      <td>0.298952</td>\n      <td>0.222386</td>\n      <td>0.001294</td>\n      <td>1.448981e-02</td>\n      <td>0.075047</td>\n      <td>0.088014</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_8</th>\n      <td>0.056321</td>\n      <td>0.019815</td>\n      <td>0.053462</td>\n      <td>4.618407e-02</td>\n      <td>0.035861</td>\n      <td>0.012867</td>\n      <td>0.038121</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_9</th>\n      <td>0.032285</td>\n      <td>0.025630</td>\n      <td>0.063551</td>\n      <td>4.624977e-02</td>\n      <td>0.024708</td>\n      <td>0.009373</td>\n      <td>0.027146</td>\n      <td>0.039281</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_10</th>\n      <td>0.097776</td>\n      <td>0.051925</td>\n      <td>0.036944</td>\n      <td>5.951396e-02</td>\n      <td>0.091324</td>\n      <td>0.041940</td>\n      <td>0.194258</td>\n      <td>0.000023</td>\n      <td>0.024323</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_11</th>\n      <td>0.042928</td>\n      <td>0.118534</td>\n      <td>0.596243</td>\n      <td>3.894092e-01</td>\n      <td>0.004882</td>\n      <td>0.014504</td>\n      <td>0.012418</td>\n      <td>0.065923</td>\n      <td>0.075820</td>\n      <td>0.006010</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_12</th>\n      <td>0.056934</td>\n      <td>0.090153</td>\n      <td>0.050037</td>\n      <td>5.743356e-02</td>\n      <td>0.036668</td>\n      <td>0.028588</td>\n      <td>0.056230</td>\n      <td>0.091424</td>\n      <td>0.021885</td>\n      <td>0.048969</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_13</th>\n      <td>0.139254</td>\n      <td>0.157467</td>\n      <td>0.013870</td>\n      <td>2.897317e-02</td>\n      <td>0.059081</td>\n      <td>0.036293</td>\n      <td>0.199142</td>\n      <td>0.095365</td>\n      <td>0.040164</td>\n      <td>0.086682</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_14</th>\n      <td>0.063517</td>\n      <td>0.070057</td>\n      <td>0.111105</td>\n      <td>9.921490e-02</td>\n      <td>0.037607</td>\n      <td>0.027350</td>\n      <td>0.044671</td>\n      <td>0.061799</td>\n      <td>0.110188</td>\n      <td>0.029598</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_15</th>\n      <td>0.045738</td>\n      <td>0.048798</td>\n      <td>0.065285</td>\n      <td>5.122155e-02</td>\n      <td>0.007000</td>\n      <td>0.018328</td>\n      <td>0.035721</td>\n      <td>0.056960</td>\n      <td>0.009858</td>\n      <td>0.021700</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_16</th>\n      <td>0.027086</td>\n      <td>0.108046</td>\n      <td>0.221426</td>\n      <td>2.110780e-01</td>\n      <td>0.062877</td>\n      <td>0.021934</td>\n      <td>0.043957</td>\n      <td>0.004659</td>\n      <td>0.082664</td>\n      <td>0.063997</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_17</th>\n      <td>0.053004</td>\n      <td>0.074902</td>\n      <td>0.023093</td>\n      <td>7.553867e-03</td>\n      <td>0.062197</td>\n      <td>0.015488</td>\n      <td>0.127245</td>\n      <td>0.173912</td>\n      <td>0.028709</td>\n      <td>0.092959</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_18</th>\n      <td>0.084856</td>\n      <td>0.242716</td>\n      <td>0.115655</td>\n      <td>2.148952e-01</td>\n      <td>0.052186</td>\n      <td>0.048710</td>\n      <td>0.098972</td>\n      <td>0.087777</td>\n      <td>0.043642</td>\n      <td>0.071635</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_19</th>\n      <td>0.002302</td>\n      <td>0.176655</td>\n      <td>0.012228</td>\n      <td>3.519107e-07</td>\n      <td>0.008556</td>\n      <td>0.038493</td>\n      <td>0.058071</td>\n      <td>0.019387</td>\n      <td>0.000167</td>\n      <td>0.009015</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_20</th>\n      <td>0.070511</td>\n      <td>0.449160</td>\n      <td>0.011069</td>\n      <td>4.465657e-02</td>\n      <td>0.046200</td>\n      <td>0.057813</td>\n      <td>0.364972</td>\n      <td>0.062595</td>\n      <td>0.023397</td>\n      <td>0.176373</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_21</th>\n      <td>0.027026</td>\n      <td>0.014113</td>\n      <td>0.354925</td>\n      <td>2.329227e-01</td>\n      <td>0.003288</td>\n      <td>0.008046</td>\n      <td>0.022908</td>\n      <td>0.041095</td>\n      <td>0.028409</td>\n      <td>0.005134</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_22</th>\n      <td>0.063283</td>\n      <td>0.215106</td>\n      <td>0.251082</td>\n      <td>2.477378e-01</td>\n      <td>0.075161</td>\n      <td>0.038939</td>\n      <td>0.162620</td>\n      <td>0.029032</td>\n      <td>0.062348</td>\n      <td>0.141405</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_23</th>\n      <td>0.048686</td>\n      <td>0.162065</td>\n      <td>0.002427</td>\n      <td>3.062225e-02</td>\n      <td>0.017281</td>\n      <td>0.043651</td>\n      <td>0.186462</td>\n      <td>0.012774</td>\n      <td>0.006940</td>\n      <td>0.096666</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_24</th>\n      <td>0.067255</td>\n      <td>0.253684</td>\n      <td>0.031596</td>\n      <td>3.727726e-03</td>\n      <td>0.075222</td>\n      <td>0.082124</td>\n      <td>0.244813</td>\n      <td>0.161848</td>\n      <td>0.073618</td>\n      <td>0.081684</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_25</th>\n      <td>0.187237</td>\n      <td>0.096366</td>\n      <td>0.157459</td>\n      <td>1.342306e-01</td>\n      <td>0.003610</td>\n      <td>0.023319</td>\n      <td>0.048820</td>\n      <td>0.036939</td>\n      <td>0.025279</td>\n      <td>0.009792</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_26</th>\n      <td>0.022813</td>\n      <td>0.064856</td>\n      <td>0.268112</td>\n      <td>3.657567e-01</td>\n      <td>0.025116</td>\n      <td>0.004680</td>\n      <td>0.008782</td>\n      <td>0.041599</td>\n      <td>0.066414</td>\n      <td>0.003721</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_27</th>\n      <td>0.038826</td>\n      <td>0.037841</td>\n      <td>0.508370</td>\n      <td>3.086287e-01</td>\n      <td>0.002098</td>\n      <td>0.001943</td>\n      <td>0.015429</td>\n      <td>0.050272</td>\n      <td>0.042531</td>\n      <td>0.001551</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_28</th>\n      <td>0.030257</td>\n      <td>0.072494</td>\n      <td>0.551398</td>\n      <td>4.864171e-01</td>\n      <td>0.047688</td>\n      <td>0.017132</td>\n      <td>0.000998</td>\n      <td>0.036668</td>\n      <td>0.055545</td>\n      <td>0.022349</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_29</th>\n      <td>0.069266</td>\n      <td>0.025689</td>\n      <td>0.004141</td>\n      <td>1.427066e-02</td>\n      <td>0.065957</td>\n      <td>0.002389</td>\n      <td>0.046231</td>\n      <td>0.104985</td>\n      <td>0.021328</td>\n      <td>0.068243</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_30</th>\n      <td>0.033108</td>\n      <td>0.026896</td>\n      <td>0.007667</td>\n      <td>8.733991e-04</td>\n      <td>0.318117</td>\n      <td>0.196493</td>\n      <td>0.050535</td>\n      <td>0.009574</td>\n      <td>0.015830</td>\n      <td>0.012623</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>...</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>feat_64</th>\n      <td>0.010499</td>\n      <td>0.005354</td>\n      <td>0.065105</td>\n      <td>4.728813e-02</td>\n      <td>0.021017</td>\n      <td>0.002764</td>\n      <td>0.011165</td>\n      <td>0.003194</td>\n      <td>0.702951</td>\n      <td>0.022536</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_65</th>\n      <td>0.110041</td>\n      <td>0.078801</td>\n      <td>0.065492</td>\n      <td>6.228472e-02</td>\n      <td>0.228349</td>\n      <td>0.066867</td>\n      <td>0.202346</td>\n      <td>0.025544</td>\n      <td>0.038163</td>\n      <td>0.182756</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_66</th>\n      <td>0.053010</td>\n      <td>0.175620</td>\n      <td>0.088017</td>\n      <td>1.296545e-01</td>\n      <td>0.048364</td>\n      <td>0.033285</td>\n      <td>0.122660</td>\n      <td>0.115175</td>\n      <td>0.001778</td>\n      <td>0.100722</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_67</th>\n      <td>0.154301</td>\n      <td>0.068667</td>\n      <td>0.110081</td>\n      <td>8.045694e-02</td>\n      <td>0.061964</td>\n      <td>0.038289</td>\n      <td>0.148598</td>\n      <td>0.320949</td>\n      <td>0.176921</td>\n      <td>0.043117</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_68</th>\n      <td>0.014674</td>\n      <td>0.012802</td>\n      <td>0.030992</td>\n      <td>2.009191e-02</td>\n      <td>0.107405</td>\n      <td>0.021619</td>\n      <td>0.040309</td>\n      <td>0.075384</td>\n      <td>0.012192</td>\n      <td>0.001693</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_69</th>\n      <td>0.007544</td>\n      <td>0.307406</td>\n      <td>0.032748</td>\n      <td>1.446082e-02</td>\n      <td>0.003294</td>\n      <td>0.074836</td>\n      <td>0.131430</td>\n      <td>0.046258</td>\n      <td>0.029335</td>\n      <td>0.077354</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_70</th>\n      <td>0.165442</td>\n      <td>0.112968</td>\n      <td>0.018774</td>\n      <td>2.079779e-02</td>\n      <td>0.118510</td>\n      <td>0.052401</td>\n      <td>0.237907</td>\n      <td>0.023089</td>\n      <td>0.056205</td>\n      <td>0.322857</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_71</th>\n      <td>0.013712</td>\n      <td>0.002336</td>\n      <td>0.053020</td>\n      <td>4.241268e-02</td>\n      <td>0.056428</td>\n      <td>0.011901</td>\n      <td>0.115813</td>\n      <td>0.081664</td>\n      <td>0.043286</td>\n      <td>0.104834</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_72</th>\n      <td>0.029983</td>\n      <td>0.023267</td>\n      <td>0.045339</td>\n      <td>2.979578e-02</td>\n      <td>0.005177</td>\n      <td>0.011090</td>\n      <td>0.014921</td>\n      <td>0.029868</td>\n      <td>0.058147</td>\n      <td>0.004225</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_73</th>\n      <td>0.140815</td>\n      <td>0.039192</td>\n      <td>0.013972</td>\n      <td>1.128547e-02</td>\n      <td>0.001609</td>\n      <td>0.025023</td>\n      <td>0.022819</td>\n      <td>0.028999</td>\n      <td>0.022679</td>\n      <td>0.000240</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_74</th>\n      <td>0.051365</td>\n      <td>0.070724</td>\n      <td>0.041559</td>\n      <td>4.909735e-02</td>\n      <td>0.017265</td>\n      <td>0.043160</td>\n      <td>0.053059</td>\n      <td>0.000431</td>\n      <td>0.007594</td>\n      <td>0.008912</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_75</th>\n      <td>0.011596</td>\n      <td>0.093689</td>\n      <td>0.044724</td>\n      <td>3.145389e-02</td>\n      <td>0.015279</td>\n      <td>0.006951</td>\n      <td>0.039865</td>\n      <td>0.031466</td>\n      <td>0.027313</td>\n      <td>0.003828</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_76</th>\n      <td>0.153808</td>\n      <td>0.259360</td>\n      <td>0.028670</td>\n      <td>1.379188e-02</td>\n      <td>0.035570</td>\n      <td>0.073867</td>\n      <td>0.375114</td>\n      <td>0.081682</td>\n      <td>0.027424</td>\n      <td>0.106752</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_77</th>\n      <td>0.123752</td>\n      <td>0.014911</td>\n      <td>0.001584</td>\n      <td>1.531773e-02</td>\n      <td>0.030462</td>\n      <td>0.006501</td>\n      <td>0.005769</td>\n      <td>0.027486</td>\n      <td>0.020185</td>\n      <td>0.019069</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_78</th>\n      <td>0.279202</td>\n      <td>0.094256</td>\n      <td>0.021979</td>\n      <td>1.449856e-02</td>\n      <td>0.070709</td>\n      <td>0.061250</td>\n      <td>0.567084</td>\n      <td>0.079623</td>\n      <td>0.015922</td>\n      <td>0.091760</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_79</th>\n      <td>0.228912</td>\n      <td>0.033668</td>\n      <td>0.020566</td>\n      <td>1.083473e-02</td>\n      <td>0.055115</td>\n      <td>0.009942</td>\n      <td>0.066753</td>\n      <td>0.083714</td>\n      <td>0.036116</td>\n      <td>0.113659</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_80</th>\n      <td>0.013303</td>\n      <td>0.155768</td>\n      <td>0.442036</td>\n      <td>4.057725e-01</td>\n      <td>0.026223</td>\n      <td>0.017648</td>\n      <td>0.028860</td>\n      <td>0.038382</td>\n      <td>0.046721</td>\n      <td>0.019042</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_81</th>\n      <td>0.032427</td>\n      <td>0.052101</td>\n      <td>0.013089</td>\n      <td>2.828377e-02</td>\n      <td>0.129333</td>\n      <td>0.044136</td>\n      <td>0.144308</td>\n      <td>0.035102</td>\n      <td>0.005847</td>\n      <td>0.135928</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_82</th>\n      <td>0.026085</td>\n      <td>0.119109</td>\n      <td>0.438458</td>\n      <td>4.365413e-01</td>\n      <td>0.057400</td>\n      <td>0.014907</td>\n      <td>0.022059</td>\n      <td>0.034409</td>\n      <td>0.039806</td>\n      <td>0.029741</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_83</th>\n      <td>0.059165</td>\n      <td>0.371691</td>\n      <td>0.019914</td>\n      <td>1.051874e-03</td>\n      <td>0.008006</td>\n      <td>0.035145</td>\n      <td>0.282069</td>\n      <td>0.033479</td>\n      <td>0.032875</td>\n      <td>0.052025</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_84</th>\n      <td>0.049634</td>\n      <td>0.009845</td>\n      <td>0.011159</td>\n      <td>5.684499e-03</td>\n      <td>0.467329</td>\n      <td>0.177777</td>\n      <td>0.062634</td>\n      <td>0.005064</td>\n      <td>0.013569</td>\n      <td>0.017939</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_85</th>\n      <td>0.008739</td>\n      <td>0.006764</td>\n      <td>0.048626</td>\n      <td>3.315343e-02</td>\n      <td>0.034062</td>\n      <td>0.004290</td>\n      <td>0.037874</td>\n      <td>0.003416</td>\n      <td>0.031462</td>\n      <td>0.086758</td>\n      <td>...</td>\n      <td>0.010210</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_86</th>\n      <td>0.107947</td>\n      <td>0.039090</td>\n      <td>0.096093</td>\n      <td>7.102916e-02</td>\n      <td>0.013879</td>\n      <td>0.010455</td>\n      <td>0.009169</td>\n      <td>0.029395</td>\n      <td>0.019144</td>\n      <td>0.159447</td>\n      <td>...</td>\n      <td>0.003459</td>\n      <td>0.109643</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_87</th>\n      <td>0.089374</td>\n      <td>0.047451</td>\n      <td>0.009838</td>\n      <td>5.054728e-03</td>\n      <td>0.013999</td>\n      <td>0.015256</td>\n      <td>0.089574</td>\n      <td>0.059929</td>\n      <td>0.016925</td>\n      <td>0.077421</td>\n      <td>...</td>\n      <td>0.013631</td>\n      <td>0.049250</td>\n      <td>0.073685</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_88</th>\n      <td>0.020830</td>\n      <td>0.047035</td>\n      <td>0.082336</td>\n      <td>6.748367e-02</td>\n      <td>0.019201</td>\n      <td>0.015437</td>\n      <td>0.033646</td>\n      <td>0.050931</td>\n      <td>0.001160</td>\n      <td>0.054635</td>\n      <td>...</td>\n      <td>0.017903</td>\n      <td>0.027886</td>\n      <td>0.426972</td>\n      <td>0.023053</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_89</th>\n      <td>0.096851</td>\n      <td>0.105527</td>\n      <td>0.174781</td>\n      <td>1.837145e-01</td>\n      <td>0.119951</td>\n      <td>0.035042</td>\n      <td>0.063511</td>\n      <td>0.007974</td>\n      <td>0.019147</td>\n      <td>0.061498</td>\n      <td>...</td>\n      <td>0.103643</td>\n      <td>0.053582</td>\n      <td>0.011822</td>\n      <td>0.066008</td>\n      <td>0.022552</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_90</th>\n      <td>0.010310</td>\n      <td>0.515022</td>\n      <td>0.015068</td>\n      <td>9.454061e-03</td>\n      <td>0.004842</td>\n      <td>0.054034</td>\n      <td>0.129578</td>\n      <td>0.026807</td>\n      <td>0.020698</td>\n      <td>0.049908</td>\n      <td>...</td>\n      <td>0.006013</td>\n      <td>0.003931</td>\n      <td>0.019803</td>\n      <td>0.014696</td>\n      <td>0.031679</td>\n      <td>0.027764</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_91</th>\n      <td>0.037264</td>\n      <td>0.026383</td>\n      <td>0.012417</td>\n      <td>1.031241e-02</td>\n      <td>0.012012</td>\n      <td>0.012465</td>\n      <td>0.068506</td>\n      <td>0.095990</td>\n      <td>0.014742</td>\n      <td>0.024025</td>\n      <td>...</td>\n      <td>0.003444</td>\n      <td>0.023091</td>\n      <td>0.024005</td>\n      <td>0.028850</td>\n      <td>0.033653</td>\n      <td>0.015917</td>\n      <td>0.014812</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_92</th>\n      <td>0.054777</td>\n      <td>0.008219</td>\n      <td>0.066921</td>\n      <td>8.763105e-02</td>\n      <td>0.065331</td>\n      <td>0.015479</td>\n      <td>0.032261</td>\n      <td>0.013608</td>\n      <td>0.069707</td>\n      <td>0.006869</td>\n      <td>...</td>\n      <td>0.048431</td>\n      <td>0.043484</td>\n      <td>0.049393</td>\n      <td>0.001424</td>\n      <td>0.070120</td>\n      <td>0.129622</td>\n      <td>0.035311</td>\n      <td>0.104226</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>feat_93</th>\n      <td>0.081783</td>\n      <td>0.054593</td>\n      <td>0.006814</td>\n      <td>1.574563e-02</td>\n      <td>0.002038</td>\n      <td>0.008521</td>\n      <td>0.034912</td>\n      <td>0.005131</td>\n      <td>0.006038</td>\n      <td>0.041316</td>\n      <td>...</td>\n      <td>0.003723</td>\n      <td>0.023390</td>\n      <td>0.029035</td>\n      <td>0.499990</td>\n      <td>0.008631</td>\n      <td>0.030650</td>\n      <td>0.039864</td>\n      <td>0.000045</td>\n      <td>0.003653</td>\n      <td>NaN</td>\n    </tr>\n  </tbody>\n</table>\n<p>93 rows \u00d7 93 columns</p>\n</div>",
   "text/plain": "           feat_1    feat_2    feat_3        feat_4    feat_5    feat_6  \\\nfeat_1        NaN       NaN       NaN           NaN       NaN       NaN   \nfeat_2   0.031332       NaN       NaN           NaN       NaN       NaN   \nfeat_3   0.027807  0.082573       NaN           NaN       NaN       NaN   \nfeat_4   0.027529  0.134987  0.583523           NaN       NaN       NaN   \nfeat_5   0.042973  0.020926  0.010880  1.729026e-02       NaN       NaN   \nfeat_6   0.043603  0.041343  0.004288  1.405895e-02  0.145355       NaN   \nfeat_7   0.298952  0.222386  0.001294  1.448981e-02  0.075047  0.088014   \nfeat_8   0.056321  0.019815  0.053462  4.618407e-02  0.035861  0.012867   \nfeat_9   0.032285  0.025630  0.063551  4.624977e-02  0.024708  0.009373   \nfeat_10  0.097776  0.051925  0.036944  5.951396e-02  0.091324  0.041940   \nfeat_11  0.042928  0.118534  0.596243  3.894092e-01  0.004882  0.014504   \nfeat_12  0.056934  0.090153  0.050037  5.743356e-02  0.036668  0.028588   \nfeat_13  0.139254  0.157467  0.013870  2.897317e-02  0.059081  0.036293   \nfeat_14  0.063517  0.070057  0.111105  9.921490e-02  0.037607  0.027350   \nfeat_15  0.045738  0.048798  0.065285  5.122155e-02  0.007000  0.018328   \nfeat_16  0.027086  0.108046  0.221426  2.110780e-01  0.062877  0.021934   \nfeat_17  0.053004  0.074902  0.023093  7.553867e-03  0.062197  0.015488   \nfeat_18  0.084856  0.242716  0.115655  2.148952e-01  0.052186  0.048710   \nfeat_19  0.002302  0.176655  0.012228  3.519107e-07  0.008556  0.038493   \nfeat_20  0.070511  0.449160  0.011069  4.465657e-02  0.046200  0.057813   \nfeat_21  0.027026  0.014113  0.354925  2.329227e-01  0.003288  0.008046   \nfeat_22  0.063283  0.215106  0.251082  2.477378e-01  0.075161  0.038939   \nfeat_23  0.048686  0.162065  0.002427  3.062225e-02  0.017281  0.043651   \nfeat_24  0.067255  0.253684  0.031596  3.727726e-03  0.075222  0.082124   \nfeat_25  0.187237  0.096366  0.157459  1.342306e-01  0.003610  0.023319   \nfeat_26  0.022813  0.064856  0.268112  3.657567e-01  0.025116  0.004680   \nfeat_27  0.038826  0.037841  0.508370  3.086287e-01  0.002098  0.001943   \nfeat_28  0.030257  0.072494  0.551398  4.864171e-01  0.047688  0.017132   \nfeat_29  0.069266  0.025689  0.004141  1.427066e-02  0.065957  0.002389   \nfeat_30  0.033108  0.026896  0.007667  8.733991e-04  0.318117  0.196493   \n...           ...       ...       ...           ...       ...       ...   \nfeat_64  0.010499  0.005354  0.065105  4.728813e-02  0.021017  0.002764   \nfeat_65  0.110041  0.078801  0.065492  6.228472e-02  0.228349  0.066867   \nfeat_66  0.053010  0.175620  0.088017  1.296545e-01  0.048364  0.033285   \nfeat_67  0.154301  0.068667  0.110081  8.045694e-02  0.061964  0.038289   \nfeat_68  0.014674  0.012802  0.030992  2.009191e-02  0.107405  0.021619   \nfeat_69  0.007544  0.307406  0.032748  1.446082e-02  0.003294  0.074836   \nfeat_70  0.165442  0.112968  0.018774  2.079779e-02  0.118510  0.052401   \nfeat_71  0.013712  0.002336  0.053020  4.241268e-02  0.056428  0.011901   \nfeat_72  0.029983  0.023267  0.045339  2.979578e-02  0.005177  0.011090   \nfeat_73  0.140815  0.039192  0.013972  1.128547e-02  0.001609  0.025023   \nfeat_74  0.051365  0.070724  0.041559  4.909735e-02  0.017265  0.043160   \nfeat_75  0.011596  0.093689  0.044724  3.145389e-02  0.015279  0.006951   \nfeat_76  0.153808  0.259360  0.028670  1.379188e-02  0.035570  0.073867   \nfeat_77  0.123752  0.014911  0.001584  1.531773e-02  0.030462  0.006501   \nfeat_78  0.279202  0.094256  0.021979  1.449856e-02  0.070709  0.061250   \nfeat_79  0.228912  0.033668  0.020566  1.083473e-02  0.055115  0.009942   \nfeat_80  0.013303  0.155768  0.442036  4.057725e-01  0.026223  0.017648   \nfeat_81  0.032427  0.052101  0.013089  2.828377e-02  0.129333  0.044136   \nfeat_82  0.026085  0.119109  0.438458  4.365413e-01  0.057400  0.014907   \nfeat_83  0.059165  0.371691  0.019914  1.051874e-03  0.008006  0.035145   \nfeat_84  0.049634  0.009845  0.011159  5.684499e-03  0.467329  0.177777   \nfeat_85  0.008739  0.006764  0.048626  3.315343e-02  0.034062  0.004290   \nfeat_86  0.107947  0.039090  0.096093  7.102916e-02  0.013879  0.010455   \nfeat_87  0.089374  0.047451  0.009838  5.054728e-03  0.013999  0.015256   \nfeat_88  0.020830  0.047035  0.082336  6.748367e-02  0.019201  0.015437   \nfeat_89  0.096851  0.105527  0.174781  1.837145e-01  0.119951  0.035042   \nfeat_90  0.010310  0.515022  0.015068  9.454061e-03  0.004842  0.054034   \nfeat_91  0.037264  0.026383  0.012417  1.031241e-02  0.012012  0.012465   \nfeat_92  0.054777  0.008219  0.066921  8.763105e-02  0.065331  0.015479   \nfeat_93  0.081783  0.054593  0.006814  1.574563e-02  0.002038  0.008521   \n\n           feat_7    feat_8    feat_9   feat_10   ...      feat_84   feat_85  \\\nfeat_1        NaN       NaN       NaN       NaN   ...          NaN       NaN   \nfeat_2        NaN       NaN       NaN       NaN   ...          NaN       NaN   \nfeat_3        NaN       NaN       NaN       NaN   ...          NaN       NaN   \nfeat_4        NaN       NaN       NaN       NaN   ...          NaN       NaN   \nfeat_5        NaN       NaN       NaN       NaN   ...          NaN       NaN   \nfeat_6        NaN       NaN       NaN       NaN   ...          NaN       NaN   \nfeat_7        NaN       NaN       NaN       NaN   ...          NaN       NaN   \nfeat_8   0.038121       NaN       NaN       NaN   ...          NaN       NaN   \nfeat_9   0.027146  0.039281       NaN       NaN   ...          NaN       NaN   \nfeat_10  0.194258  0.000023  0.024323       NaN   ...          NaN       NaN   \nfeat_11  0.012418  0.065923  0.075820  0.006010   ...          NaN       NaN   \nfeat_12  0.056230  0.091424  0.021885  0.048969   ...          NaN       NaN   \nfeat_13  0.199142  0.095365  0.040164  0.086682   ...          NaN       NaN   \nfeat_14  0.044671  0.061799  0.110188  0.029598   ...          NaN       NaN   \nfeat_15  0.035721  0.056960  0.009858  0.021700   ...          NaN       NaN   \nfeat_16  0.043957  0.004659  0.082664  0.063997   ...          NaN       NaN   \nfeat_17  0.127245  0.173912  0.028709  0.092959   ...          NaN       NaN   \nfeat_18  0.098972  0.087777  0.043642  0.071635   ...          NaN       NaN   \nfeat_19  0.058071  0.019387  0.000167  0.009015   ...          NaN       NaN   \nfeat_20  0.364972  0.062595  0.023397  0.176373   ...          NaN       NaN   \nfeat_21  0.022908  0.041095  0.028409  0.005134   ...          NaN       NaN   \nfeat_22  0.162620  0.029032  0.062348  0.141405   ...          NaN       NaN   \nfeat_23  0.186462  0.012774  0.006940  0.096666   ...          NaN       NaN   \nfeat_24  0.244813  0.161848  0.073618  0.081684   ...          NaN       NaN   \nfeat_25  0.048820  0.036939  0.025279  0.009792   ...          NaN       NaN   \nfeat_26  0.008782  0.041599  0.066414  0.003721   ...          NaN       NaN   \nfeat_27  0.015429  0.050272  0.042531  0.001551   ...          NaN       NaN   \nfeat_28  0.000998  0.036668  0.055545  0.022349   ...          NaN       NaN   \nfeat_29  0.046231  0.104985  0.021328  0.068243   ...          NaN       NaN   \nfeat_30  0.050535  0.009574  0.015830  0.012623   ...          NaN       NaN   \n...           ...       ...       ...       ...   ...          ...       ...   \nfeat_64  0.011165  0.003194  0.702951  0.022536   ...          NaN       NaN   \nfeat_65  0.202346  0.025544  0.038163  0.182756   ...          NaN       NaN   \nfeat_66  0.122660  0.115175  0.001778  0.100722   ...          NaN       NaN   \nfeat_67  0.148598  0.320949  0.176921  0.043117   ...          NaN       NaN   \nfeat_68  0.040309  0.075384  0.012192  0.001693   ...          NaN       NaN   \nfeat_69  0.131430  0.046258  0.029335  0.077354   ...          NaN       NaN   \nfeat_70  0.237907  0.023089  0.056205  0.322857   ...          NaN       NaN   \nfeat_71  0.115813  0.081664  0.043286  0.104834   ...          NaN       NaN   \nfeat_72  0.014921  0.029868  0.058147  0.004225   ...          NaN       NaN   \nfeat_73  0.022819  0.028999  0.022679  0.000240   ...          NaN       NaN   \nfeat_74  0.053059  0.000431  0.007594  0.008912   ...          NaN       NaN   \nfeat_75  0.039865  0.031466  0.027313  0.003828   ...          NaN       NaN   \nfeat_76  0.375114  0.081682  0.027424  0.106752   ...          NaN       NaN   \nfeat_77  0.005769  0.027486  0.020185  0.019069   ...          NaN       NaN   \nfeat_78  0.567084  0.079623  0.015922  0.091760   ...          NaN       NaN   \nfeat_79  0.066753  0.083714  0.036116  0.113659   ...          NaN       NaN   \nfeat_80  0.028860  0.038382  0.046721  0.019042   ...          NaN       NaN   \nfeat_81  0.144308  0.035102  0.005847  0.135928   ...          NaN       NaN   \nfeat_82  0.022059  0.034409  0.039806  0.029741   ...          NaN       NaN   \nfeat_83  0.282069  0.033479  0.032875  0.052025   ...          NaN       NaN   \nfeat_84  0.062634  0.005064  0.013569  0.017939   ...          NaN       NaN   \nfeat_85  0.037874  0.003416  0.031462  0.086758   ...     0.010210       NaN   \nfeat_86  0.009169  0.029395  0.019144  0.159447   ...     0.003459  0.109643   \nfeat_87  0.089574  0.059929  0.016925  0.077421   ...     0.013631  0.049250   \nfeat_88  0.033646  0.050931  0.001160  0.054635   ...     0.017903  0.027886   \nfeat_89  0.063511  0.007974  0.019147  0.061498   ...     0.103643  0.053582   \nfeat_90  0.129578  0.026807  0.020698  0.049908   ...     0.006013  0.003931   \nfeat_91  0.068506  0.095990  0.014742  0.024025   ...     0.003444  0.023091   \nfeat_92  0.032261  0.013608  0.069707  0.006869   ...     0.048431  0.043484   \nfeat_93  0.034912  0.005131  0.006038  0.041316   ...     0.003723  0.023390   \n\n          feat_86   feat_87   feat_88   feat_89   feat_90   feat_91   feat_92  \\\nfeat_1        NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_2        NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_3        NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_4        NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_5        NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_6        NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_7        NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_8        NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_9        NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_10       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_11       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_12       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_13       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_14       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_15       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_16       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_17       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_18       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_19       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_20       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_21       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_22       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_23       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_24       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_25       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_26       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_27       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_28       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_29       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_30       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \n...           ...       ...       ...       ...       ...       ...       ...   \nfeat_64       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_65       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_66       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_67       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_68       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_69       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_70       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_71       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_72       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_73       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_74       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_75       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_76       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_77       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_78       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_79       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_80       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_81       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_82       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_83       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_84       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_85       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_86       NaN       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_87  0.073685       NaN       NaN       NaN       NaN       NaN       NaN   \nfeat_88  0.426972  0.023053       NaN       NaN       NaN       NaN       NaN   \nfeat_89  0.011822  0.066008  0.022552       NaN       NaN       NaN       NaN   \nfeat_90  0.019803  0.014696  0.031679  0.027764       NaN       NaN       NaN   \nfeat_91  0.024005  0.028850  0.033653  0.015917  0.014812       NaN       NaN   \nfeat_92  0.049393  0.001424  0.070120  0.129622  0.035311  0.104226       NaN   \nfeat_93  0.029035  0.499990  0.008631  0.030650  0.039864  0.000045  0.003653   \n\n         feat_93  \nfeat_1       NaN  \nfeat_2       NaN  \nfeat_3       NaN  \nfeat_4       NaN  \nfeat_5       NaN  \nfeat_6       NaN  \nfeat_7       NaN  \nfeat_8       NaN  \nfeat_9       NaN  \nfeat_10      NaN  \nfeat_11      NaN  \nfeat_12      NaN  \nfeat_13      NaN  \nfeat_14      NaN  \nfeat_15      NaN  \nfeat_16      NaN  \nfeat_17      NaN  \nfeat_18      NaN  \nfeat_19      NaN  \nfeat_20      NaN  \nfeat_21      NaN  \nfeat_22      NaN  \nfeat_23      NaN  \nfeat_24      NaN  \nfeat_25      NaN  \nfeat_26      NaN  \nfeat_27      NaN  \nfeat_28      NaN  \nfeat_29      NaN  \nfeat_30      NaN  \n...          ...  \nfeat_64      NaN  \nfeat_65      NaN  \nfeat_66      NaN  \nfeat_67      NaN  \nfeat_68      NaN  \nfeat_69      NaN  \nfeat_70      NaN  \nfeat_71      NaN  \nfeat_72      NaN  \nfeat_73      NaN  \nfeat_74      NaN  \nfeat_75      NaN  \nfeat_76      NaN  \nfeat_77      NaN  \nfeat_78      NaN  \nfeat_79      NaN  \nfeat_80      NaN  \nfeat_81      NaN  \nfeat_82      NaN  \nfeat_83      NaN  \nfeat_84      NaN  \nfeat_85      NaN  \nfeat_86      NaN  \nfeat_87      NaN  \nfeat_88      NaN  \nfeat_89      NaN  \nfeat_90      NaN  \nfeat_91      NaN  \nfeat_92      NaN  \nfeat_93      NaN  \n\n[93 rows x 93 columns]"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Filtrando colunas

A partir da matriz de correlação assima, buscamos agora
identificar quais das colunas possuem uma forte correlação de acordo com a
tabela a seguir.
Como sugerido por
[Makuka,2012](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3576830/)
<center>Interpretação do resultado de correlação </center>

|Valor
absoluto|Significado|
|---|---|
|0.9 < v | Muito forte |
|0.7 < v <= 0.9 | Forte
|
|0.5 < v <= 0.7 | Moderada |
|0.3 < v <= 0.5 | Fraca |
|0.0 < v <= 0.3 |
Desprezível |

```{.python .input  n=5}
strong_correlation = correlation.where(correlation > 0.8)
strong_correlation = strong_correlation.dropna(how='all', axis=(0,1))
corr_features = strong_correlation[strong_correlation.notnull()].stack().index
corr_features_size = len(corr_features)
if corr_features_size:
    col = math.floor(math.log2(corr_features_size)) or 1
    row = math.ceil(corr_features_size/col)
    figure, axis = plt.subplots(row, col, figsize=[15,2*row])
    figure.tight_layout()
    for idx, (feature1, feature2) in enumerate(corr_features):
        if row == 1: # Has a single element
            plot = axis.scatter(df_train[feature1],df_train[feature2])
            plot = axis.set_xlabel(feature1)
            plot = axis.set_ylabel(feature2)
            plot = axis.annotate(strong_correlation[feature2][feature1],xy=(0,0))
        elif col == 1: # Has multiples elements, but is a array
            plot = axis[idx].scatter(df_train[feature1], df_train[feature2])
            plot = axis[idx].set_xlabel(feature1)
            plot = axis[idx].set_ylabel(feature2)
            plot = axis[idx].annotate(strong_correlation[feature2][feature1],xy=(0,0))
        else: # Multitle elements and is a matrix
            plot = axis[int(idx/col), idx%col].scatter(df_train[feature1], df_train[feature2])
            plot = axis[int(idx/col), idx%col].set_xlabel(feature1)
            plot = axis[int(idx/col), idx%col].set_ylabel(feature2)
            plot = axis[int(idx/col), idx%col].annotate(strong_correlation[feature2][feature1],xy=(0,0))
    plt.show()
```

```{.json .output n=5}
[
 {
  "data": {
   "image/png": "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\nRU5ErkJggg==\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7fd93b656e48>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

## Resultado

A correlação mostra que não há uma fortíssima correlação entre as
features, entretanto, há 10 colunas que estão fortemente correlacionadas. Porem
buscamos uma correlação fortíssima para não remover features com comportamentos
diferentes.

# Train/Test split

Utilizaremos 80% da base de treino para
efetivamente treinar o modelo e 20% para
averiguar a performance do modelo.

```{.python .input  n=6}
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

X = df_train
y = df_target.categories
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Feature Scaling

Trata-se do processo de transformar todos os dados da
amostra para uma unidade padrão, neste problema utilizaremos a técnica de
padronização que consiste em remover a média dos dados e colocar todos na escala
do desvio padrão [Wikipedia](https://en.wikipedia.org/wiki/Feature_scaling). Em
que $\bar{x}$ é a média e $\sigma$ é o desvio padrão.

\begin{equation}
    x' =
\frac{x - \bar{x}}{\sigma}
\end{equation}

```{.python .input  n=7}
sc_X = StandardScaler()
sc_X_train = sc_X.fit_transform(X_train)
sc_X_test = sc_X.transform(X_test)
```

Feature scaling foi aplicado nos dataframes de **features** e utilizado nos
modelos, mas o resultado não apresentou mudança. Os modelos continuaram com
exatamente as mesmas performances.

### Confusion Matrix

A matriz de confução é
uma métrica para algorítmos supervisionados em que é possível estabelecer uma
relação entre os acertos e erros durante a classificação do conjunto de
amostras. Basicamente elabora-se uma matriz em que nas colunas e linhas são as
possíveis classes. Cada célula traz a contagem de amostras que eram da Label X
(coluna) e foram classificadas na Label Y (linha). Dessa forma, na matriz, a
diagonal principal trará os acertos do classificador
[Microsoft](https://docs.microsoft.com/pt-br/sql/analysis-services/data-
mining/classification-matrix-analysis-services-data-mining). Veja o exemplo a
seguir:

|Classificador\Real|Label 1|Label 2|Label 3|
|---|-------|-------|-------|
|**Label 1**|10|10|0|
|**Label 2**|1|10|1|
|**Label 3**|0|0|3|

Plot para matriz de confusão encontrado em
[Scikit](http://scikit-
learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-
glr-auto-examples-model-selection-plot-confusion-matrix-py) e adaptado para o
problema

```{.python .input  n=8}
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(11, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
```

## Modelo Dummy Classifier

Dummy Classifier é um modelo que faz predições
usando regras simples.

O dummy é importante para termos como parâmetro de
comparação com outros modelos.Não pode ser utilizado em problemas reais porque
ele é apenas para realizar comparações e trabalha com aleatoriedade e frequencia
de repetições para realizar as predições.


Usamos dois tipos de estratégia:

*
**Stratified**: realiza predições baseadas na
distribuição das classes da base
de treino. (Ex.: 10% A, 20% B, 50% C, 20% D)
* **Most Frequent**: sempre prediz
com a classe mais frequente na base de treino

```{.python .input  n=9}
from sklearn.dummy import DummyClassifier

def dummies(X_train, y_train, X_test, y_test):
    models = ['most_frequent', 'stratified']

    for model in models:
        clf = DummyClassifier(strategy=model)
        clf.fit(X_train, y_train)
        score = clf.score(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plot_confusion_matrix(cm, classes=model)
        # Cross validation
        accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
        
        add_results(model, clf.score(X_train, y_train), clf.score(X_test, y_test))
        print(model, 'train dataset score: %.2f' % score)
        print('Média: %.2f' % accuracies.mean())
        print('Desvio padrão: %.4f' % accuracies.std())

dummies(X_train, y_train, X_test, y_test)
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Confusion matrix, without normalization\n"
 },
 {
  "data": {
   "image/png": "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\ncUqxAAAAAElFTkSuQmCC\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7fd93a926470>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "most_frequent train dataset score: 0.26\nM\u00e9dia: 0.26\nDesvio padr\u00e3o: 0.0001\nConfusion matrix, without normalization\n"
 },
 {
  "data": {
   "image/png": "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\nB9gdEALGoqoAAAAASUVORK5CYII=\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7fd94dd9f8d0>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "stratified train dataset score: 0.17\nM\u00e9dia: 0.17\nDesvio padr\u00e3o: 0.0038\n"
 }
]
```

## Boosting

A definição de boosting é que até mesmo algorítmos
fracos de
machine larning podem se tornar potentes [(KEARNS,
1988)](https://www.cis.upenn.edu/~mkearns/papers/boostnote.pdf).

Um algorítmo
fraco de aprendizagem pode ser definido como modelos ou regras que não possuem
boa acurácia ou aparentam ser ineficientes, tais como modelos *dummy*: mais
frequente, estratificado, randômico. Já algorítmos de aprendizagem forte, são
aqueles que apresentam uma boa taxa de acertos [(FREUND e
SCHAPIRE)](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=4BF3325D8222
B3234BB95971FCAD8759?doi=10.1.1.56.9855&rep=rep1&type=pdf).
**Exemplo - Corrida de cavalos**[(FREUND e
SCHAPIRE)](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=4BF3325D8222
B3234BB95971FCAD8759?doi=10.1.1.56.9855&rep=rep1&type=pdf):
Como determinar em qual cavalor apostar, considerando um conjunto de dados
disponíveis tais como informações do cavalo, do dono, das corridas anteriores e
etc. Ao perguntar para especialistas cada um deles irá falar coisas distintas e
ainda assim muito imprecisas (modelos fracos)! Mas seria possível utilizar as
regras de aposta de cada especialista e gerar uma única regra que seja capaz de
predizer o cavalor vencedor da corrida utilizando boost

## Gradient
Descent

![](http://matthewemery.ca/images/gradient_descent.gif)


Um algorítmo
de gradient descendent é uma forma de minimizar o valor de uma função
interativamente, na qual são dados um conjunto de parametros e ela busca a
partir daí o menor valor[(TOUSSAINT, 2012)](https://ipvs.informatik.uni-
stuttgart.de/mlr/marc/notes/gradientDescent.pdf). De forma que:
\begin{equation}
y_{min} = F(x_1) > F(x_2) > F(x_3) > ... > F(x_n),\ onde:\
F(x_n) < precisão
\end{equation}

Um pseudo algorítmo que pode ser proposto para
um problema de
gradient é:


# XGBoost

XGBoost é um algoritmo que implementa
*gradient
boosting* de
Decision Trees de
forma rápida e com alta performance.
**Gradient
Boosting** é
uma técnica de *machine learning* para problemas de
regressão e
classificação
que produz um modelo de predição na forma de
*ensemble* de modelos
de predições
fracas, normalmente árvores de decisões.
Boosting é um processo
sequencial, mas
como o `XGBoost` consegue implementá-lo
de forma paralela?
Sabemos que cada
árvore pode ser produzida apenas depois que
produzida a árvore
anterior, mas o
processo de criar as árvores pode ser
paralelizado utilizando
todos os núcleos a
disposição.

## Model
### Objective Function:
\begin{equation}
\text{obj}(\theta) = L(\theta)
+ \Omega(\theta)
\end{equation}
**L- Training Loss function**: Mede predição do
modelo na base de treino.
(Métrica: *Mean Squared Error*(MSE))
**Omega-
Regularization function **:
Controla a complexidade do modelo (Ajuda a evitar o
*Overfitting*)

nota: As
*objective functions* devem sempre possuir *training
loss* e *regularization*
![](https://raw.githubusercontent.com/dmlc/web-
data/master/xgboost/model/step_fit.png)

### CART
Uso de *CARTs* (Classification
And Regression Trees) no ensemble das árvores
![](https://raw.githubusercontent.com/dmlc/web-
data/master/xgboost/model/twocart.png)


Modelo de ensemble de árvores IGUAL ao
modelo Random Forest, mas onde está então a diferença?

## Training
### Additive
Training:

Precisamos agora melhorar os paramêtros da função de
**Regularization**, mas como fazer isso? Fazer isso aqui é muito mais difícil do
que em problemas de otimização tradicionais, onde você pode usar o gradiente
para isso. Não é fácil treinar todas as árvores ao mesmo tempo. Em vez disso,
usamos uma **estratégia aditiva**: consertamos o que aprendemos e adicionamos
uma nova árvore de cada vez.


\begin{split}\hat{y}_i^{(0)} &= 0\\
\hat{y}_i^{(1)} &= f_1(x_i) = \hat{y}_i^{(0)} + f_1(x_i)\\
\hat{y}_i^{(2)} &=
f_1(x_i) + f_2(x_i)= \hat{y}_i^{(1)} + f_2(x_i)\\
&\dots\\
\hat{y}_i^{(t)} &=
\sum_{k=1}^t f_k(x_i)= \hat{y}_i^{(t-1)} + f_t(x_i)
\end{split}

```{.python .input  n=10}
x = inital_value
step = 0.01

repita
xprev=x
        x = xperv - step * F(xprev)
    enquanto abs(x - xprev)
>
precisao
```

```{.json .output n=10}
[
 {
  "ename": "IndentationError",
  "evalue": "unexpected indent (<ipython-input-10-c356a565a567>, line 6)",
  "output_type": "error",
  "traceback": [
   "\u001b[0;36m  File \u001b[0;32m\"<ipython-input-10-c356a565a567>\"\u001b[0;36m, line \u001b[0;32m6\u001b[0m\n\u001b[0;31m    x = xperv - step * F(xprev)\u001b[0m\n\u001b[0m    ^\u001b[0m\n\u001b[0;31mIndentationError\u001b[0m\u001b[0;31m:\u001b[0m unexpected indent\n"
  ]
 }
]
```

```{.python .input}
%%time
from xgboost import XGBClassifier

def xgboost(X_train, y_train, X_test, y_test):    
    xgbclf = XGBClassifier(
        learning_rate=0.01,
        n_estimators=140,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        nthread=8,
        scale_pos_weight=1
        )
    print('XGBoost fit')
    xgbclf.fit(X_train, y_train)
    print('XGBoost train score')
    train_score = xgbclf.score(X_train, y_train)
    print('XGBoost test score')
    y_pred = xgbclf.predict(X_test)

    print('XGBoost confusion matrix')
    cm = confusion_matrix(y_test, y_pred)

    print('XGBoost cross validation')
    accuracies = cross_val_score(estimator=xgbclf, X=X_train, y=y_train, cv=10)
    
    print('XGBoost results')
    add_results('xgboost', xgbclf.score(X_train, y_train), xgbclf.score(X_test, y_test))
    
    plot_confusion_matrix(cm, classes=xgbclf)
    print('Resultado na base de treino %.2f' % train_score)
    print('Resultado Médio na base de teste: %.2f' % accuracies.mean())
    print('Desvio padrão: %.4f' % accuracies.std())
    

xgboost(X_train, y_train, X_test, y_test)
```

## GridSearchCV
A ferramenta GridSearch disponibilizada pelo Scikit, gera de
forma exaustiva candidatos a partir de um grid de  parâmetros especificados com
o atributo param_grid.

```{.python .input}
dt_params = [{
    'max_depth': [40, 50, 60, 80, 100, 120],
    'max_features': [70, 80, 90, 92],
    'min_samples_leaf': [2, 5, 10, 20, 30, 40]
}]

xgb_params = [{
    'max_depth': [4, 5, 6],
    'min_child_weight': [4, 5, 6]
}]

mlp_params = [{
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'hidden_layer_sizes': [10, 20, 30, 40, 50, 60, 70]
}]
```

```{.python .input}
%%time
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def search_params(classifier, params):
    clf = classifier()
    grid_search = GridSearchCV(estimator=clf,
                              param_grid=params,
                              cv = 10,
                              n_jobs=-1)

    grid_search = grid_search.fit(X_train, y_train)
    print(grid_search.best_score_, grid_search.best_params_)
    return grid_search.best_score_
```

### Aplicando GridSearchCV ao XGBClassifier:

```{.python .input}
%%time
from xgboost import XGBClassifier

# Takes long time to run
search_params(XGBClassifier, xgb_params)
```

Aplicando GridSearchCV ao Decision Tree Classifier:

```{.python .input}
search_params(DecisionTreeClassifier, dt_params)
```

## Decision Tree

Os dados são separados recursivamente formando uma árvore de
decisão baseada nas
features.Pode-se definir uma árvore de decisão, conforme diz
(MITCHELL, 1997),
como um método para aproximar valores discretos em funções,
onde a função de
aprendizagem é representada por uma árvore de decisão. Tais
árvores aprendidas
podem ser representadas - a nível de código fonte - como
conjuntos de estruturas
condicionais "se-então" para melhorar a leitura e
entendimento humano, de acordo
com (MITCHELL, 1997).

Estes algoritmos são muito
utilizados, segundo (MITCHELL, 1997), na área de
algoritmos de inferência
indutiva, e dentre as aplicações de tais algoritmos,
tem-se máquinas que
aprenderam a diagnosticar casos da medicina, ou ainda, para
avaliar o risco de
inadimplência dos requerentes de créditos em bancos.

Para visualizar de forma
mais fácil a representação de uma árvore, a figura 3,
representada abaixo,
caracteriza uma árvore de decisão em que a máquina deve
decidir com base nas
variáveis do tempo (ensolarado, nublado ou chuvoso), se
pode ou não ocorrer uma
partida de tênis. Além das variáveis de tempo, tem-se
outras variáveis que podem
ser levadas em conta dependendo da condição climática local, como umidade (alta
ou normal) e o vento (forte ou fraco).

![Workflow Random
forest](arvore_jogo_tenis.png)

O algoritmo de árvores de decisão classifica
instâncias ou dados, ordenando-os
apartir da raiz da árvore, para os nós de suas
folhas. Cada nó da árvore
exemplifica uma pergunta (teste) de alguns - atributos
- de instância, e cada
ramo descendente de um nó corresponde para um dos
possíveis valores de tal
atributo (MITCHELL, 1997). Vale a pena citar: O
algoritmo ID3 (QUINLAN, 1986)
aprende sobre árvores de decisão construindo-as de
cima para baixo (nó raiz para
as ramificações) tentando buscar respostas para a
pergunta "Qual atributo
devemos testar na raiz da árvore?", sendo assim, cada
atributo instanciado é
calculado por meio de testes estatísticos, para
determinar o quão bem (ótimo)
tal atributo, isolado dos demais, classifica os
exemplos de treinamento.

Quando o melhor atributo é selecionado e utilizado
como teste no nó principal da
árvore, cria-se um descendente para cada valor
admissível deste atributo e os
exemplos de treinamento são sorteados para o nó
filho mais apropriado. O
processo inteiro é então repetido utilizando
treinamentos associados a cada
descendente para selecionar o melhor atributo
para testar na árvore. Quando
realizado dessa forma, o algoritmo tenta de forma
“gulosa“3.4. O modelo 49
Figura 3 – Exemplo de árvore de decisão, sobre
condições para realização de um
jogo de
tênis.

```{.python .input}
from sklearn.model_selection import cross_val_score

def fit_tree(X_train, y_train, X_test, y_test, tree_description='decision_tree'):
    tree_clf = DecisionTreeClassifier(max_features=70, min_samples_leaf=10, max_depth=40)
    tree_clf.fit(X_train, y_train)

    inner_score = tree_clf.score(X_train, y_train)
    tree_fit = cross_val_score(tree_clf, X_train, y_train)
    
    add_results(tree_description, tree_clf.score(X_train, y_train), tree_clf.score(X_test, y_test))
    
    return inner_score, tree_fit.mean(), tree_fit.std()

"inner: {:.2f} cross: {:.2f} +/- {:.2f}".format(*fit_tree(X_train, y_train, X_test, y_test))
```

## Distribuição dos dados

Um dos modelos a ser utilizado será o decision tree
no método de montagem random forest. Este modelo de predição possui um problema
de viés quando uma das classes na base de treino é mais predominante do que
outra, ou seja, a distribuição das classes na base de treino devem ser
semelhantes para evitar problemas de
[overfiting](http://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-
underfitting-vs-overfitting.html).

Para tanto, precisa-se descobrir qual a
contagem de cada classe disponível na base de treino, montaremos um histograma
para verificar a diferença entre elas.

```{.python .input}
counts = [0] *len(df_target.target.cat.categories)

def reduce(target):
    counts[target.categories] += 1
    return counts[target.categories]

df_target['increase_count'] = df_target.apply(reduce, axis=1)
df_target.groupby('target').count()
df_target.groupby('target')['increase_count'].max().sum() == df_target.target.count()
```

### Filtrar dados

Agora, iremos filtrar os dados deixando apenas os primeiros
registros. O critério de filtrar os dados será pegar a classe que possue o menor
número e utilizar ele como base para remover os demais, considerando um tamanho
máximo de até 2x o da menor classe

```{.python .input}
distance_percent = 2
minimum_value = df_target.groupby('target')['increase_count'].max().min()
df_rtarget = df_target[ df_target.increase_count < minimum_value*distance_percent ]
df_rtarget.groupby('target').count()
df_rtrain = df_train.drop( df_target[df_target.increase_count >= minimum_value * distance_percent].index )
df_rtrain.shape[0] == df_rtarget.shape[0]
```

### Verificando resultado

Após aplicar uma melhor distribuição nos dados,
rodou-se novamene o algorítmo da decision tree e percebeu-se que a acurácia do
modelo diminuiu, e portanto, não será utilizado.

```{.python .input}
X_tr, X_te, y_tr, y_te = train_test_split(df_rtrain, df_rtarget.target, test_size=0.2)
"inner: {:.2f} cross: {:.2f} +/- {:.2f}".format(*fit_tree(X_tr, y_tr, X_te, y_te))
```

# Random Forest

Breiman breiman, 2001, descreve Random Forests como uma
evolução das decisions
trees, onde várias ávores são formadas para criar um
modelo com maior precisão.
Isto é feito a partir da separação dos Dados em
conjutos
de dados menores e aleatórios. Cada árvore é contruida a partir de um
pedaço
aleatório dos dados. Quando um novo dado chega, a predição é feita por
todas as
Árvores e ao fim é feita uma
votação por maioria, ou seja, a categoria
com mais votos ganha e o resultado é
dado.

![Workflow Randomforest](forest.jpg)
De acordo com breiman, 2001, as RFs corrigem a maior parte
dos problemas de
Overfitting que as Árvores de decisão apresentam. Tudo depende
do quanto as DT
contidas dentro da Random Forest. Isto é, o quanto elas
representam os dados.
##
Utilizando o algoritmo

```{.python .input}
### %%time

from sklearn.ensemble import RandomForestClassifier

def test_random(params, X_train, y_train, X_test, y_test, name='random_forest'):
    rfclf = RandomForestClassifier(**params)
    rfclf = rfclf.fit(X_train, y_train)
    
    train_score = rfclf.score(X_train, y_train)
    test_score = rfclf.score(X_test, y_test)

    add_results(name, train_score, test_score)
    return name, train_score, test_score
params = {'n_estimators': 10, 'max_features': 70, 'min_samples_leaf': 10, 'max_depth': 40}
test_random({}, X_train, y_train, X_test, y_test)
test_random(params, X_train, y_train, X_test, y_test, 'random_forest_otimized')
```

## Verificando com Cross Validation

Cross validation irá predizer um pedaço do
dataset utilizando o modelo treinado
com o resto dos dados que não fazem parte
deste dataset.

```{.python .input}
rfscores = cross_val_score(rfclf, X_train, y_train)
print ("{} de precisão".format(rfscores.mean() * 100))

```

## Importancia das features para a RF

A seguir vemos quais as influências de
cada uma das features para o uso no random forest. Quanto maior no gráfico,
maior é a importância da feature.

### Gini

O método utilizado para gerar a
importância das features no modelo é a Decrease Mean Importance, que utiliza em
seus cálculos um indicador de impureza no sistema. No caso do random forest
implementado [(LOUPPE et
al.,2013)](https://pdfs.semanticscholar.org/2635/19c5a43fbf981da5ba873062219c50f
df56d.pdf),
este indicador é o Gini Impurity que pode ser entendido como uma redução da
probabilidade de errar a classificação de uma categoria dentro de um algorítmo
de árvore [(Sebastian Raschaka)](https://sebastianraschka.com/faq/docs/decision-
tree-binary.html).

#### O indice
O indice de Gini pode ser calculado utilizando
a seguinte
fórmula[(TEKIMONO,2009)](http://people.revoledu.com/kardi/tutorial/DecisionTree/
how-
to-measure-impurity.htm):

\begin{equation}
    Gini = 1- \sum_{i=1} p_i^2
\end{equation}
Em que $p_i$ é a probabilidade da ocorrência de uma determinada
classe,
desconsiderando os atributos. Ou seja $N_i$ é o número de ocorrências da
classe
i e N é o total de elementos das classes:

\begin{equation}
    p_i =
\frac{N_i}{N}
\end{equation}

#### Para Decisions Trees

Para Classification and
Regression Trees (CART), utiliza-se o indice de Gini modificado, isto é,
calcula-se ainda as probabilidades em $p_i$, mas agora utiliza-se do indice de
Gini nos filhos da esquerda $t_l$ e direita $t_r$. Recalcula-se as
probabilidades para ambos os nós também em $p_l$ e $p_r$ utilizando como base as
possíveis classes reduzidas a $N_t$ [(LOUPPE et
al.,2013)](https://pdfs.semanticscholar.org/2635/19c5a43fbf981da5ba873062219c50f
df56d.pdf).
\begin{equation}
    i(s, t) = Gini(t) - p_l Gini(t_l) - p_r Gini(t_r) \\
p(t) =
\frac{N_{l|r}}{N_t}
\end{equation}

#### Decrease Mean Importance

Para
calcular
a importância de uma feature X ao tentar predizer uma label Y, utiliza-
se os
indices de impureza com a proporção de $N_f$ amostras em relação ao total
$N$.
$N_T$ é o total de árvores na floresta. Assim, para uma Random Forest a
conta é:
\begin{equation}
    I(X_m) = \frac{1}{N_T} \sum_{T} \sum_{t \epsilon
T:v(s)=X_m} pf(t)i(s,t) \\
    pf(f) = \frac{N_f}{N}
\end{equation}

```{.python .input}
fig, axis = plt.subplots(figsize=(15, 5))
plot = axis.bar(df_train.columns, rfclf.feature_importances_)
plot = axis.set_xticklabels(df_train.columns.values, rotation='vertical')
plot = axis.set_xlabel('feature')
plot = axis.set_ylabel('importance')
plt.show()
```

## ExtraTrees

O [Scikit Learn](http://scikit-
learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
nos apresenta um tipo diferente de random forest que pode apresentar resultados
melhores que o [RandomForestClassifier](http://scikit-
learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
.
Assim como afirma que as extra tree devem ser utilizadas apenas em algorítmos de
montagem Como o Extra Trees Classifier e Regressor.

O que diferencia uma extra
tree de uma decision tree é a forma que é feita a construção da árvore. Enquanto
uma decision tree utiliza cópia dos dados e sub amostras para realizar as
divisões de cada nó. Uma extra tree utiliza um ponto de divisão randomico e
utiliza toda a base de treino para crescer a árvore [(GEURTS, ERNST e WEHENKEL,
2005)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.65.7485&rep=rep1
&type=pdf).

```{.python .input}
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()
etscores = cross_val_score(etc, X_train, y_train)
print ("{} de precisão".format((etscores.mean() * 100)))

etc = etc.fit(X_train, y_train)
add_results('extra_trees', etc.score(X_train, y_train), etc.score(X_test, y_test))
print("Inner score", etc.score(X_train, y_train))
```

## Neurônio Artificial

![Workflow NeuralNetwork](neural.jpg)

#### Entrada
-
Sinais de entrada {x1,x2,...,xn}.
- Cada sinal de entrada e ponderado por 1
peso.{w1,w2,...,wn}.
- O peso é adquirido a partir do treino.

#### Função
agregadora
- Recebe todos os sinais e realiza a soma dos produtos dos sinais.
#### Neurônio
- Tem a função de deixar, passar ou inibir um sinal de saida de
acordo com a entrada.
- Teta é a limiar de ativacao(ponderado),'u' é o potencial
de ativação que é passado para a função (g(u)), que é a função de ativação que é
responsavel pela saida que permite o sinal passar ou não ou até mesmo
modificalo.

#### Formula

- Potencial de ativação

![Workflow
Potencialdeativacao](formula.png)

## Neural Networks
As Redes Neurais é uma estrutura de aprendizado de máquinas que tenta imitar o
padrão de aprendizado de redes neurais biológicas naturais. As redes neurais
biológicas têm neurônios interligados com dendritos que recebem entradas, então,
com base nessas entradas, eles produzem um sinal de saída através de um axônio
para outro neurônio. Vamos tentar imitar esse processo através do uso de Redes
Neurais Artificiais (ANN)

### O Perceptron
Um perceptron possui uma ou mais entradas, uma polarização, uma função de
ativação e uma única saída. O perceptron recebe entradas, as multiplica por
algum peso e as passa para uma função de ativação para produzir uma saída.
Existem muitas funções de ativação possíveis para escolher, como a função
lógica, uma função trigonométrica, uma função de etapa etc. Também nos
certificamos de adicionar um preconceito ao perceptron, isso evita problemas em
que todas as entradas podem ser iguais a zero (o que significa nenhum peso
multiplicativo teria efeito). Veja o diagrama abaixo para uma visualização de um
perceptron:

![Workflow NeuralNetwork](perceptron.jpg)

Uma vez que temos a saída, podemos compará-la com um rótulo conhecido e ajustar
os pesos de acordo (os pesos geralmente começam com valores de inicialização
aleatórios). Continuamos repetindo esse processo até atingir um número máximo de
iterações permitidas ou uma taxa de erro aceitável.

Para criar uma rede neural, simplesmente começamos a adicionar camadas de
perceptrons em conjunto, criando um modelo perceptron multicamada de uma rede
neural. Você terá uma camada de entrada que aceita diretamente suas entradas de
recursos e uma camada de saída que criará as saídas resultantes. Quaisquer
camadas intermediárias são conhecidas como camadas ocultas porque elas não
"vêem" diretamente as entradas ou saídas do recurso. Para uma visualização desta
verificação, veja o diagrama abaixo (fonte: Wikipedia).

![Workflow NeuralNetwork](ann-in-hidden-out.jpg)

### MLP Classifier
Esse algoritmo é um
classificador Perceptron de Multicamadas
usado para fazer o
treinamento de
modelos, e é uma biblioteca do Scikit-Learn.

```{.python .input  n=11}
%%time

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='adam',activation='relu',max_iter=250)
mlp.fit(X_train, y_train)
saidas = mlp.predict(X_test)
scoreTreino =  mlp.score(X_train, y_train)
scoreTeste =  mlp.score(X_test, y_test)

print('Score treino: ', scoreTreino)
print('Score teste: ', scoreTeste)

mlpscores = cross_val_score(mlp, X_train, y_train)

print('Score: {} +/- {}'.format(mlpscores.mean(), mlpscores.std()))

add_results('multi_layer_perceptron', scoreTreino, scoreTeste)
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Score treino:  0.869944648701\nScore teste:  0.791693600517\nScore: 0.7824534985494574 +/- 0.002153526936763193\nCPU times: user 5min 8s, sys: 3.32 s, total: 5min 11s\nWall time: 2min 35s\n"
 }
]
```

```{.python .input  n=12}
from sklearn.neural_network import MLPClassifier

mp = confusion_matrix(y_test,saidas);
plot_confusion_matrix(mp, classes=model)
```

```{.json .output n=12}
[
 {
  "ename": "NameError",
  "evalue": "name 'model' is not defined",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-12-160ae97e2776>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mmp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mconfusion_matrix\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0msaidas\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m;\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0mplot_confusion_matrix\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmp\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mclasses\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m: name 'model' is not defined"
  ]
 }
]
```

```{.python .input  n=13}
target_names = ['class 1','class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9',]
print(div,classification_report(y_test,saidas, target_names = target_names, sample_weight = mlp.coefs_))
```

```{.json .output n=13}
[
 {
  "ename": "NameError",
  "evalue": "name 'div' is not defined",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-13-d56f6483a463>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0mtarget_names\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m'class 1'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m'class 2'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m'class 3'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m'class 4'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m'class 5'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m'class 6'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m'class 7'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m'class 8'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m'class 9'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdiv\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mclassification_report\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0msaidas\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtarget_names\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtarget_names\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msample_weight\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmlp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcoefs_\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m: name 'div' is not defined"
  ]
 }
]
```

## Preprocessamento de dados

A rede neural pode ter dificuldade em convergir antes de atingir o número máximo
de iterações permitido se os dados não forem normalizados. Multi-layer
Perceptron é sensível ao dimensionamento de features, portanto, é altamente
recomendável dimensionar seus dados. Usaremos o StandardScaler incorporado para
padronização.

```{.python .input  n=15}
%%time
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "CPU times: user 52 ms, sys: 16 ms, total: 68 ms\nWall time: 63.2 ms\n"
 }
]
```

```{.python .input  n=16}
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

```{.python .input  n=17}
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
```

```{.python .input  n=18}
%%time
mlp.fit(X_train,y_train)
```

```{.json .output n=18}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "CPU times: user 1min 13s, sys: 860 ms, total: 1min 14s\nWall time: 37.1 s\n"
 },
 {
  "data": {
   "text/plain": "MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,\n       beta_2=0.999, early_stopping=False, epsilon=1e-08,\n       hidden_layer_sizes=(30, 30, 30), learning_rate='constant',\n       learning_rate_init=0.001, max_iter=200, momentum=0.9,\n       nesterovs_momentum=True, power_t=0.5, random_state=None,\n       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,\n       verbose=False, warm_start=False)"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=19}
predictions = mlp.predict(X_test)
```

```{.python .input  n=20}
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
```

```{.json .output n=20}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[ 184   12    1    2    0   23   25   67   79]\n [   3 2746  331   81    8   10   32    6    6]\n [   1  839  622   60    0    7   37    4    1]\n [   0  177   69  270    2   18    8    4    1]\n [   0   13    2    0  537    2    2    1    2]\n [  14   18    7   11    3 2605   46   51   47]\n [  12   51   32    7    4   42  389   31    2]\n [  37   22    1    1    4   57   26 1546   26]\n [  35   13    2    2    0   32    7   54  844]]\n"
 }
]
```

```{.python .input  n=21}
print(classification_report(y_test,predictions))
```

```{.json .output n=21}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "             precision    recall  f1-score   support\n\n          0       0.64      0.47      0.54       393\n          1       0.71      0.85      0.77      3223\n          2       0.58      0.40      0.47      1571\n          3       0.62      0.49      0.55       549\n          4       0.96      0.96      0.96       559\n          5       0.93      0.93      0.93      2802\n          6       0.68      0.68      0.68       570\n          7       0.88      0.90      0.89      1720\n          8       0.84      0.85      0.85       989\n\navg / total       0.78      0.79      0.78     12376\n\n"
 }
]
```

coefs_ é uma lista de matrizes de peso, onde a matriz de peso no índice i
representa os pesos entre a camada i e a camada i + 1.

intercepts_ é uma lista de vetores de polarização, onde o vetor no índice i
representa os valores de polarização adicionados à camada i + 1.

```{.python .input}
len(mlp.coefs_)
```

```{.python .input}
len(mlp.coefs_[0])
```

```{.python .input}
len(mlp.intercepts_[0])
```

# Conclusão

Como conclusão, tivemos a utilização do modelo Random Forest e
Extreme Gradient Boosting otimizados. Mas o gráfico a seguir irá mostrar os
resultados com a base de treino e base de teste.

```{.python .input}
columns = [x.replace('_',' ') for x in results.keys()]
train = []
test = []
width=0.4
base = np.arange(len(columns))
for key in results:
    train.append(results[key]['train'])
    test.append(results[key]['test'])
fig, ax=plt.subplots(figsize=[10,10])
fig = ax.bar(base, train, width)
fig = ax.bar(base+width, test, width)
fig = ax.set_xticks(base+width/2)
fig = ax.set_xticklabels(columns, rotation='45')
fig = ax.legend(['Base de treino', 'Base de teste'])
plt.show()
```

# Referências Bibliográficas
http://scikit-
learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.du
mmy.DummyClassifier
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-
xgboost-with-codes-python/

http://scikit-
learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
ftp://ftp.sas.com/pub/neural/FAQ3.html#A_hu
[MITCHELL](https://dl.acm.org/citation.cfm?id=505283), Tom M. Machine learning.
1997. Burr Ridge, IL: McGraw Hill, v. 45, n. 37, p. 870-877, 1997.
[QUINLAN](http://hunch.net/~coms-4771/quinlan.pdf), J.. Ross . Induction of
decision trees. Machine learning, v. 1, n. 1, p. 81-106, 1986.
[BREIMAN](https://www.stat.berkeley.edu/users/breiman/randomforest2001.pdf),
Leo. Random forests. Machine learning, v. 45, n. 1, p. 5-32, 2001.

BABATUNDE,
Oluleye, ARMSTRONG, Leisa, DIEPEVEEN,
Dean e LENG, J. Comparative analysis of
Genetic Algorithm and Particle Swam
Optimization: An application in precision
agriculture. 2015. **Asian Journal of
Computer and Information Systems**. 3.
1-12.
