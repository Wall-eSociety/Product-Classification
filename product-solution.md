```python
# Configure to show multiples outputs from a single cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
import math
```

```python

with zipfile.ZipFile('Datasets.zip') as ziped_file:
    with ziped_file.open('Datasets/train.csv') as train_file:
        df_train = pd.read_csv(train_file, header=0).set_index('id')
    with ziped_file.open('Datasets/test.csv') as test_file:
        df_test = pd.read_csv(test_file, header=0).set_index('id')
df_target = pd.DataFrame(df_train.pop('target')) # Get the target
df_target.head() # Show target classes
df_train.head() # The train dataset
df_test.head() # It hasn't target

```

# Tratamento

Será realizada as etapas de feature selection e feature
engineering.
Correlação entre features

Será realizada uma análise da correlação
entre as features. Visto que há um total de 93 colunas que não foi
disponibilizada nenhuma informação sobre o que são elas e o que representam e
portanto, esta análize ajudará a identificar as relações entre as features.

##
Correlação

A correlação entre duas variáveis é quando existe algum laço
matemático que envolve o valor de duas variáveis de alguma forma [ESTATÍSTICA II
- CORRELAÇÃO E
REGRESSÃO](http://www.ctec.ufal.br/professor/mgn/05CorrelacaoERegressao.pdf).
Uma das maneiras mais simples de se identificar a correlação entre duas
variáveis é plotando-as em um gráfico, para tentar identificar alguma relação
entre elas, entretanto, como são um total de 93 features, dificulta visualizar a
correlação em forma gráfica.

A correlação de
[Pearson](https://pt.wikipedia.org/wiki/Coeficiente_de_correla%C3%A7%C3%A3o_de%0A_Pearson)
mede o grau da correlação (e a direcção dessa correlação - se positiva ou
negativa) entre duas variáveis de escala métrica (intervalar ou de rácio/razão).
Já a correlação de
[Spearman](https://pt.wikipedia.org/wiki/Coeficiente_de_correla%C3%A7%C3%A3o_de_postos_de_Spearman)
entre duas variáveis é igual à correlação de Pearson entre os valores de postos
daquelas duas variáveis. Enquanto a correlação de Pearson avalia relações
lineares, a correlação de Spearman avalia relações monótonas, sejam elas
lineares ou não.

Visto ambos os tipos de correlação, utilizaremos a de Pearson
para avaliar se há alguma correlação linear crescente ou decrescente entre as
variáveis, pois esta relação nos possibilita remover uma delas sem prejuizos aos
modelos de machine learn

```python
shape = (df_train.shape[1], df_train.shape[1])
upper_matrix = np.tril(np.ones(shape)).astype(np.bool)
np.fill_diagonal(upper_matrix, False)
correlation = df_train.corr('pearson').abs().where(upper_matrix)
correlation
```

## Filtrando colunas

A partir da matriz de correlação assima, buscamos agora
identificar quais das colunas possuem uma forte correlação de acordo com a
tabela a seguir.
Como sugerido por [Makuka,
2012](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3576830/)
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

