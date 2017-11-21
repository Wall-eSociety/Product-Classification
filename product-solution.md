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
        df_train = pd.read_csv(train_file, header=0)
    with ziped_file.open('Datasets/test.csv') as test_file:
        df_test = pd.read_csv(test_file, header=0)
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

## Correlação

A correlação entre duas variáveis é quando existe algum laço
matemático que envolve o valor de duas variáveis de alguma forma [ESTATÍSTICA II
- CORRELAÇÃO E
REGRESSÃO](http://www.ctec.ufal.br/professor/mgn/05CorrelacaoERegressao.pdf).
Uma das maneiras mais simples de se identificar a correlação entre duas
variáveis é plotando-as em um gráfico, para tentar identificar alguma relação
entre elas, entretanto, como são um total de 93 features, dificulta visualizar a
correlação em forma gráfica.

A correlação de
[Pearson](https://pt.wikipedia.org/wiki/Coeficiente_de_correla%C3%A7%C3%A3o_de%0
A_Pearson)
mede o grau da correlação (e a direcção dessa correlação - se positiva ou
negativa) entre duas variáveis de escala métrica (intervalar ou de rácio/razão).
Já a correlação de
[Spearman](https://pt.wikipedia.org/wiki/Coeficiente_de_correla%C3%A7%C3%A3o_de_
postos_de_Spearman)
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
|0.7 < v <= 0.9 | Forte |
|0.5 < v <= 0.7 | Moderada |
|0.3 < v <= 0.5 | Fraca |
|0.0 < v <= 0.3 | Desprezível |

```python
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

## Resultado

A correlação mostra que não há uma fortíssima correlação entre as
features, entretanto, há 10 colunas que estão fortemente correlacionadas. Porem
buscamos uma correlação fortíssima para não remover features com comportamentos
diferentes.

```python
X_train = df_train
y_train = df_target.iloc[:, 0]

X_train.head()
y_train.head()
```

# Feature Selection

É o processo de selecionar um subconjunto de termos do conjunto de treinamento e
usá-lo na classificação. Serve para dois propósitos: diminuir a quantidade do
vocabulário de treinamento, tornando o classificador mais eficiente (na maioria
das vezes o custo computacional de treinar é caro); aumentar a precisão da
classificação eliminando ruído. Segundo Ikonomakis, Kotsiantis e Tampakas
(2005), é a redução da dimensionalidade do conjunto de dados que tem o objetivo
de excluir as características que são consideradas irrelevantes para a
classificação. Mais pode ser encontrado no [Relatório técnico "conceitos sobre
Aprendizado de Máquina"](http://conteudo.icmc.usp.br/pessoas/taspardo/TechReport
UFSCar2009b-MatosEtAl.pdf)

Existem diversos técnicas de redução de dimensionalidade, neste progeto será
utilizado o _Univariate feature selection_ sendo que o mesmo pode ser utilizado
como comparação para averiguar a acurácia da base com e sem a seleção de
característica.

## Univariate feature selection

Funciona selecionando os melhores recursos com base em testes estatísticos
univariados. Pode ser visto como um passo de pré-processamento para um
estimador. _Scikit-learn_ expõe rotinas de seleção de recursos, sendo utilizado
o método de transformação _SelectKBest_, no qual seleciona os valores com a
pontuação mais alta.

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy

train = SelectKBest(score_func=chi2)
univariate_fit = train.fit(X_train, y_train)

numpy.set_printoptions(precision=3)
print(univariate_fit.scores_)
features = univariate_fit.transform(X_train)

print(features[:,])
```

## Recursive feature elimination

Tendo feito a seleção de característica utilizando o método _Univariate feature
selection_ foi feito um cascateamento das _features_ geradas pelo mesmo afim de
obter uma filtragem de dados para garantir a seleção dos melhores. Este
cascateamento foi feito utilizando o método de eliminação recursiva de
características no qual consiste em selecionar recursos de forma recursiva
considerando pequenos e menores conjuntos de recursos.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(features, y_train)
print("Num Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_)

```

# Modelo Dummy Classifier

Dummy Classifier é um modelo que faz predições usando regras simples.

O dummy é importante para termos como parâmetro de comparação
com outros modelos.

```python
from sklearn.dummy import DummyClassifier

# Most Frequent: always predicts the most frequent label in the training set.
mf_clf = DummyClassifier(strategy='most_frequent')
mf_clf.fit(X_train, y_train)

# Stratified: generates predictions by respecting the training set’s class distribution.
sf_clf = DummyClassifier(strategy='stratified')
sf_clf.fit(X_train, y_train)

mf_score = mf_clf.score(X_train, y_train)
sf_score = sf_clf.score(X_train, y_train)

print('Most Frequent Dummy Score: %.4f' % mf_score)
print('Stratified Dummy Score: %.4f' % sf_score)
```

# Random Forest

Breiman breiman, 2001, descreve Random Forests como uma evolução das decisions
trees, onde várias ávores são formadas para criar um modelo com maior precisão.
Isto é feito a partir da separação dos Dados em conjutos
de dados menores e aleatórios. Cada árvore é contruida a partir de um pedaço
aleatório dos dados. Quando um novo dado chega, a predição é feita por todas as
Árvores e ao fim é feita uma
votação por maioria, ou seja, a categoria com mais votos ganha e o resultado é
dado.

![Workflow Random forest](forest.jpg)

De acordo com breiman, 2001, as RFs corrigem a maior parte dos problemas de
Overfitting que as Árvores de decisão apresentam. Tudo depende do quanto as DT
contidas dentro da Random Forest. Isto é, o quanto elas representam os dados.

Referências:

[BREIMAN](https://www.stat.berkeley.edu/users/breiman/randomforest2001.pdf),
Leo. Random forests. Machine learning, v. 45, n. 1, p. 5-32, 2001.

## Utilizando o algoritmo

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
y = df_target.iloc[:,-1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(df_train, y)

```

## Importancia das features para a RF

```python
clf.feature_importances_
```

## Verificando a acurácia com os dados de treinamento

Utilizando os dados que foram utilizados parar treinar o algoritmo como entrada
para predição nos dá noção se o modelo pode estar viciado.

```python
print (clf.score(df_train, y) * 100, end='')
print ("% de precisão")
```

## Verificando com Cross Validation

Cross validation irá predizer um pedaço do dataset utilizando o modelo treinado
com o resto dos dados que não fazem parte deste dataset.

```python
rfscores = cross_val_score(clf, df_train, y)
print (rfscores.mean() * 100, end='')
print ("% de precisão")
```

## ExtraTrees

O [Scikit Learn](http://scikit-
learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
nos apresenta um tipo diferente de random forest que pode apresentar resultados
melhores que o [RandomForestClassifier](http://scikit-
learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

```python
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier();
etscores = cross_val_score(clf, df_train, y)
print (etscores.mean() * 100, end='')
print ("% de precisão")
```

## Boosting Trees

Este algorítmo demora demais para rodar, descomente se tiver a paciencia de
esperar.
Estimativa: 10 min com I7 3.1  8Ram

```python
#from sklearn.ensemble import GradientBoostingClassifier

#gbc = GradientBoostingClassifier();
#gbcscores = cross_val_score(gbc, df_train, y)
```

```python
#print (gbcscores.mean() * 100, end='')
#print ("%")
```

# Referências Bibliográficas
http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.h
tml#sklearn.dummy.DummyClassifier
