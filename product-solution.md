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
df_target.target = pd.Categorical(df_target.target) # Transform target in Categorical type
df_target['categories'] = df_target.target.cat.codes # Add the codes in a columns
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

## Train/Test split

Utilizaremos 80% da base de treino para efetivamente treinar o modelo e 20% para
averiguar a performance do modelo.

```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

X = df_train.iloc[:, :].values
y = df_target.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

## XGBoost

### *eXtreme Gradient Boost*

XGBoost é um algoritmo que implementa *gradient boosting* de Decision Trees de
forma rápida e com alta performance.

**Gradient Boosting** é uma técnica de *machine learning* para problemas de
regressão e classificação que produz um modelo de predição na forma de
*ensemble* de modelos de predições fracas, normalmente árvores de decisões.

```python
%%time
from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print('Média: %.2f' % accuracies.mean())
print('Desvio padrão: %.4f' % accuracies.std())
print('Matriz de confusao:\n', cm)
```

```python
x_train = df_train
y_train = df_target.target

x_train.head()
y_train.head()
```

## Modelo Dummy Classifier

Dummy Classifier é um modelo que faz predições usando
regras simples.

O dummy é importante para termos como parâmetro de comparação
com outros modelos.

```python
from sklearn.dummy import DummyClassifier

models = ['most_frequent', 'stratified']

for model in models:
    clf = DummyClassifier(strategy=model)
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Cross validation
    accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
    print('Média: %.2f' % accuracies.mean())
    print('Desvio padrão: %.4f' % accuracies.std())

    # Confusion matrix
    print('Matriz de confusao de', model, '\n', cm)
    print(model, 'score: %.2f' % score)
# Most Frequent: always predicts the most frequent label in the training set.
mf_clf = DummyClassifier(strategy='most_frequent')
mf_clf.fit(x_train, y_train)

# Stratified: generates predictions by respecting the training set’s class distribution.
sf_clf = DummyClassifier(strategy='stratified')
sf_clf.fit(x_train, y_train)

mf_score = mf_clf.score(x_train, y_train)
sf_score = sf_clf.score(x_train, y_train)

print('Most Frequent Dummy Score: %.4f' % mf_score)
print('Stratified Dummy Score: %.4f' % sf_score)
```

# Decision Tree

# Adicionar descrição de como funciona!!!!

```python
from sklearn.model_selection import cross_val_score
from sklearn import tree
def fit_tree(X, Y):
    tree_classifier = tree.DecisionTreeClassifier()
    tree_classifier.fit(X, Y)
    inner_score = tree_classifier.score(X, Y)
    tree_classifier = tree.DecisionTreeClassifier()
    tree_fit = cross_val_score(tree_classifier, X, Y)
    return inner_score, tree_fit.mean(), tree_fit.std()

"inner: {:.2f} cross: {:.2f} +/- {:.2f}".format(*fit_tree(x_train, y_train))

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

```python
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

```python
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

```python
"inner: {:.2f} cross: {:.2f} +/- {:.2f}".format(*fit_tree(df_rtrain, df_rtarget.target))

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

![Workflow Random
forest](forest.jpg)

De acordo com breiman, 2001, as RFs corrigem a maior parte
dos problemas de
Overfitting que as Árvores de decisão apresentam. Tudo depende
do quanto as DT
contidas dentro da Random Forest. Isto é, o quanto elas
representam os dados.

Referências:
[BREIMAN](https://www.stat.berkeley.edu/users/breiman/randomforest2001.pdf),
Leo. Random forests. Machine learning, v. 45, n. 1, p. 5-32, 2001.

##
Utilizando o algoritmo

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(x_train, y_train)

```

## Importancia das features para a RF

A seguir vemos quais as influências de
cada uma das features para o uso no random forest. Quanto maior no gráfico,
maior é a importância da feature.

```python
fig, axis = plt.subplots(figsize=(15, 5))
plot = axis.bar(x_train.columns, clf.feature_importances_)
plot = axis.set_xticklabels(x_train.columns.values, rotation='vertical')
plot = axis.set_xlabel('feature')
plot = axis.set_ylabel('importance')
plt.show()

```

## Verificando a acurácia com os dados de treinamento

Utilizando os dados que
foram utilizados parar treinar o algoritmo como entrada
para predição nos dá
noção se o modelo pode estar viciado.

```python
print ("{} de precisão".format(clf.score(x_train, y_train) * 100))
```

## Verificando com Cross Validation

Cross validation irá predizer um pedaço do
dataset utilizando o modelo treinado
com o resto dos dados que não fazem parte
deste dataset.

```python
rfscores = cross_val_score(clf, x_train, y_train)
print ("{} de precisão".format(rfscores.mean() * 100))

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
etscores = cross_val_score(clf, x_train, y_train)
extra_tree_fit = etc.fit(x_train, y_train)
print ("{} de precisão".format((etscores.mean() * 100)))
print(extra_tree_fit.score(x_train, y_train))
```

## Boosting Trees

Este algorítmo demora demais para rodar, descomente se tiver
a paciencia de
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
http://scikit-
learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.h
tml#sklearn.dummy.DummyClassifier
