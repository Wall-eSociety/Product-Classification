```{.python .input}
# Configure to show multiples outputs from a single cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


import zipfile
import pandas as pd

with zipfile.ZipFile('Datasets.zip') as ziped_file:
    with ziped_file.open('Datasets/train.csv') as train_file:
        df_train = pd.read_csv(train_file, header=0).set_index('id')
    with ziped_file.open('Datasets/test.csv') as test_file:
        df_test = pd.read_csv(test_file, header=0).set_index('id')

df_train.head() # It has target
df_test.head() # It hasn't target
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

```{.python .input}
X = df_train.drop("target", axis=1)
Y = df_train.iloc[:,-1]
```

```{.python .input}
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)

```

## Importancia das features para a RF

```{.python .input}
clf.feature_importances_
```

## Verificando a acurácia com os dados de treinamento

Utilizando os dados que foram utilizados parar treinar o algoritmo como entrada
para predição nos dá noção se o modelo pode estar viciado.

```{.python .input}
clf.score(X,Y)
```

## Verificando com Cross Validation

Cross validation irá predizer um pedaço do dataset utilizando o modelo treinado
com o resto dos dados que não fazem parte deste dataset.

```{.python .input}
rfscores = cross_val_score(clf, X, Y)
print (rfscores.mean() * 100, end='')
print ("%") 
```

## ExtraTrees

O [Scikit Learn](http://scikit-
learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
nos apresenta um tipo diferente de random forest que pode apresentar resultados
melhores que o [RandomForestClassifier](http://scikit-
learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

```{.python .input}
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier();
etscores = cross_val_score(clf, X, Y)
print (etscores.mean() * 100, end='')
print ("%")
```
