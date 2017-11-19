```python
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

```python
X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]
```

# Dummy model

Dummy Classifier é um modelo que faz predições usando regras simples.

O modelo dummy classifier é importante para termos como parâmetro de comparação
com outros modelos.

```python
from sklearn.dummy import DummyClassifier

clf = DummyClassifier(strategy='most_frequent',random_state=0)
clf.fit(X_train, y_train)

print('Dummy Score: %.4f' % clf.score(X_train, y_train))
```

# Referências Bibliográficas
http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.h
tml#sklearn.dummy.DummyClassifier
