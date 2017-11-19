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

df_train.head() # It has target
df_test.head() # It hasn't target
```
