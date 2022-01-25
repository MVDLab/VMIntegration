
import pandas as pd
import numpy as np


x = np.arange(9.).reshape((3,3))

df = pd.DataFrame(x, columns=['a','b','c'])

print(df.dtypes)


print(np.cross(df['a']-df['b'], df['a']-df['c']))
print(df)