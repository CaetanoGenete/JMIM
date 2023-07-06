import numpy as np

from JMIM.preprocessing import label_data
from JMIM.entropy import generate_pmf

test = np.random.randn(1000, 4)
data, labels = label_data(test, method="ewd", bins=10)

pmf = generate_pmf(data, labels)

labels = ([0,1], 2, 3)

print(pmf[labels])