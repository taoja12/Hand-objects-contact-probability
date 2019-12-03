import numpy as np
import pandas as pd

txt = np.loadtxt("probs.txt")
txtDF = pd.DataFrame(txt)
txtDF.to_csv("probs.csv", index=False)