import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv("FILENAME.csv")
print(df.columns)

X = df["INDEPENDENT_VAR"]
Y = df["DEPENDENT_VAR"]

result = stats.linregress(X, Y)

result.slope
result.intercept
result.pvalue
result.stderr

predicted = result.intercept + result.slope * X
residuals = Y - predicted

stats.probplot(residuals, dist="norm", plot=plt)
plt.show()
