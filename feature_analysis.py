import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
importancia = pd.DataFrame({
    'variables': X_train.columns,
    'importancia': model_xgboost.feature_importances_
})
importancia.sort_values('importancia', ascending=True, inplace=True)
importancia0 = importancia.tail(5)
importancia0.plot.barh(x='variables', y='importancia', figsize=(8,6))
plt.xlabel('Importancia')
plt.title('Top 5 Feature Importances')
plt.grid(True)
plt.show()
