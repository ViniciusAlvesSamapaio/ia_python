import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 
tabela = pd.read_csv("advertising.csv")
sns.heatmap(tabela.corr(), cmap= "Purples", annot = True)

plt.show()

y = tabela["Vendas"]
x = tabela[["TV", "Radio", "Jornal"]]

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.7)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

modelo_regressaolinear = LinearRegression()
modelo_arvore = RandomForestRegressor()

 
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvore.fit(x_treino, y_treino)

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvore = modelo_arvore.predict(x_teste)

from sklearn.metrics import r2_score

print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvore))

tabela_a = pd.DataFrame()
tabela_a["y_teste"] = y_teste
tabela_a["previsao arvore decisao"] = previsao_arvore
tabela_a["previsao regressao linear"] = previsao_regressaolinear
print(tabela_a)

sns.lineplot(data=tabela_a)
plt.show()
