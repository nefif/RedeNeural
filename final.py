import pandas as pd
import numpy as np

estudantes = pd.read_csv("data\dataframe_d2.csv")
estudantes_numericos = pd.DataFrame(estudantes)
estudantes_numericos.replace({True: 1, False: 0}, inplace=True)
estudantes_numericos["date_unregistration"] = estudantes_numericos['date_unregistration'].replace({0:1})

estudantes_transformacao = estudantes_numericos.fillna(0)

print(estudantes_transformacao)

# estudantes_transformacao.to_csv("dataframe_filna.csv", index=False)


# Converter todas as linhas do DataFrame em matrizes 41 x 4 x 1 usando to_numpy
estudantes_matriz = estudantes_transformacao.to_numpy().reshape((-1, 41, 4, 1))

# Exibir a primeira matriz resultante como exemplo
print(estudantes_matriz[0])