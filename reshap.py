import pandas as pd

stud_info = pd.read_csv("data\studentInfo.csv").sort_values(by="code_module")
stud_asse = pd.read_csv("data\studentAssessment.csv").sort_values(by="date_submitted")
stud_reg = pd.read_csv("data\studentRegistration.csv").sort_values(by="code_module")
stud_info["final_result"] = stud_info["final_result"].replace(
    {"Distinction": 1, "Withdrawn": 0, "Pass": 1, "Fail": 0}
)

agrupamento = stud_asse.groupby("id_student").value_counts().transpose()
ordenado = stud_asse.sort_values(by="id_student")
# print(agrupamento)
""" print(stud_info)
stud_info['final_result'] = stud_info['final_result'].replace({'Distinction': 'Pass', 'Withdrawn': 'Fail'})
dados_cod = pd.get_dummies(stud_info, columns=['code_module','code_presentation','gender','region','highest_education','imd_band','age_band','disability','final_result'])
print(dados_cod) """

stud_asse["repeticao"] = stud_asse.groupby("id_student").cumcount() + 1

# Usa pivot_table para transformar o DataFrame
df_transformado = stud_asse.pivot_table(
    index="id_student",
    columns=["repeticao"],
    values=["id_assessment", "date_submitted", "is_banked", "score"],
    aggfunc="first",
)

# Renomeia as colunas conforme o padrão desejado
df_transformado.columns = [f"{col[0]}_{col[1]}" for col in df_transformado.columns]

# Reseta o índice para tornar 'id_student' uma coluna
df_transformado = df_transformado.reset_index()

# Reorganiza as colunas para a ordem desejada
column_order = ["id_student"]
column_order.extend(
    [
        f"{col}_{i}"
        for i in range(1, df_transformado.shape[1] // 4 + 1)
        for col in ["id_assessment", "date_submitted", "is_banked", "score"]
    ]
)
df_transformado = df_transformado[column_order]

# Exibe o DataFrame transformado
# print(df_transformado)
df_junto = pd.merge(
    stud_info, stud_reg, on=["id_student", "code_module", "code_presentation"]
).merge(df_transformado, on="id_student")

dados_cod = pd.get_dummies(
    df_junto,
    columns=[
        "code_module",
        "code_presentation",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ],
)

coluna_para_mover = "final_result"

colunas_ordenadas = [col for col in dados_cod.columns if col != coluna_para_mover] + [
    coluna_para_mover
]

df = dados_cod[colunas_ordenadas]


print(df)
df.to_csv("dataframe_d2.csv", index=False)
