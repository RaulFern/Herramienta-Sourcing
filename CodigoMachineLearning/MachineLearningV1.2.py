import psycopg2
import pandas as pds
from datetime import datetime
from sqlalchemy import create_engine,text
import time
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm
from urllib import parse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def Tablafinal():
    hostname="10.100.200.31"
    dbname="TM"
    uname="sa"
    pwd=parse.quote("D1str1t3x")
    engine = create_engine("mssql+pymssql://{user}:{pw}@{host}/{db}"
    .format(host=hostname, db=dbname, user=uname, pw=pwd))
    dbConnection= engine.connect();
    QueryCompr = text("SELECT *,\
       CASE WHEN vtapromedio = 0 THEN NULL ELSE invact/vtapromedio END AS MesesInventario,\
       CASE WHEN invact = 0 THEN NULL ELSE vtapromedio/invact*100 END AS Rotacion,\
       CASE\
           WHEN vtapromedio <= 20 AND invact IS NULL THEN 2\
           WHEN vtapromedio > 20 AND vtapromedio <= 21 AND invact IS NULL THEN 3\
           WHEN vtapromedio > 21 AND vtapromedio <= 100  AND invact IS NULL THEN 25\
           WHEN vtapromedio > 100 AND vtapromedio <= 350  AND invact IS NULL THEN 35\
           WHEN vtapromedio > 350 AND invact IS NULL THEN 60\
           ELSE CASE WHEN invact = 0 THEN NULL ELSE vtapromedio/NULLIF(invact, 0)*100 END\
       END AS RotacionRecal\
       FROM RECOMENDADOR r")
    df3 = pds.read_sql(QueryCompr, dbConnection)
    dbConnection.close();
    return df3

def asignar_etiqueta(indice, invtto):
    if indice <= 10:
        return "NC"
    elif indice <= 25:
        return "AL"
    elif indice <= 50 and (invtto is None or invtto == 0):
        return "COMAL"
    elif indice <= 50 and (invtto > 0):
        return "COMPROV"
    elif indice > 50 and (invtto is None or invtto == 0):
        return "COMUR"
    elif indice > 50 and (invtto > 0):
        return "COMURPROV"



def TablaPruebas():
    hostname="10.100.200.31"
    dbname="TM"
    uname="sa"
    pwd=parse.quote("D1str1t3x")
    engine = create_engine("mssql+pymssql://{user}:{pw}@{host}/{db}"
    .format(host=hostname, db=dbname, user=uname, pw=pwd))
    dbConnection= engine.connect();
    QueryCompr = text("SELECT *,\
       CASE WHEN vtapromedio = 0 THEN NULL ELSE invact/vtapromedio END AS MesesInventario,\
       CASE WHEN invact = 0 THEN NULL ELSE vtapromedio/invact*100 END AS Rotacion,\
       CASE\
           WHEN vtapromedio <= 20 AND invact IS NULL THEN 2\
           WHEN vtapromedio > 20 AND vtapromedio < 21 AND invact IS NULL THEN 3\
           WHEN vtapromedio >= 21 AND invact IS NULL THEN 25\
           ELSE CASE WHEN invact = 0 THEN NULL ELSE vtapromedio/NULLIF(invact, 0)*100 END\
       END AS RotacionRecal\
       FROM RECOMENDADOR2 r")
    df3 = pds.read_sql(QueryCompr, dbConnection)
    dbConnection.close();
    return df3
#-----------------------------------------#



tablaCalculada = Tablafinal()
print(tablaCalculada)

print("firmado")
print(tablaCalculada.columns)

conditions = [
    tablaCalculada['RotacionRecal'] <= 13,
    (tablaCalculada['RotacionRecal'] > 13) & (tablaCalculada['RotacionRecal'] <= 25) & (tablaCalculada['invtto'] > 0),
    (tablaCalculada['RotacionRecal'] > 13) & (tablaCalculada['RotacionRecal'] <= 25) & ((tablaCalculada['invtto'].isnull()) | (tablaCalculada['invtto'] == 0)),
    (tablaCalculada['RotacionRecal'] > 25) & (tablaCalculada['RotacionRecal'] <= 50) & (tablaCalculada['invtto'] > 0),
    (tablaCalculada['RotacionRecal'] > 25) & (tablaCalculada['RotacionRecal'] <= 50) & ((tablaCalculada['invtto'].isnull()) | (tablaCalculada['invtto'] == 0)),
    (tablaCalculada['RotacionRecal'] > 50) & ((tablaCalculada['invtto'].isnull()) | (tablaCalculada['invtto'] == 0)),
    (tablaCalculada['RotacionRecal'] > 50) & (tablaCalculada['invtto'] > 0)
]
labels = ['NC', 'AL PROV','AL', 'COMPROV', 'COMAL','COMPUR', 'COMPURPROV']

#tablaCalculada['objetivo'] = tablaCalculada.apply(lambda row: asignar_etiqueta(row['RotacionRecal'],row['invtto']))
tablaCalculada['objetivo'] = np.select(conditions, labels, default='Otro')

print(tablaCalculada)
tablaCalculada.to_excel("VerTablaCalculada.xlsx", index=False)
exit()
#creacion del modelo de random forest
X = tablaCalculada.drop(['producto','diseño','color','Código','Nombre PBI', 'Proveedor','objetivo'], axis=1)  # Elimina la columna de etiquetas del DataFrame
y = tablaCalculada['objetivo']

print(X)
# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Definir y entrenar el modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predecir las etiquetas para los datos de prueba
predicted_labels = model.predict(X_test)


# Evaluar el rendimiento del modelo (opcional)
accuracy = model.score(X_test, y_test)
print("Precisión del modelo:", accuracy)


print("Nuevos Calculos...")
DatosPruebas =TablaPruebas()
X_nuevo = DatosPruebas.drop(['producto','diseño','color','Código','Nombre PBI', 'Proveedor'], axis=1)
etiquetas_predichas = model.predict(X_nuevo)

print(etiquetas_predichas)

X_nuevo['etiquetas_predichas'] = etiquetas_predichas
print(X_nuevo)
X_nuevo.to_excel("ResultMachineLeraning.xlsx", index=False)

# datosproov = ExtractProov()
#
# datosArticulos = ExtractParms()
# datosArticulos = tuple(datosArticulos)
# print(datosArticulos)
#
# datosInvTTO = ExtractInvTTO(datosArticulos)
# datosInvTTO['pro_producto'] = datosInvTTO['pro_producto'].astype('int32')
# datosInvTTO['pro_producto'] = datosInvTTO['pro_producto'].astype('str')
# print(datosInvTTO)
#
# datosInvAct = ExtractInvACT(datosArticulos)
# datosInvAct['pro_producto'] = datosInvAct['pro_producto'].astype('int32')
# datosInvAct['pro_producto'] = datosInvAct['pro_producto'].astype('str')
# print(datosInvAct)
#
#
# dataf= conexionbase2pruebas(datosArticulos)
# dataf['producto'] = dataf['producto'].astype('int32')
# dataf['producto'] = dataf['producto'].astype('str')
# print(dataf)
#
# DF = pds.merge(dataf,datosproov, how='left', left_on=['producto'], right_on=['Código'])#.drop(['LicTradNum'], axis=1)
#
# DF = pds.merge(DF,datosInvTTO[['pro_producto','dpc_colorcom','invtto']], how='left', left_on=['producto','color'], right_on=['pro_producto','dpc_colorcom'])\
# .drop(['pro_producto','dpc_colorcom'], axis=1)
#
# DF = pds.merge(DF,datosInvAct[['pro_producto','dpc_colorcom','invact']], how='left', left_on=['producto','color'], right_on=['pro_producto','dpc_colorcom'])\
# .drop(['pro_producto','dpc_colorcom'], axis=1)
#
# print(DF)
# DF.to_excel("VentaPromedioFebrero2024.xlsx", index=False)
#
# #dataf= conexionbase2pruebas()
# #dataf = pds.read_excel('IMPORTACIONES JOHN URIBE 2023 (7) (2).xlsx', sheet_name = "DATOS")
# #print(dataf)
# #print(dataf.info())
#
# #dataImpo = dataf[["PR0VEED0R ","N. FACTURA","VAL0R DE FACTURA U$","MES INGR. AL INVEN.","REFERENCIA (TELA)","ESTAD0 DE IMP0RTACIÓN"]]
#
# #print(dataImpo)
#
# #dataImpo1 = dataImpo[(dataImpo['ESTAD0 DE IMP0RTACIÓN']== '05-MCÍA.EN PUERTO PEND.X NACIONALIZAR') |(dataImpo['ESTAD0 DE IMP0RTACIÓN'] == '04-MCÍA.EN VÍA (NAVEGANDO)')]
#
# #print(dataImpo1)
#
# #sns.scatterplot(x='pru_nrounidades', y='dpe_nrounidades', data=dataf, hue='pro_producto')
# #dataf.to_excel("InventarioCons270124.xlsx")
# #dataImpo1 = dataImpo1.astype({"PR0VEED0R ":'string',"N. FACTURA":'string',
# #                "VAL0R DE FACTURA U$": 'float64',
# #                "MES INGR. AL INVEN.":'string',"REFERENCIA (TELA)":'string',"ESTAD0 DE IMP0RTACIÓN":'string'})
#
# #print(dataImpo1.info())
#
# hostname="10.100.200.31"
# dbname="TM"
# uname="sa"
# pwd=parse.quote("D1str1t3x")
# engine = create_engine("mssql+pymssql://{user}:{pw}@{host}/{db}"
# .format(host=hostname, db=dbname, user=uname, pw=pwd))
#
# # Crear un objeto de conexión y cursor
# with engine.connect() as dbConnection, dbConnection.begin():
#     try:
#         dbConnection= engine.connect();
#         DF.to_sql('RECOMENDADOR', con=dbConnection, if_exists='replace', index=False)
#
#         # Confirmar la transacción
#         dbConnection.commit()
#     except Exception as e:
#         # Si hay un error, revertir la transacción
#         print(f"Error: {e}")
#         dbConnection.rollback()
#
#
#
# dbConnection.close();
#
#
# # dataImpo.to_excel("IMPO_ACT.xlsx", index=False)
# # dataImpo1.to_excel("IMPO_FINAL.xlsx", index=False)
# #dataf.to_excel(urlCarga+"InventarioConsolidado.xlsx", index=False)
