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
from ydata_profiling import ProfileReport


from pandas_profiling import ProfileReport #estudio variables
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay






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
#tablaCalculada.to_excel("VerTablaCalculada.xlsx", index=False)


#Código que responde a la descripción anterior (incorpore las lineas de code necesarias. Describa cadas sentencia de código)
#tablaCalculada.describe().transpose()
#reportepandasprofiling =ProfileReport(tablaCalculada, explorative=True) # Usanos una herrmaienta muy util en Python
#reportepandasprofiling.to_file('Reporte_profiling_obesidad.html')

report = ProfileReport(tablaCalculada, title='My Data')
report.to_file("my_report.html")

X = tablaCalculada.drop(['producto','diseño','color','Código','Nombre PBI', 'Proveedor','objetivo'], axis=1)
report2 = ProfileReport(X, title='My Data2')
report2.to_file("my_reportVaribales.html")


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
