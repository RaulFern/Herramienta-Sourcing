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

#modelos
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#mediciones
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#comprobadores
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#estudio graficadores
from pandas_profiling import ProfileReport #estudio variables
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import pydotplus
import matplotlib.image as pltimg

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

#Preparacion de los datos
X = tablaCalculada.drop(['producto','diseño','color','Código','Nombre PBI', 'Proveedor','objetivo'], axis=1)  # Elimina la columna de etiquetas del DataFrame
y = tablaCalculada['objetivo']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creacion del modelo de random forest
model = RandomForestClassifier(n_estimators=60, random_state=2)
model.fit(X_train, y_train)

# Predecir las etiquetas para los datos de prueba
predicted_labels = model.predict(X_test)



#####################nuevo modelo ##############################

#creacion del modelo de DecisionTreeClassifier

modelDT = DecisionTreeClassifier(random_state = 2 , criterion="entropy")
modelDT.fit(X_train, y_train)
predictions = modelDT.predict(X_test)
#print(modelDT)





# Evaluar el rendimiento del modelo (opcional)
print(model)
accuracy = model.score(X_test, y_test)
print("Precisión del modelo RF:", accuracy)


#se utiliza stratifiedkfold para garantizar una verificacion mas acorde a un conunto de muestras no balanceado ya que
#este metodo toma los conjuntos de datos con el mismo porcentaje de muestrasde cada clase

#validacion modelo RANDOM FOREST con Kfold
kfold = StratifiedKFold(n_splits = 5, random_state =1, shuffle=True)
cv_puntuacion = cross_val_score(model, X_train, y_train, cv= kfold, scoring="accuracy")
print("Validacion Cruzada de 5: ",cv_puntuacion)
print("Promedio Validacion Cruzada: ",cv_puntuacion.mean())


#print("Precision del modelo: ",accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
print("Matriz de confusion del modelo RF:")
mostrar = ConfusionMatrixDisplay(confusion_matrix= confusion_matrix(y_test, predicted_labels),display_labels=model.classes_ )
mostrar.plot()
plt.show()

print("Reporte de clasificacion RF: ")
print(classification_report(y_test, predicted_labels))



print(modelDT)
kfold = StratifiedKFold(n_splits = 5, random_state =1, shuffle=True)
cv_puntuacion = cross_val_score(modelDT, X_train, y_train, cv= kfold, scoring="accuracy")
print("Validacion Cruzada de 5: ",cv_puntuacion)
print("Promedio Validacion Cruzada: ",cv_puntuacion.mean())


print("Matriz de confusion del modelo DT:")
mostrar = ConfusionMatrixDisplay(confusion_matrix= confusion_matrix(y_test, predictions),display_labels=modelDT.classes_ )
mostrar.plot()
plt.show()

print("Reporte de clasificacion DT: ")
print(classification_report(y_test, predictions))


#Código para mostrar la comparación de métricas de desempeño de las dos propuestas en tabla
modelos = []
modelos.append(("DecisionTree",DecisionTreeClassifier( random_state = 2,criterion="entropy")))
modelos.append(("RandomForest",RandomForestClassifier( n_estimators=100, random_state=42)))
resultados = []
nombres =[]
promedios = []
desviacion = []
#iteramos
for nombre, modelo in modelos:
    kfoldunion = StratifiedKFold(n_splits = 5, random_state =1, shuffle=True)
    cv_puntuacionunion = cross_val_score(modelo, X_train,y_train , cv= kfoldunion, scoring="accuracy")
    resultados.append(cv_puntuacionunion)
    nombres.append(nombre)

for res in resultados:
    promedio = np.mean(res)
    std = np.std(res)
    promedios.append(promedio)
    desviacion.append(std)

concatenado = [promedios, desviacion]
dfresultados = pds.DataFrame(concatenado, index = ["CrossValidation", "StandardDesviation"], columns= nombres)
print(dfresultados)

#Código para mostrar la comparación de métricas de desempeño de las dos propuestas en gráfica
plt.boxplot(resultados, labels = nombres)
plt.title("Comparacion de modelos")
plt.show()

exit()

print("Nuevos Calculos...")
DatosPruebas =TablaPruebas()
X_nuevo = DatosPruebas.drop(['producto','diseño','color','Código','Nombre PBI', 'Proveedor'], axis=1)
etiquetas_predichas = model.predict(X_nuevo)

print(etiquetas_predichas)

X_nuevo['etiquetas_predichas'] = etiquetas_predichas
print(X_nuevo)
X_nuevo.to_excel("ResultMachineLeraning.xlsx", index=False)

#
