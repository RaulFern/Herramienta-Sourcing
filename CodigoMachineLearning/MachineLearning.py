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
           WHEN vtapromedio > 20 AND vtapromedio < 21 AND invact IS NULL THEN 3\
           WHEN vtapromedio >= 21 AND invact IS NULL THEN 25\
           ELSE CASE WHEN invact = 0 THEN NULL ELSE vtapromedio/NULLIF(invact, 0)*100 END\
       END AS RotacionRecal\
       FROM RECOMENDADOR r")
    df3 = pds.read_sql(QueryCompr, dbConnection)
    dbConnection.close();
    return df3

def asignar_etiqueta(indice):
    if indice <= 10:
        return "NC"
    elif indice <= 25:
        return "AL"
    elif indice <= 50:
        return "COM"
    else:
        return "COMA"

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

#conexion a la base de datos, para descargar el inventario consolidado, que trata de una funncion SQL el cual se lde direcciona John Uribe
def conexionbase2pruebas(paramsval):
    alchemyEngine   = create_engine('postgresql+psycopg2://{user}:{pw}@bdsait.gco.com.co/db_sait_col_prod'.format(user=uname, pw=pwd), pool_recycle=3600)
    dbConnection    = alchemyEngine.connect();
    QueryCompr = text("select producto,Diseño, color, sum(cantidad)Cantidad,sum(valorbruto)ValorBruto,\
                        sum(valorneto)ValorNeto, sum(cantidad)/6 vtapromedio from (\
                        SELECT producto,  Diseño, color, sum(cantidad)Cantidad\
                        ,sum(valorbruto)ValorBruto, sum(valorneto)ValorNeto,mescontable, current_date - fechadocumento AS diferencia_dias\
                        FROM   (SELECT min_fechadoc       FechaDocumento,\
				                    pro_producto       Producto,\
                                    dpc_disenocom      Diseño,\
                                    dpc_colorcom       Color,\
                                    min_nrounidades    Cantidad,\
                                    min_valorbruto     ValorBruto,\
                                    min_valorneto      ValorNeto,\
                                    C.mes_mescontable MesContable\
                                    FROM   sai_movimientoinv C\
                                    INNER JOIN sai_factura F\
                                    ON C.emp_empresa = F.emp_empresa\
                                    AND C.bod_bodega = F.bod_bodega\
                                    AND C.dcc_secuencia = F.dcc_secuencia\
                                    AND C.min_documento = F.fac_factura\
                                    AND C.tpd_tipodoc = F.tpd_tipodoc\
                                    AND C.cli_cliente = F.cli_cliente\
                                    AND C.cli_secuencia = F.cli_secuencia\
                                    WHERE  C.tpd_tipodoc IN( 'FV', 'FT' )\
                                    AND min_estado IN( 'I', 'A' )\
                        UNION ALL\
                        SELECT min_fechadoc       FechaDocumento,\
                        pro_producto         Producto,\
                        dpc_disenocom        Diseño,\
                        dpc_colorcom         Color,\
                        min_nrounidades * -1 Cantidad,\
                        min_valorbruto * -1  ValorBruto,\
                        min_valorneto * -1   ValorNeto,\
                        mes_mescontable\
                        FROM   sai_movimientoinv\
                        WHERE  tpd_tipodoc IN( 'MI' )\
                        AND cpi_concepto = '03'\
                        AND min_estado IN( 'I', 'A' )\
                        UNION ALL\
                        SELECT min_fechadoc       FechaDocumento,\
                        pro_producto         Producto,\
                        dpc_disenocom        Diseño,\
                        dpc_colorcom         Color,\
                        min_nrounidades * -1 Cantidad,\
                        min_valorbruto * -1  ValorBruto,\
                        min_valorneto * -1   ValorNeto,\
                        mes_mescontable\
                        FROM   sai_movimientoinv\
                        WHERE  tpd_tipodoc IN( 'DV' )\
                        AND min_estado IN( 'I', 'A' )) uni\
                        WHERE  fechadocumento >='2023-01-01'\
                        and producto in :params\
                        and  current_date - fechadocumento <= 180\
                        group by fechadocumento ,Producto, Diseño,Color,mescontable\
                        )final\
                        group by producto,Diseño, color\
                        ")
    df0 = pds.read_sql(QueryCompr, dbConnection,params={"params": paramsval})
    dbConnection.close();
    return df0


def ExtractInvTTO(paramsval):
    alchemyEngine   = create_engine('postgresql+psycopg2://{user}:{pw}@bdsait.gco.com.co/db_sait_col_prod'.format(user=uname, pw=pwd), pool_recycle=3600)
    dbConnection    = alchemyEngine.connect();
    QueryCompr = text("select prv_proveedor, prv_apellidouno, pro_producto, prb_nombrecom, dpc_colorcom, sum(dor_nrounidades)INVTTO from (\
                        select t2.prv_proveedor, t3.prv_apellidouno  ,t1.pro_producto, t4.prb_nombrecom ,dpc_colorcom,\
                        dor_nrounidades, t2.ord_proforma, t2.ord_fecha\
                        from sai_detordencompra t1\
                        left join sai_ordencompra t2 on t1.emp_empresa = t2.emp_empresa and t1.ord_ordencompra = t2.ord_ordencompra\
                        left join sai_proveedor t3 on t2.prv_proveedor = t3.prv_proveedor\
                        left join (select distinct pro_producto, prb_nombrecom  from sai_prodbodega\
		                      where emp_empresa = 'JU'\
		                            group by pro_producto, prb_nombrecom) t4 on t1.pro_producto = t4.pro_producto\
                                    where ord_fecha > '2023-10-01'\
                                    and t2.ord_estado = 'R'\
                                    and T2.prv_proveedor in ('321',\
                                    'B24','A89','AH3','366','28','B23',\
                                    '243','103','303','110','CP0','CO9','AG4','C20')\
                                    and t1.pro_producto in :params )UNI\
                                    group by prv_proveedor, prv_apellidouno, pro_producto, prb_nombrecom, dpc_colorcom\
                        ")
    df0 = pds.read_sql(QueryCompr, dbConnection,params={"params": paramsval})
    dbConnection.close();
    return df0

def ExtractInvACT(paramsval):
    alchemyEngine   = create_engine('postgresql+psycopg2://{user}:{pw}@bdsait.gco.com.co/db_sait_col_prod'.format(user=uname, pw=pwd), pool_recycle=3600)
    dbConnection    = alchemyEngine.connect();
    QueryCompr = text("select emp_empresa , pro_producto , dpc_colorcom, sum(pru_nrounidades)INVACT from sai_productoporubi\
                        where pru_nrounidades > 0\
                        and pro_producto in :params\
                        group by emp_empresa , pro_producto , dpc_colorcom\
                        ")
    df0 = pds.read_sql(QueryCompr, dbConnection,params={"params": paramsval})
    dbConnection.close();
    return df0


def ExtractParms():
    DataParam = pds.read_excel('Parámetros.xlsx', sheet_name = "Referencias")
    DataParam = DataParam['Código'].dropna()
    DataParam = DataParam.astype('int32')
    DataParam = DataParam.astype('str')
    return DataParam

def ExtractProov():
    DataProov = pds.read_excel('Parámetros.xlsx', sheet_name = "Referencias")
    DataProov = DataProov[['Código','Nombre PBI','Proveedor']].dropna()
    DataProov['Código'] = DataProov['Código'].astype('int32')
    DataProov['Código'] = DataProov['Código'].astype('str')
    return DataProov


tablaCalculada = Tablafinal()

tablaCalculada['objetivo'] = tablaCalculada['RotacionRecal'].apply(asignar_etiqueta)
print(tablaCalculada)


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
