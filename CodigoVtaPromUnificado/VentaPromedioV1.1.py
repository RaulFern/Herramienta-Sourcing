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


uname="u_sait_col"
pwd=parse.quote("@*8XS9?=RrskFE3A")

# url para la carga del archivo de inventario consolidado, se encuentra en en un servidor de Colombia
urlCarga = "//10.56.20.98/gu/Textil/JohnUribe/Sourcing/Power BI/"
#pwd=parse.quote("@*8XS9?=RrskFE3A")

#conexion a la base de datos, para descargar el inventario consolidado, que trata de una funncion SQL el cual se lde direcciona John Uribe
def conexionbase2pruebas(paramsval):
    alchemyEngine   = create_engine('postgresql+psycopg2://{user}:{pw}@bdsait.gco.com.co/db_sait_col_prod'.format(user=uname, pw=pwd), pool_recycle=3600)
    dbConnection    = alchemyEngine.connect();
    QueryCompr = text("select producto,Diseño, color,  grupoventa, sum(cantidad)Cantidad,sum(valorbruto)ValorBruto,\
                        sum(valorneto)ValorNeto, sum(cantidad)/6 vtapromedio from (\
                        SELECT producto,  Diseño, color,  grupoventa, sum(cantidad)Cantidad\
                        ,sum(valorbruto)ValorBruto, sum(valorneto)ValorNeto,mescontable, current_date - fechadocumento AS diferencia_dias\
                        FROM   (SELECT min_fechadoc       FechaDocumento,\
				                    pro_producto       Producto,\
                                    dpc_disenocom      Diseño,\
                                    dpc_colorcom       Color,\
                                    C.grp_grupo       GrupoVenta,\
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
                        grp_grupo            GrupoVenta,\
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
                        grp_grupo            GrupoVenta,\
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
                        group by fechadocumento ,Producto, Diseño,Color,GrupoVenta,mescontable\
                        )final\
                        group by producto,Diseño, color, grupoventa\
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



datosproov = ExtractProov()

datosArticulos = ExtractParms()
datosArticulos = tuple(datosArticulos)
print(datosArticulos)



dataf= conexionbase2pruebas(datosArticulos)
dataf['producto'] = dataf['producto'].astype('int32')
dataf['producto'] = dataf['producto'].astype('str')
print(dataf)

DF = pds.merge(dataf,datosproov, how='left', left_on=['producto'], right_on=['Código'])#.drop(['LicTradNum'], axis=1)

DF.to_excel("VentaPromedioFebrero2024.xlsx", index=False)

#dataf= conexionbase2pruebas()
#dataf = pds.read_excel('IMPORTACIONES JOHN URIBE 2023 (7) (2).xlsx', sheet_name = "DATOS")
#print(dataf)
#print(dataf.info())

#dataImpo = dataf[["PR0VEED0R ","N. FACTURA","VAL0R DE FACTURA U$","MES INGR. AL INVEN.","REFERENCIA (TELA)","ESTAD0 DE IMP0RTACIÓN"]]

#print(dataImpo)

#dataImpo1 = dataImpo[(dataImpo['ESTAD0 DE IMP0RTACIÓN']== '05-MCÍA.EN PUERTO PEND.X NACIONALIZAR') |(dataImpo['ESTAD0 DE IMP0RTACIÓN'] == '04-MCÍA.EN VÍA (NAVEGANDO)')]

#print(dataImpo1)

#sns.scatterplot(x='pru_nrounidades', y='dpe_nrounidades', data=dataf, hue='pro_producto')
#dataf.to_excel("InventarioCons270124.xlsx")
#dataImpo1 = dataImpo1.astype({"PR0VEED0R ":'string',"N. FACTURA":'string',
#                "VAL0R DE FACTURA U$": 'float64',
#                "MES INGR. AL INVEN.":'string',"REFERENCIA (TELA)":'string',"ESTAD0 DE IMP0RTACIÓN":'string'})

#print(dataImpo1.info())

# hostname="10.100.200.31"
# dbname="TM"
# uname="sa"
# pwd=parse.quote("D1str1t3x")
# engine = create_engine("mssql+pymssql://{user}:{pw}@{host}/{db}"
# .format(host=hostname, db=dbname, user=uname, pw=pwd))
#
# # Crear un objeto de conexión y cursor
# with engine.connect() as dbConnection, dbConnection.begin():
#     # Aquí comienza la transacción
#     try:
#         # Insertar datos en la tabla 'IMPOTRANSIT'
# dbConnection= engine.connect();
# dataImpo1.to_sql('IMPOTRANSIT', con=dbConnection, if_exists='replace', index=False)
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


# dataImpo.to_excel("IMPO_ACT.xlsx", index=False)
# dataImpo1.to_excel("IMPO_FINAL.xlsx", index=False)
#dataf.to_excel(urlCarga+"InventarioConsolidado.xlsx", index=False)
