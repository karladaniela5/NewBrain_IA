#Aquí se encuentran las instrucciones para el algoritmo CLÚSTERING JERÁRQUICO
import streamlit as st
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering 

def clusterJ(datos):
	if datos is not None:
		menuC = st.sidebar.selectbox('Elige la opción a mostrar',['Archivos cargados','Mapa de calor','Selección de variables'])
		if menuC=='Mapa de calor':
			mapaCalor(datos)

		if menuC=='Selección de variables':
			st.subheader("Selección de variables")
			variablesC = st.multiselect('Selecciona las variables por considerar', datos.columns,default = datos.columns.all())
			if variablesC is not None:
				MatrizDatos = np.array(datos[variablesC])
				st.subheader('Datos ingresados:')
				pd.DataFrame(MatrizDatos)
				estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
				MEstandarizada = estandarizar.fit_transform(MatrizDatos)
				st.dataframe(MEstandarizada)

				st.subheader("Elige una opción")
				arbolC = st.checkbox('Árbol clústering jerárquico')
				clusteres = st.checkbox('Elegir número de clústeres')

				if arbolC:
					arbolCentr(MEstandarizada)
				if clusteres:
					clusteresCentr(MEstandarizada,datos,variablesC)
					
def mapaCalor(datos):
	st.subheader("Mapa de calor")
	Correlacion = datos.corr(method='pearson')
	st.dataframe(Correlacion)
	graficaCorr = plt.figure(figsize=(14,7))
	MatrizInf = np.triu(Correlacion)
	sns.heatmap(Correlacion, cmap='RdBu_r', annot=True, mask=MatrizInf)
	st.pyplot(graficaCorr)

def arbolCentr(MEstandarizada):
	st.subheader("Árbol clústering jerárquico")
	arbolG=plt.figure(figsize=(10, 7))
	plt.title("Casos de hipoteca")
	plt.xlabel('Hipoteca')
	plt.ylabel('Distancia')
	Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
	nivelA = st.number_input('Inserta el nivel: ',min_value=0.0,max_value=7.0)
	plt.axhline(y=nivelA, color='black', linestyle='--') 
	st.pyplot(arbolG)



def clusteresCentr(MEstandarizada,datos,variablesC):
	st.subheader("Número de clústeres")
	numclusters = st.slider('Inserta la cantidad de clústeres:', 1, 7)
	MJerarquico = AgglomerativeClustering(n_clusters=7, linkage='complete', affinity='euclidean')
	MJerarquico.fit_predict(MEstandarizada)

	MJerarquico = AgglomerativeClustering(n_clusters=numclusters, linkage='complete', affinity='euclidean')
	MJerarquico.fit_predict(MEstandarizada)

	datosCJ = datos[variablesC]
	datosCJ['clusterJ'] = MJerarquico.labels_
	st.subheader("Clústeres")
	st.dataframe(datosCJ) 
									
	CentroidesH = datosCJ.groupby('clusterJ').mean()
	st.header("Centroides de los clústeres: ")
	st.dataframe(CentroidesH)



	graficaJ = st.checkbox('Gráfica de clústeres')
	if graficaJ:
		graficaJ =plt.figure(figsize=(10, 7))
		plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MJerarquico.labels_)
		plt.grid()
		st.pyplot(graficaJ)
