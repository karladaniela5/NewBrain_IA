#Aquí se encuentran las instrucciones para el algoritmo CLÚSTERING PARTICIONAL
import streamlit as st
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D

def clusterP(datos):
	if datos is not None:
		menuP = st.sidebar.selectbox('Elige la opción a mostrar',['Archivos cargados','Mapa de calor','Selección de variables'])
		if menuP=='Mapa de calor':
			mapaCalor(datos)

		if menuP=='Selección de variables':
			st.subheader("Selección de variables")
			variablesP = st.multiselect('Selecciona las variables por considerar', datos.columns,default = datos.columns.all())
			if variablesP is not None:
				MatrizDatos = np.array(datos[variablesP])
				st.subheader('Datos ingresados:')
				pd.DataFrame(MatrizDatos)

				estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
				MEstandarizada = estandarizar.fit_transform(MatrizDatos)
				st.dataframe(MEstandarizada)

				st.write("Opción (realizar primero método de codo)")
				codo = st.checkbox('Metodo del codo')
				clusteres = st.checkbox('Clústeres')

				if codo:
					clusternum=codoMetodo(MEstandarizada)
					if clusteres:
						clusteresCentr(MEstandarizada,datos,variablesP,clusternum)
					
def mapaCalor(datos):
	st.subheader("Mapa de calor")
	Correlacion = datos.corr(method='pearson')
	st.dataframe(Correlacion)
	graficaCorr = plt.figure(figsize=(14,7))
	MatrizInf = np.triu(Correlacion)
	sns.heatmap(Correlacion, cmap='RdBu_r', annot=True, mask=MatrizInf)
	st.pyplot(graficaCorr)

def codoMetodo(MEstandarizada):
	#Se utiliza random_state para inicializar el generador interno de números aleatorios
	st.subheader('Método del codo')
	codoMax = st.slider("Ingresa el valor del codo: ",4,12)
	SSE = []

	for i in range(2, codoMax):
		km = KMeans(n_clusters=i, random_state=0) 
		km.fit(MEstandarizada)
		SSE.append(km.inertia_)

	codoG = plt.figure(figsize=(10, 7))
	plt.plot(range(2, codoMax), SSE, marker='o')
	plt.xlabel('Cantidad de clusters *k*')
	plt.ylabel('SSE')
	plt.title('Elbow Method')
	st.pyplot(codoG)
	k1 = KneeLocator(range(2, codoMax), SSE, curve="convex", direction="decreasing")
	
	st.write("Localización del codo: " + str(k1.elbow))
	return k1.elbow


def clusteresCentr(MEstandarizada,datos,variablesP,numclusters):
	MParticional = KMeans(n_clusters=numclusters, random_state=0).fit(MEstandarizada)
	MParticional.predict(MEstandarizada)
	MParticional.labels_

	datosCP = datos[variablesP]
	datosCP['clusterP'] = MParticional.labels_
	st.subheader("Clústeres")
	st.dataframe(datosCP) 
									
	CentroidesP = datosCP.groupby('clusterP').mean()
	st.header("Centroides de los clústeres: ")
	st.dataframe(CentroidesP)

	st.write("Opción")
	grafica3d = st.checkbox('Gráfica 3D')
	if grafica3d:
		try:
			st.header("Gráfica 3D")
			plt.rcParams['figure.figsize'] = (10, 7)
			plt.style.use('ggplot')
			colores=['red', 'blue', 'green', 'yellow','purple']
			asignar=[]
			for row in MParticional.labels_:
				asignar.append(colores[row])
			
			graficaP = plt.figure()
			ax = Axes3D(graficaP)
			ax.scatter(MEstandarizada[:, 0],
			MEstandarizada[:, 1],
			MEstandarizada[:, 2], marker='o', c=asignar, s=60)
			ax.scatter(MParticional.cluster_centers_[:, 0], MParticional.cluster_centers_[:, 1],
			MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
			st.pyplot(graficaP)
		except:
			st.error("No fue posible obtener la gráfica en 3D. Nota: Solo se puede generar la gráfica para 4 clústeres")