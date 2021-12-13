  						

#Métricas de distancia
import streamlit as st
import numpy as np						
import pandas as pd
import matplotlib.pyplot as plt     		# Para la generación de gráficas a partir de los datos
from scipy.spatial.distance import cdist 	#Para el cálculo de distancias
from scipy.spatial import distance						
from newBrain_simp import *


def metricasDistancia(datos):
	if datos is not None:
		st.sidebar.write(txtdistancias)
		elegirMenu=st.sidebar.selectbox('Elije una opción',['Archivos cargados','Distancia Euclidiana','Distancia de Chebyshev',
								'Distancia de Manhattan','Distancia de Minkowsky'])
		if elegirMenu=='Distancia Euclidiana':
			Euclidiana(datos)
		elif elegirMenu=='Distancia de Chebyshev':
			Chebyshev(datos)
		elif elegirMenu=='Distancia de Manhattan':
			Manhattan(datos)
		elif elegirMenu=='Distancia de Minkowsky':
			Minkowsky(datos)



def Euclidiana(datos):
	if datos is not None:
		st.title('DISTANCIA EUCLIDIANA')
		DstEuclidiana=cdist(datos, datos, metric='euclidean')  #computa la distancia entre dos colecciones de objetos
		MEuclidiana =pd.DataFrame(DstEuclidiana)
		st.dataframe(MEuclidiana)
		
		st.write("Distancia entre dos objetos:")
	
		level1 = st.slider('Objeto 1:', 0, len(MEuclidiana)-1)
		level2 = st.slider('Objeto 2:', 0, len(MEuclidiana)-1)
		
		Objeto1=datos.iloc[level1]
		Objeto2=datos.iloc[level2]

		distanciaEuclidiana = distance.euclidean(Objeto1, Objeto2)

		st.write("La distancia entre los objetos es: "+str(distanciaEuclidiana))

def Chebyshev(datos):
	if datos is not None:
		st.title('DISTANCIA CHEBYSHEV')
		DstChebyshev = cdist(datos, datos, metric='chebyshev')
		MChebyshev = pd.DataFrame(DstChebyshev)
		st.dataframe(MChebyshev)

		st.write("Distancia entre dos objetos:")
	
		level1 = st.slider('Objeto 1:', 0, len(MChebyshev)-1)
		level2 = st.slider('Objeto 2:', 0, len(MChebyshev)-1)
		
		Objeto1=datos.iloc[level1]
		Objeto2=datos.iloc[level2]

		distanciaChebyshev= distance.chebyshev(Objeto1, Objeto2)

		st.write("La distancia entre los objetos es: "+str(distanciaChebyshev))


def Manhattan(datos):
	if datos is not None:
		st.title('DISTANCIA MANHATTAN')
		DstManhattan = cdist(datos, datos, metric='cityblock')
		MManhattan = pd.DataFrame(DstManhattan)
		st.dataframe(MManhattan)

		st.write("Distancia entre dos objetos:")
	
		level1 = st.slider('Objeto 1:', 0, len(MManhattan)-1)
		level2 = st.slider('Objeto 2:', 0, len(MManhattan)-1)
		
		Objeto1=datos.iloc[level1]
		Objeto2=datos.iloc[level2]

		distanciaMManhattan= distance.cityblock(Objeto1, Objeto2)

		st.write("La distancia entre los objetos es: "+str(distanciaMManhattan))

def Minkowsky(datos):
	if datos is not None:
		st.title('DISTANCIA MINKOWSKY')
		DstMinkowski = cdist(datos, datos, metric='minkowski', p=1.5)
		MMinkowski = pd.DataFrame(DstMinkowski)
		st.dataframe(MMinkowski)

		st.write("Distancia entre dos objetos:")
	
		level1 = st.slider('Objeto 1:', 0, len(MMinkowski)-1)
		level2 = st.slider('Objeto 2:', 0, len(MMinkowski)-1)
		
		Objeto1=datos.iloc[level1]
		Objeto2=datos.iloc[level2]

		distanciaMMinkowski= distance.minkowski(Objeto1, Objeto2)

		st.write("La distancia entre los objetos es: "+str(distanciaMMinkowski))