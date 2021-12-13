#Aquí se encuentran las instrucciones para el algoritmo CLASIFICAIÓN
import streamlit as st
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



def clasificacionRL(datos):

	if datos is not None:
		menuP = st.sidebar.selectbox('Elige la opción a mostrar',['Archivos cargados','Mapa de calor','Aplicación del algoritmo'])
		datos=datos.replace({'M': 0, 'B': 1})
		if menuP=='Mapa de calor':
			mapaCalor(datos)

		if menuP=='Aplicación del algoritmo':
			st.sidebar.write("Usuarios:")
			experto = st.sidebar.checkbox('Experto')

			if experto:
				st.header("Selección de variables")
				variablesPre = st.multiselect('Selecciona las variables predictorias', datos.columns,default = datos.columns.all())
				variablesCla = st.multiselect('Selecciona la variable clase (Ingrese solo una)', datos.columns,default = datos.columns.all())

				if variablesPre and variablesCla is not None:
					X = np.array(datos[variablesPre])
					Y = np.array(datos[variablesCla])
					st.subheader('Variables predictorias:')
					st.dataframe(X)
					st.subheader('Variable clase:')
					st.dataframe(Y)

					graficaCLA = st.checkbox('Gráfica', key='graficaCLA')

					if graficaCLA:

						st.header("Gráfica:")
						graficaCl=plt.figure(figsize=(10, 7))
						plt.scatter(X[:,0], X[:,1], c = datos.Diagnosis)
						plt.grid()
						plt.xlabel('Texture')
						plt.ylabel('Area')
						st.pyplot(graficaCl)

					st.header("Aplicación del algoritmo")

					test = st.slider('Ingresa la cantidad de datos que se utilizaran para el test:', min_value=0.0, max_value=1.0,value=0.2)
					X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,test_size = test,
						random_state = 1234,shuffle = True)

					Clasificacion = linear_model.LogisticRegression()
					Clasificacion.fit(X_train, Y_train)

					st.subheader("Predicciones con clasificación final")
					Predicciones = Clasificacion.predict(X_validation)
					st.dataframe(Predicciones)

					eficiencia=Clasificacion.score(X_validation, Y_validation)
					st.subheader("Score del modelo: "+ str(eficiencia))
					

					st.subheader("Matriz de clasificación:")
					Y_Clasificacion = Clasificacion.predict(X_validation)
					Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(),Y_Clasificacion,rownames=['Reales'],colnames=['Clasificación'])
					st.success("Verdaderos Positivos: "+str(Matriz_Clasificacion[0][0]))
					st.error("Falsos Negativos: "+str(Matriz_Clasificacion[0][1]))
					st.error("Falsos Positivos:  "+str(Matriz_Clasificacion[1][0]))
					st.success("Verdaderos Negativos: "+str(Matriz_Clasificacion[1][1]))
					
					st.subheader("Reporte de clasificación:")
					st.text(classification_report(Y_validation, Y_Clasificacion))

					general = st.sidebar.checkbox('General',key='general')

					if general:
						st.title('PREDICCIÓN PARA EL USUARIO GENERAL')
						Textura = st.number_input('Inserta Texture: ',min_value=0.0,max_value=100000.0)
						area = st.number_input('Inserta Area: ',min_value=0.0,max_value=100000.0)
						SmoothnessC = st.number_input('Inserta Smoothness: ',min_value=0.0,max_value=100000.0)
						CompactnessC = st.number_input('Inserta Compactness: ',min_value=0.0,max_value=100000.0)
						SymmetryC = st.number_input('Inserta Symmetry: ',min_value=0.0,max_value=100000.0)
						FractalDimensionC = st.number_input('Inserta FractalDimension: ',min_value=0.0,max_value=100000.0)

						PacienteID2 = pd.DataFrame({'Texture': [Textura],
							'Area': [area],
							'Smoothness': [SmoothnessC],
							'Compactness': [CompactnessC],
							'Symmetry': [SymmetryC],
							'FractalDimension': [FractalDimensionC]})
						
						clasP=Clasificacion.predict(PacienteID2)

						if(st.button("Obtener diagnóstico")): 
							if clasP==0:
								st.subheader("El diagnóstico es:  maligno")
							else:
								st.subheader("El diagnóstico es:  benigno")


				
def mapaCalor(datos):
	st.subheader("Mapa de calor")
	Correlacion = datos.corr(method='pearson')
	st.dataframe(Correlacion)
	graficaCorr = plt.figure(figsize=(14,7))
	MatrizInf = np.triu(Correlacion)
	sns.heatmap(Correlacion, cmap='RdBu_r', annot=True, mask=MatrizInf)
	st.pyplot(graficaCorr)