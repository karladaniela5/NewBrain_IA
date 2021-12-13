#Aquí se encuentran las instrucciones para el algoritmo ÁRBOLES
import streamlit as st
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection

def arbolesD(datos):

	if datos is not None:
		st.subheader("Selección del algoritmo")
		menuA = st.selectbox('Elige el tipo del algoritmo',['Clasificación','Regresión'])

		if menuA=='Clasificación':
			menuB = st.sidebar.selectbox('Elige la opción a mostrar',['Archivos cargados','Mapa de calor','Aplicación del algoritmo'])
			datos = datos.replace({'M': 0, 'B': 1})
			if menuB=='Mapa de calor':
				mapaCalor(datos)

			if menuB=='Aplicación del algoritmo':
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

						st.header("Aplicación del algoritmo")
												

						test = st.slider('Ingresa la cantidad de datos que se utilizaran para el test:', min_value=0.0, max_value=1.0,value=0.2)
						X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,
							test_size = test,random_state = 0,shuffle = True)
						
						st.subheader("Ingresa los valores para ")

						max_depthC = st.number_input('Inserta la profundidad máxima (max_depth): ',min_value=0,value=8)
						min_samples_splitC = st.number_input('Inserta la cantidad mínima de resultados (min_samples_split):'
							,min_value=0,value=4)
						min_samples_leafC = st.number_input('Inserta la cantidad mínima de elementos por nodo (min_samples_leaf): '
							,min_value=0,value=2)	

						ClasificacionAD = DecisionTreeClassifier(max_depth=max_depthC, min_samples_split=min_samples_splitC, 
							min_samples_leaf=min_samples_leafC)
						ClasificacionAD.fit(X_train, Y_train)
						Y_Clasificacion = ClasificacionAD.predict(X_validation)
						Valores = pd.DataFrame(Y_validation, Y_Clasificacion)

						
						st.subheader("Validación vs Predicción")
						st.write(Valores)

						eficiencia=ClasificacionAD.score(X_validation, Y_validation)
						st.subheader("Score del modelo: "+ str(eficiencia))

						st.subheader("Matriz de clasificación:")
						Y_Clasificacion = ClasificacionAD.predict(X_validation)
						Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(),Y_Clasificacion,rownames=['Reales'],colnames=['Clasificación'])
						st.success("Verdaderos Positivos: "+str(Matriz_Clasificacion[0][0]))
						st.error("Falsos Negativos: "+str(Matriz_Clasificacion[0][1]))
						st.error("Falsos Positivos:  "+str(Matriz_Clasificacion[1][0]))
						st.success("Verdaderos Negativos: "+str(Matriz_Clasificacion[1][1]))

						st.subheader("Reporte de clasificación:")
						st.write("Exactitud", ClasificacionAD.score(X_validation, Y_validation))
						st.text(classification_report(Y_validation, Y_Clasificacion))

						st.subheader("Importancia de cada variable:")
						Importancia = pd.DataFrame({'Variable': list(datos[['Texture', 'Area', 'Smoothness','Compactness', 
							'Symmetry', 'FractalDimension']]),
						'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
						st.text(Importancia)

						st.header("Árbol:")#--------------Imprimierndo árbol
						imagen = st.checkbox('Imagen')
						texto = st.checkbox('Texto')

						if imagen:
							arbolDC=plt.figure(figsize=(16,16))  
							plot_tree(ClasificacionAD,feature_names = ['Texture', 'Area', 'Smoothness',
								'Compactness', 'Symmetry', 'FractalDimension'])
							st.pyplot(arbolDC)

						if texto:
							Reporte = export_text(ClasificacionAD,feature_names = ['Texture', 'Area', 
								'Smoothness','Compactness', 'Symmetry', 'FractalDimension'])
							st.text(Reporte)

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
						
							clasP=ClasificacionAD.predict(PacienteID2)

							if(st.button("Obtener diagnóstico")): 
								if clasP==0:
									st.subheader("El diagnóstico es:  maligno")
								else:
									st.subheader("El diagnóstico es:  benigno")

#----------------------------------------------REGRESIÓN------------------------------------------------------------------------------------
		if menuA=='Regresión':
			menuB = st.sidebar.selectbox('Elige la opción a mostrar',['Archivos cargados','Mapa de calor','Aplicación del algoritmo'])
			
			if menuB=='Mapa de calor':
				mapaCalor(datos)

			if menuB=='Aplicación del algoritmo':
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

						st.header("Aplicación del algoritmo")
												

						test = st.slider('Ingresa la cantidad de datos que se utilizaran para el test:', min_value=0.0, max_value=1.0,value=0.2)
						X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,test_size = test,
							random_state = 1234,shuffle = True)
						
						st.subheader("Ingresa los valores para entrenar el modelo ")

						max_depthP = st.number_input('Inserta la profundidad máxima (max_depth): ',min_value=0,value=8)
						min_samples_splitP = st.number_input('Inserta la cantidad mínima de resultados (min_samples_split):'
							,min_value=0,value=4)
						min_samples_leafP = st.number_input('Inserta la cantidad mínima de elementos por nodo (min_samples_leaf): '
							,min_value=0,value=2)	

						PronosticoAD = DecisionTreeRegressor(max_depth=max_depthP, 
							min_samples_split=min_samples_splitP, min_samples_leaf=min_samples_leafP)	
						PronosticoAD.fit(X_train, Y_train)

						Y_Pronostico = PronosticoAD.predict(X_test)

						Valores = pd.DataFrame(Y_test, Y_Pronostico)
						st.subheader("Validación vs Predicción")
						st.text(Valores)

						eficiencia=r2_score(Y_test, Y_Pronostico)
						st.subheader("Score del modelo: "+ str(eficiencia))


						st.header("Reporte de clasificación:")
						st.write('Criterio: \n', PronosticoAD.criterion) #SON LOS ERRORES
						st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
						st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
						st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   

						st.subheader("Importancia de cada variable:")
						Importancia = pd.DataFrame({'Variable': list(datos[['Texture', 'Perimeter', 'Smoothness',
							'Compactness', 'Symmetry', 'FractalDimension']]),
						'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
						st.write(Importancia)

						st.header("Árbol:")#--------------Imprimierndo árbol
						imagen = st.checkbox('Imagen')
						texto = st.checkbox('Texto')

						if imagen:
							st.subheader("Árbol en imagen:")
							arbolP=plt.figure(figsize=(16,16))  
							plot_tree(PronosticoAD, feature_names = ['Texture', 'Perimeter', 'Smoothness',
								'Compactness', 'Symmetry', 'FractalDimension'])
							st.pyplot(arbolP)


						if texto:
							st.subheader("Árbol en texto:")
							ReporteP = export_text(PronosticoAD, feature_names = ['Texture', 'Perimeter', 
								'Smoothness','Compactness', 'Symmetry', 'FractalDimension'])
							st.text(ReporteP)
							


						generalP = st.sidebar.checkbox('General',key='general')

						if generalP:
							st.title('PREDICCIÓN PARA EL USUARIO GENERAL')
							Textura = st.number_input('Inserta Texture: ',min_value=0.0,max_value=100000.0)
							Perimeter = st.number_input('Inserta Perimeter: ',min_value=0.0,max_value=100000.0)
							SmoothnessC = st.number_input('Inserta Smoothness: ',min_value=0.0,max_value=100000.0)
							CompactnessC = st.number_input('Inserta Compactness: ',min_value=0.0,max_value=100000.0)
							SymmetryC = st.number_input('Inserta Symmetry: ',min_value=0.0,max_value=100000.0)
							FractalDimensionC = st.number_input('Inserta FractalDimension: ',min_value=0.0,max_value=100000.0)

							AreaTumorID1 = pd.DataFrame({'Texture': [Textura],
								'Perimeter': [Perimeter],
								'Smoothness': [SmoothnessC],
								'Compactness': [CompactnessC],
								'Symmetry': [SymmetryC],
								'FractalDimension': [FractalDimensionC]})
						
							clasP=PronosticoAD.predict(AreaTumorID1)

							if(st.button("Obtener predicción")): 
								st.write("El área es de: ",clasP[0])

def mapaCalor(datos):
	st.subheader("Mapa de calor")
	Correlacion = datos.corr(method='pearson')
	st.dataframe(Correlacion)
	graficaCorr = plt.figure(figsize=(14,7))
	MatrizInf = np.triu(Correlacion)
	sns.heatmap(Correlacion, cmap='RdBu_r', annot=True, mask=MatrizInf)
	st.pyplot(graficaCorr)