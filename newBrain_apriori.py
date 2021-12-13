#Aquí se encuentran las instrucciones para el algoritmo APRIORI
import streamlit as st
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori

def aprioriRA(sDatosMovies):
	if sDatosMovies is not None:

		Transacciones = sDatosMovies.values.reshape(-1).tolist()

		Lista = pd.DataFrame(Transacciones)
		Lista['Frecuencia'] = 1
		Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo #El false indica que no se muestre el 0 que aparece en la primera comluna, count los cuenta,  sort los acomoda, primero los que menos aparecen
		Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum())  #se crea la columna Porcentaje y se obtiene el porcentaje de cuantas veces se han visto las peliculas
		Lista = Lista.rename(columns={0 : 'Item'}) #se cambia el nombre de la columna
		st.subheader("Frecuencias y porcentajes")
		st.dataframe(Lista)

		st.subheader("Gráfica de las frecuencias")
		grafica = plt.figure(figsize=(20,30))
		plt.ylabel('Item')
		plt.xlabel('Frecuencia')
		plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='purple')
		st.pyplot(grafica)

		MoviesLista = sDatosMovies.stack().groupby(level=0).apply(list).tolist()
		st.subheader("Ingresa los valores")
		soporte = st.number_input('Soporte:', 0.01, 2.00)
		confianza = st.number_input('Confianza:', 0.00, 1.00)
		elevacion = st.number_input('Elevacion:', 0.0, 5.0)

		Reglas = apriori(MoviesLista,min_support=soporte,min_confidence=confianza,min_lift=elevacion)
		Resultados = list(Reglas) 

		resultado = st.checkbox('Imprimir resultados')	

		if resultado:
			st.subheader("Reglas de asociación")
			st.write("Total de reglas encontradas:",len(Resultados))
			regla = 1
			for item in Resultados:
				#El primer índice de la lista
				Emparejar = item[0]
				items = [x for x in Emparejar]
				st.write("Regla" + str(regla)+": "+ str(item[0]))

				#El segundo índice de la lista
				st.write("Soporte: " + str(item[1]))

				#El tercer índice de la lista
				st.write("Confianza: " + str(item[2][0][2]))
				st.write("Lift: " + str(item[2][0][3])) 
				st.write("=====================================")
				regla += 1 


