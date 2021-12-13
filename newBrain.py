#Importamos las bibliotecas necesarias
import streamlit as st
import pandas as pd
from PIL import Image          #Biblioteca para imágenes
from newBrain_paginas import * #Importamos la biblioteca donde se encuentran las demás páginas


#PÁGINA PRINCIPAL
st.set_page_config(page_title='NEW BRAIN')
st.markdown("<span style=“background-color:#AF7AC5”>",unsafe_allow_html=True)
#Creamos el widget donde se selecciona el algoritmo
st.sidebar.title("NEW BRAIN")

elegirMenu=st.sidebar.selectbox('Elije una algoritmo',['Página principal','Apriori-Reglas de asociación',
								'Métricas de distancia','Clústering jerárquico','Clútering particional',
								'Clasificación (R. Logistica)','Árboles de decisión(Regresión y Clasificación)'])


st.sidebar.markdown('---')

if elegirMenu=='Página principal':
	paginaInicio()
elif elegirMenu=='Apriori-Reglas de asociación':
	aReglas()
elif elegirMenu=='Métricas de distancia':
	metricas()
elif elegirMenu=='Clústering jerárquico':
	clusteringJ()
elif elegirMenu=='Clútering particional':
	clusteringP()
elif elegirMenu=='Clasificación (R. Logistica)':
	clasificacion()
elif elegirMenu=='Árboles de decisión(Regresión y Clasificación)':
	arboles()
