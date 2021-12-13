#Importamos las bibliotecas necesarias
import streamlit as st
import numpy as np							
import pandas as pd
from PIL import Image          #Biblioteca para imágenes 					
from newBrain_simp import *    #Importamos la biblioteca donde se encuentra el texto
from newBrain_metricas import * 
from newBrain_apriori import * 
from newBrain_ClusteringJ import * 
from newBrain_ClusteringP import * 
from newBrain_Clasificacion import * 
from newBrain_Arboles import * 


#Definimos nuestra página de inicio
def paginaInicio():
	st.title("NEW BRAIN")
	brainImage = 'https://www.agenciasinc.es/var/ezwebin_site/storage/images/noticias/nuevo-sistema-de-deteccion-de-patrones-neuroinspirado/6443746-1-esl-MX/Nuevo-sistema-de-deteccion-de-patrones-neuroinspirado_image_380.png'
	st.image(brainImage)
	st.write(txtinicio) 
	st.header("By: Santillán Serafín Karla Daniela")

	


def aReglas():
	st.title('APRIORI-REGLAS DE ASOCIACIÓN')
	st.write(txtapriori)
	aprioriImage = 'https://www.innovaspain.com/wp-content/uploads/2020/01/algoritmo.png'
	st.image(aprioriImage,width=700)
	sDatosMovies = st.file_uploader("Ingrese el archivo donde se encuentran sus datos, debe de tener extensión csv o xlsx", type = ["csv",'xlsx'],key='apriori') 
	if sDatosMovies is not None: #Verificamos que se ingrese un archivo
		st.success("Archivo cargado con éxito")
		sDatosMovies = pd.read_csv(sDatosMovies, header=None)	 #Leemos los datos del archivo y quitamos el encabezado
		st.write('Datos del archivo cargado:')    
		st.dataframe(sDatosMovies) #Muestra el contenido de la tabla
		aprioriRA(sDatosMovies)
	


#Definimos la página de métricas de distancia
def metricas():
	st.title('MÉTRICAS DE DISTANCIA')
	st.write(txtmetricas)
	metricaImage = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Manhattan_distance.svg/640px-Manhattan_distance.svg.png'
	st.image(metricaImage,width=400)
	datos = st.file_uploader("Ingrese el archivo donde se encuentran sus datos, debe de tener extensión csv o xlsx", type = ["csv",'xlsx'],key='metricas') 
	if datos is not None: #Verificamos que se ingrese un archivo
		st.success("Archivo cargado con éxito")
		datos = pd.read_csv(datos) 	 #Leemos los datos del archivo
		st.write('Datos del archivo cargado:')    
		st.dataframe(datos) #Muestra el contenido de la tabla
		metricasDistancia(datos)

#Definimos la página de clústering jerárquico
def clusteringJ():
	st.title('CLÚSTERING JERÁRQUICO')
	st.write(txtclustjerar)
	metricaImage = 'https://www.eescorporation.com/wp-content/uploads/2021/11/Clustering-in-Machine-Learning.jpeg'
	st.image(metricaImage,width=700)
	datos = st.file_uploader("Ingrese el archivo donde se encuentran sus datos, debe de tener extensión csv o xlsx", type = ["csv",'xlsx'],key='clusteringJ') 
	if datos is not None: #Verificamos que se ingrese un archivo
		st.success("Archivo cargado con éxito")
		datos = pd.read_csv(datos) 	 #Leemos los datos del archivo
		st.write('Datos del archivo cargado:')    
		st.dataframe(datos) #Muestra el contenido de la tabla
		clusterJ(datos)

#Definimos la página de clústering jerárquico
def clusteringP():
	st.title('CLÚSTERING PARTICIONAL')
	st.write(txtclustpart)
	metricaImage = 'https://www.eescorporation.com/wp-content/uploads/2021/11/Clustering-in-Machine-Learning.jpeg'
	st.image(metricaImage,width=700)
	datos = st.file_uploader("Ingrese el archivo donde se encuentran sus datos, debe de tener extensión csv o xlsx", type = ["csv",'xlsx'],key='clusteringP') 
	if datos is not None: #Verificamos que se ingrese un archivo
		st.success("Archivo cargado con éxito")
		datos = pd.read_csv(datos) 	 #Leemos los datos del archivo
		st.write('Datos del archivo cargado:')    
		st.dataframe(datos) #Muestra el contenido de la tabla
		clusterP(datos)


#Definimos la página de clústering jerárquico
def clasificacion():
	st.title('CLASIFICACIÓN-REGRESIÓN LOGÍSTICA')
	st.write(txtclasifi)
	metricaImage = 'https://miro.medium.com/max/618/0*1lcADFIn6tymtr_1.png'
	st.image(metricaImage,width=500)
	datos = st.file_uploader("Ingrese el archivo donde se encuentran sus datos, debe de tener extensión csv o xlsx", type = ["csv",'xlsx'],key='clasificacion') 
	if datos is not None: #Verificamos que se ingrese un archivo
		st.success("Archivo cargado con éxito")
		datos = pd.read_csv(datos) 	 #Leemos los datos del archivo
		st.write('Datos del archivo cargado:')    
		st.dataframe(datos) #Muestra el contenido de la tabla
		clasificacionRL(datos)

#Definimos la página de clústering jerárquico
def arboles():
	st.title('ÁRBOLES DE DESICIÓN')
	st.write(txtarboles)
	metricaImage = 'https://www.inbenta.com/wp-content/uploads/2017/10/inbenta.jpg'
	st.image(metricaImage,width=450)
	datos = st.file_uploader("Ingrese el archivo donde se encuentran sus datos, debe de tener extensión csv o xlsx", type = ["csv",'xlsx'],key='arboles') 
	if datos is not None: #Verificamos que se ingrese un archivo
		st.success("Archivo cargado con éxito")
		datos = pd.read_csv(datos) 	 #Leemos los datos del archivo
		st.write('Datos del archivo cargado:')    
		st.dataframe(datos) #Muestra el contenido de la tabla
		arbolesD(datos)


 