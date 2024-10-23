from flask import Flask, jsonify, request, json
from flask_mysqldb import MySQL
from config import config

from flask_cors import CORS

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import spacy

import random
import numpy as np

import warnings
import tensorflow as tf

app=Flask(__name__)
conexion=MySQL(app)
# Configura CORS para permitir todas las solicitudes desde cualquier origen
CORS(app)

#RUTAS PARA HACER CRUD SOBRE LA TABLA CURSOS
@app.route('/api/cursos/listar/<id>', methods=['GET'])
def listar_cursos(id):
    try:
        profesor=buscarNombre(id)
        cursor=conexion.connection.cursor()
        sql="SELECT id_curso,codigo,nombre,creditos,nombre_profesor FROM curso WHERE nombre_profesor = '{0}'".format(profesor) 
        cursor.execute(sql)
        datos=cursor.fetchall()
        cursos=[]
        for fila in datos:
            curso={'id_curso':fila[0],'codigo':fila[1],'materia':fila[2],'creditos':fila[3],'nombre_profesor':fila[4]}
            cursos.append(curso)
        return jsonify(cursos)
    except Exception as ex:
        return "ERROR"
def buscarNombre(id):
    cursor=conexion.connection.cursor()
    sql="SELECT nombre FROM maestros WHERE id = '{0}'".format(id)
    cursor.execute(sql)
    datos=cursor.fetchone()
    if datos is not None:
        return datos[0]
    else:
        return -1

@app.route('/api/cursos/buscar/<codigo>',methods=['GET'])
def leer_curso(codigo):
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT codigo,nombre,creditos,nombre_profesor FROM curso WHERE id_curso = '{0}'".format(codigo)
        cursor.execute(sql)
        datos=cursor.fetchone()
        if datos is not None:
            curso={'codigo':datos[0],'materia':datos[1],'creditos':datos[2],'profesor':datos[3],'id_profesor':buscarID(datos[3])}
            return jsonify(curso)
        else:
            return jsonify({'mensaje':"Curso NO Encontrado"})
    except Exception as ex:
        return "ERROR"

@app.route('/api/cursos/alta',methods=['POST'])
def registrar_curso():
    try:
        profesor=request.json['profesor']
        cursor=conexion.connection.cursor()
        sql="INSERT INTO curso (codigo, nombre, creditos, nombre_profesor) VALUES ('{0}','{1}','{2}','{3}')".format(request.json['codigo'],request.json['materia'],request.json['creditos'],request.json['profesor'])
        cursor.execute(sql)
        conexion.connection.commit()
        return jsonify({'id':buscarID(profesor)})
    except Exception as ex:
        return "ERROR"
def buscarID(nombre):
    cursor=conexion.connection.cursor()
    sql="SELECT id FROM maestros WHERE nombre = '{0}'".format(nombre)
    cursor.execute(sql)
    datos=cursor.fetchone()
    return datos[0]

@app.route('/api/cursos/eliminar/<codigo>',methods=['DELETE'])
def eliminar_curso(codigo):
    try:
        cursor=conexion.connection.cursor()
        sql="DELETE FROM curso WHERE id_curso = '{0}'".format(codigo)
        cursor.execute(sql)
        conexion.connection.commit()
        return jsonify({'mensaje':"Curso Eliminado"})

    except Exception as ex:
        return "ERROR"

@app.route('/api/cursos/actualizar/<idCurso>',methods=['PUT'])
def actualizar_curso(idCurso):
    try:
        print(request.json)
        cursor=conexion.connection.cursor()
        sql="UPDATE curso SET codigo='{1}',nombre='{2}',creditos='{3}',nombre_profesor='{4}' WHERE id_curso='{0}'".format(idCurso,request.json['codigo'],request.json['materia'],request.json['creditos'],request.json['profesor'],)
        cursor.execute(sql)
        conexion.connection.commit()
        idProfesor=buscarID(request.json['profesor'])
        return jsonify({'id':idProfesor})

    except Exception as ex:
        return "ERROR"

@app.route('/api/cursos/codigos/<id>',methods=['GET'])
def GetCodigosCursos(id):
    try:
        nombre=buscarNombre(id)
        cursor=conexion.connection.cursor()
        sql="SELECT codigo FROM curso WHERE nombre_profesor='{0}';".format(nombre) 
        cursor.execute(sql)
        datos=cursor.fetchall()
        cursos=[]
        for fila in datos:
            curso={'codigo':fila[0]}
            cursos.append(curso)
        return jsonify(cursos)
    except Exception as ex:
        raise "ERROR"

#RUTAS PARA ENROLARSE A CURSO   
@app.route('/api/alumnos/curso/alta',methods=['POST'])
def EnrolarseClase():
    try:
        alumno=request.json['idalumno']
        cursor=conexion.connection.cursor()
        sql="INSERT INTO matriculas(id_curso, id_alumno) VALUES ('{0}','{1}')".format(request.json['idcurso'],request.json['idalumno'])
        cursor.execute(sql)
        conexion.connection.commit()
        return jsonify({'alumno':alumno})
    except Exception as ex:
        return "ERROR"

@app.route('/api/alumnos/curso/listar',methods=['GET'])
def ListarCursosAlumnos():
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT id_curso,codigo,nombre,creditos,nombre_profesor FROM curso"
        cursor.execute(sql)
        datos=cursor.fetchall()
        cursos=[]
        for fila in datos:
            curso={'id_curso':fila[0],'codigo':fila[1],'materia':fila[2],'creditos':fila[3],'profesor':fila[4]}
            cursos.append(curso)
        return jsonify(cursos)
    except Exception as ex:
        return "ERROR"

#RUTAS PARA HACER CRUD SOBRE LA TABLA ACTIVIDADES
@app.route('/api/actividad/listar/<id>', methods=['GET'])
def ListarActividad(id):
    try:
        profesor=buscarNombre(id)
        cursor=conexion.connection.cursor()
        sql="SELECT nombre_profesor,codigo_clase,nombre_actividad FROM actividades_clase WHERE nombre_profesor='{0}'".format(profesor) 
        cursor.execute(sql)
        datos=cursor.fetchall()
        actividades=[]
        for fila in datos:
            actividad={'nombre_profesor':fila[0],'codigo_clase':fila[1],'nombre_actividad':fila[2]}
            actividades.append(actividad)
        return jsonify(actividades)
    except Exception as ex:
        return "ERROR"

@app.route('/api/actividad/alta',methods=['POST'])
def RegistrarActividad():
    try:
        profesor=request.json['profesor']
        clase=request.json['clase']
        idClase=BuscarIDClase(clase,profesor)
        cursor=conexion.connection.cursor()
        sql="INSERT INTO actividades_clase (nombre_profesor,codigo_clase,nombre_actividad,inciso1,inciso2,inciso3,inciso4,inciso5,inciso6,inciso7,inciso8,inciso9,inciso10,respuesta1,respuesta2,respuesta3,respuesta4,respuesta5,respuesta6,respuesta7,respuesta8,respuesta9,respuesta10,id_curso) VALUES  ('{0}','{1}','{2}','{3}','{4}','{5}','{6}','{7}','{8}','{9}','{10}','{11}','{12}','{13}','{14}','{15}','{16}','{17}','{18}','{19}','{20}','{21}','{22}','{23}')".format(request.json['profesor'],request.json['clase'],request.json['actividad'],request.json['inciso1'],request.json['inciso2'],request.json['inciso3'],request.json['inciso4'],request.json['inciso5'],request.json['inciso6'],request.json['inciso7'],request.json['inciso8'],request.json['inciso9'],request.json['inciso10'],request.json['respuesta1'],request.json['respuesta2'],request.json['respuesta3'],request.json['respuesta4'],request.json['respuesta5'],request.json['respuesta6'],request.json['respuesta7'],request.json['respuesta8'],request.json['respuesta9'],request.json['respuesta10'],idClase)#22
        cursor.execute(sql)
        conexion.connection.commit()
        return jsonify({'id':buscarID(profesor)})
    except Exception as ex:
        return "ERROR"
def BuscarIDClase(clase,profesor):
    cursor=conexion.connection.cursor()
    sql="SELECT id_curso FROM curso WHERE codigo='{0}' AND nombre_profesor='{1}'".format(clase,profesor)
    cursor.execute(sql)
    datos=cursor.fetchone()
    if datos is not None:
        return datos[0]

@app.route('/api/alumnos/actividad/listar/<id>',methods=['GET'])
def ListarActividadesAlumnos(id):
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT id_curso FROM matriculas WHERE id_alumno={0}".format(id)
        cursor.execute(sql)
        datos=cursor.fetchall()
        actividades=[]
        for fila in datos:
            idCurso=fila[0]
            datosActividades=DatosActividad(idCurso)
            for fila2 in datosActividades:
                actividad={"codigoclase":fila2[0],"actividad":fila2[1],"curso":idCurso}
                actividades.append(actividad)
        return jsonify(actividades)
    except Exception as ex:
        return "ERROR"
def DatosActividad(codigoClase):
    cursor=conexion.connection.cursor()
    sql="SELECT codigo_clase,nombre_actividad FROM actividades_clase WHERE id_curso={0}".format(codigoClase)
    cursor.execute(sql)
    datos=cursor.fetchall()
    cursos=[]
    for fila in datos:
        cursos.append(fila)
    return cursos

@app.route('/api/alumnos/actividad/realizar/<codigo>/<actividad>',methods=['GET'])
def RealizarActividad(codigo,actividad):
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT nombre_actividad,inciso1,inciso2,inciso3,inciso4,inciso5,inciso6,inciso7,inciso8, inciso9,inciso10,id_actividad FROM actividades_clase WHERE id_curso='{0}' AND nombre_actividad='{1}'".format(codigo,actividad)
        cursor.execute(sql)
        datos=cursor.fetchone()
        if datos is not None:
            curso={'actividad':datos[0],'inciso_1':datos[1],'inciso_2':datos[2],'inciso_3':datos[3],'inciso_4':datos[4],'inciso_5':datos[5],'inciso_6':datos[6],'inciso_7':datos[7],'inciso_8':datos[8],'inciso_9':datos[9],'inciso_10':datos[10],'id_actividad':datos[11]}
            return jsonify(curso)
        else:
            return jsonify({'mensaje':"Curso NO Encontrado"})
    except Exception as ex:
        return "ERROR"

@app.route('/api/alumnos/actividad/enviar/<actividad>',methods=['POST'])
def EnviarRespuestasActividad(actividad):
    try:
        ContadorAciertos=0
        arregloRespuestasPost=[request.json['respuesta1'],request.json['respuesta2'],request.json['respuesta3'],request.json['respuesta4'],request.json['respuesta5'],request.json['respuesta6'],request.json['respuesta7'],request.json['respuesta8'],request.json['respuesta9'],request.json['respuesta10']]
        arregloRespuestas=['respuesta1','respuesta2','respuesta3','respuesta4','respuesta5','respuesta6','respuesta7','respuesta8','respuesta9','respuesta10']
        for i in range(0,10):
            acierto=EvaluarRespuestas(actividad,arregloRespuestas[i],arregloRespuestasPost[i])
            ContadorAciertos+=int(acierto)
        return jsonify({'TotalAciertos':ContadorAciertos})
    except Exception as ex:
        return "ERROR"
def EvaluarRespuestas(actividad,respuestaDB,respuestaPost):
    cursor=conexion.connection.cursor()
    sql="SELECT EXISTS (SELECT 1 FROM actividades_clase WHERE {0} AND {1} = {2}) AS resultado".format(actividad,respuestaDB,respuestaPost)
    cursor.execute(sql)
    datos=cursor.fetchone()
    return datos[0]

@app.route('/api/alumnos/actividad/actualizar/<alumno>',methods=['PUT'])
def ActualizarCalificaciones(alumno):
    try:
        refuerzo=TipoRefuerzo(request.json['calificacion'],request.json['tiempo'])
        id_maestro=BuscarIDMaestro_Calificaciones(request.json['id_actividad'])
        registro=IndentificarRegistro(alumno,request.json['id_actividad'])
        if(registro==0):
            cursor=conexion.connection.cursor()
            sql="INSERT INTO calificaciones_actividades(id_actividad,id_alumno,id_clase,calificacion,tiempo,refuerzo,id_maestro) VALUES ('{0}','{1}','{2}','{3}','{4}','{5}','{6}')".format(request.json['id_actividad'],alumno,request.json['id_clase'],request.json['calificacion'],request.json['tiempo'],refuerzo,id_maestro)
            cursor.execute(sql)
            conexion.connection.commit()
            return jsonify({'mensaje':"NUEVA Calificacion Guardada"})
        if(registro==1):
            cursor=conexion.connection.cursor()
            sql="UPDATE calificaciones_actividades SET calificacion='{0}', tiempo='{3}', refuerzo='{4}' WHERE id_actividad='{1}' AND id_alumno='{2}'".format(request.json['calificacion'],request.json['id_actividad'],alumno,request.json['tiempo'],refuerzo)
            cursor.execute(sql)
            conexion.connection.commit()
            return jsonify({'mensaje':"ACTULIZACION de Calificacion Guardada"})
        else:
            return jsonify({'mensaje':"ERROR"})
    except Exception as ex:
        return "ERROR"
def TipoRefuerzo(calificacion,tiempo):
    if(int(calificacion)<8):
        return "TEORIA"
    if(int(tiempo)>55):
        return "PRACTICA"
    else:
        return "RENDIMIENTO OPTIMO"    
def BuscarIDMaestro_Calificaciones(id_actividad):
    cursor=conexion.connection.cursor()
    sql="SELECT nombre_profesor FROM actividades_clase WHERE id_actividad={0}".format(id_actividad)
    cursor.execute(sql)
    datos=cursor.fetchone()
    return buscarID(datos[0])
def IndentificarRegistro(alumno,actividad):
    cursor=conexion.connection.cursor()
    sql="SELECT EXISTS (SELECT 1 FROM calificaciones_actividades WHERE id_alumno = {0} AND id_actividad = {1}) AS resultado".format(alumno,actividad)
    cursor.execute(sql)
    datos=cursor.fetchone()
    return datos[0]

#RUTAS PARA LA TABLA CALIFICACIONES
@app.route('/api/alumnos/calificaciones/<alumno>',methods=['GET'])
def Calificaciones(alumno):
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT id_actividad,id_clase,calificacion FROM calificaciones_actividades WHERE id_alumno={0}".format(alumno) 
        cursor.execute(sql)
        datos=cursor.fetchall()
        cursos=[]
        for fila in datos:
            nombreActividad=BuscarActividad(fila[0])
            nombreClase=BuscarClase(fila[1])
            curso={'actividad':nombreActividad,'clase':nombreClase,'calificacion':fila[2]}
            cursos.append(curso)
        return jsonify(cursos)
    except Exception as ex:
        return "ERROR"
def BuscarActividad(idActividad):
    cursor=conexion.connection.cursor()
    sql="SELECT nombre_actividad FROM actividades_clase WHERE id_actividad={0}".format(idActividad)
    cursor.execute(sql)
    datos=cursor.fetchone()
    return datos[0]
def BuscarClase(idClase):
    cursor=conexion.connection.cursor()
    sql="SELECT nombre FROM curso WHERE id_curso={0}".format(idClase)
    cursor.execute(sql)
    datos=cursor.fetchone()
    return datos[0]

#RUTAS PARA USAR EL ARBOL DE DECISION
@app.route('/api/arbol-decision/global',methods=['GET'])
def ArbolDecisionGlobal():
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT numeros_primos, tiempo_numeros_primos, criterios_de_divisibilidad, tiempo_criterios_divisibilidad, ecuaciones_cuadraticas, tiempo_ecuaciones_cuadraticas, teorema_de_pitagoras, tiempo_teorema_pitagoras, algebra, tiempo_algebra, funciones, tiempo_funciones, trigonometria, tiempo_trigonometria, geometria, tiempo_geometria, calculo, tiempo_calculo, tema_tipo_refuerzo FROM calificaciones_examenes"
        cursor.execute(sql)
        datos=cursor.fetchall()
        # Obteniendo los nombres de las columnas
        columnas = [desc[0] for desc in cursor.description]
        # Creando el DataFrame
        df = pd.DataFrame(datos, columns=columnas)
        # Supongamos que 'mi_objetivo' es la columna que quieres predecir y el resto son las características
        X = df.drop('tema_tipo_refuerzo', axis=1)
        y = df['tema_tipo_refuerzo']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
        arbol = DecisionTreeClassifier(criterion='entropy', max_depth=4)
        # Entrenar el modelo con el conjunto de entrenamiento
        arbol.fit(X_train, y_train)
        # Realizar predicciones en el conjunto de prueba
        y_pred = arbol.predict(X_test)
        print("PREDICCION DEL MODELO: ", y_pred[0:5])
        # Calcular la precisión
        precision = accuracy_score(y_test, y_pred)
        print("La precisión del modelo es: ", precision)
        # Crear un diccionario con los resultados
        resultados = {
            "prediccion": y_pred.tolist(),  # Convertir el array numpy a lista
            "precision": precision
        }
        # Retornar los resultados en formato JSON
        return jsonify(resultados)
    
    except Exception as ex:
        return "ERROR"

@app.route('/api/profesor/arbol-decision/<id>',methods=['GET'])
def ArbolDecisionProfesor(id):
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT calificacion, tiempo, refuerzo FROM calificaciones_actividades WHERE id_maestro='{0}'".format(id)
        cursor.execute(sql)
        datos=cursor.fetchall()
        # Obteniendo los nombres de las columnas
        registros=CantidadRegistros(id)
        if(registros<=6):
            resultados = {
                "prediccion_1": "?",
                "prediccion_2": "?",
                "prediccion_3": "?",
                "precision": "?",
                "mensaje":"No hay datos suficientes para generar predicciones"
            }
            return jsonify(resultados)
        columnas = [desc[0] for desc in cursor.description]
        # Creando el DataFrame
        df = pd.DataFrame(datos, columns=columnas)
        X = df.drop('refuerzo', axis=1)
        y = df['refuerzo']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
        arbol = DecisionTreeClassifier(criterion='entropy', max_depth=4)
        # Entrenar el modelo con el conjunto de entrenamiento
        arbol.fit(X_train, y_train)
        # Realizar predicciones en el conjunto de prueba
        y_pred = arbol.predict(X_test)
        print("PREDICCION DEL MODELO: ", y_pred[0:5])
        # Calcular la precisión
        precision = accuracy_score(y_test, y_pred)
        print("La precisión del modelo es: ", precision)
        # Crear un diccionario con los resultados
        resultados = {
            "prediccion_1": y_pred[0],
            "prediccion_2": y_pred[1],
            "prediccion_3": y_pred[2],
            "precision": precision*100,
            "mensaje": "Se generaron 3 predicciones"
        }
        # Retornar los resultados en formato JSON
        return jsonify(resultados)
    except Exception as ex:
        return "ERROR"
def CantidadRegistros(id):
    cursor = conexion.connection.cursor()
    sql = "SELECT COUNT(*) FROM calificaciones_actividades WHERE id_maestro='{0}'".format(id)
    cursor.execute(sql)
    cantidad_registros = cursor.fetchone()[0]
    return cantidad_registros

# RUTAS PARA USAR EL CHATBOT
@app.route('/api/chatbot',methods=['POST'])
def conversacionBot():
    try:
        nlp = spacy.load("es_core_news_sm")

        def enviarRespuesta(intent):
            if intent!="unknown":
                respuesta=responsesGlobal(intent)
            else:
                respuesta="No se como responder esto..."
            return respuesta
        
        def enviarPeticion(text):
            # Procesa el texto
            doc = nlp(text.lower())
            #print(doc)
            for token in doc:
                datos=KeysWordsGlobal(token.text)
                if datos!="unknown":
                    return datos
            return "unknown"
        
        def procesarTexto(text):
            intent=enviarPeticion(text)
            respuesta=enviarRespuesta(intent)
            return respuesta
        
        print(request.json)
        texto = request.json['peticion']
        respuesta = procesarTexto(texto)
        return jsonify({"Bot: ":respuesta})

    except Exception as ex:
        return "ERROR"

def responsesGlobal(clave):
        cursor=conexion.connection.cursor()
        sql = "SELECT valor FROM responses WHERE clave = '{0}'".format(clave)
        cursor.execute(sql)
        miResultado = cursor.fetchall()
        if miResultado:
            valor = miResultado[0][0]
        return valor

def KeysWordsGlobal(valor):
        cursor=conexion.connection.cursor()
        sql = "SELECT clave FROM keywords WHERE valor = '{0}'".format(valor)
        cursor.execute(sql)
        miResultado = cursor.fetchall()
        if miResultado:
            valor = miResultado[0][0]
        else:
            valor = "unknown"
        return valor

# RUTAS PARA EJERCICIOS
@app.route('/api/ejercicios/criterios',methods=['GET'])
def ejercicioCriterios():
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM criterios where id = "+str(random.randint(1, 50))
        cursor.execute(sql)
        datos=cursor.fetchall()
        cursos=[]
        for fila in datos:
            curso={'id':fila[0],'ejercicio':fila[1],'paso_1':fila[2],'paso_2':fila[3],'paso_4':fila[4],'paso_5':fila[6],'paso_6':fila[7],'resultado_1':fila[8],'resultado_2':fila[9],'resultado_3':fila[10],'resultado_4':fila[11],'resultado_5':fila[12],'resultado_6':fila[13]}
            cursos.append(curso)
        return jsonify(cursos)
    except Exception as ex:
        return "ERROR"

@app.route('/api/ejercicios/ecuaciones_cuadraticas',methods=['GET'])
def ejercicioEcuacionesCuadraticas():
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM ecuaciones_cuadraticas where id = "+str(random.randint(1, 50))
        cursor.execute(sql)
        datos=cursor.fetchall()
        cursos=[]
        for fila in datos:
            curso={'id':fila[0],'ejercicio':fila[1],'a':fila[2],'b':fila[3],'c':fila[4],'paso_1':fila[5],'paso_2':fila[6],'paso_3':fila[7],'paso_4':fila[8],'paso_5':fila[9],'paso_6':fila[10],'paso_7':fila[11],'paso_8':fila[12],'paso_9':fila[13],'paso_10':fila[14],'paso_11':fila[15],'resultado_1':fila[16],'resultado_2':fila[17],'resultado_3':fila[18],'resultado_4':fila[19]}
            cursos.append(curso)
        return jsonify(cursos)
    except Exception as ex:
        return "ERROR"
    
@app.route('/api/ejercicios/teorema_pitagoras',methods=['GET'])
def ejercicioteoremapitagoras():
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM teorema_pitagoras where id = "+str(random.randint(1, 50))
        cursor.execute(sql)
        datos=cursor.fetchall()
        cursos=[]
        for fila in datos:
            curso={'id':fila[0],'ejercicio':fila[1],'cateto_a':fila[2],'cateto_b':fila[3],'hipotenusa':fila[4],'area':fila[5],'paso_1':fila[6],'paso_2':fila[7],'paso_3':fila[8],'paso_4':fila[9],'paso_5':fila[10],'paso_6':fila[11],'paso_7':fila[12],'resultado_1':fila[13],'resultado_2':fila[14]}
            cursos.append(curso)
        return jsonify(cursos)
    except Exception as ex:
        return "ERROR"

@app.route('/api/ejercicios/algebra',methods=['GET'])
def ejercicioAlgebra():
    try:
        tabla_Algebra=random.randint(1,2)
        
        if tabla_Algebra==1:
            cursor=conexion.connection.cursor()
            sql="SELECT * FROM algebra where id = "+str(random.randint(1, 50))
            cursor.execute(sql)
            datos=cursor.fetchall()
            cursos=[]
            for fila in datos:
                curso={'id':fila[0],'ejercicio':fila[1],'paso_1':fila[2],'resultado_1':fila[3]}
                cursos.append(curso)

        if tabla_Algebra==2:
            cursor=conexion.connection.cursor()
            sql="SELECT * FROM algebra_tercer_grado where id = "+str(random.randint(1, 50))
            cursor.execute(sql)
            datos=cursor.fetchall()
            cursos=[]
            for fila in datos:
                curso={'id':fila[0],'ejercicio':fila[1],'resultado_1':fila[2],'resultado_2':fila[3]}
                cursos.append(curso)

        return jsonify(cursos)
    
    except Exception as ex:
        return "ERROR"
    
@app.route('/api/ejercicios/funciones',methods=['GET'])
def ejerciciofunciones():
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM funciones where id = "+str(random.randint(1, 50))
        cursor.execute(sql)
        datos=cursor.fetchall()
        cursos=[]
        for fila in datos:
            curso={'id':fila[0],'ejercicio':fila[1],'paso_1':fila[2],'paso_2':fila[3],'paso_3':fila[4],'paso_4':fila[5],'paso_5':fila[6],'paso_6':fila[7],'paso_7':fila[8],'paso_8':fila[9],'paso_9':fila[10],'paso_10':fila[11],'paso_11':fila[12],'paso_12':fila[13],'paso_13':fila[14],'resultado_1':fila[15],'resultado_2':fila[16]}
            cursos.append(curso)
        return jsonify(cursos)
    except Exception as ex:
        return "ERROR"

@app.route('/api/ejercicios/trigonometria',methods=['GET'])
def ejercicioTrigonometria():
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM trigonometria where id = "+str(random.randint(1, 50))
        cursor.execute(sql)
        datos=cursor.fetchall()
        cursos=[]
        for fila in datos:
            curso={'id':fila[0],'ejercicio':fila[1],'paso_1':fila[2],'paso_2':fila[3],'paso_3':fila[4],'paso_4':fila[5],'paso_5':fila[6],'paso_6':fila[7],'paso_7':fila[8],'paso_8':fila[9],'paso_9':fila[10],'paso_10':fila[11],'paso_11':fila[12],'paso_12':fila[13],'resultado_1':fila[14],'resultado_2':fila[15],'resultado_3':fila[16],'resultado_4':fila[17]}
            cursos.append(curso)
        return jsonify(cursos)
    except Exception as ex:
        return "ERROR"

@app.route('/api/ejercicios/calculo',methods=['GET'])
def ejercicioDerivadas():
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM derivadas where id = "+str(random.randint(1, 50))
        cursor.execute(sql)
        datos=cursor.fetchall()
        cursos=[]
        for fila in datos:
            curso={'id':fila[0],'ejercicio':fila[1],'paso_1':fila[2],'paso_2':fila[3],'resultado_1':fila[4]}
            cursos.append(curso)
        return jsonify(cursos)
    except Exception as ex:
        return "ERROR"

#RUTAS PARA LA GENERACION DE RESPUESTAS
@app.route('/api/ejercicios/G_respuestas/criterios/<id>',methods=['GET'])
def GenerarRespuestasCriterios(id):
    try:
        arrayOpciones=[]
        arrayRespuestas=[]
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM criterios WHERE id= '{0}'".format(id)
        cursor.execute(sql)
        datos=cursor.fetchone()
        for i in range(8, 14):
            if(datos[i]!=''):
                arrayRespuestas.append(datos[i])
        arrayOpciones.append(arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)])
        for i in range(0,3):
            arrayOpciones.append(GOCriterios(random.randint(1,50)))
        np.random.shuffle(arrayOpciones)        
        respuestas={'opcion_a':arrayOpciones[0],'opcion_b':arrayOpciones[1],'opcion_c':arrayOpciones[2],'opcion_d':arrayOpciones[3]}
        return jsonify(respuestas)
    except Exception as ex:
        return "ERROR"

def GOCriterios(numeroRand):
    arrayRespuestas=[]
    cursor=conexion.connection.cursor()
    sql="SELECT * FROM criterios WHERE id= '{0}'".format(numeroRand)
    cursor.execute(sql)
    datos=cursor.fetchone()
    for i in range(8, 14):
        if(datos[i]!=''):
            arrayRespuestas.append(datos[i])
    return arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)]

@app.route('/api/ejercicios/G_respuestas/ecuaciones_cuadraticas/<id>',methods=['GET'])
def GenerarRespuestasEcuacionesCuadraticas(id):
    try:
        arrayOpciones=[]
        arrayRespuestas=[]
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM ecuaciones_cuadraticas WHERE id= '{0}'".format(id)
        cursor.execute(sql)
        datos=cursor.fetchone()
        for i in range(15, 19):
            if(datos[i]!=''):
                arrayRespuestas.append(datos[i])
        arrayOpciones.append(arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)])
        for i in range(0,3):
            arrayOpciones.append(GOEcuacionesCuadraticas(random.randint(1,50)))
        np.random.shuffle(arrayOpciones)        
        respuestas={'opcion_a':arrayOpciones[0],'opcion_b':arrayOpciones[1],'opcion_c':arrayOpciones[2],'opcion_d':arrayOpciones[3]}
        return jsonify(respuestas)
    except Exception as ex:
        return "ERROR"
    
def GOEcuacionesCuadraticas(numeroRand):
    arrayRespuestas=[]
    cursor=conexion.connection.cursor()
    sql="SELECT * FROM ecuaciones_cuadraticas WHERE id= '{0}'".format(numeroRand)
    cursor.execute(sql)
    datos=cursor.fetchone()
    for i in range(15, 19):
        if(datos[i]!=''):
            arrayRespuestas.append(datos[i])
    return arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)]

@app.route('/api/ejercicios/G_respuestas/teorema_pitagoras/<id>',methods=['GET'])
def GenerarRespuestasTeoremaPitagoras(id):
    try:
        arrayOpciones=[]
        arrayRespuestas=[]
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM teorema_pitagoras WHERE id= '{0}'".format(id)
        cursor.execute(sql)
        datos=cursor.fetchone()
        for i in range(12, 14):
            if(datos[i]!=''):
                arrayRespuestas.append(datos[i])
        arrayOpciones.append(arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)])
        for i in range(0,3):
            arrayOpciones.append(GOTeoremaPitagoras(random.randint(1,50)))
        np.random.shuffle(arrayOpciones)        
        respuestas={'opcion_a':arrayOpciones[0],'opcion_b':arrayOpciones[1],'opcion_c':arrayOpciones[2],'opcion_d':arrayOpciones[3]}
        return jsonify(respuestas)
    except Exception as ex:
        return "ERROR"
    
def GOTeoremaPitagoras(numeroRand):
    arrayRespuestas=[]
    cursor=conexion.connection.cursor()
    sql="SELECT * FROM teorema_pitagoras WHERE id= '{0}'".format(numeroRand)
    cursor.execute(sql)
    datos=cursor.fetchone()
    for i in range(12, 14):
        if(datos[i]!=''):
            arrayRespuestas.append(datos[i])
    return arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)]

@app.route('/api/ejercicios/G_respuestas/algebra/<id>',methods=['GET'])
def GenerarRespuestasAlgebra(id):
    try:
        arrayOpciones=[]
        arrayRespuestas=[]
        arrayRespuestas2=[]
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM algebra WHERE id= '{0}'".format(id)
        cursor.execute(sql)
        datos=cursor.fetchone()
        arrayRespuestas.append(datos[3])        
        cursor2=conexion.connection.cursor()
        sql2="SELECT * FROM algebra_tercer_grado WHERE id= '{0}'".format(id)
        cursor2.execute(sql2)
        datos2=cursor2.fetchone()
        for i in range(2, 3):
            if(datos2[i]!=''):
                arrayRespuestas2.append(datos2[i])
        arrayOpciones.append(arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)])
        arrayOpciones.append(arrayRespuestas2[random.randint(0,len(arrayRespuestas)-1)])
        for i in range(0,2):
            arrayOpciones.append(GOAlgebra(random.randint(1,50)))
        np.random.shuffle(arrayOpciones)        
        respuestas={'opcion_a':arrayOpciones[0],'opcion_b':arrayOpciones[1],'opcion_c':arrayOpciones[2],'opcion_d':arrayOpciones[3]}
        return jsonify(respuestas)
    except Exception as ex:
        return "ERROR"

def GOAlgebra(numeroRand):
    arrayRespuestas=[]
    cursor=conexion.connection.cursor()
    sql="SELECT * FROM algebra WHERE id= '{0}'".format(numeroRand)
    cursor.execute(sql)
    datos=cursor.fetchone()
    arrayRespuestas.append(datos[3])
    return arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)]

@app.route('/api/ejercicios/G_respuestas/funciones/<id>',methods=['GET'])
def GenerarRespuestasFunciones(id):
    try:
        arrayOpciones=[]
        arrayRespuestas=[]
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM funciones WHERE id= '{0}'".format(id)
        cursor.execute(sql)
        datos=cursor.fetchone()
        for i in range(14, 16):
            if(datos[i]!=''):
                arrayRespuestas.append(datos[i])
        arrayOpciones.append(arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)])
        for i in range(0,3):
            arrayOpciones.append(GOFunciones(random.randint(1,50)))
        np.random.shuffle(arrayOpciones)        
        respuestas={'opcion_a':arrayOpciones[0],'opcion_b':arrayOpciones[1],'opcion_c':arrayOpciones[2],'opcion_d':arrayOpciones[3]}
        return jsonify(respuestas)
    except Exception as ex:
        return "ERROR"
    
def GOFunciones(numeroRand):
    arrayRespuestas=[]
    cursor=conexion.connection.cursor()
    sql="SELECT * FROM funciones WHERE id= '{0}'".format(numeroRand)
    cursor.execute(sql)
    datos=cursor.fetchone()
    for i in range(14, 16):
        if(datos[i]!=''):
            arrayRespuestas.append(datos[i])
    return arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)]

@app.route('/api/ejercicios/G_respuestas/trigonometria/<id>',methods=['GET'])
def GenerarRespuestasTrigonometria(id):
    try:
        arrayOpciones=[]
        arrayRespuestas=[]
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM trigonometria WHERE id= '{0}'".format(id)
        cursor.execute(sql)
        datos=cursor.fetchone()
        for i in range(14, 17):
            if(datos[i]!=''):
                arrayRespuestas.append(datos[i])
        arrayOpciones.append(arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)])
        for i in range(0,3):
            arrayOpciones.append(GOTrigonometria(random.randint(1,50)))
        np.random.shuffle(arrayOpciones)        
        respuestas={'opcion_a':arrayOpciones[0],'opcion_b':arrayOpciones[1],'opcion_c':arrayOpciones[2],'opcion_d':arrayOpciones[3]}
        return jsonify(respuestas)
    except Exception as ex:
        return "ERROR"
    
def GOTrigonometria(numeroRand):
    arrayRespuestas=[]
    cursor=conexion.connection.cursor()
    sql="SELECT * FROM trigonometria WHERE id= '{0}'".format(numeroRand)
    cursor.execute(sql)
    datos=cursor.fetchone()
    for i in range(14, 17):
        if(datos[i]!=''):
            arrayRespuestas.append(datos[i])
    return arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)]

@app.route('/api/ejercicios/G_respuestas/derivadas/<id>',methods=['GET'])
def GenerarRespuestasDerivadas(id):
    try:
        arrayOpciones=[]
        arrayRespuestas=[]
        cursor=conexion.connection.cursor()
        sql="SELECT * FROM derivadas WHERE id= '{0}'".format(id)
        cursor.execute(sql)
        datos=cursor.fetchone()
        arrayRespuestas.append(datos[4])
        arrayOpciones.append(arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)])
        for i in range(0,3):
            arrayOpciones.append(GODerivadas(random.randint(1,50)))
        np.random.shuffle(arrayOpciones)        
        respuestas={'opcion_a':arrayOpciones[0],'opcion_b':arrayOpciones[1],'opcion_c':arrayOpciones[2],'opcion_d':arrayOpciones[3]}
        return jsonify(respuestas)
    except Exception as ex:
        return "ERROR"
    
def GODerivadas(numeroRand):
    arrayRespuestas=[]
    cursor=conexion.connection.cursor()
    sql="SELECT * FROM derivadas WHERE id= '{0}'".format(numeroRand)
    cursor.execute(sql)
    datos=cursor.fetchone()
    arrayRespuestas.append(datos[4])
    return arrayRespuestas[random.randint(0,len(arrayRespuestas)-1)]

#RUTAS PARA MANDAR RESPUESTAS
@app.route('/api/ejercicios/E_respuestas/<topico>',methods=['POST'])
def EnviarRespuesta(topico):
    try:
        cursor=conexion.connection.cursor()
        if(topico=='criterios'):
            sql="SELECT * FROM criterios WHERE id= '{0}'".format(request.json['id'])
        if(topico=='ecuaciones_cuadraticas'):
            sql="SELECT * FROM ecuaciones_cuadraticas WHERE id= '{0}'".format(request.json['id'])
        if(topico=='teorema_pitagoras'):
            sql="SELECT * FROM teorema_pitagoras WHERE id= '{0}'".format(request.json['id'])
        if(topico=='algebra'):
            sql="SELECT * FROM algebra WHERE id= '{0}'".format(request.json['id'])
            cursor2=conexion.connection.cursor()
            sql2="SELECT * FROM algebra_tercer_grado WHERE id= '{0}'".format(request.json['id'])
            cursor2.execute(sql2)
            datos2=cursor2.fetchone()
        if(topico=='funciones'):
            sql="SELECT * FROM funciones WHERE id= '{0}'".format(request.json['id'])
        if(topico=='trigonometria'):
            sql="SELECT * FROM trigonometria WHERE id= '{0}'".format(request.json['id'])
        if(topico=='derivadas'):
            sql="SELECT * FROM derivadas WHERE id= '{0}'".format(request.json['id'])
        cursor.execute(sql)
        datos=cursor.fetchone()
        
        if(topico=='algebra'):
            arrayAlgebra=[]
            for i in range(0,len(datos)-1):
                arrayAlgebra.append(datos[i])
            for i in range(0,len(datos2)-1):
                arrayAlgebra.append(datos2[i])
            for i in range(0,len(arrayAlgebra)):
                if(arrayAlgebra[i]==request.json['respuesta']):
                    estatus={'respuesta':'CORRECTA'}
                    return jsonify(estatus)

        if(topico=='derivadas'):
            if(datos[4]==request.json['respuesta']):
                estatus={'respuesta':'CORRECTA'}
                return jsonify(estatus)

        for i in range(0,len(datos)-1):
            if(datos[i]==request.json['respuesta']):
                estatus={'respuesta':'CORRECTA'}
                return jsonify(estatus)
            
        estatus={'respuesta':'INCORRECTA'}
        return jsonify(estatus)
    except Exception as ex:
        return "ERROR"

#RUTAS PARA LA RED NEURONAL
@app.route('/api/redesneuronales',methods=['POST'])
def RedNeuronalv1():
    try:
        warnings.filterwarnings('ignore', category=UserWarning, module='keras')
        if(request.json['coeficiente']==1):    
            ecuaciones=np.array([[1,18,21],[1,2,10],[1,5,4],[1,5,10],[1,2,3],[1,2,8],],dtype=float)
            resultados=np.array([3,8,-1,5,1,6,],dtype=float)
        if(request.json['coeficiente']==2):
            ecuaciones=np.array([[2,2,2],[2,-7,7],[2,7,1],[2,2,-8],[2,3,-5],],dtype=float)
            resultados=np.array([0,7,-3,-5,-4],dtype=float)
        if(request.json['coeficiente']==3):
            ecuaciones=np.array([[3,3,6],[3,15,9],[3,4,16],[3,-19,2],[3,5,-13],[3,-5,4],],dtype=float)
            resultados=np.array([1,-2,4,7,-6,3],dtype=float)
        if(request.json['coeficiente']==4):
            ecuaciones=np.array([[4,2,6],[4,2,2],[4,-17,-1],[4,-9,15],[4,15,-5],],dtype=float)
            resultados=np.array([1,0,4,6,-5],dtype=float)
        if(request.json['coeficiente']==5):
            ecuaciones=np.array([[5,27,12],[5,-6,-6],[5,-3,7],[5,22,-3],[5,5,0],],dtype=float)
            resultados=np.array([-3,0,2,-5,-1],dtype=float)
        oculta_1=tf.keras.layers.Dense(units=3,input_shape=[3])
        oculta_2=tf.keras.layers.Dense(units=3)
        salida=tf.keras.layers.Dense(units=1)
        modelo=tf.keras.Sequential([oculta_1,oculta_2,salida])
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(0.1),
            loss='mean_squared_error'
        )
        historial=modelo.fit(ecuaciones,resultados,epochs=100,verbose=False)
        a=request.json['coeficiente']
        b=request.json['variableA']
        c=request.json['variableB']
        ecuacion=np.array([[a,b,c]],dtype=float)
        resultado_prediccion = modelo.predict([ecuacion])
        print("X = " + str(round(resultado_prediccion[0][0])))
        resultado=round(resultado_prediccion[0][0])
        estatus={'prediccion':resultado}
        return jsonify(estatus)
    except Exception as ex:
        return "ERROR"

@app.route('/api/redesneuronalesv2',methods=['POST'])
def RedNeuronalv2():
    try:
        warnings.filterwarnings('ignore', category=UserWarning, module='keras')
        ecuaciones=np.array([[2,9,10],[2,7,3],[2,7,6],[2,7,6],[2,7,6],[2,4,2],[2,6,0],[2,4,2],[2,3,1],[2,5,2]],dtype=float)
        resultados=np.array([-2,-3,-2,-4,-2,-1,-3,-1,-1,-2],dtype=float)
        oculta_1=tf.keras.layers.Dense(units=3,input_shape=[3])
        oculta_2=tf.keras.layers.Dense(units=3)
        salida=tf.keras.layers.Dense(units=1)
        modelo=tf.keras.Sequential([oculta_1,oculta_2,salida])
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(0.1),
            loss='mean_squared_error'
        )
        historial=modelo.fit(ecuaciones,resultados,epochs=1000,verbose=False)
        a=request.json['coeficiente']
        b=request.json['variableA']
        c=request.json['variableB']
        ecuacion=np.array([[a,b,c]],dtype=float)
        resultado_prediccion = modelo.predict([ecuacion])
        print("X = " + str(round(resultado_prediccion[0][0])))
        resultado=round(resultado_prediccion[0][0])
        estatus={'prediccion':resultado}
        return jsonify(estatus)
    except Exception as ex:
        return "ERROR"
    
#RUTAS PARA USUARIOS
@app.route('/api/maestros/alta', methods=['POST'])
def registrar_maestro():
    try:
        cursor = conexion.connection.cursor()
        sql = "INSERT INTO maestros (nombre, username, password) VALUES ('{0}', '{1}', '{2}')".format(request.json['nombre'], request.json['username'], request.json['password'])
        cursor.execute(sql)
        conexion.connection.commit()
        return jsonify({'mensaje': "Maestro Registrado"})
    except Exception as ex:
         return "ERROR"
    
@app.route('/api/maestros/loggin',methods=['POST'])
def logginMaestros():
    try:
        cursor=conexion.connection.cursor()
        sql = "SELECT id FROM maestros WHERE username = '{0}' AND password = '{1}'".format(request.json['username'], request.json['password'])
        cursor.execute(sql)
        datos=cursor.fetchone()
        if datos is not None:
            maestro={'codigo':datos[0]}
        else:
            maestro={'codigo':-1}
        return jsonify(maestro)
    except Exception as ex:
         return "ERROR"

@app.route('/api/alumnos/alta', methods=['POST'])
def registrar_alumno():
    try:
        print(request.json)
        cursor = conexion.connection.cursor()
        sql = "INSERT INTO alumnos (nombre, username, password) VALUES ('{0}', '{1}', '{2}')".format(request.json['nombre'], request.json['username'], request.json['password'])
        cursor.execute(sql)
        conexion.connection.commit()
        return jsonify({'mensaje': "Alumno Registrado"})

    except Exception as ex:
        return "ERROR"

@app.route('/api/alumnos/loggin',methods=['POST'])
def logginAlumno():
    try:
        cursor=conexion.connection.cursor()
        sql = "SELECT id FROM alumnos WHERE username = '{0}' AND password = '{1}'".format(request.json['username'], request.json['password'])
        cursor.execute(sql)
        datos=cursor.fetchone()
        if datos is not None:
            alumno={'codigo':datos[0]}
        else:
            alumno={'codigo':-1}
        return jsonify(alumno)
    except Exception as ex:
         return "ERROR"

@app.route('/api/maestros/confirmacion/<id>',methods=['GET'])
def MaestroConfirmacion(id):
    try:
        maestro=buscarNombre(id)
        if(maestro==-1):
            confirmacion="False"
        else:
            confirmacion=maestro
        return jsonify({'confirmacion': confirmacion})
    except Exception as ex:
        return 'ERROR'

@app.route('/api/alumnos/confirmacion/<id>',methods=['GET'])
def AlumnosConfirmacion(id):
    try:
        alumno=buscarNombreEstudiante(id)
        if(alumno==-1):
            confirmacion="False"
        else:
            confirmacion=alumno
        return jsonify({'confirmacion': confirmacion})
    except Exception as ex:
        return 'ERROR'
def buscarNombreEstudiante(id):
    cursor=conexion.connection.cursor()
    sql="SELECT nombre FROM alumnos WHERE id = '{0}'".format(id)
    cursor.execute(sql)
    datos=cursor.fetchone()
    if datos is not None:
        return datos[0]
    else:
        return -1
    
def pagina_no_encontada(error):
    return "<h1>PAGINA NO ENCONTRADA...</h1>",404

if __name__=='__main__':
    app.config.from_object(config['development'])
    app.register_error_handler(404,pagina_no_encontada)
    app.run()