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

#RUTAS PARA HACER CRUD SOBRE ALGUNA TABLA DE LA DB
@app.route('/api/cursos/listar', methods=['GET'])
def listar_cursos():
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT codigo,nombre,creditos FROM curso"
        cursor.execute(sql)
        datos=cursor.fetchall()
        cursos=[]
        for fila in datos:
            curso={'codigo':fila[0],'materia':fila[1],'creditos':fila[2],}
            cursos.append(curso)
        return jsonify(cursos)
    except Exception as ex:
        return "ERROR"

@app.route('/api/cursos/buscar/<codigo>',methods=['GET'])
def leer_curso(codigo):
    try:
        cursor=conexion.connection.cursor()
        sql="SELECT codigo,nombre,creditos FROM curso WHERE codigo = '{0}'".format(codigo)
        cursor.execute(sql)
        datos=cursor.fetchone()

        if datos is not None:
            curso={'codigo':datos[0],'materia':datos[1],'creditos':datos[2],}
            return jsonify(curso)
        
        else:
            return jsonify({'mensaje':"Curso NO Encontrado"})

    except Exception as ex:
        return "ERROR"

@app.route('/api/cursos/alta',methods=['POST'])
def registrar_curso():
    try:
        print(request.json)
        cursor=conexion.connection.cursor()
        sql="INSERT INTO curso (codigo, nombre, creditos) VALUES ('{0}','{1}',{2})".format(request.json['codigo'],request.json['nombre'],request.json['creditos'])
        cursor.execute(sql)
        conexion.connection.commit()
        return jsonify({'mensaje':"Curso Registrado"})

    except Exception as ex:
        return "ERROR"

@app.route('/api/cursos/eliminar/<codigo>',methods=['DELETE'])
def eliminar_curso(codigo):
    try:
        cursor=conexion.connection.cursor()
        sql="DELETE FROM curso WHERE codigo = '{0}'".format(codigo)
        cursor.execute(sql)
        conexion.connection.commit()
        return jsonify({'mensaje':"Curso Eliminado"})

    except Exception as ex:
        return "ERROR"

@app.route('/api/cursos/actualizar/<codigo>',methods=['PUT'])
def actualizar_curso(codigo):
    try:
        print(request.json)
        cursor=conexion.connection.cursor()
        sql="UPDATE curso SET nombre = '{0}', creditos = '{1}' WHERE codigo = '{2}'".format(request.json['nombre'],request.json['creditos'],codigo)
        cursor.execute(sql)
        conexion.connection.commit()
        return jsonify({'mensaje':"Curso Actualizado"})

    except Exception as ex:
        return "ERROR"

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

#RUTAS PARA LA RED NEURONAL (NO INTEGRAR A PYTHONANYWHERE)
@app.route('/api/redesneuronales',methods=['POST'])
def RedNeuronalv1():
    try:
        warnings.filterwarnings('ignore', category=UserWarning, module='keras')
        ecuaciones=np.array([[1,18,21],[1,2,10],[1,5,4],[1,5,10],[1,2,3],[1,2,8],],dtype=float)
        resultados=np.array([3,8,-1,5,1,6,],dtype=float)
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

def pagina_no_encontada(error):
    return "<h1>PAGINA NO ENCONTRADA...</h1>",404

if __name__=='__main__':
    app.config.from_object(config['development'])
    app.register_error_handler(404,pagina_no_encontada)
    app.run()