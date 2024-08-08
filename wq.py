# -*- coding: utf-8 -*-
import streamlit as st
import random
import json
import pickle
import numpy as np
import pandas as pd
import nltk # type: ignore
import mysql.connector
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

nltk.download("wordnet")
nltk.download("punkt")

lemmatizer = WordNetLemmatizer()

# Importamos los archivos generados en el código anterior
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.keras")

# Pasamos las palabras de la oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    try:
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        max_index = np.where(res == np.max(res))[0][0]
        category = classes[max_index]
        return category
    except Exception as e:
        st.error(f"Error al predecir la clase: {e}")
        return "No entiendo tu pregunta"

# Obtenemos una respuesta aleatoria
def get_response(tag, intents_json):
    list_of_intents = intents_json["intents"]
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

# Función para almacenar la interacción y la calificación
def store_interaction(user_input, response, rating):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="vpsclx_intranet"
    )

    mycursor = mydb.cursor()

    sql = "INSERT INTO interacciones (user_input, response, rating) VALUES (%s, %s, %s)"
    val = (user_input, response, rating)

    try:
        mycursor.execute(sql, val)
        mydb.commit()
        st.success("Interacción almacenada exitosamente.")
    except mysql.connector.Error as error:
        st.error(f"Error al insertar datos: {error}")
    finally:
        mycursor.close()
        mydb.close()

# Función para obtener la precisión del modelo
def get_model_accuracy():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="vpsclx_intranet"
    )

    mycursor = mydb.cursor()

    sql = "SELECT user_input, response, rating FROM interacciones"
    mycursor.execute(sql)
    results = mycursor.fetchall()

    true_labels = []
    predicted_labels = []

    for result in results:
        user_input, response, rating = result
        true_labels.append(rating)
        predicted_labels.append(1 if rating > 0 else 0)

    accuracy = accuracy_score(true_labels, predicted_labels)
    mycursor.close()
    mydb.close()

    return accuracy

# Función para obtener respuestas basadas en interacciones pasadas
def get_response_from_interactions(user_input):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="vpsclx_intranet"
    )

    mycursor = mydb.cursor()

    sql = "SELECT response, rating FROM interacciones WHERE user_input = %s ORDER BY rating DESC"
    val = (user_input,)

    mycursor.execute(sql, val)
    results = mycursor.fetchall()

    mycursor.close()
    mydb.close()

    if results:
        return results[0][0]  # Devuelve la respuesta con el rating más alto
    else:
        return None

# Función para entrenar el modelo con nuevas interacciones
def train_model():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="vpsclx_intranet"
    )

    mycursor = mydb.cursor()

    sql = "SELECT user_input, response, rating FROM interacciones"
    mycursor.execute(sql)
    results = mycursor.fetchall()

    training_data = []

    for result in results:
        user_input, response, rating = result
        training_data.append((user_input, response, rating))

    mycursor.close()
    mydb.close()

    # Preparar datos para entrenamiento
    X = []
    y = []

    for user_input, response, rating in training_data:
        bow = bag_of_words(user_input)
        X.append(bow)
        if response not in classes:
            classes.append(response)
        y.append(classes.index(response))

    X = np.array(X)
    y = np.array(y)

    # Verificar que las etiquetas estén en el rango válido
    if np.max(y) >= len(classes):
        st.error("Error: Las etiquetas están fuera del rango válido.")
        return

    # Crear y entrenar el modelo
    model = Sequential()
    model.add(Dense(128, input_shape=(len(words),), activation='relu'))
    model.add(Dropout(0.5))
    for _ in range(10):  # 10 capas ocultas
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=5, verbose=1)

    # Guardar el modelo entrenado
    model.save("chatbot_model.keras")

    # Guardar las clases actualizadas
    with open("classes.pkl", "wb") as f:
        pickle.dump(classes, f)

# Interfaz de Streamlit
st.title("Chatbot Streamlit 🧠")

# Inicializar el estado de los mensajes
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! ¿En qué puedo ayudarte?"}]

# Mostrar los mensajes existentes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Obtener la entrada del usuario
user_input = st.chat_input("Escribe tu mensaje:")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Obtener respuesta basada en interacciones pasadas
    response_from_interactions = get_response_from_interactions(user_input)

    if response_from_interactions:
        response = response_from_interactions
    else:
        response = get_response(predict_class(user_input), intents)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # Calificación de la respuesta
    rating = st.text_input("Califica la respuesta del chatbot (0-10):")

    # Almacenar la interacción y la calificación
    if rating:
        store_interaction(user_input, response, rating)

# Mostrar la precisión del modelo
accuracy = get_model_accuracy()
st.write(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Entrenar el modelo con nuevas interacciones
train_model()
