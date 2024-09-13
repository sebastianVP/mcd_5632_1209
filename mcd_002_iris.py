import streamlit as st
import pandas as pd
import os
from  sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


st.write('''
# APLICACION IRIS PARA PREDICCCION DE TIPOS DE ESPECIES
         
Esta aplicacion predice el tipo de flor en base a sus mediciones del sepal y petal
         ''')

st.sidebar.header('Parametros de entrada por el Usuario')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length',4.3,7.9,5.4)
    sepal_width  = st.sidebar.slider('Sepal width',2.0,4.4,3.4)

    petal_length= st.sidebar.slider('Petal length',1.0,6.9,1.3)
    petal_width = st.sidebar.slider('Petal width',0.1,2.5,0.2)

    data = {'sepal_length':sepal_length,
        'sepal_width':sepal_width,
        'petal_length':petal_length,
        'petal_width':petal_width}
    test_df = pd.DataFrame(data,index=[0])
    return test_df



df = user_input_features()

st.subheader("Parametros de entrada por el Usuario")
st.write(df)

iris= datasets.load_iris()
#var = os.getcwd()
#print(var)
#new = os.path.join(var,'modelo_iris_1209')
iris = datasets.load_iris()
X= iris.data
Y= iris.target
clf =RandomForestClassifier()
clf.fit(X,Y)
'''
clf= pickle.load(open(new,'rb'))
'''
prediction = clf.predict(df)
prediction_proba= clf.predict_proba(df)

st.subheader('Mostrando Etiquetas de Especies y su correspondiente Index')
df_ = pd.DataFrame(iris.target_names)
st.write(df_)


st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)