# Importamos las librerías necesarias

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargamos el modelo de clasificación
modelo = tf.keras.models.load_model('clasificador_perrogato.keras')

# Título de la página
st.set_page_config(page_title = '🐶🐱 Clasificador de perro-gato')
st.title('🐶🐱 Clasificador de perro-gato')

# Creamos un campo desde el que se pueda subir una imagen
campo_imagen = st.file_uploader('Inserta una imagen en este cuadro:', type = 'jpg')

# Clasificamos la imagen
clase = []
if campo_imagen is not None:
    # Mostramos la imagen
    imagen = Image.open(campo_imagen)
    st.image(imagen)

    # A la imagen introducida por el usuario le debemos hacer las mismas transformaciones que a las imágenes del conjunto test
    # La cambiamos de tamaño a 200 x 200 píxeles
    imagen = imagen.resize((200, 200))
    # Asumiendo que los colores están expresados en la escala de 0 a 255, los pasamos a la escala [0, 1]
    imagen = np.array(imagen) / 255.0
    # Añadimos una dimension (en primer lugar) donde iría la clase de la imagen
    imagen = np.expand_dims(imagen, axis = 0)
    
    # Hacemos la clasificación
    clase = modelo.predict(imagen)

# Imprimimos por pantalla el resultado
if len(clase):
    st.title('✅ Resultado:')
    score = float(tf.keras.ops.sigmoid(clase[0][0]))
    if clase[0][0] > 0.5:
        st.info(f'Esta imagen es un {100 * (1 - score):.2f} % gato y un {100 * score:.2f} % perro.')
        st.info('¡Es un perro! 🐕')
    else:
        st.info(f'Esta imagen es un {100 * score:.2f} % gato y un {100 * (1 - score):.2f} % perro.')
        st.info('¡Es un gato! 🐈')
