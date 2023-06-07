import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import collections

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('VGG16_150-13-0.65.hdf5')
  return model


with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Skylumia Task
         """
         )

file = st.file_uploader("Upload the image to be classified", type=["jpg", "png","jpeg"])


st.set_option('deprecation.showfileUploaderEncoding', False)
col1, col2 = st.columns(2)

if file is None:
    st.text("Please upload an image file")
else:
    with col1:
      image = np.array(Image.open(file))
      IMG_SIZE = (150,150) # for VGG16 model needs this specific img shape
      img = cv2.resize(image,IMG_SIZE)
      img=img/255.0 # scall down image 
      img = np.expand_dims(img, axis=0)
      st.image(img,width = 300)
      predict_x= model.predict(img)
      classes_x=np.argmax(predict_x,axis=1)
      numbers = range(7)
      classes = ['0', '1', '2', '3', '4', '5', '6', '7']
      dir_clases = dict(zip(numbers,classes))
    with col2:
      st.write(f'Model is **{predict_x[0][classes_x[0]]*100:.2f}%** sure that it is **{dir_clases[classes_x[0]]}**')
      dir_clases = dict(zip(predict_x[0],classes))
      
      od = collections.OrderedDict(sorted(dir_clases.items(),reverse=True)) # sort dict in descending order.
      
      # print to 5 accurate results.
      st.write('Other **top 5 possibislities** :')
      temp_increment = 1
      for key,values in od.items():
        if temp_increment != 1:
          st.write(f'{key*100:.2f}% - {values}')
        temp_increment += 1
        if temp_increment >=7:
          break