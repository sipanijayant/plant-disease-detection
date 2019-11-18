from django.shortcuts import render
from keras.preprocessing import image
import numpy as np
from .deeplearning import graph, model, output_list
import base64
import requests
from bs4 import BeautifulSoup
import re
import xml.etree.ElementTree
 
output_dict_next = {'Apple___Apple_scab': 'apple-scab/',
               'Apple___Black_rot': 'black-knot-fungus/',
               'Apple___Cedar_apple_rust': 'apple-scab/',
               'Apple___healthy': 'its compleatly healthy',
               'Blueberry___healthy':"its compleatly healthy",
               'Cherry_(including_sour)___Powdery_mildew': "downy-mildew/",
               'Cherry_(including_sour)___healthy': "its compleatly healthy",
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "bacterial-leaf-spot/",
               'Corn_(maize)___Common_rust_': "common-rust/",
               'Corn_(maize)___Northern_Leaf_Blight': "late-blight/",
               'Corn_(maize)___healthy': "its compleatly healthy",
               'Grape___Black_rot': "black-knot-fungus/",
               'Grape___Esca_(Black_Measles)': "bacterial-leaf-spot/",
               'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'bacterial-leaf-spot/',
               'Grape___healthy': "its compleatly healthy",
               'Orange___Haunglongbing_(Citrus_greening)': "",
               'Peach___Bacterial_spot': "bacterial-leaf-spot/",
               'Peach___healthy': "its compleatly healthy",
               'Pepper,_bell___Bacterial_spot': "bacterial-leaf-spot/",
               'Pepper,_bell___healthy': "its compleatly healthy",
               'Potato___Early_blight': "early-blight/",
               'Potato___Late_blight': "late-blight/",
               'Potato___healthy': "its compleatly healthy",
               'Raspberry___healthy': "its compleatly healthy",
               'Soybean___healthy': "its compleatly healthy",
               'Squash___Powdery_mildew': "powdery-mildew/",
               'Strawberry___Leaf_scorch': "bacterial-leaf-spot/",
               'Strawberry___healthy': "its compleatly healthy",
               'Tomato___Bacterial_spot': "bacterial-leaf-spot/",
               'Tomato___Early_blight': "early-blight/",
               'Tomato___Late_blight': "late-blight/",
               'Tomato___Leaf_Mold': "bacterial-leaf-spot/",
               'Tomato___Septoria_leaf_spot': "bacterial-leaf-spot/",
               'Tomato___Spider_mites Two-spotted_spider_mite': "",
               'Tomato___Target_Spot': "bacterial-leaf-spot/",
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "mosaic-virus/",
               'Tomato___Tomato_mosaic_virus': "mosaic-virus/",
               'Tomato___healthy': "its compleatly healthy"}

               
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext
 
def treatment(disease):
 
    URL = "https://www.planetnatural.com/pest-problem-solver/plant-disease/"
    r = requests.get(URL+disease)
    soup = BeautifulSoup(r.content, 'html.parser')
    #print(soup.prettify())
 
    table = soup.find('div', attrs = {'class':'post-entry'})
    #print(table.prettify())
 
    list_items = table.find_all('li')
    if disease=='mosaic-virus/':
        del list_items[0:5]
    d=[]
    c=[]
 
 
    for artist_name in list_items:
        #print(artist_name.prettify())
        name=artist_name.contents
        a=[]
        b=[]
        #print(name)
        #for i in name:
        #    print(i)
        for i in name:
           a.append(cleanhtml(str(i)))
        for i in a:
            b.append(i.replace('\xa0',''))
        #print(b)
        c.append(b)
 
    for i in range(len(c)):
        d.append(str(i+1)+'. '+''.join(c[i]))
 
    return d
 
def index(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        b64_img = base64.b64encode(myfile.file.read()).decode('ascii')
        img = image.load_img(myfile, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255

        with graph.as_default():
            prediction = model.predict(img)

        prediction_flatten = prediction.flatten()
        max_val_index = np.argmax(prediction_flatten)
        result = output_list[max_val_index]
        disease=output_dict_next[result]
        disease_final=disease
        llist=treatment(disease)
        c=[]
        c.append(disease_final)
        for i in llist:
            c.append(i+'\n')
        result = ''.join(map(str, c))
        return render(request, "plant_app/index.html", {
            'result': result, 'file_url': b64_img })

    return render(request, "plant_app/index.html")
