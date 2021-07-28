from flask import Flask, url_for, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array




app = Flask(__name__)


UPLOAD_FOLDER='C:/Users/msi/Anaconda3/envs/flask_coton/flask_coton/static'




def load_model1():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model=load_model('model_inception.h5')


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    #if image.mode != "RGB":
        #image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # return the processed image
    return image





@app.route("/",methods=["POST"])
def upload_predict():
	if request.method=="POST":
		image_file=request.files["image"]
		if image_file:
			image_location=os.path.join(
				UPLOAD_FOLDER,
				image_file.filename)
			image_file.save(image_location)
			image=prepare_image(load_img(image_location), target=(224, 224))
			model=load_model('model_inception.h5')
			model.predict(image)
			a=np.argmax(model.predict(image), axis=1)
			if a[0]==0:
			  predict= 'feuille de coton malade'
			elif a[0]==2:
			  predict= "feuille de coton saine"
			elif a[0]==1:
			  predict= 'plante de coton malade'
			elif a[0]==3:
			  predict= "plante de coton saine"
			else:
				predict= 'veuillez insérer une image valide'
			return jsonify({'ETAT': predict}),200
	return jsonify({'ETAT': 'No image posted'}),400



if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
       "please wait until server has fully started"))

    load_model1()
    app.run(debug=True)



