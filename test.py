from keras.models import Model, load_model
from keras.preprocessing import image
import numpy as np
import os

from tensorflow.compat.v1 import ConfigProto, InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = InteractiveSession(config=config)


class SignalNet():
	def __init__(self):
		self.base_model = load_model("signal_model.h5")


	def extract(self,image_path):
		img = image.load_img(image_path, target_size = (224,224))
		x = image.img_to_array(img)
		x = np.expand_dims(x,axis=0)
		#Get prediction
		features = self.base_model.predict(x)
		return np.argmax(features[0])

model = SignalNet()

for signalBands in os.listdir('train'):
	correctCount = 0
	print(signalBands)
	for img in os.listdir('train/'+signalBands):
		print(model.extract("train/"+signalBands+'/'+img))
		
	#print(filecol, "Accuracy:", correctCount/len(os.listdir('ourDataset/'+filecol)))