import numpy as np 
from scipy.misc import imread,imresize 
def load_image(filename,size=None):
	"""load and resize an image from a disk"""
	img = imread(filename)
	if size is not None:
		orig_shape = np.array(img.shape[:2])
		min_index = np.argmin(orig_shape)
		scale_factor = float(size)/orig_shape[min_index]
		new_shape = (orig_shape*scale_factor).astype(int)
		img = imresize(img,scale_factor)
	return img 

SQUEEZENET_MEAN = np.array([0.485,0.456,0.406],dtype = np.float32) 
SQUEEZENET_STD = np.array([0.229,0.224,0.225],dtype=np.float32)

def preprocess_image(img):
	return (img.astype(np.float32)/255.0-SQUEEZENET_MEAN)/SQUEEZENET_STD

def deprocess_image(img,rescale=False):
	img = (img * SQUEEZENET_STD + SQUEEZENET_MEAN)
	if rescale:
		vmin,vmax = img.min(),img.max()
		img = (img-vmin)/(vmax-vmin)
	return np.clip(img*255,0.0,255.0).astype(np.uint8)