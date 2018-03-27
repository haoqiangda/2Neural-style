from scipy.misc import imread,imresize
import numpy as np 
import matplotlib.pyplot as plt 
from image_utils import load_image,preprocess_image,deprocess_image
import tensorflow as tf 
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config = config)
	return session 
def rel_error(x,y):
	return np.max(np.abs(x-y)/(np.maximum(1e-8,np.abs(x)+np.abs(y))))
def check_scipy():
	import scipy 
	vnum = int(scipy.__version__.split('.')[1])
	assert vnum>=16,"You must install scipy >=0.16.0"
check_scipy()
from squeezenet import SqueezeNet
import tensorflow as tf 
tf.reset_default_graph()
sess = get_session()
SAVE_PATH = 'datasets/squeezenet.ckpt'
print(SAVE_PATH)
#if not os.path.exists(SAVE_PATH):
    #raise ValueError("You need to download SqueezeNet!")
model = SqueezeNet(save_path=SAVE_PATH, sess=sess)
#load data for testing 
content_img_test = preprocess_image(load_image('tubingen.jpg',size=192))[None]
style_img_test = preprocess_image(load_image('starry_night.jpg',size=192))[None]
answers = np.load('style-transfer-checks.npz')

def content_loss(content_weight,content_curr,content_orig):
	return content_weight*tf.reduce_sum(tf.squared_difference(content_curr,content_orig))

def gram_matrix(features,normalize=True):
	"""Inputs: the shape of features is (1,H,W,C)"""
	features = tf.transpose(features,[0,3,1,2])
	shape = tf.shape(features)
	features = tf.reshape(features,(shape[0],shape[1],-1))
	transpose_features = tf.transpose(features,[0,2,1])
	output = tf.matmul(features,transpose_features)
	if normalize:
		output = tf.div(output,tf.cast(shape[0]*shape[1]*shape[2]*shape[3],tf.float32))
	return output

def style_loss(style_weights,feats,style_layers,style_targets):
   #Input:
   #---style_layers:List of layer indices into feats giving the layers to include in the style loss 
   #---feats: List of the features at every layer of the current image,as produced by the extract_features function
   #---style_targets:List of the same length as style layers ,where style_targets[i] is 
   #a tensor giving the Gram matrix the source style image computed at layer style_layers[i]
	style_losses = 0
	for i in range(len(style_layers)):
		cur_index = style_layers[i]
		cur_feat = feats[cur_index]
		cur_style_weight = style_weights[i]
		cur_style_target = style_targets[i]
		gram = gram_matrix(cur_feat)
		style_losses += cur_style_weight*tf.reduce_sum(tf.squared_difference(gram,cur_style_target))
	return style_losses

def TV_loss(img,tv_weight):
	shape = tf.shape(img)  # the shape of the img is (1,H,W,C)
	img_row_before = tf.slice(img,[0,0,0,0],[-1,-1,shape[2]-1,-1])
	img_row_after = tf.slice(img,[0,0,1,0],[-1,-1,shape[2]-1,-1])
	img_col_before = tf.slice(img,[0,0,0,0],[-1,shape[1]-1,-1,-1])
	img_col_after = tf.slice(img,[0,1,0,0],[-1,shape[1]-1,-1,-1])
	tv_loss =  tv_weight*(tf.reduce_sum(tf.squared_difference(img_col_after,img_col_before))+\
			tf.reduce_sum(tf.squared_difference(img_row_after,img_row_before)))
	return tv_loss

def style_transfer(content_img,style_img,content_size,style_size,content_layer,style_layers,
					content_weight,style_weights,tv_weight,init_random=False):
	content_pre_img = preprocess_image(load_image(content_img,size=content_size))
	feats = model.extract_features(model.image)  #extract features of every layer from the input image
	content_targets = sess.run(feats[content_layer],{model.image:content_pre_img[None]})
	style_pre_img = preprocess_image(load_image(style_img,size = style_size))
	style_feats = [feats[idx] for idx in style_layers]
	#to transfer gram
	style_target=[] 
	for style_feat_var in style_feats:
		style_target.append(gram_matrix(style_feat_var))
	style_targets = sess.run(style_target,{model.image:style_pre_img[None]})

	if init_random:
		img_var = tf.Variable(tf.random_uniform(content_pre_img[None].shape,0,1),name="image")
	else:
		img_var = tf.Variable(content_pre_img[None], name="image")
	# to compute loss  
	#print(img_var[None].shape)
	feat = model.extract_features(img_var)
	conloss = content_loss(content_weight,feat[content_layer],content_targets)
	styloss = style_loss(style_weights,feat,style_layers,style_targets)
	tvloss = TV_loss(img_var,tv_weight)
	loss = conloss+styloss+tvloss
	
	#params 
	initial_lr = 3.0
	decayed_lr = 0.1 
	decayed_lr_at = 180
	max_iters = 200 

	lr_var = tf.Variable(initial_lr,name="lr")
	# Create train_op that updates the generated image when run
	with tf.variable_scope("optimizer") as opt_scope:
		train_op = tf.train.AdamOptimizer(lr_var).minimize(loss,var_list=[img_var])
	# Initialize the generated image and optimization variables
	opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=opt_scope.name)
	sess.run(tf.variables_initializer([lr_var,img_var]+opt_vars))
	# Create an op that will clamp the image values when run
	clamp_image = tf.assign(img_var,tf.clip_by_value(img_var,-1.5,1.5))
	
	#plot 
	f,s = plt.subplots(1,2)
	s[0].axis('off')
	s[1].axis('off')
	s[0].set_title('content source img')
	s[1].set_title('style source img')
	s[0].imshow(deprocess_image(content_pre_img))
	s[1].imshow(deprocess_image(style_pre_img))
	plt.show()
	plt.figure()

	for i in range(max_iters):
		#take a optimization step to update the img
		sess.run(train_op)
		if i < decayed_lr_at:
			sess.run(clamp_image)
		if i == decayed_lr_at:
			sess.run(tf.assign(lr_var,decayed_lr))
		if i % 100 ==0:
			print('Iteration:{}'.format(i))
			img = sess.run(img_var)
			plt.imshow(deprocess_image(img[0],rescale=True))
			plt.axis('off')
			plt.show()
	print('Iteration:{}'.format(i))
	img = sess.run(img_var)
	plt.imshow(deprocess_image(img[0],rescale=True))
	plt.axis('off')
	plt.show()
	



