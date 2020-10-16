import tensorflow as tf

def gram(input_tensor):
	result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
	input_shape = tf.shape(input_tensor)
	num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
	return result/(num_locations)


def wct(content, style, alpha=1, eps=1e-8):
	'''TensorFlow version of Whiten-Color Transform
	   Assume that content/style encodings have shape 1xHxWxC

	   See p.4 of the Universal Style Transfer paper for corresponding equations:
	   https://arxiv.org/pdf/1705.08086.pdf
	'''
	# Remove batch dim and reorder to CxHxW
	content_t = tf.transpose(tf.squeeze(content), (2, 0, 1))
	style_t = tf.transpose(tf.squeeze(style), (2, 0, 1))

	Cc, Hc, Wc = tf.unstack(tf.shape(content_t))
	Cs, Hs, Ws = tf.unstack(tf.shape(style_t))

	# CxHxW -> CxH*W
	content_flat = tf.reshape(content_t, (Cc, Hc*Wc))
	style_flat = tf.reshape(style_t, (Cs, Hs*Ws))

	# Content covariance
	# keep_dims wurde in keepdims umbenannt, das is doch scheiße
	mc = tf.reduce_mean(content_flat, axis=1, keepdims=True)
	fc = content_flat - mc
	fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc*Wc, tf.float32) - 1.) + tf.eye(Cc)*eps

	# Style covariance
	ms = tf.reduce_mean(style_flat, axis=1, keepdims=True)
	fs = style_flat - ms
	fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs*Ws, tf.float32) - 1.) + tf.eye(Cs)*eps

	# tf.svd is slower on GPU, see https://github.com/tensorflow/tensorflow/issues/13603
	with tf.device('/cpu:0'):  
		Sc, Uc, _ = tf.linalg.svd(fcfc)
		Ss, Us, _ = tf.linalg.svd(fsfs)

	## Uncomment to perform SVD for content/style with np in one call
	## This is slower than CPU tf.svd but won't segfault for ill-conditioned matrices
	# @jit
	# def np_svd(content, style):
	#     '''tf.py_func helper to run SVD with NumPy for content/style cov tensors'''
	#     Uc, Sc, _ = np.linalg.svd(content)
	#     Us, Ss, _ = np.linalg.svd(style)
	#     return Uc, Sc, Us, Ss
	# Uc, Sc, Us, Ss = tf.py_func(np_svd, [fcfc, fsfs], [tf.float32, tf.float32, tf.float32, tf.float32])

	# Filter small singular values
	k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.int32))
	k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.int32))

	# Whiten content feature
	Dc = tf.linalg.diag(tf.pow(Sc[:k_c], -0.5))
	fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:,:k_c], Dc), Uc[:,:k_c], transpose_b=True), fc)

	# Color content with style
	Ds = tf.linalg.diag(tf.pow(Ss[:k_s], 0.5))
	fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:,:k_s], Ds), Us[:,:k_s], transpose_b=True), fc_hat)

	# Re-center with mean of style
	fcs_hat = fcs_hat + ms

	# Blend whiten-colored feature with original content feature
	blended = alpha * fcs_hat + (1 - alpha) * (fc + mc)

	# CxH*W -> CxHxW
	blended = tf.reshape(blended, (Cc,Hc,Wc))
	# CxHxW -> 1xHxWxC
	blended = tf.expand_dims(tf.transpose(blended, (1,2,0)), 0)

	return blended