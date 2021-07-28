# test_data_load.py


import numpy as np
import tensorflow as tf
from hyperparams import Hyperparams as hp
from data_load import *
from model import Text2Mel, GraphModel


#data1, data2, data3, = get_batch()
data3 = get_batch()

z = list(data3.as_numpy_iterator())
print(len(z))
print(z[0])

for step, (fname, text, mel, mag) in enumerate(data3):
	print("Step {}: {} {} {} {}".format(step, fname, text, mel, mag))
	print(mel.get_shape())
	print(step)
	if step % 10 == 0 and step > 0:
		break

'''
graph = GraphModel()
graph.train_model("")
graph.save_model("./text2mel_test")
'''


#'''
early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
	"./text2mel_test_chkpt", monitor="loss", save_best_only=True
)
text2mel = Text2Mel()
text2mel.compile(
	optimizer=tf.keras.optimizers.Adam(lr=hp.lr), metrics=["accuracy"],
)
text2mel.fit(
	data3, epochs=10#epochs=50,
	#callbacks=[early_stop, checkpoint]#epochs=1, steps_per_epoch=100#steps_per_epoch=hp.num_iterations
)
text2mel.save("./text2mel_test")
#'''

'''
devices = tf.config.list_physical_devices()
device_name = devices[-1].name
dev = device_name.split(":")[-2] + ":" + device_name.split(":")[-1]
with tf.device(dev):
	optimizer = tf.keras.optimizers.Adam(lr=hp.lr)
	text2mel = Text2Mel()
	text2mel.compile(
		optimizer=optimizer, metrics=["accuracy"],
	)
	global_step = 0
	#while global_step < hp.num_iterations:
	while global_step < 10000:
		# Iterate over batches of the dataset (batch_size = 1 for
		# now).
		for step, (fname, text, mel, mag) in enumerate(data3):
			#print(step)
			#print(mel.get_shape())
			# Compute s from mel.
			s = tf.concat(
				(tf.zeros_like(mel[:, :1, :]), mel[:, :-1, :]), 1
			)

			# Open a GradientTape to record the operations run
			# during a forward pass, which enables
			# auto-differentiation.
			with tf.GradientTape() as tape:
				# Run a forward pass of the text2mel model. The
				# operations that model applies to its inputs are
				# going to be recorded on the GradientTape.
				y, y_logits, alignments, max_attentions = text2mel(
					(text, s), training=True
				)

				# Compute the loss value for this minibatch.
				# Mel L1 loss.
				loss_mels = tf.reduce_mean(tf.abs(y - mel))

				# Mel binary divergence loss.
				loss_bd1 = tf.reduce_mean(
					tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logits,
						labels=mel
					)
				)

				# Guided attention loss.
				A = tf.pad(alignments, 
					[(0, 0), (0, text2mel.hp.max_N), (0, text2mel.hp.max_T)], mode="CONSTANT",
					constant_values=-1.0
				)[:, :text2mel.hp.max_N, :text2mel.hp.max_T]
				#attention_masks = tf.to_float(tf.not_equal(A, -1))
				attention_masks = tf.cast(tf.not_equal(A, -1), tf.float32)
				loss_att = tf.reduce_sum(
					tf.abs(A * text2mel.gts) * attention_masks
				)
				mask_sum = tf.reduce_sum(attention_masks)
				loss_att /= mask_sum

				# Total loss.
				loss = loss_mels + loss_bd1 + loss_att
				loss_value = loss

			# Use the gradient tape to automatically retrieve the
			# gradients of the trainable variables with respect to
			# the loss.
			grads = tape.gradient(
				loss_value, text2mel.trainable_weights
			)

			# Run one step of gradient descent by updating the
			# value of the variables to minimize the loss.
			optimizer.apply_gradients(
				zip(grads, text2mel.trainable_weights)
			)

			# Log every 1000 batches.
			if global_step % 1000 == 0:
				print(
					"Training loss at step {}: {}".format(
						global_step, float(loss_value)
					)
				)

			global_step += 1

	text2mel.save("./text2mel_test")
'''