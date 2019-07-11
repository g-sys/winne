#!/usr/bin/env python

import os
import random
import time
import tensorflow as tf

def vocab_encode(text, vocab):
	return [vocab.index(x) + 1 for x in text if x in vocab]

def vocab_decode(array, vocab):
	return "".join([vocab[x - 1] for x in array])

class CharRNN(object):
	def __init__(self, model):
		self.model = model
		self.path = "names.txt"
		self.vocab = " &$%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ\\^_abcdefghijklmnopqrstuvwxyz{|}"
		self.seq = tf.placeholder(tf.int32, [None, None])
		self.temp = tf.constant(1.5)
		self.hidden_sizes = [128, 256]
		self.batch_size = 64
		self.lr = 0.0003
		self.skip_step = 10
		self.num_steps = 50		 # for RNN unrolled
		self.len_generated = 200
		self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

	def create_rnn(self, seq):
		layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_sizes]
		cells = tf.nn.rnn_cell.MultiRNNCell(layers)
		batch = tf.shape(seq)[0]
		zero_states = cells.zero_state(batch, dtype=tf.float32)
		self.in_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]]) for state in zero_states])
		# this line to calculate the real length of seq; all seq are padded to be of the same length, which is num_steps
		length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)
		self.output, self.out_state = tf.nn.dynamic_rnn(cells, seq, length, self.in_state)

	def create_model(self):
		seq = tf.one_hot(self.seq, len(self.vocab))
		self.create_rnn(seq)
		self.logits = tf.layers.dense(self.output, len(self.vocab), None)
		loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[:, :-1], labels=seq[:, 1:])
		self.loss = tf.reduce_sum(loss)
		# sample the next character from Maxwell-Boltzmann Distribution
		# with temperature temp. It works equally well without tf.exp
		self.sample = tf.multinomial(tf.exp(self.logits[:, -1] / self.temp), 1)[:, 0]
		self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

	def train(self):
		saver = tf.train.Saver()
		start = time.time()
		min_loss = None
		with tf.Session() as sess:
			writer = tf.summary.FileWriter("graphs/gist", sess.graph)
			sess.run(tf.global_variables_initializer())
			ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/" + self.model + "/checkpoint"))
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
			iteration = self.gstep.eval()
			lines = [line.strip() for line in open(self.path, "r").readlines()]
			print("Finished reading data.")
			prev_chunk = None
			while True:
				random.shuffle(lines)
				for text in lines:
					#print("Iteration " + str(iteration))
					text = vocab_encode(text, self.vocab)
					chunk = text
					chunk += [0] * (self.num_steps - len(chunk))
					#print(chunk)
					if not prev_chunk:
						prev_chunk = chunk
						break
					batch_loss, _ = sess.run([self.loss, self.opt], {self.seq: [chunk, prev_chunk]})
					if (iteration + 1) % self.skip_step == 0:
						print("Iter {}. \n		Loss {}. Time {}".format(iteration, batch_loss, time.time() - start))
						self.online_infer(sess)
						start = time.time()
						checkpoint_name = "checkpoints/" + self.model + "/char-rnn"
						if min_loss is None:
							saver.save(sess, checkpoint_name, iteration)
						elif batch_loss < min_loss:
							saver.save(sess, checkpoint_name, iteration)
							min_loss = batch_loss
					iteration += 1

	def online_infer(self, sess):
		# Generate sequence one char at a time, based on the previous char
		for seed in ["Chateau", "I", "R", "T", "E", "N", "M", "G", "A", "W"]:
			sentence = seed
			state = None
			for _ in range(self.len_generated):
				batch = [vocab_encode(sentence[-1], self.vocab)]
				feed = {self.seq: batch}
				if state is not None:		# for first decoder step the state is None
					for i in range(len(state)):
						feed.update({self.in_state[i]: state[i]})
				index, state = sess.run([self.sample, self.out_state], feed)
				sentence += vocab_decode(index, self.vocab)
			print("\t" + sentence)

def main():
	model = "winne"
	if not os.path.exists("checkpoints"):
		os.mkdir("checkpoints")
	if not os.path.exists("checkpoints/" + model):
		os.mkdir("checkpoints/" + model)
	tf.reset_default_graph()
	lm = CharRNN(model)
	print("Initialized RNN.\nCreating model.")
	lm.create_model()
	print("Created model.\nTraining.")
	lm.train()
	print("Training complete.")
	
if __name__ == "__main__":
	main()

