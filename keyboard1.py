import random
import re
import tkinter as tk

class Uniform(object):
	"""Barebones example of a language model class."""

	def __init__(self):
		self.vocab = set()

	def train(self, filename):
		"""Train the model on a text file."""
		for line in open(filename):
			for w in line:
				self.vocab.add(w)

	# The following two methods make the model work like a finite
	# automaton.

	def start(self):
		"""Resets the state."""
		pass

	def read(self, w):
		"""Reads in character w, updating the state."""
		pass

	# The following two methods add probabilities to the finite automaton.

	def prob(self, w, u):
		"""Returns the probability of the next character being w given the
		current state."""
		return 1/(len(self.vocab)+1) # +1 for <unk>

	def probs(self):
		"""Returns a dict mapping from all characters in the vocabulary to the
probabilities of each character."""
		return {w: self.prob(w) for w in self.vocab}

class CharacterBased(object):
	"""Character-based language model class."""

	def __init__(self):
		self.vocab = set()
		self.gram_len = 9
		self.state = "<s>"*self.gram_len
		self.model = {}	# dict (n-gram) of dict (gram as you see them) of counts
		self.n_1plus = {}
		self.delta = 0.01

	def true_len(self, s):
		count = s.count("<s>")
		s =re.sub("<s>", "", s)
		count += s.count("</s>")
		s = re.sub("</s>", "", s)
		count += len(s)
		return count

	def train(self, filename):
		"""Train the model on a text file."""
		total_chars = 0
		for i in range(self.gram_len):
			self.n_1plus[i+1] = {}
			self.model[i+1] = {}
		for line in open(filename):
			line = line.strip('\n')
			total_chars += len(line)
			for i in range(self.gram_len):
				n = i+1
				# add <s>
				for x in range(n):
					y = x+1
					l = len(line)
					if l >= n-y:
						word = "<s>"*y + line[0:n-y]
					else:
						word = "<s>"*y + line + "</s>"*(n-y-l) 
					self.vocab.add(word)
					if word not in self.model[n]:
						self.model[n][word] = 0
					self.model[n][word] += 1
					if word not in self.n_1plus[n]:
						self.n_1plus[n][word] = {}
					if l > n-y:
						if line[n-y] not in self.n_1plus[n][word]:
							self.n_1plus[n][word][line[n-y]] = 0
						self.n_1plus[n][word][line[n-y]] += 1
					else:
						if "</s>" not in self.n_1plus[n][word]:
							self.n_1plus[n][word]["<s>"] = 0
						self.n_1plus[n][word]["<s>"] += 1
				# add </s>
				for x in range(n):
					y = x+1
					l = len(line)
					if l >= n-y:
						word = line[l-n+y-1:] + "</s>"*y 
					else:
						word = "<s>"*(n-y-l) + line + "</s>"*y
					self.vocab.add(word)
					if word not in self.model[n]:
						self.model[n][word] = 0
					self.model[n][word] += 1
					if word not in self.n_1plus[n]:
						self.n_1plus[n][word] = {}
					if "</s>" not in self.n_1plus[n][word]:
						self.n_1plus[n][word]["</s>"] = 0
					self.n_1plus[n][word]["</s>"] += 1
				# middle section
				for w in range(len(line)-n):
					word = line[w:w+n]
					self.vocab.add(word)
					if word not in self.model[n]:
						self.model[n][word] = 0
					self.model[n][word] += 1
					if w != len(line)-n-1:
						if word not in self.n_1plus[n]:
							self.n_1plus[n][word] = {}
						if line[w+n] not in self.n_1plus[n][word]:
							self.n_1plus[n][word][line[w+n]] = 0
						self.n_1plus[n][word][line[w+n]] += 1
		self.n = self.gram_len*total_chars
		self.delta = self.vocab.__len__() / (self.n) # witten-bell delta

	# The following two methods make the model work like a finite
	# automaton.

	def start(self):
		"""Resets the state."""
		self.state = "<s>"*self.gram_len

	def read(self, w):
		"""Reads in character w, updating the state."""
		if len(self.state) >= 3:
			if self.state[0:3] == "<s>":
				self.state = self.state[3:]+w
				return
		if len(self.state) >= 4:
			if self.state[0:4] == "</s>":
				self.state = self.state[4:]+w
				return
		self.state = self.state[1:] + w
	# The following two methods add probabilities to the finite automaton.

	def prob(self, w, u):
		"""Returns the probability of the next character being w given the
		current state."""
		# base case, no context (unigram)
		if self.true_len(u) == 0:	
			N = len(self.vocab)+1
			lamb = N/(N + self.n*self.delta)
			c = 0	# +1 for <unk>
			if self.true_len(w) in self.model:
				if w in self.model[self.true_len(w)]:
					c = self.model[self.true_len(w)][w]
			p = lamb*c/N + (1-lamb)/self.n
			return p
		try:
			# temp vars for readability
			count_u = self.model[self.true_len(u)][u]
			count_uw = self.n_1plus[self.true_len(u)][u][w]
			lamb = count_u / (count_u + len(self.n_1plus[self.true_len(u)][u].keys()))
			# recurse
			if len(u) >= 4:
				if u[0:4] == "</s>":
					p = lamb*count_uw/count_u + (1-lamb)*self.prob(w, u[4:])
				elif u[0:3] == "<s>":
					p = lamb*count_uw/count_u + (1-lamb)*self.prob(w, u[3:])
				else:
					p = lamb*count_uw/count_u + (1-lamb)*self.prob(w, u[1:])
					
			elif len(u) >= 3:
				if u[0:3] == "<s>":
					p = lamb*count_uw/count_u + (1-lamb)*self.prob(w, u[3:])
				else:
					p = lamb*count_uw/count_u + (1-lamb)*self.prob(w, u[1:])
			else:
				p = lamb*count_uw/count_u + (1-lamb)*self.prob(w, u[1:])
		except KeyError:
			# recurse
			if len(u) >= 4:
				if u[0:4] == "</s>":
					p = self.prob(w, u[4:])
				elif u[0:3] == "<s>":
					p = self.prob(w, u[3:])
				else:
					p = self.prob(w, u[1:])
			elif len(u) >= 3:
				if u[0:3] == "<s>":
					p = self.prob(w, u[3:])
				else:
					p = self.prob(w, u[1:])
			else:
				p = self.prob(w, u[1:])
		return p		# return value

	def probs(self):
		"""Returns a dict mapping from all characters in the vocabulary to the
probabilities of each character."""
		chars = 'qwertyuiopasdfghjklzxcvbnm,. '
		p = {}
		for i in self.model[1]:
			p[i] = self.prob(i, self.state)
		return p
	def allprobs(self):
		return {w: self.prob(w, self.state) for w in self.vocab}

class Application(tk.Frame):
	def __init__(self, model, master=None):
		self.model = model

		tk.Frame.__init__(self, master)
		self.pack()

		self.INPUT = tk.Text(self)
		self.INPUT.pack()

		self.chars = ['qwertyuiop',
				'asdfghjkl',
				'zxcvbnm,.',
				' ']

		self.KEYS = tk.Frame(self)
		for row in self.chars:
			r = tk.Frame(self.KEYS)
			for w in row:
				# trick to make button sized in pixels
				f = tk.Frame(r, height=32)
				b = tk.Button(f, text=w, command=lambda w=w: self.press(w))
				b.pack(fill=tk.BOTH, expand=1)
				f.pack(side=tk.LEFT)
				f.pack_propagate(False)
			r.pack()
		self.KEYS.pack()

		self.TOOLBAR = tk.Frame()

		self.BEST = tk.Button(self.TOOLBAR, text='Best', command=self.best, 
							  repeatdelay=500, repeatinterval=1)
		self.BEST.pack(side=tk.LEFT)

		self.WORST = tk.Button(self.TOOLBAR, text='Worst', command=self.worst, 
							   repeatdelay=500, repeatinterval=1)
		self.WORST.pack(side=tk.LEFT)

		self.RANDOM = tk.Button(self.TOOLBAR, text='Random', command=self.random, 
								repeatdelay=500, repeatinterval=1)
		self.RANDOM.pack(side=tk.LEFT)

		self.QUIT = tk.Button(self.TOOLBAR, text='Quit', command=self.quit)
		self.QUIT.pack(side=tk.LEFT)

		self.TOOLBAR.pack()

		self.update()
		self.resize_keys()

	def resize_keys(self):
		for bs, ws in zip(self.KEYS.winfo_children(), self.chars):
			wds = [150*self.model.prob(w, self.model.state)+15 for w in ws]
			wds = [int(wd+0.5) for wd in wds]

			for b, wd in zip(bs.winfo_children(), wds):
				b.config(width=wd)

	def press(self, w):
		self.INPUT.insert(tk.END, w)
		self.INPUT.see(tk.END)
		self.model.read(w)
		self.resize_keys()

	def best(self):
		_, w = max((p, w) for (w, p) in self.model.probs().items())
		self.press(w)

	def worst(self):
		_, w = min((p, w) for (w, p) in self.model.probs().items())
		self.press(w)

	def random(self):
		s = 0.
		r = random.random()
		p = self.model.probs()
		#chars = 'qwertyuiopasdfghjklzxcvbnm,. '
		#p = {}
		#for i in chars:
		#	p[i] = self.model.prob(i, self.model.state)
		for w in p:
			s += p[w]
			if s > r:
				break
		self.press(w)

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument(dest='train')
	args = parser.parse_args()

	##### Replace this line with an instantiation of your model #####
	m = CharacterBased()
	m.train(args.train)
	m.start()

	root = tk.Tk()
	app = Application(m, master=root)
	app.mainloop()
	root.destroy()
