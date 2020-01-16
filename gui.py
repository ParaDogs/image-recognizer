from tkinter import *
import neuro
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image


def init():
	data_file = open("mnist_test.csv", 'r')
	data_list = data_file.readlines()
	data_file.close()

	input_nodes = 28 * 28
	hidden_nodes = 200
	output_nodes = 10
	learning_rate = 0.15

	if os.path.isfile('neuro.pickle'):
		with open('neuro.pickle', 'rb') as f:
			n = pickle.load(f)
	else:
		n = neuro.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
		training_data_file = open("mnist_train.csv", 'r')
		training_data_list = training_data_file.readlines()
		training_data_file.close()

		epochs = 1
		for e in range(epochs):
			for record in training_data_list:
				all_values = record.split(',')
				inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
				targets = np.zeros(output_nodes) + 0.01
				targets[int(all_values[0])] = 0.99
				n.train(inputs, targets)
		with open('neuro.pickle', 'wb') as f:
			pickle.dump(n, f)
	return n


class App(tk.Tk):
	def __init__(self):
		tk.Tk.__init__(self)
		self.neuro = init()
		self.title('Digits recognition')
		self.x = self.y = 0
		# Creating elements
		self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
		self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 20), justify=LEFT)
		self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
		self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
		# Grid structure
		self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
		self.label.grid(row=0, column=1, pady=2, padx=2)
		self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
		self.button_clear.grid(row=1, column=0, pady=2)
		# self.canvas.bind("<Motion>", self.start_pos)
		self.canvas.bind("<B1-Motion>", self.draw_lines)

	def clear_all(self):
		self.canvas.delete("all")

	def classify_handwriting(self):
		HWND = self.canvas.winfo_id()  # get the handle of the canvas
		rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
		img = ImageGrab.grab(rect)
		# resize image to 28x28 pixels
		img = img.resize((28, 28))
		# convert rgb to grayscale
		img = img.convert('L')
		img = np.array(img)
		# reshaping to support our model input and normalizing
		img = img.reshape(1, 28 * 28)
		inputs = ((255 - np.asfarray(img)) / 255.0 * 0.99) + 0.01
		outputs = self.neuro.query(inputs)
		chance = dict(enumerate(outputs))
		chance = [[str(key), "{0:.2f}".format(float(val) * 100)] for key, val in chance.items()]
		chance.sort(key=lambda xx: float(xx[1]), reverse=True)
		result = ""
		for x in chance:
			result += "{}: {:>8}%\n".format(x[0], x[1])
		self.label.configure(text=result)

	def draw_lines(self, event):
		self.x = event.x
		self.y = event.y
		r = 10
		self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


app = App()
mainloop()
