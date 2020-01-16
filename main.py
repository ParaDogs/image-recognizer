import neuro
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt

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

m = 2  # номер элемента в контрольной выборке, для которого хочешь найти класс
all_values = data_list[m].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28, 28))
print(all_values[0])
plt.imshow(image_array, cmap='Greys', interpolation='None')
inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
outputs = n.query(inputs)
label = np.argmax(outputs)
print("ответ", label)
