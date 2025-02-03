import time

import matplotlib.pyplot as plt
import numpy
import numpy as np
# библиотека scipy.special содержит сигмоиду expit()
import scipy.special


# определение класса нейронной сети
class NeuralNetwork:
	# инициализировать нейронную сеть
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		# задать количество узлов во входном, скрытом и выходном слое
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		# коэффициент обучения
		self.lr = learningrate

		# использование сигмоиды в качестве функции активации
		self.activation_function = lambda x: scipy.special.expit(x)

		# Матрицы весовых коэффициентов связей wih (между входным и скрытым
		# слоями) и who (между скрытым и выходным слоями).
		# Весовые коэффициенты связей между узлом i и узлом j следующего слоя
		# обозначены как w_i_j:
		# wll w21
		# wl2 w22 и т.д.
		#self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
		#self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		pass

	# тренировка нейронной сети
	def train(self, inputs_list, targets_list):
		# преобразовать список входных значений в двухмерный массив
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T
		# рассчитать входящие сигналы для скрытого слоя
		hidden_inputs = numpy.dot(self.wih, inputs)
		# рассчитать исходящие сигналы для скрытого слоя
		hidden_outputs = self.activation_function(hidden_inputs)
		# рассчитать входящие сигналы для выходного слоя
		final_inputs = numpy.dot(self.who, hidden_outputs)
		# рассчитать исходящие сигналы для выходного слоя
		final_outputs = self.activation_function(final_inputs)

		#ошибка = целевое значение - фактическое значение
		output_errors = targets - final_outputs
		# ошибки скрытого слоя - это ошибки output_errors,
		# распределенные пропорционально весовым коэффициентам связей
		# и рекомбинированные на скрытых узлах
		hidden_errors = numpy.dot(self.who.T, output_errors)
		# обновить весовые коэффициенты связей между скрытым и выходным слоями
		self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
		# обновить весовые коэффициенты для связей между входным и скрытым слоями
		self.wih += self.lr * numpy.dot((hidden_errors *  hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))


	def trainWithEpoch(self, inputs, targets, epochs:int):
		for е in range(epochs):
			for i in range(len(targets)):
				n.train(inputs[i], targets[i])

	# опрос нейронной сети
	def query(self, inputs_list):
		#преобразовать список входных значений в двухмерный массив
		inputs = numpy.array(inputs_list, ndmin=2).T
		# рассчитать входящие сигналы для скрытого слоя
		hidden_inputs = numpy.dot(self.wih, inputs)
		# рассчитать исходящие сигналы для скрытого слоя
		hidden_outputs = self.activation_function(hidden_inputs)
		# рассчитать входящие сигналы для выходного слоя
		final_inputs = numpy.dot(self.who, hidden_outputs)
		# рассчитать исходящие сигналы для выходного слоя
		final_outputs = self.activation_function(final_inputs)
		return final_outputs




#преднастроенные (неизменные) параметры сети
input_nodes = 784  #кол-во входных узлов сети
output_nodes = 10 #кол-во выходных узлов сети

#пределы изменения изменяемых параметров сети
#переменный параметр: кол-в скрытых узов сети
hidden_nodes=[10, 100]
#переменный параметр: коэф обучения
learning_rate=[0.1]
#переменный параметр - количество эпох (циклов обучения)
epoch=[1, 2, 4]


#подготовительная работа. А именно загрузка тренировочных и тестовых датасетов в память

#функция подготовки данных для работы нейросети
def prepareDate(data_list, inputs, targets):
	"""Фун подготовки данных
	:param data_list - список строчек из csv файла с данными
	:param inputs - пустой список куда мы поместим массив векторов входных данных для нейросети
	:param targets - пустой список куда мы поместим массив выходных данных из нейросети"""
	for record in data_list:
		all_values = record.split(',')
		input = (numpy.asarray(all_values[1:], dtype=numpy.float32) / 255.0 * 0.99) + 0.01
		target = numpy.zeros(output_nodes) + 0.01
		target[int(all_values[0])] = 0.99
		inputs.append(input)
		targets.append(target)
		pass

# загрузить в память список тренировочный набор данных CSV-файла набора MNIST
training_data_file = open("dataset/mnist_train_100.csv", 'r')
#training_data_file = open("dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines ()
training_data_file.close()
inputsTrain = []
targetsTrain = []
prepareDate(training_data_list, inputsTrain, targetsTrain)
# загрузить в память список тестовый набор данных CSV-файла набора MNIST
test_data_file = open("dataset/mnist_test_10.csv", 'r')
#test_data_file = open("dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
inputsTest = []
targetsTest = []
prepareDate(test_data_list, inputsTest, targetsTest)

#собираемые данные
datasTime=numpy.zeros((len(epoch), len(hidden_nodes), len(learning_rate)), dtype=np.float32)
datasEffective=numpy.zeros((len(epoch), len(hidden_nodes), len(learning_rate)), dtype=np.float32)

for epochIndex, epochValue in enumerate(epoch):
	for hiddenNodesIndex, hiddenNodesValue in enumerate(hidden_nodes):
		for learningRateIndex, learningRateValue in enumerate(learning_rate):
			print(f"epochCount = {epochValue}, hiddenNeironCount={hiddenNodesValue}, learningRate={learningRateValue}:")
			start_time = time.time()

			# создать экземпляр нейронной сети
			n = NeuralNetwork(input_nodes, hiddenNodesValue, output_nodes, learningRateValue)

			# тренировка нейроной сети
			n.trainWithEpoch(inputsTrain, targetsTrain, epochValue)

			# тестирование нейросети
			all = 0.0
			ok = 0.0
			for i in range(len(inputsTest)):
				targetResult = numpy.argmax(targetsTest[i])
				output = n.query(inputsTest[i])
				outputResult = numpy.argmax(output)
				all = all + 1.0
				if (targetResult == outputResult):
					ok = ok + 1.0
				else:
					#print("\terror: targetResult="+str(targetResult)+", outputResult="+str(outputResult))
					pass

			end_time = time.time()
			elapsed_time = end_time - start_time
			print(f"\tThe task took {elapsed_time:.2f} seconds to complete. Effective={ok / all}")

			#сохраним данные собранные по работе сети в данной итерации при данных параметрах
			datasTime[epochIndex][hiddenNodesIndex][learningRateIndex]=elapsed_time
			datasEffective[epochIndex][hiddenNodesIndex][learningRateIndex]=ok/all
			del n


#визуализируем собранную статистику в графическом виде
for epochIndex, epochValue in enumerate(epoch):
	for hiddenNodesIndex, hiddenNodesValue in enumerate(hidden_nodes):
		plt.title("кол циклов обучения = "+str(epochValue))
		plt.plot(learning_rate, datasEffective[epochIndex][hiddenNodesIndex], label="эпох= "+str(epochValue)+",нейронов на скр слое = "+str(hiddenNodesValue), marker='o', markersize=12)
plt.legend()
plt.xlabel("коэфф обучения")
plt.ylabel("эффективность обучения")
#plt.show()
plt.savefig('neiro1_'+str(time.time())+'_epoch_'+str(epochValue)+'.png')

#print(datas)

"""
# одиночная оценка
record = test_data_list[0]
all_values = record.split(',')
inputs = (numpy.asarray(all_values[1:], dtype=numpy.float32) / 255.0 * 0.99) + 0.01
print("result="+all_values[0])
outputs=n.query(inputs)
print(outputs)
label = numpy.argmax(outputs)
print("network says ", label)

image_array = numpy.asarray(all_values[1:], dtype=numpy.float32).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation ='None')
matplotlib.pyplot.show()
"""
