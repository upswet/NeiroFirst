import dataclasses

import matplotlib
import numpy
# библиотека scipy.special содержит сигмоиду expit()
import scipy.special
from PIL import Image
import matplotlib.pyplot as plt
import time
from abc import ABC, abstractmethod


class Activatefunction:
	def fun(self, x):
		"""функция активации"""
		pass
	def deriv(self, x, y):
		"""производная функции активации
		:param x - переменная х
		:param y - ранее вычисленая фун активации от х (то есть fun(x))"""
		pass
class Sigmoid(Activatefunction):
	def fun(self, x):
		#return scipy.special.expit(x);
		return 1 / (1 + numpy.exp(-x))
	def deriv(self, x, y):
		fx = y if not (not y) else self.fun(x)
		return fx * (1 - fx)

class AbstractLayer:
	neironCount: int  # кол-во нейронов на данном слое
	signalOutput:list # исходящий сигнал с данного слоя
	signalInput: list # входящий сигнал на данный слой
	errors: list #ошибка при обр распр ошибки
	type=None
class InputLayer(AbstractLayer):
	type='INPUT'
class HiddenLayer(AbstractLayer):
	type='HIDDEN'
	activ: Activatefunction #функция активации для нейронов данного слоя
	w: list =[]  # Матрица связей нейронов предыдущего слоя и данного
class OutputLayer(AbstractLayer):
	type='OUTPUT'
	activ: Activatefunction #функция активации для нейронов данного слоя
	w:list #Матрица связей нейронов предыдущего слоя и данного

# определение класса нейронной сети
class NeuralNetwork:
	layers=[] #слои нейронной сети

	# коэффициент обучения
	learningRate = 0.1
	def setLearningRate(self, learningRate:float):
		self.learningRate=learningRate
		return self

	def __init__(self):
		pass

	def addInputLayer(self, neironCount:int):
		self.layers=[]
		if (len(self.layers)>0):
			raise Exception("Входной слой можно добавить только в качестве первого слоя нейросети")
		layer = InputLayer()
		layer.neironCount=neironCount
		self.layers.append(layer)
		return self
	def addHiddenLayer(self, neironCount:int, activ: Activatefunction):
		if (len(self.layers)==0):
			raise Exception("Скрытый слой можно добавить только если в нейросети уже есть хотя бы один другой слой")
		if (self.layers[-1].type=='OUTPUT'):
			raise Exception("Нельзя добавить скрытый слой после входного слоя")
		layer = HiddenLayer()
		layer.activ=activ
		layer.neironCount=neironCount
		self.layers.append(layer)
		return self
	def addOutputLayer(self, neironCount:int, activ: Activatefunction):
		if (len(self.layers)==0):
			raise Exception("Выходной слой можно добавить только если в нейросети уже есть хотя бы один другой слой")
		layer = OutputLayer()
		layer.activ=activ
		layer.neironCount=neironCount
		self.layers.append(layer)
		return self

	def initialization(self):
		"""Инициализация весов связей между нейронами разных слоёв"""
		for i in range(1, len(self.layers)):
			prevLayerCount=self.layers[i-1].neironCount
			currentLayerCount=self.layers[i].neironCount
			self.layers[i].w = numpy.random.normal(0.0, pow(currentLayerCount, -0.5), (currentLayerCount, prevLayerCount))


	def forward(self, inputVec):
		"""Прямое распространение сигнала
		:param inputVec - массив входных значений для первого слоя нейронов"""
		# преобразовать список входных значений в двухмерный массив
		# Это исходящий сигнал из первого слоя
		self.layers[0].signalInput = numpy.array(inputVec, ndmin=2).T
		self.layers[0].signalOutput=self.layers[0].signalInput
		for i in range(1, len(self.layers)):
			layer=self.layers[i]
			# рассчитать входящие сигналы для текущего слоя
			layer.signalInput = numpy.dot(layer.w, self.layers[i-1].signalOutput)
			# рассчитать исходящие сигналы для текущего слоя (они же будут входящими для следующего)
			layer.signalOutput = layer.activ.fun(layer.signalInput)
		return self.layers[-1].signalOutput

	def backPropagation(self, targetVec, output):
		"""Обратное распространение ошибки с модификацией весов"""
		target = numpy.array(targetVec, ndmin=2).T
		self.layers[-1].errors = target - output
		for i in reversed(range(1, len(self.layers))):
			self.layers[i].w += self.learningRate * numpy.dot((self.layers[i].errors * self.layers[i].activ.deriv(self.layers[i].signalInput, None)), numpy.transpose(self.layers[i-1].signalOutput))
			self.layers[i-1].errors = numpy.dot(self.layers[i].w.T, self.layers[i].errors)

	def train(self, inputsVec, targetsVec):
		"""Тренировка сети на одном векторе
		:param inputVec - вектор входных данных, то есть сигналов подающихся на входные нейроны
		:param targetVec - вектор целевых выходных данных для расчёта ошибки"""
		for i in range(len(inputsVec)):
			output=self.forward(inputsVec[i])
			self.backPropagation(targetsVec[i], output)

	def trains(self, inputsVec, targetsVec, epoch:int):
		"""Многократная тренировка на массивах данных
		:param inputs - массив входных векторов
		:param targets - массив целевых векторов
		:param epoch - количество эпох (полноценных повторов циклов обучения)"""
		for i in range(epoch):
			self.train(inputsVec, targetsVec)




#функция подготовки данных для работы нейросети
def prepareDate(data_list, inputs, targets):
	"""Фун подготовки данных
	:param data_list - список строчек из csv файла с данными
	:param inputs - пустой список куда мы поместим массив векторов входных данных для нейросети
	:param targets - пустой список куда мы поместим массив выходных данных из нейросети"""
	for record in data_list:
		all_values = record.split(',')
		input = (numpy.asarray(all_values[1:], dtype=numpy.float32) / 255.0 * 0.99) + 0.01
		target = numpy.zeros(10) + 0.01
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

"""
n = NeuralNetwork()
n.setLearningRate(0.1)
n.addInputLayer(784)
n.addHiddenLayer(100, Sigmoid())
#n.addHiddenLayer(100, Sigmoid())
n.addOutputLayer(10, Sigmoid())
n.initialization()

n.trains(inputsTrain, targetsTrain, 1)
all=0.0
ok=0.0
for i in range(len(targetsTest)):
	all+=1.0
	outputResult=numpy.argmax(n.forward(inputsTest[i]))
	targetResult=numpy.argmax(targetsTest[i])
	if (targetResult==outputResult):
		ok+=1.0
print(f"effective=={ok/all}")
"""

#пределы изменения изменяемых параметров сети
#переменный параметр: кол-в скрытых узов сети
hidden_nodes=[30, 100]
#переменный параметр: кол-во скрытых слоёы
hidden_count=[1, 3]
#переменный параметр - количество эпох (циклов обучения)
epoch=[1]

#собираемые данные
datasTime=numpy.zeros((len(epoch), len(hidden_count), len(hidden_nodes)), dtype=numpy.float32)
datasEffective=numpy.zeros((len(epoch), len(hidden_count), len(hidden_nodes)), dtype=numpy.float32)

for epochIndex, epochValue in enumerate(epoch):
	for hiddenCountIndex, hiddenCountValue in enumerate(hidden_count):
		for hiddenNodesIndex, hiddenNodesValue in enumerate(hidden_nodes):
			print(f"epochCount = {epochValue}, hiddenLayerCount={hiddenCountValue}, hiddenNeironCount={hiddenNodesValue}:")
			start_time = time.time()

			# создать экземпляр нейронной сети
			n = NeuralNetwork()
			n.setLearningRate(0.1)
			n.addInputLayer(784)
			for j in range(hiddenCountValue):
				n.addHiddenLayer(hiddenNodesValue, Sigmoid())
			n.addOutputLayer(10, Sigmoid())
			n.initialization()

			# тренировка нейроной сети
			n.trains(inputsTrain, targetsTrain, epochValue)

			# тестирование нейросети
			all = 0.0
			ok = 0.0
			for i in range(len(inputsTest)):
				targetResult = numpy.argmax(targetsTest[i])
				output = n.forward(inputsTest[i])
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
			datasTime[epochIndex][hiddenCountIndex][hiddenNodesIndex]=elapsed_time
			datasEffective[epochIndex][hiddenCountIndex][hiddenNodesIndex]=ok/all
			n.layers=[]
			del n


#визуализируем собранную статистику в графическом виде
for epochIndex, epochValue in enumerate(epoch):
	for hiddenCountIndex, hiddenCountValue in enumerate(hidden_count):
		plt.title("кол циклов обучения = "+str(epochValue))
		plt.plot(hidden_nodes, datasEffective[epochIndex][hiddenCountIndex], label="эпох= "+str(epochValue)+",скр слоёв = "+str(hiddenCountValue), marker='o', markersize=12)
plt.legend()
plt.xlabel("кол-во нейронов на скрытом слое")
plt.ylabel("эффективность обучения")
#plt.show()
plt.savefig('neiro2_'+str(time.time())+'_epoch_'+str(epochValue)+'.png')
