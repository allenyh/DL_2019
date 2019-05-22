
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


class Neuron(object):
	def __init__(self, w_num):
		self.input = []
		self.output = 0
		self.weights = []
		self.bias = 0
		self.delta = 0
		self.bias = np.random.randn()
		for i in range(w_num):
			self.weights.append(np.random.randn())
	
	def forward(self, inp):
		if len(inp) != len(self.weights):
			raise Exception("input and weight's length mismatch!")
		self.input = inp
		self.output = 0
		for i in range(len(self.input)):
			self.output += self.weights[i] * self.input[i]
		self.output = self.sigmoid(self.output + self.bias)
		return self.output
	
	def sigmoid(self, x):
		return 1/(1+np.exp(-x))
	
	def derivative_sigmoid(self, x):
		return np.multiply(x, 1.0 - x)
	
	def cal_delta(self, error):
		self.delta = error * self.derivative_sigmoid(self.output)

	def update(self, learnRate):
		for i in range(len(self.weights)):
			self.weights[i] -= learnRate * self.delta * self.input[i]
		self.bias -= learnRate * self.delta


# In[3]:


class NeuralLayer(object):
	def __init__(self, input_ch, neuron_num):
		self.neurons = []
		for i in range(neuron_num):
			neuron = Neuron(input_ch)
			self.neurons.append(neuron)

	def forward(self, inp):
		output = []
		for i in range(len(self.neurons)):
			output.append(self.neurons[i].forward(inp))
		return output

	def get_deltas(self, pre_layer):
		deltas = []
		pre_l_neurons = pre_layer.neurons
		for i in range(len(self.neurons)):
			error = 0
			for pre_l_neuron in pre_l_neurons:
				error += pre_l_neuron.delta * pre_l_neuron.weights[i]
			self.neurons[i].cal_delta(error)
			deltas.append(self.neurons[i].delta)
		return deltas

	def update(self, learnRate):
		for neuron in self.neurons:
			neuron.update(learnRate)


# In[4]:


class NeuralNetwork(object):
	def __init__(self, learnRate, debug=False):
		self.layers = []
		self.learnRate = learnRate
		self.debug = debug

	def train(self, dataset):
		inputs, labels = dataset
		for i in range(len(inputs)):
			self.forward(inputs[i])
			self.backpropagate(labels[i])
			self.update()
		return labels[i] - self.layers[-1].neurons[0].output

	def forward(self, inp):
		x = inp
		for i in range(len(self.layers)):
			x = self.layers[i].forward(x)
			if self.debug:
				print("Layer {0} output {1}".format(i, x))
		return x

	def backpropagate(self, label):
		last_layer = None
		deltas = []
		for i in range(len(self.layers), 0, -1):
			current_layer = self.layers[i-1]
			if last_layer is None:
				for i in range(len(current_layer.neurons)):
					error = -(label - current_layer.neurons[i].output)
					current_layer.neurons[i].cal_delta(error)
			else:
				deltas = current_layer.get_deltas(last_layer)
			last_layer = current_layer
			if self.debug:
				print("Layer {0} deltas {1}".format(i, deltas))

	def update(self):
		for layer in self.layers:
			layer.update(self.learnRate)

	def predict(self, inp):
		return self.forward(inp)

	def add_layer(self, layer):
		self.layers.append(layer)
        
	def set_lr(self, lr):
		self.learnRate = lr


# In[5]:


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return inputs, labels

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return inputs, labels


# In[6]:


def show_result(inputs, labels, pred_y):
	plt.subplot(1,2,1)
	plt.title('Ground truth', fontsize=18)
	for i in range(len(inputs)):
		if labels[i] - 0 < 0.2:
			plt.plot(inputs[i][0], inputs[i][1], 'ro')
		else:
			plt.plot(inputs[i][0], inputs[i][1], 'bo')
	plt.subplot(1,2,2)
	plt.title('Predict result', fontsize=18)
	for i in range(len(inputs)):
		if pred_y[i] - 0 < 0.2:
			plt.plot(inputs[i][0], inputs[i][1], 'ro')
		else:
			plt.plot(inputs[i][0], inputs[i][1], 'bo')
	plt.show()


# In[7]:


learnRate = 0.5
epoch = 1500
dataset = generate_linear(100)

input_channel = len(dataset[0][0])
layers = [input_channel,4,4,1]
neuron_layer = []
nn = NeuralNetwork(learnRate = learnRate, debug=False)
for i in range(len(layers)-1):
    weight = []
    bias = np.random.randn()
    for j in range(layers[i]*layers[i+1]):
        weight.append(np.random.randn())
    layer = NeuralLayer(input_ch=layers[i], neuron_num=layers[i+1])
    nn.add_layer(layer)
for i in range(epoch):
    loss = nn.train(dataset)
    loss_sum = 0
    if i % 100 == 0:
        loss_sum = np.mean(loss_sum + loss) 
        print("epoch {0} loss {1}".format(i, loss_sum))
    else:
        loss_sum += loss


# In[11]:


#test
inputs, labels = dataset
pred_y = []
error_count = 0
for i in range(len(inputs)):
    prediction = nn.predict(inputs[i])[0]
    #print("input {0} , label {1}, predict {2}".format(inputs[i], labels[i], prediction))
    pred_y.append(prediction)
    if prediction - labels[i] > 0.2:
        error_count += 1
print("error/total: {0}/{1}".format(error_count, len(inputs)))
    


# In[12]:


for i in pred_y:
    print(i)


# In[13]:


show_result(inputs, labels, pred_y)

