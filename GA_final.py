from bisect import bisect

import csv, math, random


class GA:
	# Static Parameters
	MAX_GENERATIONS =  50
	CROSS_OVER_RATE = 0.25
	MUTATION_RATE = 0.1
	POPULATION_SIZE = 30
	#filename = "Glass_New.csv"
	# filename = "SPECTF_New.csv"
	filename = "accident.csv"
	# Dyamic Parameters
	chromosomes = []
	chromosome_length = -1
	FINAL_CHROMOSOME = []
	FINAL_ACCURACY = -1

	def __init__(self):
		print("GA")
		self.setParameters()

	def initialiseParameters(self):
		self.chromosomes = []


	def setParameters(self):
		with open(self.filename, 'rt') as f:
			reader = csv.reader(f)
			self.chromosome_length = len(next(reader))-1 # -1 because we need to exclude the class label


	def initialisePopulation(self):
		print("\nInitialise Population")
		for i in range(0, self.POPULATION_SIZE):
			randoms = [random.randrange(0,2) for j in range(0, self.chromosome_length)]
			self.chromosomes.append(randoms)
		# print("chromosomes", self.chromosomes)


	def evaluation(self):
		print("\nEvaluation")
		fitness_values = [] # stores fitness function value for each of the chromosomes
		chromosome_accuracy = {} # dictionary to store chromosome number and its corresponding accuracy

		for cr in self.chromosomes:
			nb = NaiveBayes(self.filename, cr)
			nb.loadDataSet()
			nb.trainModel()
			accuracy = nb.ACCURACY
			chromosome_accuracy[accuracy] = cr
			print("Total Accuracy obtained is : %s \n" % accuracy)
			fitness_values.append(self.fitnessFunction(accuracy, "MAX_PROBLEM"))

		m = max(chromosome_accuracy)
		if m > self.FINAL_ACCURACY:
			self.FINAL_ACCURACY = m
			self.FINAL_CHROMOSOME = chromosome_accuracy[m]

		# go for Selection
		self.selection(fitness_values)


	# Maximisation problem F(x)=f(x) ; Minimisation problem F(x) = 1/1+f(x)
	def fitnessFunction(self, accuracy, type):
		if type == "MAX_PROBLEM":
			return accuracy
		else:
			return float(1/(1+accuracy))


	# Roulette Wheel based Selection
	def selection(self, fitness_values):
		print("\nselection")
		probability_fitness = []
		fitness_sum = sum(fitness_values)

		# Individual Probability
		probability_fitness = [float(value/fitness_sum) for value in fitness_values]

		# Cumulative Probability
		cumulative_probability = []
		cumulative_probability.append(probability_fitness[0])
		for i in range(1,len(probability_fitness)):
			cumulative_probability.append(cumulative_probability[i-1] + probability_fitness[i])

		# print("cumulative_probability : ", cumulative_probability)

		# random number between 0 and 1
		rnd = [random.uniform(0,1) for i in range(0, len(cumulative_probability))]
		# select the chromosomes
		selected_chromosomes=[]
		for num in rnd:
			position = bisect(rnd,num)
			if position == 30:
				position = 29 # since last postion is 29 ,  0 based indexing
			selected_chromosomes.append(position)

		# make a copy of the chromosomes
		chromosomes_copy = []
		chromosomes_copy = self.chromosomes

		# replace the old chromosome with the new one
		print("\nselected_chromosomes", selected_chromosomes)
		print("\n chromosomes ",  self.chromosomes)

		for i in range(0, len(selected_chromosomes)):
			self.chromosomes[i] = chromosomes_copy[selected_chromosomes[i]]

		# delete the chromosomes copy
		del chromosomes_copy


	def crossover(self):
		print("\ncrossover")
		randoms = [random.uniform(0,1) for i in range(0, self.chromosome_length)]
		less_than_CR = [postion for postion, x in enumerate(randoms) if x <= self.CROSS_OVER_RATE]
		print("less_than_CR", less_than_CR)
		del randoms

		less_than_CR_copy = []
		less_than_CR_copy = less_than_CR

		cross_over = [random.randrange(0,self.chromosome_length-1) for i in range(0,len(less_than_CR))]

		for i in range(0,len(cross_over)-1):
			for j in range(cross_over[i], self.chromosome_length):
				self.chromosomes[i][j] = self.chromosomes[i+1][j]

		# crossover of last and first chromosome
		if cross_over: # check if list is not empty
			for j  in range(cross_over[-1], self.chromosome_length):
				self.chromosomes[-1][j] = self.chromosomes[0][j]


	def mutation(self):
		print("\nmutation")
		# 1 chromosome : 9 genes :: 30 chromosomes : 270 genes
		# select MUTATION_RATE of the total genes
		genes = self.chromosome_length*self.POPULATION_SIZE
		randoms = [random.randrange(0,genes) for i in (0,math.floor(self.MUTATION_RATE * genes))]

		# flip the randomly chosen genes
		for gene in randoms:
			row = math.floor(gene/self.chromosome_length)
			col = gene%self.chromosome_length
			self.chromosomes[row][col] = self.chromosomes[row][col] ^ 1 # XOR with 1 will flip the bit

		# print("chromosomes after current generation : ", self.chromosomes)

	def trainModel(self):
		generations = 0
		while generations <= self.MAX_GENERATIONS:
			print("\n----------------- GENERATION %s ----------------------" %generations)
			generations += 1
			self.initialiseParameters()
			self.initialisePopulation()
			self.evaluation()
			# self.selection() # selection called form evaluation
			self.crossover()
			self.mutation()

		print("\nFINAL CHROMOSOMES")
		for i in self.chromosomes:
			print(i)

		print("\nCHROMOSOME : %s  ACCURACY : %s" %(self.FINAL_CHROMOSOME, self.FINAL_ACCURACY))



''' Naive Bayes Classifier '''
class NaiveBayes:
	######### Change these values only ##############
	filename = ''
	MAX_ITERATIONS = 50

	# DYNAMIC Parameters
	ATTRIBUTES = -1
	TUPLES = -1
	FOLD_LENGTH = -1

	# STATIC Parameters
	feature_set = []
	data = []
	testing_data = []
	actual_output = []
	offset = 0.1  # 10 percent
	FOLDS = 10
	ACCURACY = 0.0

	# OUTPUT MEASURES
	true_positives = 0
	true_negatives = 0
	false_positives = 0
	false_negatives = 0

	def __init__(self, filename, feature_set):
		print("NAIVE BAYES")
		self.data = []
		self.feature_set = feature_set
		self.filename = filename
		self.ATTRIBUTES = feature_set.count(1) # its important to set the ATTRIBUTES here itself

	def loadDataSet(self):
		tuples = 0
		flag = False
		with open(self.filename, 'rt') as f:
			reader = csv.reader(f)
			for row in reader:
				if flag == False:
					if not row[0].isdigit():
						flag = True
						continue
				tuples += 1
				self.data.append(row)
		self.setParameters(tuples)

		self.formatDataSet()

		# Shuffle the data
		# random.shuffle(self.data)


	def setParameters(self, tuples):
		self.TUPLES = tuples
		# self.ATTRIBUTES = len(self.data[0]) - 1
		self.FOLD_LENGTH = math.ceil(self.offset * self.TUPLES)
		# print("tuples=%s attributes=%s" % (self.TUPLES, self.ATTRIBUTES))


	def formatDataSet(self):
		temp_data = []
		# print("data", self.data)
		# print("features", self.feature_set)

		for i in range(len(self.data)):
			temp_data.append([self.data[i][j] for j in range(0, len(self.feature_set)) if self.feature_set[j] == 1])
			temp_data[i].append(self.data[i][-1]) # appending the class label

		self.data = temp_data
		del temp_data

		# convert strings to float
		for i in range(len(self.data)):
			# Yes : 1 No : 0
			if self.data[i][-1] == 'Yes' or self.data[i][-1] == '1':
				self.data[i][-1] = 1
			elif self.data[i][-1] == 'No'or self.data[i][-1] == '0':
				self.data[i][-1] = 0

			self.data[i][:-1] = [float(x) for x in self.data[i][:-1]]  # convert to float except the last element

		# self.printData("data", self.data)


	def separateClassLabels(self, data):
		seperated = {}
		for row in data:
			if row[-1] not in seperated:
				seperated[row[-1]]=[]
			seperated[row[-1]].append(row[:-1])
		return seperated


	def probabilityOfClassLabel(self, data, label):
		total=0
		for index in data:
			total += len(data[index])
		return len(data[label]) / total


	def mean(self, numbers):
		return sum(numbers)/float(len(numbers))


	def stdDeviation(self, numbers):
		mu = self.mean(numbers)
		n = len(numbers)
		variance = sum([ pow(x-mu,2) for x in numbers]) / float(n-1)
		sigma = math.sqrt(variance)
		return sigma


	def findProbability(self, x, mean, stdDev):
		exponent = math.exp(-(pow(x-mean,2)/(2*pow(stdDev,2))))
		return exponent / (math.sqrt(2*math.pi) * stdDev)


	def trainModel(self):
		# traverse over each fold
		accu = 0
		for fold in range(0, self.FOLDS):
			# print('############################ FOLD : %s  ##############################\n' % (fold+1))
			self.true_positives = self.false_positives = self.true_negatives = self.false_negatives = 0
			# clear the previous testing data
			self.testing_data.clear()
			training_data = list(self.data)
			# print("\n data set", self.data)
			# print("\ntraining_data", training_data)

			# produce the corresponding testing data
			for j in range(0, self.FOLD_LENGTH):
				if self.FOLD_LENGTH * fold + j < self.TUPLES:
					self.testing_data.append(self.data[self.FOLD_LENGTH * fold + j])
					del training_data[self.FOLD_LENGTH * fold]

			seperated_data = {} # dictionary to store seperated class labels
			seperated_data = self.separateClassLabels(training_data)

			# print("\nseperated data", seperated_data)

			mean_list={} # dictionary to store a list of mean of attributes for each class label
			stdDev_list={} # dictionary to store a list of standard deviation of attributes for each class label

			for label in seperated_data:
				if label not in mean_list:
					mean_list[label]=[]
					stdDev_list[label]=[]
				for i in range(0, self.ATTRIBUTES):
					attr_list = [x[i] for x in seperated_data[label]]
					mean_list[label].append(self.mean(attr_list))
					stdDev_list[label].append(self.stdDeviation(attr_list))

			# Now predict for the testing data
			self.predict(seperated_data, mean_list, stdDev_list)

			# Cumulative accuracy
			# self.printData("accuracy", self.findAccuracy())
			accu += self.findAccuracy()

			# Average accuracy for current Fold
			self.ACCURACY = accu / self.FOLDS


	def predict(self, seperated_data, mean_list, stdDev_list):
		# print("############################ TESTING ################################")
		probability = {}
		for row in self.testing_data:
			# access each class label
			for label in seperated_data:
				probability[label] = self.probabilityOfClassLabel(seperated_data,label)
				# traverse each attribute except the last column
				for j in range(0,len(row)-1):
					probability[label] *= self.findProbability(row[j],mean_list[label][j],stdDev_list[label][j])

			temp_value = max(probability, key = lambda i : probability[i])

			######## In General ########
			# if temp_value == row[-1]:
				# true++
			# else
				# false++
			############################

			#### In case of only 2 class labels ####
			if temp_value == 1:
				if temp_value == row[-1]:
					self.true_positives += 1
				else:
					self.false_positives += 1

			else:
				if temp_value == row[-1]:
					self.true_negatives += 1
				else:
					self.false_negatives += 1


	def findAccuracy(self):
		# print("\ntp:%s fp:%s tn:%s fn:%s" %(self.true_positives, self.false_positives, self.true_negatives, self.false_negatives))

		# PRECISION & RECALL for positive class
		precision_positive = -1
		recall_positive = -1
		if self.true_positives + self.false_positives != 0:
			precision_positive = self.true_positives / (self.true_positives + self.false_positives)

		if self.true_positives + self.false_negatives != 0:
			recall_positive = self.true_positives / (self.true_positives + self.false_negatives)

		# print("Positive_Precision = %s  Positive_Recall = %s" %(precision_positive, recall_positive))

		# PRECISION & RECALL for negative class
		precision_negative = -1
		recall_negative = -1
		if self.true_negatives + self.false_negatives != 0:
			precision_negative = self.true_negatives / (self.true_negatives + self.false_negatives)

		if self.false_positives + self.true_negatives != 0:
			recall_negative = self.true_negatives / (self.false_positives + self.true_negatives)

		# print("Negative_Precision = %s  Negative_Recall = %s\n" %(precision_negative, recall_negative))

		return (self.true_positives + self.true_negatives) / \
			   (self.true_positives + self.false_positives + self.false_negatives + self.true_negatives) * 100


	def printData(self, type, data):
		print(type,"=", data, "\n")
# Class ends



if __name__ == "__main__":
	model = GA()
	model.trainModel()
