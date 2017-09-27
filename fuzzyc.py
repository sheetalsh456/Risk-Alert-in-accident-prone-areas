from __future__ import division
import csv, math, random
from copy import deepcopy

class Fuzzy_C_Means:
    filename = 'accident.csv'
    MAX_ITERATIONS = 50
    ATTRIBUTES = -1
    TUPLES = -1
    FOLD_LENGTH = -1
    feature_set = []
    data = []
    testing_data = []
    actual_output = []
    offset = 0.1  
    FOLDS = 1
    ACCURACY = 0.0
    C = 5  
    fuzziness_factor = 1.5
    stopping_threshold = 0.001
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    def __init__(self, threshold, m):
        print("Fuzzy_C_Means")
        self.data = []
        self.stopping_threshold = threshold
        self.fuzziness_factor = m

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

    def setParameters(self, tuples):
        self.TUPLES = tuples
        self.ATTRIBUTES = len(self.data[0]) - 1
        self.FOLD_LENGTH = math.ceil(self.offset * self.TUPLES)

    def formatDataSet(self):
        for i in range(len(self.data)):
            if self.data[i][-1] == 'Yes' or self.data[i][-1] == '1':
                self.data[i][-1] = 1
            elif self.data[i][-1] == 'No' or self.data[i][-1] == '0':
                self.data[i][-1] = 0
            self.data[i][:-1] = [float(x) for x in self.data[i][:-1]]  

    def findDistance(self, x, y):
        return math.sqrt(sum([(i-j)**2 for i,j in zip(x,y)]))

    def generate_randoms(self, C):
        r = [ random.random() for i in range(0, C) ]
        s = sum(r)
        r = [ i/s for i in r]
        return r

    def findMean(self, a):
        rows = len(a)
        cols = len(a[0])
        column_sums = [sum(row[i] for row in a) for i in range(0, cols)]
        mean_values = [i/rows for i in column_sums]
        return mean_values

    def terminate(self, old_membership_matrix, new_membership_matrix):
        if not new_membership_matrix or not old_membership_matrix:
            return False
        for i in range(0, len(new_membership_matrix)):
            for j in range(0, self.C):
            	diff = abs(new_membership_matrix[i][j] - old_membership_matrix[i][j])
                if diff>self.stopping_threshold:
                    return False
        return True

    def trainModel(self):
        iterations = 1
        old_membership_matrix = []
        new_membership_matrix = []
        for i in range(0, self.TUPLES):
            new_membership_matrix.append(self.generate_randoms(self.C))
        while iterations <= self.MAX_ITERATIONS and not self.terminate(old_membership_matrix, new_membership_matrix):
            if iterations != 1:
                old_membership_matrix = deepcopy(new_membership_matrix)
            iterations += 1
            clusters = {}
            for i in range(0, self.C):
                if i not in clusters:
                    clusters[i] = []
            for c in clusters:
                for j in range(0, self.ATTRIBUTES):
                    numer = 0
                    denom = 0
                    for i in range(0, self.TUPLES):
                        mem_value = new_membership_matrix[i][c]**self.fuzziness_factor
                        numer += mem_value * self.data[i][j]
                        denom += mem_value
                    clusters[c].append(numer/denom)  
            for i in range(0, self.TUPLES):
                for j in range(0, self.C):
                    denom = 0
                    for k in clusters:
                        dist1 = self.findDistance(self.data[i],clusters[j])
                        dist2 = self.findDistance(self.data[i],clusters[k])
                        denom += (dist1 / dist2) ** (2 / (self.fuzziness_factor -1 ))
                    new_membership_matrix[i][j] = float(1/denom)
        self.display(clusters, new_membership_matrix, iterations)
        self.predict(clusters, new_membership_matrix)

    def display(self, clusters, new_membership_matrix, iterations):
        print "Number of iterations :", iterations
        print "Centroids :"
        with open("cluster.txt","w") as f4:
                f4.write("0")
        for i in range(0, len(new_membership_matrix)):
            id=0
            
            for j in range(0, self.C):
                if(new_membership_matrix[i][j] > new_membership_matrix[i][id]):
                    id=j
            with open("cluster.txt","a") as f4:
                f4.write("%f\n" % (id+1))
            print "\nPoint", i+1 ," Cluster", id+1,
            for j in range(0, self.C):
                print "        ", new_membership_matrix[i][j],
            print "        ", sum(new_membership_matrix[i]),

    def predict(self, clusters, membership_matrix):
        for i in range(0, len(self.data))	:
        	max_index = membership_matrix[i].index(max(membership_matrix[i]))

        	if self.data[i][-1] == 0:
        		if max_index == 0:
        			self.true_negatives += 1
        		else:
        			self.false_positives += 1

        	elif self.data[i][-1] == 1:
        		if max_index == 0:
        			self.false_negatives += 1
        		else:
        			self.true_positives += 1


if __name__ == "__main__":
	fuzziness_factors = [1.25, 1.5, 1.1]
	threshold = [0.001, 0.01, 0.1]
	for m in fuzziness_factors:
		for theta in threshold:
			print "THRESHOLD : ", theta, "FUZZINESS : ", m
			model = Fuzzy_C_Means(threshold, m)
			model.loadDataSet()
			model.trainModel()
