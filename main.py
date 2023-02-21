#thank you Jesus Christ.
#import pandas as pd
#import collections
#import random

#from sklearn.linear_model import LinearRegression
#linefitter = LinearRegression()
#linefitter.fit(x,y)

#import numpy as np
#import pandas as pd
#from matplotlib import pyplot as plt

#imported data
#data = pd.read_csv("honeyproduction.csv")
#print(data.head(10))

#data01 = data.groupby("year")["totalprod"].mean()
#print(data01)

#column = ["year","totalprod"]
#data02 = pd.DataFrame(data01, columns= column).drop(columns= "year").reset_index()
#new_date = pd.period_range(start= 1998, end= 2012, freq= "Y")
#print(data02.dtypes)

##x = data02["year"].reshape(-1,1)
#y = data02.totalprod

#plt.scatter(x, y)
#plt.show()

#linear = LinearRegression()
#linear.fit(x,y)

#new_y = linear.predict(y)

#from google.cloud import bigquery

#create client object that will recieve data from the bigquery database
#client = bigquery.client() #very important

#how to access, start by crafting a data  set reference
#dataset_ref = client.dataset("hacker_news", project = "bigquery-public-data")

#then request info from the api
#dataset = client.get_dataset(dataset_ref)

#tables = list(client.list_table(dataset))

#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])

#plt.show()


#import numpy as np
#import pandas as pd

#data = pd.read_csv("https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/manhattan.csv")

#taking alook at the first few rows.

#print(data.head(5))

#turn it to a dataframe

#new_data = pd.DataFrame(data)
#training data used to fit the model
#test data used to test the model at the end.
#[][][][][][x][x][x] - 80% training, 20% testing

#from sklearn.model_selection import train_test_split
#split data into x and y

#x = new_data[["bedrooms", "bathrooms", "size_sqft", "min_to_subway",
#              "floor", "building_age_yrs", "no_fee", "has_roofdeck", "has_washer_dryer", "has_doorman",
#              "has_elevator", "has_dishwasher", "has_patio", "has_gym"]]
#y = new_data[["rent"]]

#now use the imported library to split the data

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, train_size= 0.8, random_state= 6)

#now to training visualization, did my first succesful machine learning with multiple and simple linear regression
# start by creating a 2d graph to explain the context of the price movement

#import matplotlib.pyplot as plt
#plt.scatter(y_test,y_predicted, alpha= 0.4) #creating a scatter plot

#creating x-axis label and y-axis label
#plt.xlabel("Original price")
#plt.ylabel("Predicted price")

#creating a tittle
#plt.title("Machine learning with linear regression")
#plt.legend(["old", "new"])
#plt.show() #to display the plot


#higlighting correletions plot them individualy and look at the graphs trend
#plt.scatter(new_data[['size_sqft']], new_data[['rent']], alpha=0.4)
#plt.show()

# one of the best ways to evaluate our models is via score or r2
#print(model.score(x_train, y_train).__round__(2)) # this is the model = LinearRegression()
#what commit is this
#thank you Jesus Christ.

#from scipy.spatial import distance
#print(distance.euclidean([1,2], [4,0]))
#print(distance.cityblock([1,2], 4,0))
#print(distance.hamming([5,4,9], [1,7,9]))

#star_wars = [125, 1977]
#raiders = [115, 1981]
#mean_girls = [97, 2004]
#def distance(movie1, movie2):
#    distance = 0
#    for x in range(len(movie1)):
#        distance += (movie1[x] - movie2[x]) ** 2
#    return distance ** 0.5

#print(distance(star_wars, raiders))
#print(distance(star_wars, mean_girls))

#min max nomalization
#def min_max_normalize(lst):
#    minimum = min(lst)
#    maximum = max(lst)
#    normalized = list()

#    for value in lst:
#        vast = (value - minimum)/ (maximum - minimum)
 #       normalized.append(vast)
 #   return normalized


#def classify(unknown, dataset, k):
##    distance = []
#    for tittle in range(len(dataset)):
#        movie = dataset[tittle]
#        distance_to_point = distance(movie, unknown)

#        distance.append([distance_to_point, tittle])
#        distance.sort()
#        neigbors = distance[0:k]
        #distance = ((dataset[tittle] - unknown[tittle]) **2)**0.5
        #distance ** 0.5
#    return neigbors

#print(classify([.4, .2, .9], movie_dataset, 5))


#def classify(unknown, dataset, lables, k):
#    num_good = 0
#    num_bad = 0
#    for movie in neighbors:
#        title = movie[1]
#    return title

#jump = [ 1, 2, 3, 4, 5, 6]
#print(3 in jump)

#from sklearn.neighbors import KNeighborsClassifier
#from movies import movie_dataset, labels

#classifier = KNeighborsClassifier(n_neighbors= 3)

## [0.5, 0.2, 0.1],
  #[0.9, 0.7, 0.3], #fit takes two inputs, the training point and label
  #[0.4, 0.5, 0.7]
#]

#training_labels = [0, 1, 1]

#classifier.fit(training_points, training_labels)

#unknown_points = [
 #   [0.2, 0.1, 0.7],
  ## [0.5, 0.8, 0.1]
#]

#guesses = classifier.predict(unknown_points)
#print(guesses)

#def distance(v, b):
 #  for i in range(len(v)):
  #     dis +=  (v[i] - b[i])** 2
   # return dis **0.5

#print(distance([0,1], [1,2]))
##print(distance([1,0], [1,2]))
#print(distance([3,1], [1,2]))
#print(distance([1,3], [1,2]))

#Breast Cancer classifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.datasets import load_breast_cancer

#breast_cancer = load_breast_cancer()
#print(breast_cancer.data[0])
#print(breast_cancer.feature_names)
#print(breast_cancer)

#print(breast_cancer.target)
#print(breast_cancer.target_names)

#now we need to split the data into training and validation
#from sklearn.model_selection import train_test_split
#train_test_split(breast_cancer.data, breast_cancer.target, test_size= 0.2, random_state= 100)

#training_data, validation_data, training_labels,validation_labels = train_test_split(breast_cancer.data, breast_cancer.target, test_size= 0.2, random_state= 10)
#print(len(training_data) == len(training_labels))

#classifier = KNeighborsClassifier(n_neighbors=3)
#classifier.fit(training_data, training_labels)

#print(classifier.score(validation_data, validation_labels))

#checking for accuracy of the  neighbors
#accuracy = []
#neighbor = range(1,101)
#for k in range(1, 101):
#    classifier = KNeighborsClassifier(n_neighbors=k)
#    classifier.fit(training_data, training_labels)
#    accuracy.append(classifier.score(validation_data, validation_labels))

#import matplotlib.pyplot as plt

#plt.plot(neighbor, accuracy)
#plt.ylabel("accuracy")
#plt.xlabel("neigbors")
#plt.show()

#thank you Jesus Christ

#logistic regressions
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegression

#logistic = LogisticRegression()
#logistic.fit(training_data,training_labels)
#logvalue = logistic.predict(validation_data)
#print(logvalue)
#print(validation_labels)

#plt.plot(training_data, training_labels)
#plt.(validation_data, validation_labels)
#plt.plot(validation_data, logvalue)
#plt.show()

#def calculate_odd(x):
#    new = x/100
#    print(round(new/(1-new),2))

#def calculate_odd_log(x):
#    new = x/100
#    some = round(new/(1-new),2)
#    print(numpy.log(some))

#calculate_odd(40)
#calculate_odd_log(40)

#calculate_odd_log(90)
#calculate_odd(90)


#fitting a logistic regression
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split

#from sklearn import datasets

#iris = datasets.load_iris()
#print(iris.keys())
#print(iris.data.shape)
#print(iris["target"])
#print(iris["target_names"])

#x_training, x_test, y_training, y_test = train_test_split(iris.data, iris.target, test_size= 0.2, train_size= 0.8, random_state= 10)

#model = LogisticRegression()
#model.fit(x_training, y_training)

#print(x_training.shape)
#print(y_training.shape)

#print(model.score(x_test, y_test)) # perfect fit

#new_values = model.predict(x_test)
#proba = model.predict_proba(x_test)

#print(proba[:,1])
#print(new_values)

#x = model.coef_
#y = model.intercept_

#print(accuracy_score(y_test, new_values))
#print(confusion_matrix(y_test, new_values))
#print(precision_score(y_test, new_values))
#import numpy as np
#import  pandas as pd

#calculaed_log = x + y * x_training
#pre_probability =  np.exp(calculaed_log)/(1 + np.exp(calculaed_log))

#from matplotlib import pyplot as plt
#plt.plot(new_values, y_test)
#plt.plot(new_values)
#plt.legend()
#plt.show()

#from sklearn.preprocessing import StandardScaler
#scale = StandardScaler()
#x_scaled = scale.fit_transform() # at onces


#import numpy as np
#import pandas as pd

#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import StandardScaler

#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import f1_score
#from sklearn.metrics import recall_score

#titanic_import = pd.read_csv("train.csv")
#names = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
#titanic = pd.DataFrame(titanic_import, columns= names)
#print(titanic.head(5))

#print(titanic.columns)
#change sex to have a 1, 0 depending on sex.
#sex = titanic[["Sex"]]
#print(sex)

#def change_unit(x):
#    for i in x:
#        if i == "female":
#            return 1
#        else:
#            return 0
#titanic["Sex"].apply(change_unit)
#print(titanic["Sex"])

#def change(any):
#    for i in range(len(any)):
#        if any[i] == "female":
#            return 1
#        else:
#            return 0
#    return

#john = ["female", "male", "female", "male"]
#j = pd.DataFrame(john)
#print(john.apply(change))
#print(change(titanic.Sex))
#jax = [1, 0 for x in john if x == "female" else]
#print(jax)
#print(titanic['Sex'].change())
#titanic["SEX"] = titanic.apply(change)
#print(titanic["SEX"])
#print(titanic['Sex'].map({'female':1, 'male':0})) # worked
#titanic['Gender'] = titanic['Sex'].map({'female':1, 'male':0})
#print(titanic.Gender)

#titanic["age"] =  titanic['Age'].fillna(value= titanic['Age'].mean()).round(1)
#print(titanic['Age'].fillna(value= titanic['Age'].mean()).round(1))
#print(titanic.age)

#titanic_updated = titanic.drop(columns=['Sex', 'Age'])
#print(titanic_updated.columns)

#titanic_updated['FirstClass'] = titanic_updated['Pclass'].map({1:1, 2:0, 3:0})
#titanic_updated['SecondClass'] = titanic_updated['Pclass'].map({2:1, 1:0, 3:0})
#titanic_updated['ThirdClass'] = titanic_updated['Pclass'].map({3:1, 1:0, 2:0})

#print(titanic_updated.columns)

#select columns
#features = titanic_updated[['Gender', 'age', 'FirstClass', 'SecondClass','ThirdClass']]
#survival = titanic_updated[['Survived']]

#x_train, x_test, y_train, y_test = train_test_split(features, survival, test_size= 0.2, train_size= 0.8, random_state=10)

#standardize
#standar_x = StandardScaler()
#x_train_scaled = standar_x.fit_transform(x_train)
#x_test_scaled = standar_x.transform(x_test)

#logistic_reg = LogisticRegression()
#logistic_reg.fit(x_train_scaled, y_train)
#print(logistic_reg.score(x_train_scaled,y_train))
#print(logistic_reg.score(x_test_scaled,y_test))

#print(logistic_reg.coef_)
#print(logistic_reg.intercept_)

#predicted_prob = logistic_reg.predict_proba(x_test_scaled)
#predicted_y = logistic_reg.predict(x_test_scaled)
#print(predicted_prob[:,1])

#check_accuracy
#print(accuracy_score(y_test, predicted_y))
#print(precision_score(y_test, predicted_y))
#print(recall_score(y_test,predicted_y))

#Jack = np.array([0.0,20.0,0.0,0.0,1.0])
#Rose = np.array([1.0,17.0,1.0,0.0,0.0])
#Me = np.array([0.0,23.0,1.0,0.0,0.0])

#combined = np.array([Jack, Rose, Me])
#trans = standar_x.transform(combined)

#print(logistic_reg.predict(trans))



#import numpy as np
#import pandas as pd
#import collections


#split_labels_1 = [
#  ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "vgood"],
#  [ "good", "good"],
#  ["vgood", "vgood"]
#]

#split_labels_2 = [
#  ["unacc", "unacc", "unacc", "unacc","unacc", "unacc", "good", "good", "good", "good"],
#  ["vgood", "vgood", "vgood"]
#]

#unsplit_labels = ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "good", "good", "vgood", "vgood", "vgood"]


#def gini(dataset):
#  impurity = 1
#  label_counts = collections.Counter(dataset)
#  for label in label_counts:
#    prob_of_label = label_counts[label] / len(dataset)
#    impurity -= prob_of_label ** 2
#  return impurity

#data ={'a':5, 'b':5, 'c':5, 'd':5}
#things = {'a':10, 'b':10}
#a = 20

#for x in things:
#  impurity = 1
#  impurity -= (data[x]/a) ** 2

#print(impurity)




#cali = gini(unsplit_labels)

#for x in split_labels_2:
#  cali -= gini(x)

#print(cali)


#def information_gain(starting_labels, split_labels):
#  info_gain = gini(starting_labels)
    # Multiply gini(subset) by the correct percentage below
#  for subset in split_labels:
#    info_gain - gini(subset) * len(subset)/len(starting_labels)
#  return info_gain

#print(information_gain(unsplit_labels, split_labels_1))

#from sklearn.tree import DecisionTreeClassifier
#clasifier = DecisionTreeClassifier()

#from sklearn.metrics import confusion_matrix


#Titanic_ref = pd.read_csv('train.csv', header= 0)
#titanic = pd.DataFrame(Titanic_ref)
#print(titanic.head(5))
#titanic_corr = round(titanic.corr(), 2)
#print(titanic_corr)

#import seaborn as sns
#import numpy as np
#import matplotlib.pyplot as plt
#sns.set_theme(style="white")

#sns.heatmap(titanic_corr);

#print(titanic.Fare)
#print(titanic.Parch)

#import numpy as np
#import pandas as pd
#import random
#random.seed(4)

#indices = [random.randint(0,999) for _ in range(1000)]
#print(indices)

#data_subset = []
#labels_subset  = []

#for index in indices:
#    data_subset.append(car_data[index])
#    labels_subset.append(car_labels[index])

#subset_tree = build_tree(data_subset, labels_subset)
#print(subset_tree)
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier

#import numpy as np
#import pandas as pd
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import tree
#from sklearn.model_selection import train_test_split

#names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
#         'marital-status', 'occupation', 'relationship', 'race', 'sex',
#         'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
#         'class']

#adults = pd.read_csv('adult.data', header= 0, names= names)
#print(adults.shape)
#print(adults.info)
#print(adults.describe)

#print(adults.head(5))
#print(adults.columns)

#adults_x = adults.drop(columns= ['fnlwgt', 'education-num'])
#print(adults_x.columns)
#print(adults.corr())

#decide = DecisionTreeClassifier(max_depth=5)
#classifier = RandomForestClassifier(n_estimators=2000, random_state= 0)
#Ney = KNeighborsClassifier(n_neighbors= 4)
#-------------------------------------------------------------------------------------------changing values
#import collections
#ocupa = collections.Counter(adults['occupation'])
#on = collections.Counter(adults['native-country'])
#edu = collections.Counter(adults['education'])
#print(collections.Counter(adults['education-num']))
#work = collections.Counter(adults['workclass'])
#married = collections.Counter(adults['marital-status'])
#relation = collections.Counter(adults['relationship'])
#race = collections.Counter(adults['race'])
#sex = collections.Counter(adults['sex'])
#clas = collections.Counter(adults['class'])


#def change(data):
#    start = 0
#    for x in data:
#        data[x] = start
#        start += 1
#    return data

#change(ocupa)
#change(con)
#change(work)
#change(married)
#change(relation)
#change(race)
#change(sex)
#change(clas)
#change(edu)
#--------------------------------------------------- data definition

#adults_x['class']  = adults_x['class'].map(clas)
#adults_x['occupation'] = adults_x['occupation'].map(ocupa)
#adults_x['native-country'] = adults_x['native-country'].map(con)
#adults_x['workclass'] = adults_x['workclass'].map(work)
#adults_x['marital-status'] = adults_x['marital-status'].map(married)
#adults_x['relationship'] = adults_x['relationship'].map(relation)
#adults_x['race'] = adults_x['race'].map(race)
#adults_x['sex'] = adults_x['sex'].map(sex)
#adults_x['education'] = adults_x['education'].map(edu)
#--------------------------------------------------checking corr
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#adults['workclass'] = le.fit_transform(adults[['workclass']])
#print(adults.head(12))
#----------------------------------------------- label

#labels = adults_x[['class']]
#data = adults_x[['age', 'workclass','education','occupation', 'relationship', 'race', 'sex',
#            'capital-gain', 'capital-loss','hours-per-week', 'native-country']]

#------------------------------------------------- training
#train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size= 0.2, random_state= 1, train_size= 0.8)

#forrest = RandomForestClassifier(random_state= 1,max_leaf_nodes=3000, max_depth=2000)
#orrest.fit(train_data, train_label)

#from sklearn.linear_model import LogisticRegression, LinearRegression
#from sklearn.tree import DecisionTreeClassifier


#classisfier = LogisticRegression()

#classisfier.fit(train_data, train_label)

#---------------------------accuracy
#print(classisfier.score(test_data, test_label)) #tick

#-------------------------------------prediction
#predicte_forrest = classisfier.predict(test_data)

#------------------------------------------------------------check accuracy
#from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

#print(precision_score(test_label, predicte_forrest))
#print(accuracy_score(test_label, predicte_forrest)) #tick
#print(confusion_matrix(test_label, predicte_forrest))


#----------------------------------trying to change
#print(adults_x.corr())


#thank You Grace. In Jesus Christ name.
#------------------------------------------------------- Cleared the data Space for Clustering K-means
#import numpy as np
#import pandas as pd

#from sklearn import datasets #- data loading from sklearn

#iris = datasets.load_iris()
#print(iris.data) # how to see the data by row
#print(iris.DESCR)
#print(iris.target)

#import seaborn as sns #---------------------------- visualize dat before k-means
#import matplotlib.pyplot as plt
#important to read the DESCR if you load the data from sklearn dataset
#sepal_length = iris.data[:,0]
#petal_length = iris.data[:,2]

#sns.scatterplot(iris.data, x = sepal_length, y = petal_length, alpha = 0.5)
#plt.show()
#--------------------------------------------------------- Implementing K-mean step 1
#-------- Create data samples to the nearest Centroid
#-------- Update the centroid based on the assigned data sample
#-------------- All this follows from the alreayd plotted data

#k = 3 #-------------- because we have 3 species

#import random #- never used, used np instead
#x = np.random.uniform(min(sepal_length), max(sepal_length), k)
#y = np.random.uniform(min(petal_length), max(petal_length), k)

#centroids = np.array(list(zip(x,y))) #-------------------- used to join and combine the files

#--------------------------------------------------------------------------------------------- Plot again
#sns.scatterplot(iris.data, x = sepal_length, y = petal_length, alpha = 0.5)
#sns.scatterplot(data= centroids, x = x, y = y)
#plt.show()

#------------------------ calculate distance, ucadin
#from scipy.spatial import distance
#print(distance.euclidean(x, y))

#-- create function
#def distance(x,y):
#    distance = 0
#    for i in range(len(x)):
#        distance += (x[i] - y[i]) ** 2
#    return distance ** 0.5

# step 2 create label sizes to fit the the length of the data

#labels = np.zeros(len(iris.data))
#distances = np.zeros(k)

#-------------------------------------create loop
#sep_petal = np.array(list(zip(sepal_length, petal_length)))

#for i in range(len(iris.data)):
#    k = 0
#    distances[k] = distance(sep_petal[i], centroids[k])
#    cluster = np.argmin(distances)
#    labels[i] = cluster
#    k +=1

#print(labels)
#print(distances)

#---------------------------------------- 2nd stage ebds
#from copy import deepcopy
#centroids_old = deepcopy(centroids)

#for i in range(k):
#    points = []
#    for j in range(len(sep_petal)):
#        if labels[j] == i:
#            points.append(sep_petal[j])

# Now using ---------------------------------------------------------------------------------------------sklearn
#from sklearn.cluster import KMeans

#moodel = KMeans(n_clusters= k)
#moodel.fit(iris.data)

#labels = moodel.predict(iris.data)
#print(labels)

#all_data = np.array(list(zip(sepal_length, petal_length, labels)))
#------------------------------------ plot revised

#plt.scatter(x = sepal_length, y = petal_length, c= labels)
#plt.show()

#--------------------------------------------------------------- Checking for accuracy
#things = pd.Series(iris.target)
#print(things)

#species = things.map({0:'setosa', 1:'versicolor', 2:'virginica'})
#species.map({0:'setosa', 1:'versicolor', 2:'virginica'})

#df = pd.DataFrame({'label':labels, 'species':species})
#accuracy = pd.crosstab(df['label'], df['species'])
#print(accuracy)


#--------------------------- determining the best fit of cluster
#k = list(range(1,10))
#groups = []
#for i in k:
#    model = KMeans(n_clusters=i)
#    model.fit(iris.data)
#    groups.append(model.inertia_)

#print(groups)

#plt.plot(k, groups, '-o')
#plt.show()
# Thank You Christ Jesus. for Everything

#----------------------------------------------------------- Starting homework
#import numpy as np
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

#from sklearn.cluster import KMeans
#from sklearn import datasets

#-------------------------------- Data set imported

#digits = datasets.load_digits()
#print(digits.DESCR)
#print(digits.data)
#print(digits.target)

#plt.gray()
#plt.matshow(digits.images[100])
#plt.show()

#print(digits.target[100])

#----------------------------------------------------------------- made a plot and checked if its correct.
# got to find out which fit is best
#to_plot = []
#ku = range(1,23)
#for i in ku:
#    model = KMeans(n_clusters=i)
#    model.fit(digits.data)
#    to_plot.append(model.inertia_)

#plt.plot(ku,to_plot, '-o')
#plt.show()
# --------------------------------------------------- I think the best k is 8 or 9
#model = KMeans(n_clusters= 10, init='k-means++', random_state= 42)
#model.fit(digits.data)
#predicted = model.predict(digits.data)

#print(predicted)
# check for accuracy
#print(pd.crosstab(digits.target, predicted))

#--------------------- this is a new fig printing
#fig = plt.figure(figsize=(5, 2))
#fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

#for i in range(10):
#    ax = fig.add_subplot(6,2,1+i) # initializing subplot
#    ax.imshow(model.cluster_centers_[i].reshape((8,8)), cmap=plt.cm.binary)

#plt.show()

#plt.scatter(predicted, range(10))
#plt.show()

#print(digits.data)
#--------------------------------------- trying plotting
#x = digits.data[:,0]
#y = digits.data[:,1]
#plt.scatter(x,y, c = predicted, alpha= 0.5)
#plt.show()
#Thank you Jesus Christ
#class Tweet:
#    pass

#a = Tweet()
#a.message = '140 characters'
#print(a.message)


#--------------------------------------------------- Learned class on youtube
# Neurons
#class Perceptron: # has got to be in caps
#    def __init__(self, num_inputs = 2, weights = [2,1]):
#        self.num_inputs = num_inputs
#        self.weights = weights

#   def weighted_sum(self, inputs):
#        weighted = 0
#        for i in range(self.num_inputs):
#            weighted += self.weights[i] * inputs[i]
#        return weighted
#    def activation(self, weighted):
#        if weighted >= 0:
#            return 1
#        else:
#            return -1
#    def training(self, training_set):
#        foundline = False
#        while not foundline:
#            total_error = 0
#            for inputs in training_set:
#                prediction = self.activation(self.weighted_sum(inputs))
#                actual = training_set[inputs]
#                error = actual - prediction

#                total_error += abs(error)
#                for i in range(self.num_inputs):
#                    self.weights[i] += error * inputs[i]
#            if total_error == 0:
#                foundline = True

#small_training_set = {(0,3):1, (3,0):-1, (0,-3):-1, (-3,0):1} #---------------data

#cool_perceptron = Perceptron()
#print(cool_perceptron.weighted_sum([24,55]))
#print(cool_perceptron.activation(53))
#print(cool_perceptron.training(small_training_set))
#print(cool_perceptron.weights)

#from sklearn.linear_model import Perceptron
#from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split

# Load the iris dataset
#X, y = load_iris(return_X_y=True)

# Split the data into a training set and a test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Perceptron object
#clf = Perceptron()

# Train the model on the training data
#clf.fit(X_train, y_train)

# Evaluate the model on the test data
#accuracy = clf.score(X_test, y_test)

#print('Test accuracy:', accuracy)

#from sklearn.neural_network import MLPClassifier
#from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split

# Load the iris dataset
#X, y = load_iris(return_X_y=True)

# Split the data into a training set and a test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an MLPClassifier object
#clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, alpha=1e-4,
#                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                    learning_rate_init=.1)

# Train the model on the training data
#clf.fit(X_train, y_train)

# Evaluate the model on the test data
#accuracy = clf.score(X_test, y_test)

#print('Test accuracy:', accuracy)
#------------------------------------------------------------------------------ AI Minimax
#class Game:
#    def __init__(self, board, x, o):
#        self.board = board
#        self.info = {'Game is played by': [x,o]}
#        self.x = x
#        self.o = o
#        self.position = 0

#    def position_play(self, pos, value):
#        for _ in range(len(self.board)):
#            self.board[pos] = value
#        return self.board

#    def available_moves(self):
        #positions = 0
#        for i in self.board:
#           if type(i) == int:
#                self.positions + 1
#            else:
#                positions + 0
#        return self.positions

#    def winner(self):
#        import collections
#        tokens = collections.Counter(self.board)
#        tokens.sort_values()
#        return tokens[len(tokens) - 1]

#    def game_is_over(self):
#        if self.positions > 0:
#            return False
#        else:
#            return True

#----------------------------------------------------------------- Tester
#pin = [1, 2, 3 ,4 ,5,6, 7,8,9]
#start = Game(pin, 'x', 'o')
#start.position_play(4, 'x')
#start.position_play(0, 'o')
#start.position_play(1,'x')
#start.position_play(6, 'o')
#start.position_play(8, 'x')
#tart.position_play(3,'o')

#------------------------------------checking statues
#print(start.available_moves())
#print(start.game_is_over())
#print(start.winner())
#def tester(board):
#    position = 0
#    for i in board:
#        if type(i) == int:
#            position += 1
#        else:
#            position += 0
#    return position
#print(tester(pin))
#print(pin[len(pin) - 1])
#---------------------------------------------------------------------- End tester
#---------------------- Introduction to NLP ( Natural Language Processing )

#import re # for regex
#import nltk # for Natural Language Processing Tool Kit
#from nltk.tokenize import word_tokenize
#from nltk.stem import PorterStemmer
#from nltk.stem import WordNetLemmatizer

#text = "So many squids are jumping out of suitcases these days that you can barely go anywhere without seeing one burst forth from a tightly packed valise. I went to the dentist the other day, and sure enough I saw an angry one jump out of my dentist's bag within minutes of arriving. She hardly even noticed."

#cleaned = re.sub('\W+', ' ', text)
#tokenized = word_tokenize(cleaned) #tokenize

#blunt
#stemmer = PorterStemmer()
#stemmed = [stemmer.stem(token) for token in tokenized]

#lemitize
#lemitizer = WordNetLemmatizer()
#lametize = [lemitizer.lemmatize(stem) for stem in stemmed]
#print('stem')
#print(stemmed)


#-------------------------------------- NTLK

# importing regex and nltk
#import re, nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer
# importing Counter to get word counts for bag of words
#from collections import Counter
# importing a passage from Through the Looking Glass
#from looking_glass import looking_glass_text
# importing part-of-speech function for lemmatization
#from part_of_speech import get_part_of_speech

# Change text to another string:
#text = looking_glass_text

#cleaned = re.sub('\W+', ' ', text).lower()
#tokenized = word_tokenize(cleaned)

#stop_words = stopwords.words('english')
#filtered = [word for word in tokenized if word not in stop_words]

#normalizer = WordNetLemmatizer()
#normalized = [normalizer.lemmatize(token, get_part_of_speech(token)) for token in filtered]
# Comment out the print statement below
#print(normalized)

# Define bag_of_looking_glass_words & print:
#-------------------------------------------------------------- Feauture Engineering
#from sklearn.preprocessing import OrdinalEncoder
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler

#import numpy as np
#import pandas as pd
#from copy import deepcopy
#import collections


#cars = pd.read_csv('cars.csv')
#print(cars.dtypes)

#vehicles = pd.read_csv('vehicles.csv')
#cars = deepcopy(vehicles)
#print(cars.dtypes)
#print(vehicles.head(10))
#print(vehicles.columns)
#print(vehicles.dtypes)
#print(vehicles['condition'].value_counts())
#print(collections.Counter(vehicles['condition']))


#chosen = vehicles['condition'].values.reshape(-1,1)
#encoder = OrdinalEncoder(categories=[['excellent', 'new','like new', 'fair', 'good',
#                                      'salvage', 'nan']])
#vehicles['contain_rating'] = encoder.fit_transform(chosen)
#print(vehicles['contain_rating'].value_counts())

#---------------------------------------------------------------label encoding
#import numpy as np
#import pandas as pd

#women = pd.read_csv('Womens.csv')
#print(women.dtypes)
#print(women.columns)
#print(women.info())
#print(women['Rating'].value_counts())
#print(women['Recommended IND'].value_counts())

#from sklearn.preprocessing import OrdinalEncoder
#change = OrdinalEncoder(categories=[['Loved it', 'Like it', 'Was okay', 'Not great',
#                                     'Hated it']])
#revised = change.fit_transform(reviews['rating'])
#----------------------------------------------------- Feature Selection
#import pandas as pd


#print(df)

#x = df.drop(columns=['exam_score'])
#print(x)
#y = df['exam_score']
#print(y)

#x_num = x.drop(columns=['edu_goal'])

#print(x_num)

#from sklearn.feature_selection import VarianceThreshold
#select = VarianceThreshold(threshold= 0)
#print(select.fit_transform(x_num))
#print(x_num.columns[select.get_support(indices= True)])

#import matplotlib.pyplot as plt
#import seaborn as sns

#corr_matrix = x_num.corr(method='pearson')  # 'pearson' is default

#sns.heatmap(corr_matrix, annot=True, cmap = 'RdBu_r')
#plt.show()
#------------------------------------------------------------------------------- wrapper
#import numpy as np
#import pandas as pd

#df = pd.DataFrame(data={
#   'edu_goal': ['bachelors', 'bachelors', 'bachelors', 'masters', 'masters', 'masters', 'masters', 'phd', 'phd',
#                  'phd'],
#     'hours_study': [1, 2, 3, 3, 3, 4, 3, 4, 5, 5],
#     'hours_TV': [4, 3, 4, 3, 2, 3, 2, 2, 1, 1],
#     'hours_sleep': [10, 10, 8, 8, 6, 6, 8, 8, 10, 10],
#     'height_cm': [155, 151, 160, 160, 156, 150, 164, 151, 158, 152],
#     'grade_level': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
#     'exam_score': [71, 72, 78, 79, 85, 86, 92, 93, 99, 100]
#})

#print(df)

#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegression
#-------------------------------------------------------for linear and logistic models
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_score
#------------------- checking model offeciency
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
#----------- changing feautures
#from sklearn.ensemble import RandomForestClassifier
#domran = RandomForestClassifier(max_depth= 1000)
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.cluster import KMeans
#-------------------------------- unsupervised and supervised
#from sklearn.model_selection import train_test_split
#from sklearn.feature_selection import VarianceThreshold
#print(df['edu_goal'].value_counts())
#encode = LabelEncoder()
#df['edu_goal'] = encode.fit_transform(df['edu_goal'])
#print(df['edu_goal'].value_counts())
#--------------------- changed label
#y= df[['exam_score']]
#y = df['exam_score']
#x = df.iloc[:,:-1]
#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, train_size= 0.8, random_state= 10)
#.....................sequential forward selection
#mxd = LinearRegression()
#mxd.fit(x_train, y_train)
#predicted = mxd.predict(x_test)
#print(mxd.score(x_test,y_test))
#print(precision_score(y_test, predicted))
#print(confusion_matrix(y_test, predicted))
#-------------------------------------------- using sequential to choose a number of features
#from sklearn.feature_selection import SequentialFeatureSelector
#sfs = SequentialFeatureSelector(mxd, k_features = 3, forward = True,
#                                floating = False, scoring = 'accuracy',
#                                cv = 0)
#sfs.fit(x,y)
#print(sfs.subsets_)

#sfs = SequentialFeatureSelector(estimator=domran,
#                                k_features=5,
#                                forward=True,
#                                scoring='accuracy',
#                                cv=5)

# Fit the feature selector to the training data
#sfs.fit(X_train, y_train)

# Get the selected features
#selected_features = sfs.k_feature_idx_
#print(selected_features)

#------------------------ afresh
#import numpy as np
#import pandas as pd

#df = pd.DataFrame(data={
#   'edu_goal': ['bachelors', 'bachelors', 'bachelors', 'masters', 'masters', 'masters', 'masters', 'phd', 'phd',
#                  'phd'],
#     'hours_study': [1, 2, 3, 3, 3, 4, 3, 4, 5, 5],
#     'hours_TV': [4, 3, 4, 3, 2, 3, 2, 2, 1, 1],
#     'hours_sleep': [10, 10, 8, 8, 6, 6, 8, 8, 10, 10],
#     'height_cm': [155, 151, 160, 160, 156, 150, 164, 151, 158, 152],
#     'grade_level': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
#     'exam_score': [71, 72, 78, 79, 85, 86, 92, 93, 99, 100]
#})
#print(df.shape[1])
#from sklearn.preprocessing import LabelEncoder
#labelchange = LabelEncoder()
#df['edu_goal'] = labelchange.fit_transform(df['edu_goal'])

#--chose data
#print(df.head(5))
#x = df.iloc[:,:-1]
#y = df['exam_score']
#print(x)
#print(y)

#-------------------------------------- Changed Labels to Int
#now for Wrapper Sequential forward selection
#from mlxtend.feature_selection import  SequentialFeatureSelector
#from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn.ensemble import RandomForestClassifier

#lr = LogisticRegression(max_iter= 1000)
#Rc = RandomForestClassifier(max_depth=1000, max_leaf_nodes=1000)
#sequential forward selection

#chosen = SequentialFeatureSelector(Rc, k_features= 4, forward= True,
#                                   scoring= 'accuracy', cv=0, floating= False)
#chosen.fit(x,y)
#print(pd.DataFrame(chosen.subsets_).stack())

#import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
#plot_sfs(chosen.get_metric_dict())
#plt.show()
#----------------------------   Homework
#import numpy as np
#import pandas as pd

# data importations
#obesity = pd.read_csv('obesity.csv')
#print(obesity.info())
#print(obesity.head(5))
#print(obesity.columns)
#------------------------------- i intend to perform both ordinal and label then standardize the data
#from sklearn.preprocessing import OrdinalEncoder
#encoder = OrdinalEncoder(categories=[['Sometimes', 'Frequently', 'Always', 'no']])
#to_use = obesity['CAEC'].values.reshape(-1,1)
#obesity['CAEC'] = encoder.fit_transform(to_use)

#------------------- trying to see which makes sense
#from sklearn.preprocessing import LabelEncoder
#labeler = LabelEncoder()
#obesity['CAEC'] = labeler.fit_transform(obesity['CAEC'])
#---------------------- better off using Ordinal if the data aligns
#obesity['Gender'] = obesity['Gender'].map({'Male':0, 'Female':1})
#print(obesity['Gender'].value_counts())
#obesity['family_history_with_overweight'] = obesity['family_history_with_overweight'].map({'yes':0, 'no':1})
#print(obesity['family_history_with_overweight'].value_counts())
#obesity['FAVC'] = obesity['FAVC'].map({'yes':0, 'no':1})
#print(obesity['FAVC'].value_counts())
#print(obesity['CAEC'].value_counts())
#obesity['SMOKE'] = obesity['SMOKE'].map({'no':0, 'yes':1})
#print(obesity['SMOKE'].value_counts())
#obesity['SCC'] = obesity['SCC'].map({'no':0, 'yes':1})
#print(obesity['SCC'].value_counts())

#calc = OrdinalEncoder(categories=[['Sometimes', 'no', 'Frequently', 'Always']])
#clcc = obesity['CALC'].values.reshape(-1,1)
#obesity['CALC'] = calc.fit_transform(clcc)
#print(obesity['CALC'].value_counts())

#mmtran = OrdinalEncoder(categories=[['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike']])
#mtn = obesity['MTRANS'].values.reshape(-1,1)
#obesity['MTRANS'] = mmtran.fit_transform(mtn)
#print(obesity['MTRANS'].value_counts())

#nobsd = OrdinalEncoder(categories=[['Obesity_Type_I', 'Obesity_Type_III', 'Obesity_Type_II', 'Overweight_Level_I', 'Overweight_Level_II', 'Normal_Weight', 'Insufficient_Weight']])
#ndsd = obesity['NObeyesdad'].values.reshape(-1,1)
#obesity['NObeyesdad'] = nobsd.fit_transform(ndsd)
#obesity['NObeyesdad'] = obesity['NObeyesdad'].map({'Obesity_Type_I':1, 'Obesity_Type_III':1, 'Obesity_Type_II':1, 'Overweight_Level_I':0, 'Overweight_Level_II':0, 'Normal_Weight':3, 'Insufficient_Weight':3})
#print(obesity['NObeyesdad'].value_counts())

#---------------------------------------------- time to standardize the data & split it too
#from sklearn.preprocessing import StandardScaler
#standardize = StandardScaler()

#x = obesity.iloc[:,:-1]
#y = obesity.iloc[:,-1]

#x_stand = standardize.fit_transform(x)

#names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
#       'CALC', 'MTRANS']
#new_x = pd.DataFrame(x_stand, columns= names)
#print(new_x.head(5))

#for x in new_x:
#    print(new_x[x].mean())
#    print(new_x[x].std())
#original = standardize.inverse_transform(x_stand)
#print(pd.DataFrame(original, columns=names)) # -------------------- learned this cool thing, that you can tranform the data back

#I will try standerdized
#from sklearn.feature_selection import VarianceThreshold # - wont use this today
#from mlxtend.feature_selection import SequentialFeatureSelector
#from sklearn.ensemble import RandomForestClassifier

#R_forest = RandomForestClassifier(max_leaf_nodes=1000, max_depth= 1000)

#forward_se = SequentialFeatureSelector(R_forest, forward= True, scoring='accuracy', cv = 0,
#                                       floating= True, k_features= 5)
#forward_se.fit_transform(new_x, y)
#print(pd.DataFrame(forward_se.subsets_).stack())
#print(forward_se.k_feature_names_)

#-------------------------------------------- Now trying un-standardized data

#R_forest = RandomForestClassifier(max_leaf_nodes=1000, max_depth= 1000)

#forward_se = SequentialFeatureSelector(R_forest, forward= False, scoring='accuracy', cv = 0,
#                                       floating= True, k_features= 5)
#forward_se.fit_transform(new_x, y)
#print(pd.DataFrame(forward_se.subsets_).stack())
#print(forward_se.k_feature_names_)
#-------------------------------------------------------------------------- Clean

#from sklearn import datasets
#data, y, coefficients = datasets.make_regression(n_samples = 100, n_features = 2, coef = True, random_state = 23)
#x1 = data[:,0]
#x2 = data[:,1]
#print(coefficients)
#print(data)

#from sklearn.linear_model import Lasso
#after learning Sequential forward and backwar i think the rest of the tools are somewhat redundant to be honest.
#----------- Doing homework

#import numpy as np
#import pandas as pd

#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split

#tenis = pd.read_csv('tennis_stats.csv')
#print(tenis.info())
#-------- perform feature engineering

#print(tenis.columns)
#teniss = tenis.drop(columns='Year')
#tenis_ball = teniss.groupby('Player').mean().reset_index().drop(columns = 'Player')
#tenis_ball['Ranking'] = tenis_ball['Ranking'].astype('int')
#print(tenis_ball.head(5))

#------------------------ Begin
#x = round(tenis_ball.iloc[:,:-1], 2)
#print(x)
#y = tenis_ball['Ranking']
#print(y)

#- Performing  Sequwnciatial selection
#from mlxtend.feature_selection import SequentialFeatureSelector
#from sklearn.ensemble import RandomForestClassifier

#rand_forest = RandomForestClassifier(max_leaf_nodes=1000, max_depth=1000)
#mlx = SequentialFeatureSelector(rand_forest, scoring='accuracy', k_features= 7, cv= 0, floating= True, forward= False)

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, train_size= 0.8, random_state= 10)
#mlx.fit_transform(x_train,y_train)

#print(pd.DataFrame(mlx.subsets_).stack())
#--------------------------- Naive Bayes
# to use the naive bays we need to use a classifier called countvectorizier
#import string
#import numpy as np
#import pandas as pd

#use_this = ["Please note that this is a research paper and the data contained within is for academic use only. The information presented has not been verified for accuracy and should not be used for any other purpose. The views and opinions expressed are those of the authors with accordance to the learning throughout the semester and class."]
#print(use_this.strip().replace(".", ","))

#from sklearn.feature_extraction.text import CountVectorizer
#review = "This crib was amazing"

#counter = CountVectorizer()
#counter.fit(use_this)

#find = counter.transform(review)
#print(find)

#print([10] * 1000) # new technique
#from reviews import baby_counter, baby_training, instant_video_counter, instant_video_training, video_game_counter, video_game_training
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB

#review = "this game was violent"

#baby_review_counts = baby_counter.transform([review])
#instant_video_review_counts = instant_video_counter.transform([review])
#video_game_review_counts = video_game_counter.transform([review])

#baby_classifier = MultinomialNB()
#instant_video_classifier = MultinomialNB()
#video_game_classifier = MultinomialNB()

#baby_labels = [0] * 1000 + [1] * 1000
#instant_video_labels = [0] * 1000 + [1] * 1000
#video_game_labels = [0] * 1000 + [1] * 1000


#baby_classifier.fit(baby_training, baby_labels)
#instant_video_classifier.fit(instant_video_training, instant_video_labels)
#video_game_classifier.fit(video_game_training, video_game_labels)

#print("Baby training set: " +str(baby_classifier.predict_proba(baby_review_counts)))
#print("Amazon Instant Video training set: " + str(instant_video_classifier.predict_proba(instant_video_review_counts)))
#print("Video Games training set: " + str(video_game_classifier.predict_proba(video_game_review_counts)))


#------------------------ Naive Bayes that i wont use Ever.
#from sklearn import datasets
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.feature_extraction.text import CountVectorizer

#train_emails = datasets.fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'train')
#test_emails = datasets.fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'test')
#print(train_emails.target)
#print(email_data.DESCR)

#counter = CountVectorizer()
#counter.fit(train_emails + test_emails)
#train_counts = counter.transform(train_emails.data)
#test_counts = counter.transform(test_emails.data)

#classifier = MultinomialNB()
#classifier.fit(train_counts, train_emails.targets)
#print(classifier.score(test_counts, test_emails.targets))

#-------------------------------------------------------------- Support Vector Machines
#from sklearn.datasets import make_circles
#from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split

#Makes concentric circles
#points, labels = make_circles(n_samples=300, factor=.2, noise=.05, random_state = 1)

#Makes training set and validation set.
#training_data, validation_data, training_labels, validation_labels = train_test_split(points, labels, train_size = 0.8, test_size = 0.2, random_state = 100)

#classifier = SVC(kernel = "linear", random_state = 1)
#classifier.fit(training_data, training_labels)
#print(classifier.score(validation_data, validation_labels))
#------------------------------------------------------------------------------------------------ Hyparameters
#import numpy as np
#import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.datasets import load_breast_cancer

#cancer = load_breast_cancer()

#x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size= 0.8, test_size= 0.2, random_state= 10)

#model = LogisticRegression(solver= 'liblinear',max_iter= 1000)
#formualte parameters
#parameters = {'penalty': ['l1', 'l2'], 'C': [len(range(0,2000))]}

#from sklearn.model_selection import GridSearchCV
#clf = GridSearchCV(model, parameters)
#clf.fit(x_train, y_train)

#print(clf.best_estimator_)

#from sklearn.model_selection import RandomizedSearchCV

#use different parameters
#from scipy.stats import uniform
#use = {'penalty':['l1', 'l2'], 'C': uniform(loc = 0, scale = 100)}
#solver = RandomizedSearchCV(model, use)
#solver.fit(x_train,y_train)
#print(solver.best_estimator_)
#print(solver.best_params_)
#--------------------------------------------- Homework


#import numpy as np
#import pandas as pd
#from scipy.stats import uniform

#---------- for number and file work

#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import load_breast_cancer
#from sklearn.model_selection import train_test_split

#cancer = load_breast_cancer()
#x = cancer.data
#y = cancer.target
#print(x)
#print(y)

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, train_size= 0.8, random_state= 10)


#model = RandomForestClassifier()
#paramater = {'n_estimators': [1,10,100,100], 'criterion':['gini', 'entropy'],
#             'max_depth':[1,10,100,100], 'min_samples_split':[1,10,100,100],
#             'min_samples_leaf':[1,10,100,100], 'max_features':[1,10,100,100],
#             'random_state':[1,10,100,100]}
#Hyperamater = RandomizedSearchCV(model, paramater)
#Hyperamater.fit(x_train, y_train)
#print(Hyperamater.best_estimator_)
#------------------------------------------Ensemble
#import numpy as np
#import pandas as pd

##df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
#d#f['accep'] = df['accep'].map({'unacc':0, 'acc':1, 'good':1, 'vgood':1})

#df['persons'] = df['persons'].map({'2':2, '4':4, 'more':5})
#df['doors'] = df['doors'].map({'2':2, '3':3, '4':4, '5more':5})
#print(df['persons'].value_counts())
#print(df['doors'].value_counts())



#df = df.astype({'doors':int, 'persons':int})
#df['accep'] = ~(df['accep']=='unacc')
#x = df.iloc[:,:-1]
#from sklearn.preprocessing import LabelEncoder
#lab = LabelEncoder()
#df['buying'] = lab.fit_transform(df['buying'])
#df['maint'] = lab.fit_transform(df['maint'])
#df['lug_boot'] = lab.fit_transform(df['lug_boot'])
#df['safety'] = lab.fit_transform(df['safety'])
#print(df.head(5))

#x = pd.get_dummies(df.iloc[:,0:6], drop_first=True)
#x = df.iloc[:,:-1]
#y = df['accep']
#print(x.head(15))
#print(y.head(5))

#x['doors'] = x['doors'].astype(int)
#x['persons'] = x['persons'].astype(int)

#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
#print(df['lug_boot'].value_counts())
#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, train_size= 0.7, random_state= 10)

#from sklearn.linear_model import LogisticRegression, LinearRegression
#from sklearn.metrics import accuracy_score, consensus_score, confusion_matrix
#logi = LinearRegression()
#logi.fit(x_train, y_train)
#pred = logi.predict(x_test)
#print(confusion_matrix(y_test, pred))
#----------------- sOME Error with the data
# nothing today because i am moving prety fast. All thanks to Christ Jesus name.
#cleared for tokenization

#import nltk
#from nltk.tokenize import word_tokenize, sent_tokenize
#import re

#nltk.download('punkt')
#nltk.download('stopwords')
#ecg_text = 'An electrocardiogram is used to record the electrical conduction through a person\'s heart. The readings can be used to diagnose cardiac arrhythmias.'
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words('english'))

#tokenize_by_word = word_tokenize(ecg_text.lower())
#tokenize_by_sent = sent_tokenize(ecg_text.lower())
#print(tokenize_by_sent)
#print(tokenize_by_word)
#print('#-------------------------')

#new_word = [word for word in tokenize_by_word if word not in stop_words]
#print(new_word)


#from nltk.stem import PorterStemmer
#stemmer = PorterStemmer()

#stemmed = [stemmer.stem(word) for word in new_word]
#print(stemmed)

#nltk.download('wordnet')
#from nltk.stem import WordNetLemmatizer
#lametize = WordNetLemmatizer()

#lammetized = [lametize.lemmatize(word) for word in stemmed]
#print(lammetized)
#--------------------------------- Parsing

#import re
#regular_expression = re.compile('.{7}')
#result = regular_expression.match('Dorothy')
#print(result.group(0))

#result_1 = re.match('.{7}', 'Dorothy').group(0)
#print(result_1)

#text = "Everything is green here, while in the country of the Munchkins blue was the favorite color. But the people do not seem to be as friendly as the Munchkins, and I'm afraid we shall be unable to find a place to pass the night."
#print(re.search('\w{10}', text).group(0))
#print(re.findall('\w{10}', text).group(0))

#--------------------- Part of speech taging

#from nltk import pos_tag
#from word_tokenized_oz import word_tokenized_oz
#from nltk.tokenize import sent_tokenize
#from nltk.tokenize import word_tokenize

#text_token = word_tokenize(text, 'english')
#print(text_token)

#from nltk.stem import PorterStemmer, LancasterStemmer
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer

#use = set(stopwords.words('english'))
#new_text = [word for word in text_token if word not in use]

#stemming = PorterStemmer()
#lamment = WordNetLemmatizer()

#def likethis(x):
#    import numpy
#    import pandas as pd
#    import collections
#    from nltk.stem import PorterStemmer, LancasterStemmer
#    from nltk.stem import WordNetLemmatizer
#    stemming = PorterStemmer()
#    lamment = WordNetLemmatizer()
#    text = []
#    final = []
#    for word in x:
#        one = stemming.stem(word)
#        text.append(one)
        #return text
#    for s in text:
#        two = lamment.lemmatize(s)
#        final.append(two)
#    return pd.DataFrame(collections.Counter(final)).sort_values()

#print(likethis(new_text))

#for x in word_tokenized_oz:
#    to = pos_tag(x)
#    pos_tagged_oz.append(to)
#    return pos_tagged_oz
#-------------------------------------------- Regez
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
#text = "Everything is green here, while in the country of the Munchkins blue was the favorite color. But the people do not seem to be as friendly as the Munchkins, and I'm afraid we shall be unable to find a place to pass the night."

#bow_vectorizer = CountVectorizer()
#new = bow_vectorizer.fit_transform(text)

#naive = MultinomialNB()
#naive.fit(new)
#pred = naive.predict("hello there again my friend from the country")
#print(pred)
#------------------------------------------------------------------------- TF - IDF
#import numpy as np
#import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#import preprocessing
#import re

#poem = '''
#Success is counted sweetest
#By those who ne'er succeed.
#To comprehend a nectar
#Requires sorest need.

#Not one of all the purple host
#Who took the flag to-day
#Can tell the definition,
#So clear, of victory,

#As he, defeated, dying,
#On whose forbidden ear
#The distant strains of triumph
#Break, agonized and clear!'''
#----------------------------------------------------------------------------------------------------------------------  Pyspark big data
#rom pyspark.sql import SparkSession
#spark = SparkSession.builder.getOrCreate() # is how you initiate spark
#print(spark)
#student_data = [("Chris",1523,0.72,"CA"),
#                ("Jake", 1555,0.83,"NY"),
#                ("Cody", 1439,0.92,"CA"),
#                ("Lisa",1442,0.81,"FL"),
#                ("Daniel",1600,0.88,"TX"),
#                ("Kelvin",1382,0.99,"FL"),
#                ("Nancy",1442,0.74,"TX"),
#                ("Pavel",1599,0.82,"NY"),
#                ("Josh",1482,0.78,"CA"),
#                ("Cynthia",1582,0.94,"CA")]
#rdd_par = spark.sparkContext.parallelize(#data) ---- is how you initiate how it works in paralell
#spark.stop() #to quit spark

#student_rdd = spark.sparkContext.parallelize(student_data,5)
#student_arthmetric = student_rdd.map(lambda x: (x[0], x[1], int(x[2] * 100), x[3]))
#filterd_student = student_arthmetric.filter(lambda x: x[2] > 80)
#print(filterd_student.collect())
#rint(filterd_student.take(4))
#print(filterd_student.reduce(x[2]-200))
#student_transformed = student_rdd.map(lambda x: x * 100)
#print(student_rdd.getNumPartitions())
#print(student_transformed.collect())
#------------------------------------------------------- Begin using Spark SQL wiht Data



#from pyspark.sql import SparkSession
#spark = SparkSession.builder.config('spark.app.name', 'Learning_Sql_pspark')\
#    .getOrCreate()

#now to create data
#sample_data = spark.sparkContext.parallelize([
#    ["en", "Statue_of_Liberty", "2022-01-01", 263],
#    ["en", "Replicas_of_the_Statue_of_Liberty", "2022-01-01", 11],
#    ["en", "Statue_of_Lucille_Ball" ,"2022-01-01", 6],
#    ["en", "Statue_of_Liberty_National_Monument", "2022-01-01", 4],
#    ["en", "Statue_of_Liberty_play"  ,"2022-01-01", 3],
#])
#-------------------------- the data is in dd so i need to convert it to df

#df_data = sample_data.toDF(['Language', 'Statue', 'date', 'Amount'])
#print(df_data.show(truncate= False))

# reads files to rdd official data df_data.rdd
#data = spark.read.option('header', True)\
#    .option('delimiter', ' ')\
#    .option('inferSchema', True)\
#    .csv('Womens.csv')

#print(data.show(5,truncate= False))

#--------------------------------------------------------------- control flow testing
#def gcd(m, n):
#    r = 0
#    if n > m:
#        r = m
#        m = n
#        n = r
#    r = m % n
#    while r != 0:
#        m = n
#        n = r
#        r = m % n
#print(gcd(5,2))
# def search(c, alphabet):
#    first = 0
#    last = len(alphabet)
#    while first <= last:
#        mid = first + ((last - first)/2)
#        if alphabet[mid] < c:
#            first = mid + 1
#        elif alphabet[mid] > c:
#            last = mid - 1
#        else:
#            return mid
#    return -1
#man = "pac"
#rint(search(2, "pac"))
#------------------------------------------------ class work ( redundant )
#print(True | False)
#print(True | True)
#print(False | False)
#print(False | True)
#print("------------------------------")
#print(False & True)
#print(False & False)
#print(True & False)
#print(True & True)
#print("--------------------------")
#print(False == 0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------








