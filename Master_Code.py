print('\f')
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import csr_matrix
import codecs
import pandas as pd
import time
import random
from pylab import* 
from scipy import*


with open('D:/HackOff/Trial.json') as data_file:    
    data = json.load(data_file)

#Converting JSONS to a countsMatrix
#creating a dictionary having ingradients as values for each cuisine as keys
def create_dict_cuisine_ingred(json):
    dictCuisineIngred = {}
    cuisines = []
    ingredients = []
    
    for i in range(len(json)):
        
        # just changing the name of one of the cuisines so
        # it is more readable in the final visualization
        cuisine = json[i]['cuisine']
        if cuisine == 'indian':
            cuisine = 'Indian'

        ingredientsPerCuisine = json[i]['ingredients']
        
        if cuisine not in dictCuisineIngred.keys():
            cuisines.append(cuisine)
            dictCuisineIngred[cuisine] = ingredientsPerCuisine
            
        else: 
            currentList = dictCuisineIngred[cuisine]
            currentList.extend(ingredientsPerCuisine)
            dictCuisineIngred[cuisine] = currentList
                 
        ingredients.extend(ingredientsPerCuisine)
         
    ingredients = list(set(ingredients)) # unique list of ALL ingredients
    numUniqueIngredients = len(ingredients)
    numCuisines = len(cuisines)
    
    return dictCuisineIngred, numCuisines, numUniqueIngredients, cuisines, ingredients

#Using the dictionary above in creating CountMatrix
def create_term_count_matrix(dictCuisineIngred, numCuisines, numIngred, cuisines, ingredients):
    termCountMatrix = np.zeros((numCuisines,numIngred))
    i = 0
    
    for cuisine in cuisines:
        ingredientsPerCuisine = dictCuisineIngred[cuisine]

        for ingredient in ingredientsPerCuisine:
            j = ingredients.index(ingredient) #in order to know which column to put the term count in, we will ago according to the terms' order in the ingredients array
            termCountMatrix[i,j] += 1

        i += 1

    return termCountMatrix


dictCuisineIngred, numCuisines, numIngred, cuisines, ingredients = create_dict_cuisine_ingred(data)
countsMatrix = create_term_count_matrix(dictCuisineIngred, numCuisines, numIngred, cuisines, ingredients)


#Generating TF-idf Matrix from the countMatrix above
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
#get_ipython().magic(u'matplotlib inline')


def tf_idf_from_count_matrix(countsMatrix):
    
    countsMatrix = sparse.csr_matrix(countsMatrix)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(countsMatrix) # normalizes vectors to mean 0 and std 1 and computes tf-idf
    tfidf.toarray() 
    return tfidf.toarray()
    
tfIdf_Matrix = tf_idf_from_count_matrix(countsMatrix)


# running PCA to reduce to 2 dimensions
###########################################
pca = PCA(n_components=2)
# print(pca.explained_variance_ratio_)   
reduced_data = pca.fit_transform(tfIdf_Matrix)

# converting to pandas dataframe for convenience:
###########################################
pca2dataFrame = pd.DataFrame(reduced_data)
pca2dataFrame.columns = ['PC1', 'PC2']


#Using Kmeans for Clustering and creating Five CLusters
from sklearn.cluster import KMeans

def kmeans_cultures(numOfClusters):
    
    kmeans = KMeans(init='k-means++', n_clusters=numOfClusters, n_init=10)
    kmeans.fit(reduced_data)
    return kmeans.predict(reduced_data)

labels = kmeans_cultures(5)


#Caluculating the effect on the CLuster for each Cuisine in the cluster based on the Jaccard Similarity Betweent the Ingredients in the Cuisine  stored in a Vector and Union of all Ingredients in the cluster stored in a Vector.
i = 0 
j = 0 

effect_on_cluster = [0 for cuisine in cuisines]

for cuisineA in cuisines:  

    A_intersection = 0
    numInClusterBesidesA = 0
    setA = set(dictCuisineIngred[cuisineA])
    setB_forA = []
    j = 0
    
    for cuisineB in cuisines:
        if cuisineB != cuisineA: # if it is A itself - we obviously wouldn't want this (will be exactly 1)
            if labels[j] == labels[i]: #determines if then they are both in the same cluster
                setB_forA.extend(set(dictCuisineIngred[cuisineB]))
                numInClusterBesidesA += 1
        j += 1
    
    A_intersection = len(set(setA & set(setB_forA))) / float(len(set(setA.union(setB_forA))))
    effect_on_cluster[i] = A_intersection
       
    i += 1

#VISUALIZING THE CLUSTERS using the pylab, scipy and matplotlib packages    
from pylab import* 
from scipy import *
import matplotlib.pyplot as plt

rdata = reduced_data
i=0
figureRatios = (15,20)
x = []
y = []
color = []
area = []

#creating a color palette:
colorPalette = ['#009600','#2980b9', '#ff6300','#2c3e50', '#660033'] 
#colorPalette = ['#009600','#2c3e50', '#660033']
# green,blue, orange, grey, purple

plt.figure(1, figsize=figureRatios)

for data in rdata:
    x.append(data[0]) 
    y.append(data[1])  
    color.append(colorPalette[labels[i]]) 
    area.append(effect_on_cluster[i]*27000) # magnifying the bubble's sizes (all by the same unit)
    # plotting the name of the cuisine:
    #x=""
    #x=labels[i]+" "+cusines[i]
    text(data[0], data[1],cuisines[i], size=10.6,horizontalalignment='center', fontweight = 'bold', color='w')
    i += 1

plt.scatter(x, y, c=color, s=area, linewidths=2, edgecolor='w', alpha=0.80) 

plt.axis([-0.45,0.65,-0.55,0.55])
#plt.axes().set_aspect(0.8, 'box')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.axis('off') # removing the PC axes

plt.show()


start_time = time.time()
meal_id,cuisine,ingredients,ing,main_set,train_set,test_set =[],[],[],[],[],[],[]
predicted_cuisine = ''

#creates different lists needed in the program
def lists_creater(filename):
    with codecs.open( filename,encoding = 'utf-8') as f:
        data = json.load(f)
        
    for i in range(0,len(data)):
        meal_id.append(data[i]["id"])
        cuisine.append(data[i]["cuisine"])
        ingredients.append(data[i]["ingredients"])
        
    for i in ingredients:
        temp =u''
        for f in range(len(i)):
            temp = temp+u" "+i[f]
        ing.append(temp.encode('utf-8'))
    #print(ing)
        
    return meal_id
    return cuisine
    return ingredients
    return ing

#vectorizes the document and converts them into features   
def ing_vectorizer(exis_ing,user_ing):
    exis_ing.append(user_ing)
    vectorizer = TfidfVectorizer(use_idf = True, stop_words = 'english',max_features = 4000)
    #X=np.array(exis_ing)
    #X=X.reshape(1,-1)
    ing_vect = vectorizer.fit_transform(exis_ing)
    return (ing_vect.todense())

#creates train set(existing data from json file) and test set (user entered ingredient)
def set_creator(main_set):
    train_set = main_set[:len(main_set)-1]
    test_set = main_set[len(main_set)-1]
    return (train_set,test_set)

#trains the model with the json data for future prediction
def KNN_trainer(train_set,cuisine,n):  
    n = int(n)
    #train_set=train_set(1,-1)
    #X=np.array(train_set)
    #X=X.shape
    close_n = KNeighborsClassifier(n_neighbors=n)
    return close_n.fit(train_set,cuisine)

#user entered ingredient is given to the model to predict cuisine and return n nearest neighbors
def KNN_predictor(test_set,close_n,no_of_neigh):
    no_of_neigh = int(no_of_neigh)
    print ("")
    #test_set=test_set(1,-1)
    #list of probabilities of Top N Closest Foods
    predicted_cuisine = close_n.predict_proba(test_set)[0]
    #Top most matched Cuisine among Top N CLosest Foods
    predicted_single_cuisine = close_n.predict(test_set)
    #List of predicted Cuisines of Top N Closest Foods
    predicted_class = close_n.classes_
    print ("The model predicts that the ingredients resembles %s" %(predicted_single_cuisine[0]))
    print ("")
    for i in range(len(predicted_cuisine)):
        if not(predicted_cuisine[i] == 0.0):
            print ("The ingredients resemble %s with %f percentage" %(predicted_class[i],predicted_cuisine[i]*100))
    
    print ("")
    print ("The %d closest meals are listed below : " % no_of_neigh)
    match_perc,match_id = close_n.kneighbors(test_set)
    for i in range(len(match_id[0])):
        print (meal_id[match_id[0][i]])
        #print (ingredients[match_id[0][i]])
    print ("")
    
    print("--- It took %s seconds ---" %(time.time() - start_time))
    print ("")
    print ("")
    return predicted_single_cuisine

#handles the sequential execution of the program
def seq_exec():
    user_ing = input("Enter the ingredients that you want to compare : ")
    main_set = ing_vectorizer(ing,user_ing)
    train_set,test_set = set_creator(main_set)
    no_of_neigh = input("Enter the number of closest items you want to find : ")
    close_n = KNN_trainer(train_set,cuisine,no_of_neigh)
    print ("Model has been successfully trained..")
    print ("Trying to predict the cuisine and n closest meal items...")
    KNN_predictor(test_set,close_n,no_of_neigh)
    ing.pop()
    try:
        nextStep = int(input("Enter 1 if you want to search again or 2 if you want to quit.."))
        if not(nextStep == 1 or nextStep == 2):
            raise ValueError()
        elif (nextStep == 1):
            seq_exec()
        elif (nextStep == 2):
            quit()
    except ValueError:
        print ("Invalid Option. Enter correctly")
        seq_exec()
        
if __name__ == '__main__':
    print ("Reading all the data files and creating lists....")
    lists_creater(filename = "D:/HackOff/trial.json")
    seq_exec()
    


pickle.dump(KNN_trainer,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
