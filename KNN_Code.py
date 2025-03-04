import numpy as np                    #for working with the data
import pandas as pd                   #for importing dataframes
from random import randint            #for generating random integers for train/test-split
import matplotlib.pyplot as plt       #for ploting the data
import math                           #for usign pi & the sqrt-function

#############################################################################################
######################## Data: Import & Manipulation ########################################
#############################################################################################

names = ["Id","Cl.thickness","Cell.size","Cell.shape",
         "Marg.adhesion","Epith.c.size", "Bare.nuclei",
         "Bl.cromatin","Normal.nucleoli","Mitoses","Class"]

#Import Data:
data = pd.read_csv("BreastCancer.csv",header=None, names=names)
#Deleting rows with missing values:
data = data.dropna()
#print(data.head())

#converting data into numpy-array:
data = data.to_numpy()
#deleting names row:
data = np.delete(data, 0, axis=0)
#deleting Id column as it doesn't contain relevant information:
data = np.delete(data, 0, axis=1)        
#column that contains labels:                                         
class_col = 9

##############################################################################################
####################### KNN-FUNCTIONS ########################################################
##############################################################################################

def standardize(data):
    variance = []
    mean = []
    
    #calculating variance and arithmetic mean for each column:
    for i in range(len(data[0])):
        column = data[:,i]                                            
        column = column.astype(np.float)
        data[:,i]  = column
        
        var = np.var(column)
        variance.append(var)
        
        me = np.mean(column)                                          
        mean.append(me)
    
    #standardization
    for j in range(len(data[:,1])):                                   
        for i in range(len(data[1])-1):
            data[j,i] = ((data[j,i] - mean[i])
                         /math.sqrt(variance[i]))       
    
    #returning standardized data
    return(data)

def manhattan_dist(x,y):
    #Converting into numpy-array with floats:
    x = np.array(x)
    x = x.astype(np.float)
    y = np.array(y)
    y = y.astype(np.float)
    #Calculating manhattan distance:
    dist = sum(abs(x-y))
    return(dist)

def eukl_dist(x,y):
    #Converting into numpy-array with floats:
    x = np.array(x)
    x = x.astype(np.float)
    y = np.array(y)
    y = y.astype(np.float)
    #Calculating euclidian distance:
    dist = math.sqrt(sum((x-y)*(x-y)))
    return(dist)

def gauss_kernel(distance):
    #calculating value for gaussian_kernel
    kernel = (1/(math.sqrt(2*(math.pi))))*(math.exp(-((distance*distance)/2)))
    return(kernel)

def triangular_kernel(distance):
    #calculating value of triangular kernel
    if abs(distance) > 1:
        kernel = 0
    else:
        kernel = 1 - abs(distance)
    return(kernel)

def nearest_neighbours(x,y,k,dist):
    #x:a single testdate, y: training data
    #dist=0 for euklidian distance
    #dist=1 for manhattan distance
    testdata = x
    data = y
    compl_testdata = testdata
    compl_data = data
    
    #deleting class column for calculation of distance
    testdata = np.delete(testdata, class_col)         
    data = np.delete(data, class_col, axis=1)
    
    #adding new column with distances of the neighbours
    list = ["Distance to datapoint"]                  
    list = np.matrix(list)
    
    #calculating distance for each of the points in data:
    
    #if manhattan distance is chosen:
    if dist == 1:
        for i in range(len(data)):                        
            distance = manhattan_dist(testdata, data[i])
            list = np.append(list,[[distance]],axis = 0)
    
    #if euclidian distance is chosen:
    elif dist == 0:
        for i in range(len(data)):                        
            distance = eukl_dist(testdata, data[i])
            list = np.append(list,[[distance]],axis = 0)
    
    #deleting names row in list:
    list = np.delete(list, 0, axis = 0)               
    
    #adding the distances-column to the data
    compl_data = np.append(compl_data, list, axis=1)  
    
    #converting distances into float so they are sortable
    compl_data[:,class_col+1] = (compl_data[:,class_col+1]
                                 .astype(np.float))
    
    #converting class_col into float
    compl_data[:,class_col] = (compl_data[:,class_col]
                                 .astype(np.float))
        
    #starting a list of k-nearest-neighbours
    knn = []                                          
    
    for i in range(k):
        #searching for minimal distance:                                              
        minElement = np.amin(compl_data[:,class_col+1])

        #searching for index where minimal distance stands:                                              
        minElementIndex = np.where(compl_data[:,class_col+1]
                                   == np.amin(compl_data
                                              [:,class_col+1]))
        
        #Only taking the one with smaller index
        minElementIndex = minElementIndex[0][0]
        
        #adding this row to the knn-list:
        knn.append(compl_data[minElementIndex])
        
        #deleting minimum from old list, searching for next one:
        compl_data = np.delete(compl_data,
                               minElementIndex, axis = 0)
    
    #return list of k-nearest-neighbours                                                  
    return(knn)


#classify a single observation:
def classify(testdata,data,k,weighted):    
    #using the k-nearest neighbours:
    knn = nearest_neighbours(testdata,data,k,0)
    
    #path if WKNN is chosen
    if weighted == True:
        #adding new column with weights of the neighbours:
        list_2 = ["Weight"]
        list_2 = np.matrix(list_2)
    
        #Calculating sum of Kernels
        kernel_sum = 0
    
        #parameter for kernel standardization
        gamma = 1
    
        for i in range(len(knn)):
            distance = knn[i][:,class_col+1]
            kernel_sum = kernel_sum + gauss_kernel(
                distance.item(0,0)/gamma)
    
        #Calculating the weights
        for i in range(len(knn)):
            distance = knn[i][:,class_col+1]
            weight = gauss_kernel(
                (distance.item(0,0)/gamma))/kernel_sum
            list_2 = np.append(list_2,[[weight]],axis = 0)
    
        #deleting names row in list_2:
        list_2 = np.delete(list_2, 0, axis = 0)
    
        #converting weights into float:
        list_2 = list_2.astype(np.float)

        #adding the weights to the list
        for i in range(len(knn)):
            knn [i] = np.append(knn[i], list_2[i], axis=1)
    
        #calculating the sum of classes * weights
        class_estimator = 0
        weighted_sum = 0
        for i in range(len(knn)):
            weighted_sum = weighted_sum + (
                knn[i][:,class_col])*(knn[i][:,class_col+2])
        print(weighted_sum)
        
        #making the prediction
        if weighted_sum >= 0.5:
            class_estimator = 1
    
    #path for normal KNN
    else:   
        #array with the classes of the k-nearest neighbours:
        knn_labels = []
        
        for i in range(len(knn)):
            label = knn[i][0,class_col]                   
            knn_labels.append(label)
        
        #counting which label appears the most
        class_estimator = max(
            set(knn_labels), key = knn_labels.count)
        
                                                      
    #return the class prediction:
    return(class_estimator)

#test/train-split: value in [0,1]:
def test_train_split(data, value):                                                                     
    testdata_amount = round(len(data)*value)
    testdata = []
    
    for i in range(testdata_amount):
        r = randint(0,len(data)-1-i)
        testdata.append(data[r])
        data = np.delete(data, r, axis=0)
    
    #returning splitted data
    return(data, testdata)                            

#testing missclassificationrate for chosen k (not relevant, for testing purpose)
def miss_rate(data,k):                                
    data, testdata = test_train_split(data, 0.2)
    n_correct = 0
    n_wrong = 0
    
    for i in range(len(testdata)):
        estimated_label = classify(testdata[i],data,k)
        true_label = testdata[i][class_col]
        if estimated_label == true_label:
            n_correct = n_correct + 1
        else:
            n_wrong = n_wrong + 1
    
    #missclassification_rate in percentage
    missclassification_rate = (n_wrong/(len(testdata)))*100    
    
    return(missclassification_rate)


def false_estimations(data,k,dist,weighted):
    #array that will contain abs. false classifications:
    miss = [0 for i in range(k)]
    
    #train/test - split:
    data, testdata = test_train_split(data, 0.2)   
    
    #path for WKNN
    if weighted == True:
        for date in testdata:
            
            #extracting true label
            true_label = date[class_col]
            #creating knn-array for all dates
            date_knn = nearest_neighbours(date,data,k,dist)
            
            for j in range(1,k+1,1):
                sub_date_knn = date_knn[:j]
                #adding new column with weights of the neighbours:
                list_2 = ["Weight"]
                list_2 = np.matrix(list_2)
    
                #Calculating sum of Kernels
                kernel_sum = 0
    
                #parameter for gaussian kernel
                gamma = 1
                
                #parameter for triangular kernel
                max_distance = sub_date_knn[j-1][:,class_col+1]
                max_dist = max_distance.item(0,0)
                
                #Calculating sum of the kernel-values
                for i in range(len(sub_date_knn)):
                    distance = sub_date_knn[i][:,class_col+1]
                    if max_dist != 0:
                        kernel_sum = (
                            kernel_sum + triangular_kernel(
                            distance.item(0,0)/max_dist))
                    elif max_dist == 0:
                        kernel_sum = (
                            kernel_sum + triangular_kernel(0))
    
                #Calculating the weights
                for i in range(len(sub_date_knn)):
                    distance = sub_date_knn[i][:,class_col+1]
                    
                    #for the rare case that all neighbours
                    #share the same distance
                    if kernel_sum != 0:
                        if max_dist != 0:
                            weight = (triangular_kernel(
                                (distance.item(0,0)
                                 /max_dist))/kernel_sum)
                        elif max_dist == 0:
                            weight = (
                                triangular_kernel(0)/kernel_sum)
                    else:
                        weight = 1
                    
                    #adding the weights to weights-column
                    list_2 = np.append(
                        list_2,[[weight]],axis = 0)
    
                #deleting names row in list_2:
                list_2 = np.delete(list_2, 0, axis = 0)
    
                #converting weights into float:
                list_2 = list_2.astype(np.float)

                #adding the weights to the list
                for i in range(len(sub_date_knn)):
                    sub_date_knn[i] = np.append(
                        sub_date_knn[i], list_2[i], axis=1)
                
                #begin class estimation
                class_est = 0
                weighted_sum = 0
                for i in range(len(sub_date_knn)):
                    weighted_sum = weighted_sum + (
                        sub_date_knn[i][:,class_col])*(
                            sub_date_knn[i][:,class_col+2])
                
                #Calculating sum of weights * classes
                if weighted_sum >= 0.5:
                    class_est = 1
                
                #comparing estimated label with true label:
                if class_est != true_label:
                    miss[j-1] = miss[j-1] + 1
    
    #path for KNN
    else:
        #right/wrong classification for each date for rising k:
        for date in testdata:
            #calculating the k-nearest-neighbours:
            date_knn = nearest_neighbours(date,data,k,dist)
            
            #knn-labels only:
            date_knn_labels = []
            for i in range(len(date_knn)):
                label = date_knn[i][0,class_col]                   
                date_knn_labels.append(label)
           
            #Creating 1-k nearest neighbours:
            for j in range(1,k+1,1):
                
                #Creating subgroups of knn-labels
                sub_date_knn_labels = date_knn_labels[:j]

                #counting which label appears the most:
                class_est = max(set(sub_date_knn_labels),
                                key = sub_date_knn_labels.count)
                true_label = date[class_col]
                
                #comparing estimated label with true label:
                if class_est != true_label:
                    miss[j-1] = miss[j-1] + 1
    
    #returning total false classifications for each k:
    return(miss)


#classification accuracy for different k after n repetitions:
def k_accuracy(data,k,repetitions,dist,weighted):
    #train/test-split with 20% testdata for knowing the
    #length of testsample
    data2, testdata = test_train_split(data, 0.2)
    
    #starting a list with total false classification for diff. k:
    total_miss = [0 for i in range(k)]
    
    #repeating multiple times:
    for n in range(repetitions):
        miss = false_estimations(data,k,dist,weighted)
        for i in range(k):
            total_miss[i] = total_miss[i] + miss[i]
        
        #printing the progress
        print(n+1,"of",repetitions,"repetitions completed...")
    
    #calculating rounded total mean accuracy in percent:
    total_mean_accuracy = [0 for i in range(k)]
    for i in range(k):
        total_mean_accuracy[i] = round((total_miss[i] /
                                        (len(testdata)
                                         *repetitions))*100,2)
    
    #searching for k with minimal error:
    min_k = min(total_mean_accuracy)
    k_index = [i for i in range(len(total_mean_accuracy))
               if total_mean_accuracy[i] == min_k]
    
    #adding +1 as Python starts counting with '0':
    true_k_index = [x+1 for x in k_index]
    
    #printing results:
    print("k =",true_k_index,
          " seems to be best for classification results after",
          repetitions,"tries\n",
          "Average classification error:",
          min(total_mean_accuracy),"%")
    
    #returning list of classification errors for rising k:
    return(total_mean_accuracy)

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

#TESTING-STUFF:
#Choosing 25th date for testing:
#testdata = data[24]
#deleting it from original data:
#data = np.delete(data, 24, axis=0)
#how many neighbours should be picked:
#k_value = 10
#Comparsion between true and estimated class:
#print("True class:",testdata[class_col],
#      "| Estimator for class:",classify(
#          testdata, data, k_value, False))
#print(nearest_neighbours(testdata,data,50,0))

#print(standardize(data))

##############################################################################################################################

#Change max k and repetitions for testing the model:
k_value = 300
rep = 10
 
#plotting knn with euclidian distance:
plt.plot([i+1 for i in range(k_value)],k_accuracy(standardize(data),k_value,rep,0,False),"b")
 
#plotting (w)knn in the same plot with diffent distance for example:
plt.plot([i+1 for i in range(k_value)],k_accuracy(data,k_value,rep,0,False),"yellow")

#plot design:
plt.ylabel('Fehlklassifizierung(%)')
plt.xlabel("Anzahl der nächsten Nachbarn")
plt.ylim(bottom=0)

#open new window with plot:
plt.show()
