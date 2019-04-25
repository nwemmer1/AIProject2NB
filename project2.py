#
# Nathan Wemmer
# nww8@zips.uakron.edu
#   3025415
#   Project 2 - Python, SciKitLearn

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import pickle
import collections
import io
import pydotplus
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import classification_report

#==================================================================
#This is the prompt to keep the user in a loop until they type in a valid character
def menuOptionNumber(prompt):
    while True:
        try:
            choice = float(input(prompt))
            break
        except ValueError:
            pass
    return choice
#==================================================================
#This shows the menu and makes sure they type in a valid menu
def showMenu(menuOptions):
    for i in range(len(menuOptions)):
        print("{:d}. {:s}".format(i+1, menuOptions[i]))
    choice = 0
    while not(np.any(choice == np.arange(len(menuOptions))+1)):
        choice = menuOptionNumber("Please choose a menu option: ")
    return choice
#==================================================================
def menuItemOne():
    #print('Please enter the file names of attributes and training examples.')
    import csv
    userInputtedFile = raw_input("Please enter the file name of attributes and training examples: ");
    global dataset
    dataset = pd.read_csv(userInputtedFile)
    dataLabels = dataset.select_dtypes(include = [np.number] )
    global idChoice
    #This is the id number of the posititon you want to add in the file (append and delete or replace)
    idChoice = input("Please enter the desired ID number: ")
    global featuresList
    #basically just a list of labels
    featuresList = list(dataLabels)
    global target_names
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    #model for the tree
    global classifierModel
    classifierModel = GaussianNB()

    #with open(userInputtedFile) as inputFile:
    #    read = csv.DictReader(inputFile)
    #    i = read.next()
    #    rest = [row for row in read]
    #
    #    print i

    x = dataset.drop('Class', axis = 1) #all values
    y = dataset.values[:,-1] #all values but the last column
    #x = dataset
    #y = dataset[idChoice]
    print("The current files shape is formatted (x,y) with x = number of data entries, y = number of attributes. ")
    print(dataset.shape)
    global x_train, y_train, x_test, y_test
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

    #trainer, tester = train_test_split(dataLabels, test_size = 0.2, random_state=0)
#This is the training data from the decision tree to create it
    global NBModel
    NBModel = classifierModel.fit(x_train, y_train)

    y_predicted = classifierModel.predict(x_test)

    #i attempted here to save the graph and view after but got really stressed out and stopped since it was extra
    #vTree = raw_input('Please enter a name for the saved file. ')
    #view = tree.export_graphviz(classifierModel,out_file=None, feature_names = featuresList, rounded=False, special_characters=True)
    #graph = graphviz.Source(view)
    #view.render(vTree)




    #I was playing around with the reports and tests
    #y_predict = classifierModel.predict(x_test)
    #print(confusion_matrix(y_test, y_predict))
    #print(classification_report(y_test, y_predict))

#==================================================================


def menuItemTwo():
    #prompt the user to enter a name for the file
    promptedUserName_pkl = raw_input("Please enter the file name for saving the tree model file: ")
    #file here
    treeModel_pkl = open(promptedUserName_pkl, 'wb')
    #list of labels, the replacement spot, and the classifier
    #open them all and store that location
    global treeModel_cl, treeModel_fe, treeModel_pi
    treeModel_fe = open(promptedUserName_pkl + 'featuresList', 'wb')
    treeModel_pi = open(promptedUserName_pkl + 'repl', 'wb')
    treeModel_cl = open(promptedUserName_pkl + 'cl', 'wb')
    #save all the different objects and put them away for later
    pickle.dump(NBModel, treeModel_pkl) #tree
    pickle.dump(featuresList,treeModel_fe) #labels list
    pickle.dump(idChoice, treeModel_pi) #choice of location for id
    pickle.dump(classifierModel, treeModel_cl) #classifier model
    #close them all now
    treeModel_pkl.close()
    treeModel_fe.close()
    treeModel_pi.close()
    treeModel_cl.close()
    print("The tree has just been saved. ")

#==================================================================

def interactivelyTraverse():
    global dataset
    #global dTree
    #tree_traverse = mod.tree_

    features_traverse = featuresList #These are the features to print out for user input
    pi_traverse = target_names #These are the names of the classes they are in (count determines this)
    #name_traverse = [ features_traverse[i] if i != _tree.TREE_UNDEFINED else "undefined!"
    # for i in tree_traverse.feature]
    appendList = list()
    for feature in features_traverse:
        print("Please input a value for")
        print(feature)
        inVal = input(" ")
        appendList.append(inVal)

    inValArr = np.asarray(appendList)
# This is where my error will happen, i do not know how to shape the array to predict it against
    arrayp = [[int(inValArr[0]), int(inValArr[1]),int(inValArr[2]),int(inValArr[3])]]
    prediction = classifierModel.predict(arrayp)
    #print(prediction)

    if (prediction == 0):
        print("Not likely.")
    elif (prediction == 1):
        print("Likely")
    else:
        print("Neither likely nor not likely. Must've been an issue.")
"""
    def recursivelyTraverse(node, depth, count):
        if (tree_traverse.feature[node] != _tree.TREE_UNDEFINED):
            var_Name = name_traverse[node]
            treeNodeThreshold = tree_traverse.threshold[node]
            print("Please input a value for ")
            print(var_Name)
            print("Threshold for this node is: ")
            print(treeNodeThreshold)
            inVal = int(input())
            if (inVal <= treeNodeThreshold):
                recursivelyTraverse(tree_traverse.children_right[node], depth+1, count+1)
            else:
                recursivelyTraverse(tree_traverse.children_left[node], depth+1, count+1)

        else:
            print("A leaf in the tree has been reached.")
            if (count==1):
                print("Input resulted in ")
                print(pi_traverse[0])
            elif (count == 2):
                print("Input resulted in ")
                print(pi_traverse[1])
            elif (count == 3):
                print("Input resulted in ")
                print(pi_traverse[2])
            elif (count == 4):
                print("Input resulted in ")
                print(pi_traverse[3])
            elif (count == 5):
                print("Input resulted in ")
                print(pi_traverse[4])
            else:
                print("Input resulted in ")
                print(pi_traverse[5])


    recursivelyTraverse(0,1,0)
    return
"""
def menuItemThree():
    while True:
        #learned the cool multiple line stuff
        submenuChoice = input("""
        3.1 Enter a new case interactively.
        3.2 Quit.
        What would you like to do? (Please enter 1 or 2):
        """)

        if submenuChoice == 1:
            print("This is a list of the features:\n")
            popList = featuresList
            popList.pop()
            print(list(popList))
#Has several parts in my code where i had to use raw_input, learned the difference between the two
            #dataForInput = raw_input("Please enter the data for the above fields, separated with commas: ").split(',')
            #dataForInput = [x.strip(' ') for x in dataForInput]
            #dataList = []

            #dataList.append(dataForInput)
            #print("The data has been appended to the data list.\n")
            interactivelyTraverse()

#I commented out a bunch of code that was from before, (in attempt to) fix
#the program to do what it should have done (i think).

#These are the list of attributes found in the columns first row
            #dd = pd.DataFrame(dataList, columns=featuresList)
            #dd[idChoice]=0
            #print(dd)
#predicts the model and returns on whether or not the tree decides it is viable or not
            #y_predicted = classifierModel.predict(dd[featuresList])
            #print("Currently running against the model.")
            #print(y_predicted[0])

            #if y_predicted[0] == 0:
            #    print("Tree has decided true.")
            #elif y_predicted[0] == 1:
            #    print("Tree has decided false.")
        elif submenuChoice == 2:
            return 0
#==================================================================
def menuItemFour():

    inputtedFile4 = raw_input("Please enter the name of the classifier file to load. ")
    inputtedFile4_fe = open(inputtedFile4 + 'featuresList', 'rb')
    inputtedFile4_pi = open(inputtedFile4 + 'repl', 'rb')
    inputtedFile4_cl = open(inputtedFile4 + 'cl', 'rb')
    interactivelyTraverse()
    global feat, pi, cl
    feat = pickle.load(inputtedFile4_fe)
    pi = pickle.load(inputtedFile4_pi)
    cl = pickle.load(inputtedFile4_cl)
    inputtedFile4_cl.close()
    inputtedFile4_fe.close()
    inputtedFile4_pi.close()
#==================================================================
#These are in their own array of instructions
menuItems = np.array(["Learn a classifier from training data.",
"Save the classifier learned in menu item 1.",
"Applying the classifier to new cases.",
"Load a classifier model saved previously and apply the model to new cases interactively as in menu 3.",
 " Quit."])

while True:
    #show menu entil they type a 5
    choice = showMenu(menuItems);

    if choice == 1:
        #Choice #1 - Learn a decision tree from training data.
        menuItemOne()
    elif choice == 2:
        #Choice #2 - Save tree learned in item 1
        menuItemTwo()
    elif choice == 3:
        #Choice #3 - Applying the decision tree to new cases
        menuItemThree()
    elif choice == 4:
        #Choice #4 - Load a tree model saved previously and apply model to new cases interactively as in menu item 3
        menuItemFour()
    elif choice == 5:
        break #Choice #5 - Quit
    elif choice < 1 or choice > 5:
        choice = showMenu(menuItems)
