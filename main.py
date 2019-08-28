import csv
import math
import sys
from collections import defaultdict
import numpy as np



# preprocess the date, remove all special characters
def training_date_preprocessing(data_location):
    columns = defaultdict(list)
    with open(data_location) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k, v) in row.items():
                columns[k].append(v)
    i = 0
    content = [[] for i in range(5)]
    for x in columns['Description']:
        if i < columns["AdoptionSpeed"].__len__():
            speed = columns["AdoptionSpeed"][i]
            x = x.replace("\n"," ").replace("*"," ").replace("'"," ").replace("~"," ").replace("/"," ").replace("_","").\
                replace(")","").replace("(","").replace("-","").replace(".","").replace(",","").replace("?","").\
                replace(":","").replace("!", "").replace("\"", "").replace("\'", "").replace("@", "").replace("#", "").\
                replace("=", "").replace(">", "").replace("<", "").replace("$", "").replace("?", "").replace("+", "").\
                replace("-", "").replace("&", "").replace("^", "").replace("%", "").replace("$", "").replace("{", "").\
                replace("}", "").replace("[", "").replace("]", "").replace("|", "").replace("\\", "").replace(";", "")
            content[int(speed)].append(x)
            i+=1
    return content
#========================================================= unigram =============================================

# create [class]:[unigram_word] dictionary
def generateWordDict(content):
    allWordDict = {0:[],1:[],2:[],3:[],4:[]}
    uniqueWordDict = set()
    for x in range(5):
        description_list = content[x]
        for description in description_list:
            for z in description.split(" "):
                allWordDict[x].append(z)
                uniqueWordDict.add(z)
    
    return [allWordDict,uniqueWordDict]

# create [unigram_word][class]:[frequency] dictionary
def generateWordFrequencyDict(allWordDict):
    wordFrequencyDict = {}
    for i in range(5):
        word_list = allWordDict[i]
        for word in word_list:
            if word in wordFrequencyDict:
                wordFrequencyDict[word][i] += 1
            else:
                wordFrequencyDict[word] = [0 for k in range(5)]
                wordFrequencyDict[word][i] = 1
    return wordFrequencyDict

# create [unigram_word][class]:[probability] dictionary
def generateWordProbabilityDict(allWordDict,uniqueWordDict,wordFrequencyDic):
    total_unique_words = len(uniqueWordDict)
    wordProbabilityDict = {}
    for i in range(5):
        for word in uniqueWordDict:
            if word in wordFrequencyDic:
                word_num_in_class_i = wordFrequencyDic[word][i]
            else:
                word_num_in_class_i = 0
            total_words_in_class_i = len(allWordDict[i])
            p = -(np.log((word_num_in_class_i + 1)) - np.log((total_words_in_class_i + total_unique_words)))
            if word not in wordProbabilityDict:
                wordProbabilityDict[word] = [0 for k in range(5)]
            wordProbabilityDict[word][i] = p
    return wordProbabilityDict

# calculate perplexity of unigram
def countAccuaracyUsingPerplexity(raw_data,wordProbabilityDict,wordFrequencyDict,allWordDict,uniqueWordDict):
    correct = 0
    wrong = 0
    for x in range(5):
        allDescription = raw_data[x]
        for description in allDescription:
            if len(description)>0:
                wordList = description.split(" ")
                minPerplexity = sys.maxint
                possibleClass = -1
                for k in range(5):
                    p = 0
                    size = 0
                    total_unique_words = len(uniqueWordDict)
                    for word in wordList:
                        size += 1
                        if word in wordProbabilityDict:
                            p += wordProbabilityDict[word][k]
                        else:
                            if word in wordFrequencyDict:
                                word_num_in_class_i = wordFrequencyDict[word][k]
                            else:
                                word_num_in_class_i = 0
                            total_words_in_class_i = len(allWordDict[k])
                            p += -(np.log((word_num_in_class_i + 1)) - np.log(
                                (total_words_in_class_i + total_unique_words)))
                    if size!=0:
                        result = math.exp(p/size)
                        if result < minPerplexity:
                            minPerplexity = result
                            possibleClass = k
                if possibleClass!=-1:
                    if possibleClass == x:
                        correct+=1
                    else:
                        wrong+=1
    return correct * 1.0/(correct+wrong) * 100

#========================================================= bigram =============================================


# create [class]:[bigram_word] dictionary
def generateWordDict_bigram(content):
    allWordDict = {0: [], 1: [], 2: [], 3: [], 4: []}
    uniqueWordDict = set()
    for x in range(5):
        description_list = content[x]
        for description in description_list:
            allWords = description.split(" ")
            for z in range(len(allWords) - 1):
                bigram = allWords[z] + " " + allWords[z + 1]
                allWordDict[x].append(bigram)
                uniqueWordDict.add(bigram)
    return [allWordDict, uniqueWordDict]
    

# create [bigram_word][class]:[frequency] dictionary
def generateWordFrequencyDict_bigram(allWordDict):
    wordFrequencyDict = {}
    for i in range(5):
        word_list = allWordDict[i]
        for word in word_list:
            if word in wordFrequencyDict:
                wordFrequencyDict[word][i] += 1
            else:
                wordFrequencyDict[word] = [0 for k in range(5)]
                wordFrequencyDict[word][i] = 1
    return wordFrequencyDict
    
# create [bigram_word][class]:[probability] dictionary
def generateWordProbabilityDict_bigram(allWordDict,uniqueWordDict,wordFrequencyDic):
    total_unique_words = len(uniqueWordDict)
    wordProbabilityDict = {}
    for i in range(5):
        for word in uniqueWordDict:
            if word in wordFrequencyDic:
                word_num_in_class_i = wordFrequencyDic[word][i]
            else:
                word_num_in_class_i = 0
            total_words_in_class_i = len(allWordDict[i])
            p = -(np.log((word_num_in_class_i + 1)) - np.log((total_words_in_class_i + total_unique_words)))
            if word not in wordProbabilityDict:
                wordProbabilityDict[word] = [0 for k in range(5)]
            wordProbabilityDict[word][i] = p
    return wordProbabilityDict

# calculate perplexity of bigram
def countAccuaracyUsingPerplexity_bigram(raw_data,wordProbabilityDict,wordFrequencyDict,allWordDict,uniqueWordDict):
    correct = 0
    wrong = 0
    for x in range(5):
        allDescription = raw_data[x]
        for description in allDescription:
            if len(description)>0:
                wordList = description.split(" ")
                bigramList = []
                for z in range(len(wordList) - 1):
                    bigram = wordList[z] + " " + wordList[z + 1]
                    bigramList.append(bigram)
                minPerplexity = sys.maxint
                possibleClass = -1
                for k in range(5):
                    p = 0
                    size = 0
                    total_unique_words = len(uniqueWordDict)
                    for word in bigramList:
                        size += 1
                        if word in wordProbabilityDict:
                            p += wordProbabilityDict[word][k]
                        else:
                            if word in wordFrequencyDict:
                                word_num_in_class_i = wordFrequencyDict[word][k]
                            else:
                                word_num_in_class_i = 0
                            total_words_in_class_i = len(allWordDict[k])
                            p += -(np.log((word_num_in_class_i + 1)) - np.log(
                                (total_words_in_class_i + total_unique_words)))
                    if size!=0:
                        result = math.exp(p/size)
                        if result < minPerplexity:
                            minPerplexity = result
                            possibleClass = k
                if possibleClass!=-1:
                    if possibleClass == x:
                        correct+=1
                    else:
                        wrong+=1
    return correct * 1.0/(correct+wrong) * 100

#========================================================= trigram =============================================


# create [class]:[trigram_word] dictionary
def generateWordDict_trigram(content):
    allWordDict = {0: [], 1: [], 2: [], 3: [], 4: []}
    uniqueWordDict = set()
    for x in range(5):
        description_list = content[x]
        for description in description_list:
            allWords = description.split(" ")
            for z in range(len(allWords) - 2):
                trigram = allWords[z] + " " + allWords[z + 1] + " " +allWords[z + 2]
                allWordDict[x].append(trigram)
                uniqueWordDict.add(trigram)
    return [allWordDict, uniqueWordDict]


# create [trigram_word][class]:[frequency] dictionary
def generateWordFrequencyDict_trigram(allWordDict):
    wordFrequencyDict = {}
    for i in range(5):
        word_list = allWordDict[i]
        for word in word_list:
            if word in wordFrequencyDict:
                wordFrequencyDict[word][i] += 1
            else:
                wordFrequencyDict[word] = [0 for k in range(5)]
                wordFrequencyDict[word][i] = 1
    return wordFrequencyDict


# create [trigram_word][class]:[probability] dictionary
def generateWordProbabilityDict_trigram(allWordDict, uniqueWordDict, wordFrequencyDic):
    total_unique_words = len(uniqueWordDict)
    wordProbabilityDict = {}
    for i in range(5):
        for word in uniqueWordDict:
            if word in wordFrequencyDic:
                word_num_in_class_i = wordFrequencyDic[word][i]
            else:
                word_num_in_class_i = 0
            total_words_in_class_i = len(allWordDict[i])
            p = -(np.log((word_num_in_class_i + 1)) - np.log((total_words_in_class_i + total_unique_words)))
            if word not in wordProbabilityDict:
                wordProbabilityDict[word] = [0 for k in range(5)]
            wordProbabilityDict[word][i] = p
    return wordProbabilityDict

# calculate perplexity of trigram_word
def countAccuaracyUsingPerplexity_trigram(raw_data, wordProbabilityDict, wordFrequencyDict, allWordDict, uniqueWordDict):
    correct = 0
    wrong = 0
    for x in range(5):
        allDescription = raw_data[x]
        for description in allDescription:
            if len(description) > 0:
                wordList = description.split(" ")
                trigramList = []
                for z in range(len(wordList) - 2):
                    trigram = wordList[z] + " " + wordList[z + 1]+ " " + wordList[z + 2]
                    trigramList.append(trigram)
                minPerplexity = sys.maxint
                possibleClass = -1
                for k in range(5):
                    p = 0
                    size = 0
                    total_unique_words = len(uniqueWordDict)
                    for word in trigramList:
                        size += 1
                        if word in wordProbabilityDict:
                            p += wordProbabilityDict[word][k]
                        else:
                            if word in wordFrequencyDict:
                                word_num_in_class_i = wordFrequencyDict[word][k]
                            else:
                                word_num_in_class_i = 0
                            total_words_in_class_i = len(allWordDict[k])
                            p += -(np.log((word_num_in_class_i + 1)) - np.log(
                                (total_words_in_class_i + total_unique_words)))
                    if size != 0:
                        result = math.exp(p / size)
                        if result < minPerplexity:
                            minPerplexity = result
                            possibleClass = k
                if possibleClass != -1:
                    if possibleClass == x:
                        correct += 1
                    else:
                        wrong += 1
    return correct * 1.0 / (correct + wrong) * 100


#========================================================= main function for each n-gram =============================================

# main function for unigram model
def unigram_bag_of_words():
    training_data_location = "data/train.csv"
    testing_data_location = "data/test.csv"
    training_raw_data = training_date_preprocessing(training_data_location)
    [allWordDict,uniqueWordDict] = generateWordDict(training_raw_data)
    wordFrequencyDict = generateWordFrequencyDict(allWordDict)
    wordProbabilityDict = generateWordProbabilityDict(allWordDict,uniqueWordDict,wordFrequencyDict)
    testing_raw_data = training_date_preprocessing(testing_data_location)
    accuracy_for_testingSet = countAccuaracyUsingPerplexity(testing_raw_data, wordProbabilityDict,wordFrequencyDict,allWordDict,uniqueWordDict)
    print (str(accuracy_for_testingSet) + "%")
    
# main function for bigram model
def bigram_of_words():
    training_data_location = "data/train.csv"
    testing_data_location = "data/test.csv"
    training_raw_data = training_date_preprocessing(training_data_location)
    [allWordDict,uniqueWordDict] = generateWordDict_bigram(training_raw_data)
    wordFrequencyDict = generateWordFrequencyDict_bigram(allWordDict)
    wordProbabilityDict = generateWordProbabilityDict_bigram(allWordDict,uniqueWordDict,wordFrequencyDict)
    testing_raw_data = training_date_preprocessing(testing_data_location)
    accuracy_for_testingSet = countAccuaracyUsingPerplexity_bigram(testing_raw_data, wordProbabilityDict,wordFrequencyDict,allWordDict,uniqueWordDict)
    print (str(accuracy_for_testingSet) + "%")
    
    
# main function for trigram model
def trigram_of_words():
    training_data_location = "data/train.csv"
    testing_data_location = "data/test.csv"
    training_raw_data = training_date_preprocessing(training_data_location)
    [allWordDict,uniqueWordDict] = generateWordDict_trigram(training_raw_data)
    wordFrequencyDict = generateWordFrequencyDict_trigram(allWordDict)
    wordProbabilityDict = generateWordProbabilityDict_trigram(allWordDict,uniqueWordDict,wordFrequencyDict)
    testing_raw_data = training_date_preprocessing(testing_data_location)
    accuracy_for_testingSet = countAccuaracyUsingPerplexity_trigram(testing_raw_data, wordProbabilityDict,wordFrequencyDict,allWordDict,uniqueWordDict)
    print (str(accuracy_for_testingSet) + "%")

if __name__ == "__main__":
    unigram_bag_of_words()
    bigram_of_words()
    trigram_of_words()
    
