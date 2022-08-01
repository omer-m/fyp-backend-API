
from cProfile import label
import json
import re
from unittest import result
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
from flask import request
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, RepeatedKFold
from datetime import datetime
import subprocess

app = Flask(__name__)
CORS(app)


def readCSV_DF(file):
    DF = pd.read_csv(file, low_memory=False)

    print(DF.columns)

    return DF

def writeTweet_TXT(tweets):
    with open('./static/ark-tweet-nlp-0.3.2/unseenTweets.txt', "w") as file:
        for each in tweets:
            file.write('%s\n' %str(each))
    file.close()

def changeFormate(twt,pos):
    tweetTXT_token= []
    POS_withoutSpaces = []
    POS_pair = []
    for i in range(len(twt)):
        temp = ''
        temp=twt[i].split(" ")
        tweetTXT_token.append(temp)
        
        POS_withoutSpaces.append(pos[i].replace(" ",""))
      
           
        temp = []
        for j in range(len(tweetTXT_token[i])):
           
            temp.append(tweetTXT_token[i][j])
            temp.append(POS_withoutSpaces[i][j])
            
        POS_pair.append(temp)
            
        
    return POS_pair , POS_withoutSpaces 
    
# low level lexicon feature 
def getLexiconFeatures(df):
    X = df.iloc[:, -8:]

    return X

def countFeature(singleTweet, lexiconArray):
    count = 0
    for item in lexiconArray:
        if item in singleTweet:
            count = count + 1
    return count

def computeLexiconFeatures(twtCOL):

    human_sheet = './static/final_features/humanLexicon.csv'
    infra_sheet = './static/final_features/infransLexicon.csv'
    human_lexicon = readCSV_DF(human_sheet)
    infra_lexicon = readCSV_DF(infra_sheet)
    human_nounLexicon = list(set(human_lexicon.loc[:, 'humanNouns']))
    human_verbLexicon = list(set(human_lexicon.loc[:, 'humanVerbs']))
    human_adverbLexicon = list(set(human_lexicon.loc[:, 'humanAdverbs']))
    human_adjectiveLexicon = list(set(human_lexicon.loc[:, 'humanAdjective']))

    infra_nounLexicon = list(set(infra_lexicon.loc[:, 'infraNouns']))
    infra_verbLexicon = list(set(infra_lexicon.loc[:, 'infraVerb']))
    infra_adverbLexicon = list(set(infra_lexicon.loc[:, 'infraAdverbs']))
    infra_adjectiveLexicon = list(set(infra_lexicon.loc[:, 'infraAdjective']))

    human_nounCount = []
    human_verbCount = []
    human_adverbCount = []
    human_adjectiveCount = []
    infra_nounCount = []
    infra_verbCount = []
    infra_adverbCount = []
    infra_adjectiveCount = []

    tweetDF = pd.DataFrame()
    # twt = twtCOL
    for twt in twtCOL:
        if 'nan' != str(twt):
            print(twt)
            twt = twt.split()
            human_nounCount.append(countFeature(twt, human_nounLexicon))
            human_verbCount.append(countFeature(twt, human_verbLexicon))
            human_adverbCount.append(countFeature(twt, human_adverbLexicon))
            human_adjectiveCount.append(countFeature(twt, human_adjectiveLexicon))
            infra_nounCount.append(countFeature(twt, infra_nounLexicon))
            infra_verbCount.append(countFeature(twt, infra_verbLexicon))
            infra_adverbCount.append(countFeature(twt, infra_adverbLexicon))
            infra_adjectiveCount.append(countFeature(twt, infra_adjectiveLexicon))
        else:
            human_nounCount.append("0")
            human_verbCount.append("0")
            human_adverbCount.append("0")
            human_adjectiveCount.append("0")
            infra_nounCount.append("0")
            infra_verbCount.append("0")
            infra_adverbCount.append("0")
            infra_adjectiveCount.append("0")

    tweetDF.insert(len(tweetDF.columns), "human_nounCount", human_nounCount)
    tweetDF.insert(len(tweetDF.columns), "human_verbCount", human_verbCount)
    tweetDF.insert(len(tweetDF.columns),
                   "human_adverbCount", human_adverbCount)
    tweetDF.insert(len(tweetDF.columns),
                   "human_adjectiveCount", human_adjectiveCount)

    tweetDF.insert(len(tweetDF.columns), "infra_nounCount", infra_nounCount)
    tweetDF.insert(len(tweetDF.columns), "infra_verbCount", infra_verbCount)
    tweetDF.insert(len(tweetDF.columns),
                   "infra_adverbCount", infra_adverbCount)
    tweetDF.insert(len(tweetDF.columns),
                   "infra_adjectiveCount", infra_adjectiveCount)

    return tweetDF

# Syntactic Features
def getSyntacticFeatures(df):
    X = df.iloc[:, -12:]

    return X

def java_POS(tweets):

    writeTweet_TXT(tweets)
    print("in javaPOS")
    filepath = './static/javaPOS_run.sh'

    p = subprocess.Popen(filepath, shell=True, stdout = subprocess.PIPE)
    rawPOS=p.communicate()

    rawPOS=rawPOS[0].decode("utf-8") 
    rawPOS=rawPOS.splitlines()

    rawPOSList = list()

    for line in rawPOS:
        line = line.split("\t")
        newline = list()
        for each in line:
            if each == '':
                continue
            newline.append(each)
        rawPOSList.append(newline)
    
    pos=[]
    tweetTXT = []
    print(len(rawPOSList))
    for i in range(0,len(rawPOSList)):
        pos.append(rawPOSList[i][1])
        tweetTXT.append(rawPOSList[i][0])
    print("pos done")
    pos_twt , posCol=changeFormate(tweetTXT,pos)
    print("pos : ",posCol )
    return posCol

# fuction to count Syntactic feature

def countLen(col):
    tweet_len = []
   
    for row in col:
        count=len(row) - row.count(',') - row.count(" ")
        tweet_len.append(count)
        
    return tweet_len
        
def countVerb(col):
    
    verb_cont=[]
    
    for row in col:
        
        count=0
        count = count + row.count("V")
        
        
        verb_cont.append(count)
        
    return verb_cont
        
def countPronoun(col):
    
    Pronoun_count = []
    
    for row in col:

        count=0       
        count = count + row.count("O")
        Pronoun_count.append(count)
        
    return Pronoun_count

def countCommonNoun (col):
    
    CommonNoun_count = []
    
    for row in col:

        count=0
        count = count + row.count("N")
        CommonNoun_count.append(count)
        
    return CommonNoun_count
        
def countAdverb(col):
    
    Adverb_count = []
    
    for row in col:

        count=0
       
        count = count + row.count("R")
        
        Adverb_count.append(count)
        
    return  Adverb_count
        
def countAdjective(col):
    
    Adjective_count = []
    
    for row in col:
       

        count=0

    
        count = count + row.count("A")
        
        Adjective_count.append(count)
        
    return  Adjective_count
        
def countInterjection(col):
    
    Interjection_count = []
    
    for row in col:
    
        count=0

        count = count + row.count("!")
        
        Interjection_count.append(count)
        
    return  Interjection_count      

def countDeterminer(col):
    
    Determiner_count = []
    
    for row in col:

        count=0
    
        count = count + row.count("D")

        Determiner_count.append(count)
        
    return  Determiner_count
        
def countURL(col):
    
    URL_count = []

    for row in col:

        count=0


        count = count + row.count('U')


        URL_count.append(count)

    return URL_count

def countAbbreviations(col):
    abbr_count = []

    for row in col:

        count=0
        count = count + row.count('G')
        abbr_count.append(count)
    
    return abbr_count

def countPrePost(col):
    prePost_count = []
    
    for row in col:
        count =0 
        count = count + row.count('P')
        prePost_count.append(count)
    
    return prePost_count

def countNominalVerbal(col):
    NominalVerbal_count = []
    
    for row in col:
        count =0 
        count = count + row.count('L')
        NominalVerbal_count.append(count)
    
    return NominalVerbal_count

# pass pos col this fuction will return df (12 SyntacticFeatures )
def computeSyntacticFeatures(PosCOL):

    print("in computeSyntacticFeatures")
    tweet_length = countLen(PosCOL)
    Determiners_Count = countDeterminer(PosCOL)
    Verbs_Count = countVerb(PosCOL)
    Pronouns_Count = countPronoun(PosCOL)
    NominalVerbal_Count = countNominalVerbal(PosCOL)
    URLs_Count = countURL(PosCOL)
    Noun_Count = countCommonNoun(PosCOL)
    Adverb_Count = countAdverb(PosCOL)
    Adjective_Count = countAdjective(PosCOL)
    Abbreviation_count= countAbbreviations(PosCOL)
    Interjection_count = countInterjection(PosCOL)
    PrePost_count = countPrePost(PosCOL)
    
    DF = pd.DataFrame()

    DF.insert(len(DF.columns),"tweet_length",tweet_length) 
    DF.insert(len(DF.columns),"Determiners_Count",Determiners_Count)  
    DF.insert(len(DF.columns),"Verbs_Count",Verbs_Count)  
    DF.insert(len(DF.columns),"Pronouns_Count",Pronouns_Count) 
    DF.insert(len(DF.columns),"NominalVerbal_Count",NominalVerbal_Count) 
    DF.insert(len(DF.columns),"URLs_Count",URLs_Count) 
    DF.insert(len(DF.columns),"Noun_Count",Noun_Count)  
    DF.insert(len(DF.columns),"Adverb_Count",Adverb_Count)  
    DF.insert(len(DF.columns),"Adjective_Count",Adjective_Count) 
    DF.insert(len(DF.columns),"Unknown_count",Abbreviation_count) 
    DF.insert(len(DF.columns),"Interjection_count",Interjection_count) 
    DF.insert(len(DF.columns),"PrePost_count",PrePost_count) 

    print(DF)
    
    return DF

# top 10 word Frequency Features
def getTop10FrequencyFeatures(df):
    X = df.iloc[:-1, -10:]

    return X

def BOW(tweets, col_name):

    all_tweet_frequncy = []
    for tweet in tweets:
        tweet = str(tweet).split()
        single_tweet = []
        for word in col_name:
            single_tweet.append(tweet.count(word))
        all_tweet_frequncy.append(single_tweet)

    NEWDF = pd.DataFrame(all_tweet_frequncy)
    NEWDF.columns = col_name

    return NEWDF


def computeTop10Features(twtCoL, disasterName):
    print(disasterName)
    fileName = "./static/final_features/"+disasterName+"_top10_frequency.csv"
    df = readCSV_DF(fileName)
    top10Word = df.columns
    df_final = BOW(twtCoL, top10Word)
    print(df_final)
    print(twtCoL)
    return df_final


def multiclassLabel(col):
    newCol = []
    for i in col:
        if i == "infrastructure_damage":
            newCol.append(2)
        if i == "human_damage":
            newCol.append(1)
        if i == "non_damage":
            newCol.append(0)
    return newCol

def binaryclassLabel(col):
    newCol = []
    for i in col:
        if i == "damage":
            newCol.append(1)
        if i == "non_damage":
            newCol.append(0)
    return newCol

def createModel(*arguments):

    X = pd.DataFrame()
    X = pd.concat([arguments[0], arguments[1], arguments[2]], axis=1)
    Y = arguments[3]
    result = dict()
    result["accuracy"] = 'None'
    result["precision_macro"] = 'None'
    result["recall_macro"] = 'None'
    result["f1_macro"] = 'None'
    result['pklURL'] = 'None'

    if arguments[4] == 'RF':
        model = RandomForestClassifier()
    elif arguments[4] == 'SVM':
        model = svm.SVC(kernel='linear')

    # selectedValidation
    if arguments[5] == 'kFold':
        cv = KFold(n_splits=10)
        temp = cross_validate(model, X, Y, scoring=arguments[6][1:],
                              cv=cv, n_jobs=-1, error_score='raise')

        for i in arguments[6][1:]:
            result[i] = round(np.mean(temp['test_'+i]), 3)

        with open('./static/'+arguments[7]+'.pkl', 'wb') as f:
            pickle.dump(model, f)
        url = './static/pklFiles/'+arguments[7]+'.pkl'


        result['pklURL'] = url

    elif arguments[5] == 'holdOut':
        X_train, X_test, y_train, y_test = train_test_split(X, Y)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for i in arguments[6][1:]:
            if i == 'accuracy':
                result["accuracy"] = round(
                    metrics.accuracy_score(y_test, y_pred), 3)
            elif i == 'precision_macro':
                result["precision_macro"] = round(metrics.precision_score(
                    y_test, y_pred, average='macro'), 3)
            elif i == 'recall_macro':
                result["recall_macro"] = round(metrics.recall_score(
                    y_test, y_pred, average='macro'), 3)
            elif i == 'f1_macro':
                result["f1_macro"] = round(metrics.f1_score(
                    y_test, y_pred, average='macro'), 3)

        with open('./static/pklFiles/'+arguments[7]+'.pkl', 'wb') as f:
            pickle.dump(model, f)
        url = './static/pklFiles/'+arguments[7]+'.pkl'

        result['pklURL'] = url

    return result


@app.route("/")
def home():
    return 'api called'


@app.route("/readFile")
def readCSVfile():

    df = pd.read_csv(
        ".\static\\california_wildfires_pos_clean_final.csv")
    newdf = df.head(5)
    df_json = newdf.to_dict()
    # print(df_json)

    omer = ["233", "233", "233", "233", "233"]
    awais = ["123", "abc", "233", "233", "233"]

    return df_json  # jsonify({'l1': omer, 'l2':awais})


# @app.route("/model/t", methods=['GET', 'POST'])
# def test():
#     return(request.json['URL'])


@app.route("/model", methods=['GET', 'POST'])
def TrainMLmodel():
    # print(request.json['URL'])
    # selectedDataset="MMD"
    selectedDisaster = request.json['selectedDisasterName']
    selectedFeature = request.json['selectedFeatureName']
    # selectedPreProcessing="All"
    selectedCType = request.json['selectedCTypeName']
    selectedMLmodel = request.json['selectedMLmodelName']
    selectedValidation = request.json['selectedValidationName']
    selectedMetric = request.json['selectedMetricName']
    # ['None', 'accuracy',
    #                   'precision_macro',
    #                   'recall_macro', 'f1_macro']

    namePKL = selectedDisaster+"_" + selectedFeature + "_" + selectedMLmodel+"_" + selectedCType

    df_SyntacticFeatures = pd.read_csv(
        "./static//final_features//"+selectedDisaster+"_SyntacticFeatures.csv")
    df_Top10Features = pd.read_csv(
        "./static//final_features//"+selectedDisaster+"_top10_frequency.csv")
    df_LexiconFeature = pd.read_csv(
        "./static//final_features//"+selectedDisaster+"_lexiconFeature.csv")

    SyntacticFeatures = getSyntacticFeatures(df_SyntacticFeatures)
    Top10Features = getTop10FrequencyFeatures(df_Top10Features)
    LexiconFeature = getLexiconFeatures(df_LexiconFeature)

    BinaryLabel = binaryclassLabel(
        df_SyntacticFeatures.loc[:, "binary_label"])  # check later
    MultiLabel = multiclassLabel(df_SyntacticFeatures.loc[:, "multi_label"])

    if selectedFeature == "allFeature" and selectedCType == "multi":

        Result = createModel(SyntacticFeatures, Top10Features, LexiconFeature,
                             MultiLabel, selectedMLmodel, selectedValidation, selectedMetric, namePKL)

    elif selectedFeature == "SyntacticFeatures" and selectedCType == "multi":

        Result = createModel(SyntacticFeatures, pd.DataFrame(), pd.DataFrame(),
                             MultiLabel, selectedMLmodel, selectedValidation, selectedMetric, namePKL)

    elif selectedFeature == "top10_frequency" and selectedCType == "multi":

        Result = createModel(Top10Features, pd.DataFrame(), pd.DataFrame(),
                             MultiLabel, selectedMLmodel, selectedValidation, selectedMetric, namePKL)

    elif selectedFeature == "lexiconFeature" and selectedCType == "multi":

        Result = createModel(LexiconFeature, pd.DataFrame(), pd.DataFrame(),
                             MultiLabel, selectedMLmodel, selectedValidation, selectedMetric, namePKL)

    elif selectedFeature == "allFeature" and selectedCType == "binary":

        Result = createModel(SyntacticFeatures, Top10Features, LexiconFeature,
                             BinaryLabel, selectedMLmodel, selectedValidation, selectedMetric, namePKL)

    elif selectedFeature == "SyntacticFeatures" and selectedCType == "binary":

        Result = createModel(SyntacticFeatures, pd.DataFrame(), pd.DataFrame(),
                             BinaryLabel, selectedMLmodel, selectedValidation, selectedMetric, namePKL)

    elif selectedFeature == "top10_frequency" and selectedCType == "binary":

        Result = createModel(Top10Features, pd.DataFrame(), pd.DataFrame(),
                             BinaryLabel, selectedMLmodel, selectedValidation, selectedMetric, namePKL)

    elif selectedFeature == "lexiconFeature" and selectedCType == "binary":

        Result = createModel(LexiconFeature, pd.DataFrame(), pd.DataFrame(),
                             BinaryLabel, selectedMLmodel, selectedValidation, selectedMetric, namePKL)

    return Result


@app.route("/modelTest", methods=['GET', 'POST'])
def modelTest():

    unseenData = request.json['unseenData']
    url = request.json['selectedmodel_url']
    disName = request.json['dName']
    featureName = request.json['fName']

    # unseenData = "10 people are dead"
    # url = "./static/pklFiles/california_wildfires_lexiconFeature_RF_binary.pkl"
    # disName= "california_wildfires"
    # featureName = "lexiconFeature"

    print(request.json['selectedmodel_url'])
    # print("static\pklFiles\california_wildfires_allFeature_RF_binary.pkl")

    selected_model = pickle.load(open(url, 'rb'))
   
    LexiconFeaturesValue = computeLexiconFeatures(unseenData)

    Top10FrequencyFeaturesValue = computeTop10Features(unseenData, disName)

    posCol = java_POS(unseenData)
    SyntacticFeaturesValue = computeSyntacticFeatures(posCol)

    All_feature = pd.DataFrame()
    All_feature = pd.concat([SyntacticFeaturesValue, Top10FrequencyFeaturesValue , LexiconFeaturesValue ], axis=1)  

    if featureName == "lexiconFeature":
        label = selected_model.predict(LexiconFeaturesValue)
    elif featureName == "top10_frequency":
        label = selected_model.predict(Top10FrequencyFeaturesValue)
    elif featureName == "SyntacticFeatures":
        label = selected_model.predict(SyntacticFeaturesValue)
    else:
        label = selected_model.predict(All_feature)
    # unseenData,selectedmodel_url,dName,fName
    print(label)
    newLabel = []

    typeC = url[-10:-4]

    for i in range(0,len(label)):
        if typeC == "binary":
            if label[i] == 0:
                newLabel.append( "non_damage")
            elif label[i] == 1:
                newLabel.append( "damage")
                
        elif typeC == "_multi":
            if label[i] == 0:
                newLabel.append( "non_damage")
            elif label[i] == 1:
                newLabel.append( "human_damage")
            elif label[i] == 2:
                newLabel.append( "infrastructure_damage")

    print(newLabel)
    return jsonify(newLabel)

@app.route("/modelMultiTest", methods=['GET', 'POST'])
def modelMultiTest():
    unseenData = request.json['unseenData']
    url = request.json['selectedmodel_url']
    disName = request.json['dName']
    featureName = request.json['fName']

    selected_model = pickle.load(open(url, 'rb'))
    
    newLabel = ""
    label = []
    typeC = url[-10:-4]

    if typeC == "binary":
        if label[0] == 0:
            newLabel = "non_damage"
        elif label[0] == 1:
            newLabel = "damage"
            
    elif typeC == "_multi":
        if label[0] == 0:
            newLabel = "non_damage"
        elif label[0] == 1:
            newLabel = "human_damage"
        elif label[0] == 2:
            newLabel = "infrastructure_damage"

    print(newLabel)
    return jsonify(newLabel)

if __name__ == "__main__":
    app.run(debug=True)
