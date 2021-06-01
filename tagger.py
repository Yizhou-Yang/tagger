# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import re
import math
import numpy as np

#this is vanilla viberti taught in lecture
#take log for the probability before nomalizing, and preload all probablilies with 0.01 to avoid all zero columns.
#this implementation currently does not use numpy... so it has efficiency issues. But it runs within a minute on my machine, so hopefully it isnt too slow...
#reads sentences, and does tagging for every .!;? (which must be PUN) to mitigate some efficiency issues.
#preloads some symbols that are repeatly misclassified
#have an accuracy of around 80%... a bit far from 90% but this is best I can do(already late)

#A  is transition matrix, B is emission matrix,P is the initial probability 
#assume there are no more than 200000 distinct words in both training and test.

taglist = ["AJ0","AJC","AJS","AT0","AV0","AVP","AVQ","CJC","CJS","CJT","CRD","DPS","DT0","DTQ","EX0","ITJ","NN0","NN1","NN2","NP0","ORD","PNI","PNP","PNQ","PNX","POS","PRF","PRP","PUL","PUN","PUQ","PUR","TO0","UNC","VBB","VBD","VBG","VBI","VBN","VBZ","VDB","VDD","VDG","VDI","VDN","VDZ","VHB","VHD","VHG","VHI","VHN","VHZ","VM0","VVB","VVD","VVG","VVI","VVN","VVZ","XX0","ZZ0","AJ0-NN1","AJ0-VVD","AJ0-VVG","AJ0-VVN","AV0-AJ0","AVP-PRP","AVQ-CJS","CJS-AVQ","CJS-PRP","CJT-DT0","CRD-PNI","DT0-CJT","NN1-AJ0","NN1-NP0","NN1-VVB","NN1-VVG","NN2-VVZ","NP0-NN1","PNI-CRD","PRP-AVP","PRP-CJS","VVB-NN1","VVD-AJ0","VVD-VVN","VVG-AJ0","VVG-NN1","VVN-AJ0","VVN-VVD","VVZ-NN2","AJ0-AV0"]
tagdict = {}
reversedict = {}
worddict = {}
#preloaded tags, to be built manually
preload = {"a":"AT0","that":"CJT","of":"PRF","at":"PRP","the":"AT0",",":"PUN","had":"VHD","there":"EX0","was":"VBD","he":"PNP","with":"PRP","to":"TO0","both":"AV0","all":"DT0","around":"AVP","on":"PRP","him":"PNP","for":"CJS","now":"AV0","make":"VVI","before":"AV0","about":"PRP","one":"CRD","it":"PNP"}

A = []
B = []
P = []
countP = 0
countA = []
countB = []
numwords = 0

prob_trellis = []
path_trellis = []

def train(training_file):
    global countP
    global numwords
    rd = open(training_file,"r",errors='replace')
    outList = rd.readlines()
    last = None
    for line in outList:
        split = line.split()
        word = split[0]
        tag = split[2]
        if(len(split)>2):
            split = line.split(" : ")
            word = split[0]
            tag = split[1].split()[0]
        #update P
        #print(tag)
        i = tagdict.get(tag)
        P[i]+= 1
        countP+=1
        #update A
        if(last!=None):
            A[tagdict.get(last)][tagdict.get(tag)]+=1
            countA[tagdict.get(last)]+=1
        last = tag
        #update B
        wordindex = worddict.get(word,-1)
        if(wordindex==-1):
            worddict.update({word:numwords})
            wordindex = numwords
            numwords+=1
        countB[tagdict.get(tag)]+=1
        B[tagdict.get(tag)][wordindex]+=1

#find the most probable word lists
#path_trellis is an array of numbers,translate it into word tags. 
def findx(s,num,obs):
    maxprob = 0
    maxindex = -1
    for x in range(len(taglist)):
        prob = prob_trellis[x][num-1]*A[x][s]*B[s][obs]
        if(prob>maxprob):
            maxprob = prob
            maxindex = x
    return maxindex

 
def findmax(prob_trellis,index):
    maxprob = 0
    maxindex = 0
    for x in range(len(taglist)):
        prob = prob_trellis[x][index]
        if(prob>=maxprob):
            maxprob = prob
            maxindex = x
    return maxindex

#clean a,b,p
def clean():
    #total = []
    for i in range(len(P)):
        P[i] = P[i]/countP

    for i in range(len(tagdict)):
        sumB = 0
        for j in range(len(tagdict)):
            if(countA[i]==0):
                A[i][j]=0.000001
                continue
            A[i][j] = A[i][j]/countA[i]

        for j in range(len(worddict)):
            if(countB[i]==0):
                B[i][j]=0.000001
                continue
            B[i][j] = B[i][j]/countB[i]

    #print(total)
    
def v_sentence(sentence,wr,punctuation):

    default = worddict.get(sentence[0])
    if default == None:
        default = 19

    for s in range(len(taglist)):
        #print(P[s] * B[s][worddict.get(outList[0],-1)])
        #print(counttag[s])
        prob_trellis[s][0] = P[s] * B[s][default]
        path_trellis[s][0] = [s]
        #handle never-before-seen words
        #print(prob_trellis[s][0])
    #for s in range(len(taglist)):
    #    print(B[s][worddict.get(outList[0])])

    # o is the item number, obs is the observation
    for num in range(1,len(sentence)):
        #if it is one of our preloads

        obs = worddict.get(sentence[num])
    

        if(obs==None):
            obs = default

        total = 0
        for s in range(len(taglist)):

            if preload.get(sentence[num-1])!=None:
                tag = preload.get(sentence[num-1])
                x = tagdict.get(tag)
            
            else:
                x = findx(s,num,obs)

            if(x==-1):
                print(sentence)
                print(num)
                exit()

            #every round, not every state can be reached by some other state
            prob_trellis[s][num] = prob_trellis[x][num-1]*A[x][s]*B[s][obs]
            total += prob_trellis[s][num]
            new_path = list(path_trellis[x][num-1])
            new_path.append(s)
            path_trellis[s][num] = new_path
        #nomalize prob_trellis[s][num]
        for s in range(len(taglist)):
            prob_trellis[s][num] = prob_trellis[s][num]/total

        #for s in range(len(taglist)):
         #   print(path_trellis[s][num])

    maxnum = findmax(prob_trellis,len(sentence)-1)
    writesentence(path_trellis[maxnum][len(sentence)-1],wr,sentence,punctuation)

def viberti(test_file,output_file):
    global numwords
    rd = open(test_file,"r")
    wr = open(output_file,"w")
    outList = rd.readlines()
    #clean the outlist
    for i in range(len(outList)):
        outList[i] = outList[i][:len(outList[i])-1]
    
    #print(outList)

    sentence = []
    for item in outList:
        if item == '.' or item == '!' or item == '?' or item == ';':
            v_sentence(sentence,wr,item)
            sentence = []
        else:
            sentence.append(item)


#write the calculated word list to output
def writesentence(output,wr,sentence,punctuation):
    for i in range(len(output)):
        wr.write(sentence[i]+' : '+reversedict.get(output[i])+'\n')
    wr.write(punctuation+' : '+'PUN'+'\n')

def word(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    #
    # YOUR IMPLEMENTATION GOES HERE
    #
    for i in range(len(taglist)):
        tagdict.update({taglist[i]:i})
        reversedict.update({i:taglist[i]})

    for i in range(len(taglist)):
        P.append(0.01)
        countA.append(0.0)
        countB.append(0.0)
        temp = []
        for j in range(len(taglist)):
            temp.append(0.01)
        A.append(temp)

        temp = []
        for j in range(200000):
            temp.append(0.01)
        B.append(temp)
        prob_trellis.append(temp)

        temp = []
        for j in range(200000):
            temp.append([])
        path_trellis.append(temp)

    #print(len(B))
    #print(len(B[0]))

    for training_file in training_list:
        train(training_file)

    clean()
    viberti(test_file,output_file)

    


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Ouptut file: " + output_file)

    # Start the training and tagging operation.
    word (training_list, test_file, output_file)