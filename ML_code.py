"""
Author: Dr Sandip Bhattacharya (sbhattac@tcd.ie)
Python3 or higher should be used 
Format of the input file pockets_zT.dat is: same HHname E_LW EGW E_WW zT(ptype) bandgap (from my paper J. Mater. Chem. C, 2016,4, 11261-11268)

The code uses Machine Learning (ML) methods to achieve the following:
a) Evaluate the feature importances to obtain physical insights into which features are required for good materials properties.
b) Build a ML model that can predict the necessary values of the electronic properties required for a high zT

The code is demonstrated for random forest regression (using sklearn). However, neural networks or SVM methods can easily be implemented as illustrated within the code.

For more details on the purpose of this project please refer to the README file

"""
import pylab, random,pickle,re
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn import *
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE,RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt
import scipy
import matplotlib.pyplot as plt

zT0=1.4

def minkowskiDist(v1, v2, p):
    """Assumes v1 and v2 are equal-length arrays of numbers
       Returns Minkowski distance of order p between v1 and v2"""
    dist = 0.0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i])**p
    return dist**(1/p)


class HHs(object):
    featureNames = ('E_LW', 'E_GW', 'gap')
    def __init__(self, E_LW, E_GW, gap, highzT, name,zT):
        self.name = name
        self.zT = zT
        self.featureVec = [E_LW, E_GW, gap]
        self.label = highzT
    def distance(self, other):
        return minkowskiDist(self.featureVec, other.featureVec, 2)
    def getE_LW(self):
        return self.featureVec[0]
    def getE_GW(self):
        return self.featureVec[1]
    def getgap(self):
        return self.featureVec[2]
    def getName(self):
        return self.name
    def getzT(self):
        return self.zT
    def getFeatures(self):
        return self.featureVec[:]
    def getLabel(self):
        return self.label
        
def getHHData(fname,zT0):
    data = {}
    data['E_LW'], data['E_GW'], data['gap'], data['highzT'] = [], [], [], []
    data['zT'],data['name'] = [], []
    f = open(fname)
    lines = f.readlines()
    f.close()

    for line in lines:
        tmp=line.split("\n")[0].split(" ")
        data['name'].append(tmp[0])
        data['zT'].append(float(tmp[-3]))
        if float(float(tmp[-3]))>zT0:
           data['highzT'].append(1)
        else:
           data['highzT'].append(0)
        data['E_GW'].append(float(tmp[2]))
        data['E_LW'].append(float(tmp[1]))
        pip=float(tmp[-2])
        if pip<0:
           data['gap'].append(0.0)
        else:
           data['gap'].append(pip)
    return data
                
def buildHHExamples(fileName,zT0):
    data = getHHData(fileName,zT0)
    examples = []
    for i in range(len(data['name'])):
        p = HHs(data['E_LW'][i], data['E_GW'][i],data['gap'][i], data['highzT'][i],data['name'][i],data['zT'][i])
        examples.append(p)
    #print('Finished processing', len(examples), 'HHs\n')    
    return examples
    
def findNearest(name, exampleSet, metric):
    for e in exampleSet:
        if e.getName() == name:
            example = e
            break
    curDist = None
    for e in exampleSet:
        if e.getName() != name:
            if curDist == None or metric(example, e) < curDist:
                nearest = e
                curDist = metric(example, nearest)
    return nearest

def accuracy(truePos, falsePos, trueNeg, falseNeg):
    numerator = truePos + trueNeg
    denominator = truePos + trueNeg + falsePos + falseNeg
    return numerator/denominator

def sensitivity(truePos, falseNeg):
    try:
        return float(truePos)/float(truePos + falseNeg)
    except ZeroDivisionError:
        return float('nan')
    
def specificity(trueNeg, falsePos):
    try:
        return float(trueNeg)/float(trueNeg + falsePos)
    except ZeroDivisionError:
        return float('nan')
    
def posPredVal(truePos, falsePos):
    try:
        return truePos/(truePos + falsePos)
    except ZeroDivisionError:
        return float('nan')
    
def negPredVal(trueNeg, falseNeg):
    try:
        return trueNeg/(trueNeg + falseNeg)
    except ZeroDivisionError:
        return float('nan')
       
def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint = True):
    accur = accuracy(truePos, falsePos, trueNeg, falseNeg)
    sens = sensitivity(truePos, falseNeg)
    spec = specificity(trueNeg, falsePos)
    ppv = posPredVal(truePos, falsePos)
    if toPrint:
        print(' Accuracy =', round(accur, 3))
        print(' Sensitivity =', round(sens, 3))
        print(' Specificity =', round(spec, 3))
        print(' Pos. Pred. Val. =', round(ppv, 3))
    return (accur, sens, spec, ppv)
   
def findKNearest(example, exampleSet, k):
    kNearest, distances = [], []
    #Build lists containing first k examples and their distances
    for i in range(k):
        kNearest.append(exampleSet[i])
        distances.append(example.distance(exampleSet[i]))
    maxDist = max(distances) #Get maximum distance
    #Look at examples not yet considered
    for e in exampleSet[k:]:
        dist = example.distance(e)
        if dist < maxDist:
            #replace farther neighbor by this one
            maxIndex = distances.index(maxDist)
            kNearest[maxIndex] = e
            distances[maxIndex] = dist
            maxDist = max(distances)      
    return kNearest, distances
    
def KNearestClassify(training, testSet, label, k):
    """Assumes training & testSet lists of examples, k an int
       Predicts whether each example in testSet has label
       Returns number of true positives, false positives,
          true negatives, and false negatives"""
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for testCase in testSet:
        nearest, distances = findKNearest(testCase, training, k)
        #conduct vote
        numMatch = 0
        for i in range(len(nearest)):
            if nearest[i].getLabel() == label:
                numMatch += 1
        if numMatch > k//2: #guess label
            if testCase.getLabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else: #guess not label
            if testCase.getLabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg

def leaveOneOut(examples, method, toPrint = True):
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for i in range(len(examples)):
        testCase = examples[i]
        trainingData = examples[0:i] + examples[i+1:]
        results = method(trainingData, [testCase])
        truePos += results[0]
        falsePos += results[1]
        trueNeg += results[2]
        falseNeg += results[3]
    if toPrint:
        getStats(truePos, falsePos, trueNeg, falseNeg)
    return truePos, falsePos, trueNeg, falseNeg

def split80_20(examples):
    sampleIndices = random.sample(range(len(examples)),
                                  len(examples)//5)
    trainingSet, testSet = [], []
    for i in range(len(examples)):
        if i in sampleIndices:
            testSet.append(examples[i])
        else:
            trainingSet.append(examples[i])
    return trainingSet, testSet
    
def randomSplits(examples, method, numSplits, toPrint = True):
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    random.seed(0)
    for t in range(numSplits):
        trainingSet, testSet = split80_20(examples)
        results = method(trainingSet, testSet)
        truePos += results[0]
        falsePos += results[1]
        trueNeg += results[2]
        falseNeg += results[3]
    getStats(truePos/numSplits, falsePos/numSplits,
             trueNeg/numSplits, falseNeg/numSplits, toPrint)
    return truePos/numSplits, falsePos/numSplits,\
             trueNeg/numSplits, falseNeg/numSplits
    
#knn = lambda training, testSet:KNearestClassify(training, testSet, 1, 3)  #1 (>zT0)
#numSplits = 10

#true_count=0
#for i in range(0,len(examples)):
#    if examples[i].getLabel()==1:
#       true_count+=1

#print('Accuracy without any ML',round(true_count/len(examples),3))


#print('\n Average of', numSplits,'80/20 splits using KNN (k=3)')
#truePos, falsePos, trueNeg, falseNeg =randomSplits(examples, knn, numSplits)

#print('Average of LOO testing using KNN (k=3)')
#truePos, falsePos, trueNeg, falseNeg =leaveOneOut(examples, knn)

def buildModel(examples, toPrint = True):
    featureVecs, labels = [],[]
    for e in examples:
        featureVecs.append(e.getFeatures())
        labels.append(e.getLabel())
    LogisticRegression = sklearn.linear_model.LogisticRegression
    model = LogisticRegression().fit(featureVecs, labels)
    if toPrint:
        print('model.classes_ =', model.classes_)
        for i in range(len(model.coef_)):
            print('For label', model.classes_[1])
            for j in range(len(model.coef_[0])):
                print('   ', HHs.featureNames[j], '=',model.coef_[0][j])
    return model

def buildModel_nn(examples, toPrint = True):
    featureVecs, labels = [],[]
    for e in examples:
        featureVecs.append(e.getFeatures())
        labels.append(e.getLabel())
    clf = MLPClassifier#(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    model=clf().fit(featureVecs, labels)
    #clf = MLPClassifier(hidden_layer_sizes=(1200,),max_iter=400)
    #model=clf.fit(featureVecs, labels)

    if toPrint:
        print('model.classes_ =', model.classes_)
        for i in range(len(model.coefs_)):
            print('For label', model.classes_[1])
            for j in range(len(model.coefs_[0])):
                print('   ', HHs.featureNames[j], '=',model.coefs_[0][j])
    return model

def relevant_vectors(examples):
    vec_ELW,vec_EGW,vec_gap,vec_zT,vec_Name=[],[],[],[],[]
    for e in examples:
        vec_ELW.append(e.getE_LW())
        vec_EGW.append(e.getE_GW())
        vec_gap.append(e.getgap())
        vec_zT.append(e.getzT()) 
        vec_Name.append(e.getName())
    return vec_ELW,vec_EGW,vec_gap,vec_zT,vec_Name

def get_predicted_zT(model,vec_ELW,vec_EGW,vec_gap):
    pred_zT=[]
    for i in range(len(vec_ELW)):
        pred_zT.append(model.predict([[vec_ELW[i],vec_EGW[i],vec_gap[i]]])[0])
    return pred_zT

def buildModel_rf(examples, toPrint = True):
    featureVecs, labels = [],[]
    for e in examples:
        featureVecs.append(e.getFeatures())
        #labels.append(e.getLabel())
        labels.append(e.getzT())   #for RandomForestRegressor
    #clf = sklearn.ensemble.RandomForestClassifier(max_features="log2",n_estimators=1000)
    clf = sklearn.ensemble.RandomForestRegressor(max_features=0.32,n_estimators=1000,max_depth=30,bootstrap=True,random_state=123,oob_score=True)
    mod=clf.fit(featureVecs, labels)
    importances = mod.feature_importances_
    std = np.std([tree.feature_importances_ for tree in mod.estimators_],axis=0)
    av = np.average([tree.feature_importances_ for tree in mod.estimators_],axis=0)
    #print("All tree.feature_importances_ Value i.e. Avg (Std):")
    #for i in range(len(std)):
    #    print(r"%d. %s: %f (%f)"%(i,HHs.featureNames[i],av[i],std[i]))


    #print("\n Actual Feature importances") 
    #indices = np.argsort(importances)[::-1]
    #for f in range(len(importances)):
    #    print ("{0}. {1} ({2})".format (f + 1, HHs.featureNames[indices[f]], importances[indices[f]]))

    if toPrint:
       std = np.std([tree.feature_importances_ for tree in mod.estimators_],axis=0)
       indices = np.argsort(importances)[::-1]
       print("Feature importances (model.feature_importances_) Std:%s:"%std)
       for f in range(len(importances)):
           print ("{0}. {1} ({2})".format (f + 1, HHs.featureNames[indices[f]], importances[indices[f]]))

    #rfe = RFE(estimator=classifier, n_features_to_select=15, step=1)
    #model = rfe.fit(featureVecs, labels)
    if toPrint:
       rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(2),scoring='accuracy')
       model=rfecv.fit(featureVecs, labels)
       if toPrint:
          print("Optimal number of features : %d" % model.n_features_)
          print("\n model.ranking_:")
          for i in range(len(model.ranking_)):
              if model.ranking_[i] == 1:
                 print (HHs.featureNames[i]) 
       return model,importances
    if toPrint==False:
       return mod,importances  #careful of model returned

def applyModel(model, testSet, label, prob = 0.5):
    testFeatureVecs = [e.getFeatures() for e in testSet]
    probs = model.predict_proba(testFeatureVecs)
    #print(testFeatureVecs) 
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for i in range(len(probs)):
        if probs[i][1] > prob:
            if testSet[i].getLabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else:
            if testSet[i].getLabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg

def lr(trainingData, testData, prob = 0.5):
    model = buildModel(trainingData, False)
    results = applyModel(model, testData, 1, prob)  #1 (>zT0)
    return results

def neural_network(trainingData, testData, prob = 0.5):
    model = buildModel_nn(trainingData, False)
    results = applyModel(model, testData, 1, prob)  #1 (>zT0)
    return results

def random_forest(trainingData, testData, prob = 0.5):
    model = buildModel_rf(trainingData, False)[0]
    results = applyModel(model, testData, 1, prob)  #1 (>zT0)
    return results


##Delete this, rethink this part
def plotanalysis(model,zT0,gap,nn=0):
    var=np.linspace(-0.7,0.7,100)
    E_LW0=[]
    E_GW0=[]
    prob=[]

    for i in var:
        for j in var:
            E_LW0.append(i),E_GW0.append(j)
            test=np.array([i,j,gap])
            prob.append(model.predict_proba(test.reshape(1, -1))[0][1])
    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(E_LW0, E_GW0, c=prob, vmin=0.0, vmax=1.0, s=35, cmap=cm,lw = 0)
    plt.colorbar(sc)
    plt.axvline(x=0.0,color='k',linewidth=0.7)
    plt.axhline(y=0.0,color='k',linewidth=0.7)
    plt.xlim(-0.7,0.7)
    plt.ylim(-0.7,0.7)
    plt.xlabel(r'$\Delta E_{\mathrm{LW}}$',color='k',fontsize=20)
    plt.ylabel(r'$\Delta E_{\mathrm{GW}}$',color='k',fontsize=20)
    plt.title("Prob. of zT>%s for gap=%s"%(zT0,gap))
    plt.tight_layout()
    if nn==0:
       plt.savefig("probplot_%s.pdf"%gap)
    else:
       plt.savefig("probplotnn_%s.pdf"%gap)
    plt.close()
    #plt.show()

def plot_correlations(vec_dftzT,predicted_zT,model):
    fig=plt.figure()
    ax=plt.subplot(1, 1, 1)
    rmse = sqrt(mean_squared_error(vec_dftzT, predicted_zT))
    mea=mean_absolute_error(vec_dftzT, predicted_zT)
    spearmanr=scipy.stats.spearmanr(vec_dftzT, predicted_zT)[0]
    pearsons=scipy.stats.pearsonr(vec_dftzT, predicted_zT)[0]
    textstr = '$\mathrm{RMSE}=%.2f$\n$\mathrm{MAE}=%.2f$\n$\mathrm{Spearman}=%.2f$\n$\mathrm{Pearson}=%.2f$\n$\mathrm{OOB score}=%.2f$'%(rmse,mea, spearmanr,pearsons,model.oob_score_)
    plt.scatter(vec_dftzT, predicted_zT, s=60,c='r',alpha=0.5,marker=r'o')
    plt.plot([0,np.amax(vec_dftzT)], [0,np.amax(vec_dftzT)],'k-',linewidth=1)
    plt.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top')
    plt.xlabel(r'zT(DFT)',color='k',fontsize=20)
    plt.ylabel(r'zT(ML)',color='k',fontsize=20)
    plt.xlim(-0.02,np.amax(vec_dftzT))
    plt.ylim(-0.02,np.amax(vec_dftzT))
    plt.tight_layout()
    plt.savefig("correlation.pdf")
    plt.close()
    return rmse,mea,spearmanr,pearsons


#Look at weights
numSplits = 10

#trainingSet, testSet = split80_20(examples)
#model_rf,importances_rf = buildModel_rf(trainingSet, True)
#model_rf,importances_rf = buildModel_rf(examples, True) #complete
zT0=1.2
examples = buildHHExamples('pockets_zT.dat',zT0)
model,rf_importances=buildModel_rf(examples,False)



vec_ELW,vec_EGW,vec_gap,vec_dftzT,vec_Name=relevant_vectors(examples)
predicted_zT=get_predicted_zT(model,vec_ELW,vec_EGW,vec_gap)


rmse,mea,spearmanr,pearsons=plot_correlations(vec_dftzT,predicted_zT,model)


stop #remove this to get Random Forest Feature importances

pattern_imp={}
interv=np.linspace(0.7,1.7,10)

for zT0 in interv:
    pattern_imp[zT0]={}
    #for feature in HHs.featureNames:
    #    pattern_imp[zT0][feature]={}

for zT0 in interv:
    examples = buildHHExamples('pockets_zT.dat',zT0)
    model_rf,importances_rf = buildModel_rf(examples, False)
    for j,feature in enumerate(HHs.featureNames):
        pattern_imp[zT0][feature]=importances_rf[j]



#HHs.featureNames



fig=plt.figure()
ax=plt.subplot(1, 1, 1)
select_features=['E_LW', 'E_GW', 'gap']

clist=["k-","b-","r-","c-","g-","m-","k--","b--","r--","c--","g--","m--","y-","y--","k.","g."]

for j,feature in enumerate(select_features):
    fimp=[]
    for zT0 in interv:
        fimp.append(pattern_imp[zT0][feature])
    ax.plot(interv,fimp,clist[j],linewidth=2,label=feature)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.988, box.height])
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    del fimp
plt.legend(loc='best')
plt.xlim(0.69,1.71)
#plt.ylim(0.015,0.18)
plt.xlabel(r'$zT_{\mathrm{cut-off}}$ ', color='k',fontsize=20)
plt.ylabel(r'Feature Imps.', color='k',fontsize=20)
plt.title("Random Forest with Bootstrapping")
plt.savefig('graph.pdf')
plt.close()




