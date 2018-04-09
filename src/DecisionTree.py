'''
Created on 2018年4月5日

@author: WZD
'''
from sklearn import tree
#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
#from IPython.display import Image  
import pydotplus 
#from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2,RFE,SelectFromModel

if __name__ == '__main__':
    '''
    data preprocessing
    '''
    data = pd.read_csv('./Credit.csv',index_col=0)
    data_no_NaN = data.dropna(axis = 0)
    data_X = data_no_NaN.drop(labels=['Label','Loanaccount'],axis=1)  
     
    feature_name = data_X.columns.values
    #print(feature_name)
    #convert non-numerical features into numerical features
    for string in ['GENDER','MARITAL_STATUS','LOANTYPE','PAYMENT_TYPE']:
        mapping = {label:idx+1 for idx,label in enumerate(set(data_X[string]))}
        data_X[string] = data_X[string].map(mapping)
        
    #print(data_X.shape) 
    data_y = data_no_NaN.values[:,0].astype(int)
    #print(data_y.shape)
   
    #print(np.sum([i==1 for i in data_y]))
    #===========================================================================
    # note:data needn't normalizer when we using decision tree method
    #===========================================================================
    '''
    split training and test sets
    '''
    trainX,testX,trainY,testY = train_test_split(data_X.values,data_y,test_size=0.33,random_state=42)
    #train model
    model_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6, presort=True)                                    
    model_tree.fit(trainX,trainY)
    #predict
    y_pre = model_tree.predict(testX)
    y_pro =  model_tree.predict_proba(testX) #two-dimensional variable. shape=[num_sample,2]
    y_prob = [1-y_pro[i,0] if y_pro[i,0]>y_pro[i,1] else y_pro[i,1] for i in range(y_pro.shape[0])]
    
    #evaluation
    tn,fp,fn,tp = confusion_matrix(y_true=testY, y_pred=y_pre).ravel()
    print('准确率：',(tp+tn)/(tn+fp+fn+tp))
    print('查全率：',tp/(tp+fn))
    print('查准率：',tp/(tp+fp))
    
    get_ks = lambda y_pred,y_true: ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic      
    print('ks:',get_ks(y_pre,testY))
    
    #plot ROC
    fpr,tpr,thresholds = roc_curve(y_true=testY, y_score=y_prob)
    roc_auc = auc(fpr,tpr)  
    plt.title('Receiver Operating Characteristic')  
    plt.plot(fpr,tpr,'b',label='AUC = %0.2f'% roc_auc)  
    plt.legend(loc='lower right')  
    plt.plot([0,1],[0,1],'r--')      
    plt.ylabel('True Positive Rate')  
    plt.xlabel('False Positive Rate')  
    plt.show()   
    
    #Somer’s D concordance statistics
    pr_0 = []
    pr_1 = []    
    for i in range (testY.shape[0]):
        if testY[i]==0:
            pr_0.append(y_prob[i])
        else:
            pr_1.append(y_prob[i])            
    score = 0.0
    for i in range(len(pr_1)):
        for j in range(len(pr_0)):
            if pr_1[i]>pr_0[j]:
                score += 1
            else:
                score +=-1
    print('Somer’s D concordance statistics:',score/(len(pr_1)*len(pr_0)))
    
    #plot decision tree model
    dot_data = tree.export_graphviz(model_tree,out_file=None,feature_names=feature_name,class_names=['0','1'],filled=True, rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)     
    graph.write_pdf("F:/tree.pdf") 
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    