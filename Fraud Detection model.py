#import random
#for i in range(30):
#    print('%.4g'% random.uniform(360,560))
    
    
import random
for i in range(10000):
    print (0)
    



https://github.com/aqibsaeed/Anomaly-Detection


http://aqibsaeed.github.io/2016-07-17-anomaly-detection/




import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'Ubuntu'
#plt.rcParams['font.monospace'] = 'Ubuntu Mono'
#plt.rcParams['font.size'] = 12
#plt.rcParams['axes.labelsize'] = 11
#plt.rcParams['axes.labelweight'] = 'bold'
#plt.rcParams['axes.titlesize'] = 12
#plt.rcParams['xtick.labelsize'] = 9
#plt.rcParams['ytick.labelsize'] = 9
#plt.rcParams['legend.fontsize'] = 11
#plt.rcParams['figure.titlesize'] = 13

from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score


def read_dataset(filePath,delimiter=','):
    return genfromtxt(filePath, delimiter=delimiter)

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma
    
def multivariateGaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)

def selectThresholdByCV(probs,gt):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs),max(probs),stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon) 
        f = f1_score(gt, predictions,average='binary')
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon
    
    return best_f1, best_epsilon
    
    
tr_data = read_dataset('/Users/uma_sahoo/Desktop/Hack/tr_server_data.csv') 
cv_data = read_dataset('/Users/uma_sahoo/Desktop/Hack/cv_server_data.csv') 
gt_data = read_dataset('/Users/uma_sahoo/Desktop/Hack/gt_server_data.csv')

n_training_samples = tr_data.shape[0]
n_dim = tr_data.shape[1]

print('Number of datapoints in training set: %d' % n_training_samples)
print('Number of dimensions/features: %d' % n_dim)


print(tr_data[1:5,:])

plt.xlabel('Dollar amount (USD)')
plt.ylabel('No.Of failures ')
plt.plot(tr_data[:,0],tr_data[:,1],'bx')
plt.show()


mu, sigma = estimateGaussian(tr_data)
p = multivariateGaussian(tr_data,mu,sigma)


#selecting optimal value of epsilon using cross validation
p_cv = multivariateGaussian(cv_data,mu,sigma)
fscore, ep = selectThresholdByCV(p_cv,gt_data)
#print(fscore, ep)
p_test = multivariateGaussian([[320,3]],mu,sigma)
if (p_test < ep):
    print('Anomaly')
else:
    print('Not an anomaly')

#selecting outlier datapoints 
outliers = np.asarray(np.where(p < ep))

plt.figure()
plt.xlabel('Dollar amount (USD)')
plt.ylabel('No.Of failures')
plt.plot(tr_data[:,0],tr_data[:,1],'bx')
plt.plot(tr_data[outliers,0],tr_data[outliers,1],'ro')
plt.show()



from sklearn import svm
# use the same dataset
tr_data = read_dataset('/Users/uma_sahoo/Desktop/Hack/tr_server_data.csv')
clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
clf.fit(tr_data)
pred = clf.predict(tr_data)

# inliers are labeled 1, outliers are labeled -1
normal = tr_data[pred == 1]
abnormal = tr_data[pred == -1]
plt.figure()
plt.plot(normal[:,0],normal[:,1],'bx')
plt.plot(abnormal[:,0],abnormal[:,1],'ro')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()
print(clf.predict([[330,1]]))
