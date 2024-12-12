import numpy as np
from sklearn.svm import SVC
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import warnings
import time
from joblib import Parallel, delayed
import multiprocessing
import sys
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
pd.options.mode.chained_assignment = None

Noise = 'No'  ## no noise variables
if Noise == 'No':
    p = 4
else:
    p = 4 + Noise

SIZE = 10000

K = 1  ## 1:K Matching
b = 6 ## stratum coef
tolerence = 0.01
fold = 3 ## number of cv fold

## create testing set
r1 = np.random.RandomState(1)
Val_X = r1.standard_normal((SIZE,p))


Val_X_strat = Val_X[:,0]

z = 1 + 2 * Val_X[:,0] + Val_X[:,1]
pr = 1 / (1+np.exp(-z))

r2 = np.random.RandomState(2)
Val_A = r2.binomial(1,pr) * 2 - 1


r3 = np.random.RandomState(3)
Val_N = r3.standard_normal((SIZE))


Val_Q = 2 * Val_X[:,2] - Val_X[:,3] + (Val_X[:,0]  - Val_X[:,1]) * Val_A \
        + b * np.sign(Val_X_strat) + Val_N

Val_X_True = np.sign(Val_X[:,0]  - Val_X[:,1])
Potential = 2 * Val_X[:,2] - Val_X[:,3] + (Val_X[:,0]  - Val_X[:,1]) * Val_X_True \
            + b * np.sign(Val_X_strat) + Val_N



########################################################################################
################################# Simulations ##########################################
########################################################################################

num_cores = multiprocessing.cpu_count()
from sklearn.model_selection import KFold

### custom scoring function
def my_scorer(est, X, y):
    pred = est.predict(X)
    A = np.array(X.index)
    Q = np.array(y.index)
    logist = linear_model.LogisticRegression()
    logist.fit(X, A)
    prob = logist.predict_proba(X)[:, 1]
    PS = prob * A + (1 - A) / 2

    Q0 = Q[pred == A]
    PS0 = PS[pred == A]
    val = np.sum(Q0 / PS0) / np.sum(1 / PS0)
    return val


#########################
def Learning(t):

    r = np.random.RandomState(t * t)
    X = r.standard_normal((sample, p))
    N = r.standard_normal((sample))

    X_strat = X[:, 0]
    z1 = 1 + 2 * X[:, 0] + X[:, 1]
    pr1 = 1 / (1 + np.exp(-z1))
    A = r.binomial(1, pr1) * 2 - 1

    Q = 2 * X[:, 2] - X[:, 3] + (X[:, 0] - X[:, 1]) * A + b * np.sign(X_strat) + N

    Cs = np.array([0.1, 0.5, 1, 2, 5, 10, 100])
    gammas = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10])
    kernels = ['linear', 'rbf']

    ######################### Q-Learning ########################
    #############################################################
    Qreg = linear_model.LinearRegression(n_jobs=-1)

    logist0 = linear_model.LogisticRegression()
    logist0.fit(X[:, 0:p], A)
    prob_QL = logist0.predict_proba(X[:, 0:p])[:, 1]
    PS0 = prob_QL * A + (1 - A) / 2
    Weights_QL = 1 / PS0

    QL_Int = X[:, 0:p] * np.transpose(np.tile(A, (p, 1)))
    Qreg.fit(np.concatenate((X[:, 0:p], QL_Int, A.reshape(-1, 1)), axis=1), Q, sample_weight=Weights_QL)
    coeff = Qreg.coef_
    Pred_QL = np.sign(
        sum(np.transpose(coeff[p:(2 * p)] * np.array(Val_X[:, 0:p]))) + np.ones(SIZE) * coeff[2 * p])
    Potential_QL = 2 * Val_X[:, 2] - Val_X[:, 3] + (Val_X[:, 0] - Val_X[:, 1]) * Pred_QL + b * np.sign(Val_X_strat) + Val_N

    ######################### AOL ###############################
    #############################################################


    regr = linear_model.LinearRegression(n_jobs=-1)
    regr.fit(X[:, 0:p], Q)
    res = Q - regr.predict(X[:, 0:p])

    logist = linear_model.LogisticRegression()
    logist.fit(X[:, 0:p], A)
    prob_OWL = logist.predict_proba(X[:, 0:p])[:, 1]
    PS = prob_OWL * A + (1 - A) / 2

    OWL_new_label = A * np.sign(res)
    Weights_OWL = np.array(abs(res) / PS)

    X_ = pd.DataFrame(X)
    X_.index = A

    OWL_new_label_ = pd.Series(OWL_new_label)
    OWL_new_label_.index = Q

    clf = SVC(tol=tolerence)

    grid = GridSearchCV(estimator=clf, param_grid=dict(C=Cs, gamma=gammas, kernel=kernels), cv= KFold(n_splits=fold, shuffle=True, random_state=t), scoring=my_scorer,
                        fit_params={'sample_weight': Weights_OWL})
    grid.fit(X_.iloc[:, 0:p], OWL_new_label_)

    Pred_OWL = grid.predict(Val_X[:, 0:p])
    Potential_OWL = 2 * Val_X[:, 2] - Val_X[:, 3] + (Val_X[:, 0] - Val_X[:, 1]) * Pred_OWL + b * np.sign(Val_X_strat) + Val_N

    ###################### M-Learning ###########################
    #############################################################
    X1 = X[np.where(A == -1)]
    X2 = X[np.where(A == 1)]

    A1 = A[A == -1]
    A2 = A[A == 1]

    Q1 = Q[np.where(A == -1)]
    Q2 = Q[np.where(A == 1)]

    Q1_K = np.tile(Q1, K)
    Q2_K = np.tile(Q2, K)

    A1_K = np.tile(A1, K)
    A2_K = np.tile(A2, K)

    X1_K = np.tile(X1, (K, 1))
    X2_K = np.tile(X2, (K, 1))

    ## Match for X1
    X2_pair_value = []
    for i in range(len(X1)):
        sim = pairwise_distances(X1[i, 0:p].reshape(1, -1), X2[:, 0:p], metric='l2')
        ind = sim[0].argsort()[:K]
        X2_pair_value.append(Q2[ind])
    Q1_paired_array = np.asarray(np.transpose(X2_pair_value)).reshape(-1)

    ## Match for X2
    X1_pair_value = []
    for i in range(len(X2)):
        sim = pairwise_distances(X2[i, 0:p].reshape(1, -1), X1[:, 0:p], metric='l2')
        ind = sim[0].argsort()[:K]
        X1_pair_value.append(Q1[ind])
    Q2_paired_array = np.asarray(np.transpose(X1_pair_value)).reshape(-1)

    X_MatchO = np.concatenate((X1_K, X2_K), axis=0)

    DIFF = np.append((Q1_K - Q1_paired_array), (Q2_K - Q2_paired_array))
    Weights_MatchO = abs(DIFF)

    X_MatchO_new_label = np.append(A1_K, A2_K) * np.sign(DIFF)

    X_MatchO_ = pd.DataFrame(X_MatchO)
    X_MatchO_.index = np.append(A1_K, A2_K)

    X_MatchO_new_label_ = pd.Series(X_MatchO_new_label)
    X_MatchO_new_label_.index = np.append(Q1_K,Q2_K)

    clf = SVC(tol=tolerence)

    grid = GridSearchCV(estimator=clf, param_grid=dict(C=Cs, gamma=gammas, kernel=kernels), cv= KFold(n_splits=fold, shuffle=True, random_state=t), scoring=my_scorer,
                        fit_params={'sample_weight': Weights_MatchO})
    grid.fit(X_MatchO_.iloc[:, 0:p], X_MatchO_new_label_)

    Pred_MatchO = grid.predict(Val_X[:, 0:p])
    Potential_MatchO = 2 * Val_X[:, 2] - Val_X[:, 3] + (Val_X[:, 0] - Val_X[:, 1]) * Pred_MatchO + b * np.sign(Val_X_strat) + Val_N

    ###################### MLearning Stratified by Predictive Value ############################
    ############################################################################################

    reg = RandomForestRegressor(n_jobs=-1,random_state=t)
    reg.fit(X[:,0:p],Q)
    Q_pred = reg.predict(X[:,0:p])

    X_up = X[np.where(Q_pred >= np.percentile(Q_pred,50))]
    X_low = X[np.where(Q_pred < np.percentile(Q_pred,50))]

    A_up = A[np.where(Q_pred >= np.percentile(Q_pred,50))]
    A_low = A[np.where(Q_pred < np.percentile(Q_pred,50))]

    Q_up = Q[np.where(Q_pred >= np.percentile(Q_pred,50))]
    Q_low = Q[np.where(Q_pred < np.percentile(Q_pred,50))]


    X_quart = [X_up,  X_low]
    A_quart = [A_up,  A_low]
    Q_quart = [Q_up,  Q_low]


    X_MatchO_temp = []
    DIFF1 = []
    A_Final = []
    Q_Final = []

    for i in range(len(A_quart)):
        X1 = X_quart[i][np.where(A_quart[i] == -1)]
        X2 = X_quart[i][np.where(A_quart[i] == 1)]

        A1 = A_quart[i][A_quart[i] == -1]
        A2 = A_quart[i][A_quart[i] == 1]

        Q1 = Q_quart[i][np.where(A_quart[i] == -1)]
        Q2 = Q_quart[i][np.where(A_quart[i] == 1)]

        Q1_K = np.tile(Q1, K)
        Q2_K = np.tile(Q2, K)

        A1_K = np.tile(A1, K)
        A2_K = np.tile(A2, K)

        X1_K = np.tile(X1, (K, 1))
        X2_K = np.tile(X2, (K, 1))


        ## Match for X1
        X2_pair_value = []
        for i in range(len(X1)):
            sim = pairwise_distances(X1[i, 0:p].reshape(1, -1), X2[:, 0:p], metric='l2')
            ind = sim[0].argsort()[:K]
            if (K > 1 & len(ind) == 1):
                ind = np.tile(ind, K)
            X2_pair_value.append(Q2[ind])
        Q1_paired_array = np.asarray(np.transpose(X2_pair_value)).reshape(-1)

        ## Match for X2
        X1_pair_value = []
        for i in range(len(X2)):
            sim = pairwise_distances(X2[i, 0:p].reshape(1, -1), X1[:, 0:p], metric='l2')
            ind = sim[0].argsort()[:K]
            if (K>1 & len(ind) == 1):
                ind = np.tile(ind,K)
            X1_pair_value.append(Q1[ind])
        Q2_paired_array = np.asarray(np.transpose(X1_pair_value)).reshape(-1)

        A_temp = np.append(A1_K, A2_K)
        X_MatchO_temp.append(np.concatenate((X1_K, X2_K), axis=0))

        Q_temp = np.append(Q1_K,Q2_K)
        DIFF_temp = np.append((Q1_K - Q1_paired_array), (Q2_K - Q2_paired_array))
        A_Final = np.append(A_Final,A_temp)
        Q_Final = np.append(Q_Final, Q_temp)
        DIFF1 = np.append(DIFF1,DIFF_temp)


    X_MatchO1 = np.concatenate(X_MatchO_temp,axis=0)
    Weights_MatchO1 = abs(DIFF1)
    X_MatchO_new_label1 = np.asarray(A_Final) * np.sign(DIFF1)

    X_MatchO1_ = pd.DataFrame(X_MatchO1)
    X_MatchO1_.index = np.asarray(A_Final)

    X_MatchO_new_label1_ = pd.Series(X_MatchO_new_label1)
    X_MatchO_new_label1_.index = np.asarray(Q_Final)

    clf = SVC(tol=tolerence)

    grid1 = GridSearchCV(estimator=clf, param_grid=dict(C=Cs, gamma=gammas, kernel=kernels),  cv= KFold(n_splits=fold, shuffle=True, random_state=t), scoring=my_scorer,
                        fit_params={'sample_weight': Weights_MatchO1})
    grid1.fit(X_MatchO1_.iloc[:,0:p], X_MatchO_new_label1_)


    Pred_MatchO1 = grid1.predict(Val_X[:,0:p])
    Potential_MatchO1 = 2 * Val_X[:, 2] - Val_X[:, 3] + (Val_X[:, 0] - Val_X[:, 1]) * Pred_MatchO1 + b * np.sign( Val_X_strat) + Val_N

    return [[np.mean(Potential_QL),np.mean(Potential_OWL),np.mean(Potential_MatchO),np.mean(Potential_MatchO1)],
            [np.mean(Pred_QL == Val_X_True), np.mean(Pred_OWL == Val_X_True),np.mean(Pred_MatchO == Val_X_True),np.mean(Pred_MatchO1 == Val_X_True)]]


samples = [100,200,500,1000]

results = {}
iters =  100

start = time.time()
for sample in samples:
    results[str(sample)] = Parallel(n_jobs= (num_cores-1))(delayed(Learning)(t) for t in range(iters))


cost = time.time() - start

print("Time consumed: ", cost)


############################################### BOX PLOT ###################################################
############################################################################################################



############################################### Value #############################################
###################################################################################################

numSize= 4
sample_size = ['100', '200', '500', '1000']


data = [np.array(results['100'])[:, 0][:, 0], np.array(results['100'])[:, 0][:, 1], np.array(results['100'])[:, 0][:, 2], np.array(results['100'])[:, 0][:, 3],
        np.array(results['200'])[:, 0][:, 0], np.array(results['200'])[:, 0][:, 1], np.array(results['200'])[:, 0][:, 2], np.array(results['200'])[:, 0][:, 3],
        np.array(results['500'])[:, 0][:, 0], np.array(results['500'])[:, 0][:, 1], np.array(results['500'])[:, 0][:, 2], np.array(results['500'])[:, 0][:, 3],
        np.array(results['1000'])[:, 0][:, 0], np.array(results['1000'])[:, 0][:, 1], np.array(results['1000'])[:, 0][:, 2], np.array(results['1000'])[:, 0][:, 3]]

pos = 1 + np.array([1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19]) *.75
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.canvas.set_window_title('A Boxplot Example')
plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5,positions=pos)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')


ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)


ax1.set_axisbelow(True)
ax1.set_xlabel('Sample Size')
ax1.set_ylabel('Empirical Value')

boxColors = ['green','pink', 'royalblue','red']
numBoxes = numSize * 4

medians = list(range(numBoxes))
for i in range(numBoxes):
    box = bp['boxes'][i]
    boxX = []
    boxY = []

    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])

    boxCoords = list(zip(boxX, boxY))

    k = i % 4
    boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
    ax1.add_patch(boxPolygon)

    med = bp['medians'][i]
    medianX = []
    medianY = []

    for j in range(2):
        medianX.append(med.get_xdata()[j])
        medianY.append(med.get_ydata()[j])

        plt.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]

    plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
             color='w', marker='*', markeredgecolor='k')


ax1.set_xlim(0.5, numBoxes + 0.5)
top = 1.5
bottom = -.8
ax1.set_ylim(bottom, top)
xtickNames = plt.setp(ax1, xticklabels=np.repeat(sample_size, 4))
plt.setp(xtickNames, rotation=60, fontsize=10)


upperLabels = [str(np.round(s, 2)) for s in medians]
weights = ['bold', 'bold','bold','bold']
for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
    k = tick % 2
    ax1.text(pos[tick], top - (top*0.05), upperLabels[tick],
             horizontalalignment='center', size='x-small', weight=weights[k],
             color='black')

plt.figtext(0.80, 0.15, 'Q-Learning', weight='roman',
            backgroundcolor=boxColors[0], color='black',
            size='x-small')

plt.figtext(0.80, 0.115, 'AO-Learning ',
            backgroundcolor=boxColors[1], color='black', weight='roman',
            size='x-small')

plt.figtext(0.80, 0.08,   'M-Learning ',
            backgroundcolor=boxColors[2], color='black', weight='roman',
            size='x-small')
plt.figtext(0.80, 0.045,   'M-Learning Stratfied  ',
            backgroundcolor=boxColors[3],
            color='black', weight='roman', size='x-small')


plt.show()
fig.savefig(str(Noise)+'Noise_S1_Obs_Value_cts' + '_REV.pdf', bbox_inches='tight',dpi=900)


### Output to txt file
orig_stdout = sys.stdout
f = open('S1_Obs_Cts.txt', 'w')
sys.stdout = f

Methods = ['QL','AMOL','ML', 'ML strat']
for i in range(len(Methods)):
    for ss in sample_size:
        print (Methods[i],ss, ' : ', '\n',
               np.round(np.mean(np.array(results[ss])[:, 0][:, i]),3),
               np.round(np.std(np.array(results[ss])[:, 0][:, i]),3),
               np.round(np.median(np.array(results[ss])[:, 0][:, i]),3)
        ),

print ('*****************************************************************','\n',
       '*****************************************************************')
for i in range(len(Methods)):
    for ss in sample_size:
        print (Methods[i],ss, ' : ', '\n',
               np.round(np.mean(np.array(results[ss])[:, 1][:, i]),3),
               np.round(np.std(np.array(results[ss])[:, 1][:, i]),3),
               np.round(np.median(np.array(results[ss])[:, 1][:, i]),3)
        )

sys.stdout = orig_stdout
f.close()