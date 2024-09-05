import numbers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from itertools import product
from itertools import combinations

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import KBinsDiscretizer


class DMC(BaseEstimator, ClassifierMixin):
    _parameter_constraints: dict = {
        "N": [Interval(numbers.Integral, 1, None, closed="left")],
        "T": [Interval(numbers.Integral, 2, None, closed="left"), StrOptions({"auto"})],
        "m": [Interval(numbers.Real,1,None, closed="neither")],
        "discretization": [StrOptions({"kmeans", "DT", "KBins", "cmeans"})],
        "L": ["array-like", None],
        "box": ["array-like", None],
        "random_state": ["random_state"],
        "min_samples_leaf": [Interval(numbers.Integral, 1, None, closed="left"), StrOptions({"auto"})],
        "ccp_alpha": [Interval(numbers.Real, 0, None, closed="left")],
        "n_bins": [Interval(numbers.Integral, 5, None, closed="left"), StrOptions({"auto"})]

        }
    def __init__(
            self,
            T="auto",
            m=1.5,
            N=1000,
            discretization='kmeans', 
            L=None,
            box=None,
            random_state=None,
            option_info=False,
            min_samples_leaf="auto", #for treee
            ccp_alpha=0, #for treee
            n_bins="auto" #for kbins
    ):
        """
        Initialize the DMC model.

        Parameters:
        N : int, default=1000
            Maximum number of iterations for the algorithm
        T : int or str, default='auto'
            Number of  discrete profiles. Must be an integer greater than or equal to 2. or 'auto' for automatic determination.
        discretization : str, default='kmeans'
            Method of discretization to use. Must be 'kmeans' or 'DT'(decision tree).
        L : array-like or None, default=None
            Loss function, default is zero-one loss.
        box : array-like or None, default=None
            Box constraints for the piStar.
        random_state : int, RandomState instance or None, default=None
            Seed for random number generator for reproducibility.
        min_samples_leaf: int [0,infity), default="auto"
            The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least  (only for DT):
        ccp_alpha: non negative float, default="auto"
            Complexity parameter used for Minimal Cost-Complexity Pruning

        n_bins:int default="auto"
            The number of bins to produce.s
        """
        self.piStar = None
        self.piTrain = None
        self.pHat = None
        self.T = T
        self.N = N
        self.L = L
        self.m = m
        self.discretization = discretization
        self.box = box
        self.label_encoder = LabelEncoder()

        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.option_info = option_info
        self.n_bins = n_bins
        
        self._validate_params()

    def fit(self, X, y, **paramT):
        self.random_state = check_random_state(self.random_state)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame): # Convert y to a numpy array if is a Dataframe
            y = y.to_numpy().ravel()  # Use ravel() to make sure that y is one-dimensional

        y_encoded = self.label_encoder.fit_transform(y)

        K = len(np.unique(y_encoded))
        if self.L is None:
            self.L = np.ones((K, K)) - np.eye(K)

        if self.discretization == 'kmeans':
            if self.T == 'auto':
                if self.option_info is True:
                    print('Calculate T_optimal... ', end='')
                self.T = self.get_T_optimal(X, y_encoded, **paramT)['T']
                if self.option_info is True:
                    print('Finish')
            self.discretization_model = KMeans(n_clusters=self.T,random_state=self.random_state)
            self.discretization_model.fit(X)
            self.discrete_profiles = self.discretization_model.labels_
            self.pHat = compute_pHat(self.discrete_profiles, y_encoded, K, self.T)
            
        elif self.discretization == "DT":
            # Consider the situation when t="auto"
            clf = DecisionTreeClassifier()
            path=clf.cost_complexity_pruning_path(X, y_encoded)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities

            if self.min_samples_leaf == 'auto' or self.ccp_alpha == 'auto':
                parameters_tree = self.get_Tree_optimal(X, y_encoded, alphas=ccp_alphas)
            if self.min_samples_leaf == 'auto':
                self.min_samples_leaf = parameters_tree['min_samples_leaf']
            if self.ccp_alpha == 'auto':
                self.ccp_alpha = parameters_tree['ccp_alpha']
                #self.min_samples_leaf=parameters_tree['min_samples_leaf']

            self.discretization_model = DecisionTreeClassifier(ccp_alpha=self.ccp_alpha, 
                                           class_weight="balanced", 
                                           criterion='entropy', 
                                           min_samples_leaf= self.min_samples_leaf,
                                        max_features= 'sqrt',
                                           splitter='random',
                                           random_state=self.random_state).fit(X, y_encoded)
            
            self.discrete_profiles=self.discretisation_DT(X, self.discretization_model)
            self.T=self.discretization_model.get_n_leaves()
            self.pHat = compute_pHat(self.discrete_profiles, y_encoded, K, self.T)
            
        elif self.discretization == "KBins":
            if self.n_bins == "auto":
                self.n_bins = self.get_nbins_optimal(X,y_encoded)["n_bins"]
            print("fin")
            self.discretization_model = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot', strategy='uniform')
            self.discretization_model.fit(X)
            onehotarrays = self.discretization_model.transform(X).toarray()
            self.discrete_profiles, self.T = self.map_binary_to_int(onehotarrays)
            self.pHat = compute_pHat(self.discrete_profiles, y_encoded, K, self.T)
            
        elif self.discretization == "cmeans":
            self.cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                X.T,  # 转置数据，因为算法期望数据是以列为特征的
                c=self.T,  # 聚类的数量
                m=self.m,  # 隶属度的模糊系数
                error=0.005,  # 停止条件
                maxiter=2000,  # 最大迭代次数
                init=None  # 初始化聚类中心
            )
            self.pHat = compute_pHat_with_cmeans(u, y_encoded, K)
        
        self.piTrain = compute_pi(y_encoded, K)
        self.piStar = compute_piStar(self.pHat, y_encoded, K, self.L, self.T, self.N, 0, self.box)[0]
        self._is_fitted = True
        self.classes_ = np.unique(y_encoded)
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X, pi=None):
        check_is_fitted(self, ['X_', 'y_', 'classes_'])
        if pi is None:
            pi = self.piStar
        #print(predict_profile_label(pi, self.pHat, self.L))
        if self.discretization=="kmeans":
            discrete_profiles=self.discretization_model.predict(X)

        elif self.discretization=="DT":
            discrete_profiles=self.discretisation_DT(X, self.discretization_model)

        elif self.discretization=="KBins":
            onehotarrays=self.discretization_model.transform(X).toarray()
            discrete_profiles,self.T=self.map_binary_to_int(onehotarrays)

        elif self.discretization == 'cmeans':
            u_pred, _, _, _, _, _ = fuzz.cluster.cmeans_predict(X.T, self.cntr, m=self.m, error=0.005, maxiter=1000)
            prob = delta_proba_U(u_pred, self.pHat, pi, self.L)
            return np.argmax(prob,axis=1)

        return self.label_encoder.inverse_transform(
            predict_profile_label(pi, self.pHat, self.L)[discrete_profiles]
        )

    def predict_prob(self, X, pi=None):
        #I think we have to change this
        check_is_fitted(self)
        if pi is None:
            pi = self.piStar
        if self.discretization == 'kmeans':
            lambd = (pi.reshape(-1, 1) * self.L).T @ self.pHat
            prob = lambd / np.sum(lambd, axis=0)
            return prob[:, self.discretization_model.predict(X)].T
        elif self.discretization == 'cmeans':
            u_pred, _, _, _, _, _ = fuzz.cluster.cmeans_predict(X.T, self.cntr, m=self.m, error=0.005, maxiter=1000)
            prob = delta_proba_U(u_pred, self.pHat, pi, self.L)
            return prob

    def get_T_optimal(self, X, y, T_start=10, T_end=100, T_step=10):
        param_grid = {
            'T': np.linspace(T_start, T_end, T_step, dtype=int)
        }
        grid_search = GridSearchCV(estimator=self, param_grid=param_grid, cv=2)
        grid_search.fit(X, y)
        return grid_search.best_params_
    
    def get_Tree_optimal(self,X,y,alphas):
        print("hola")
        parameters={"min_samples_leaf":np.linspace(4, 100, 10, dtype=int),
                    "ccp_alphas":alphas}
        grid_search = GridSearchCV(estimator=self, param_grid=parameters, cv=2)
        grid_search.fit(X, y)
        return grid_search.best_params_
    # Function set_params and get_params are used to gridsearchCV in sklearn
    def get_nbins_optimal(self,X,y):
        print("hola")
        param_grid = {
            'n_bins': np.linspace(2, 7, 1, dtype=int)
        }
        grid_search = GridSearchCV(estimator=self, param_grid=param_grid, cv=2)
        grid_search.fit(X, y)
        return grid_search.best_params_
    
    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {"T": self.T, "N": self.N,"discretization":self.discretization,"L":self.L,
                "random_state":self.random_state,"box":self.box,"option_info":self.option_info,
                "min_samples_leaf":self.min_samples_leaf,"ccp_alpha":self.ccp_alpha,
                "n_bins":self.n_bins}
       
    def discretisation_DT(self,X, modele) :
        '''
        Parameters
        ----------
        X : DataFrame
        Features.
        modele : Decision Tree Classifier Model
        Decidion Tree model.

        Returns
        -------
        Xdiscr : Vector
            Discretised features.

        '''
        Xdiscr = DecisionTreeClassifier.apply(modele, X, check_input=True)
         # Obtener los índices únicos y su inversa
        valores_unicos, inversa = np.unique(Xdiscr, return_inverse=True)
    
        # Crear un mapeo de índices únicos a valores enteros consecutivos
        mapeo = {valor: indice  for indice, valor in enumerate(valores_unicos)}
    
        # Mapear los valores originales de Xdiscr a sus equivalentes enteros consecutivos
        Xdiscr_enteros = np.array([mapeo[valor] for valor in Xdiscr])
        return Xdiscr_enteros

    def map_binary_to_int(self,array_binary,l=None):

        '''
        funcion que mapea un array binario (0,1,...,0,1) hacia un entero 0,1,..,2^l-1
        input:
        array_binary: array like:
        array binario

        l: int:
        length of the array
        '''
        l=len(array_binary[0])
        combinaciones = list(product([0, 1], repeat=l))

        diccionario_combinaciones = {tuple(comb): i for i, comb in enumerate(combinaciones)}
        enteros_mapeados = [diccionario_combinaciones[tuple(fila)] for fila in list(array_binary)]
        return np.array(enteros_mapeados), len(combinaciones)


def compute_pi(y: np.ndarray, K: int):
    """
    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Labels

    K : int
        Number of classes

    Returns
    -------
    pi : ndarray of shape (K,)
        Proportion of classes
    """
    pi = np.zeros(K)
    total_count = len(y)

    for k in range(K):
        pi[k] = np.sum(y == k) / total_count
    return pi


def compute_pHat(XD: np.ndarray, y: np.ndarray, K: int, T: int):
    """
    Parameters
    ----------
    XD : ndarray of shape (n_samples,)
        Labels of profiles for each data point

    y : ndarray of shape (n_samples,)
        Labels

    K : int
        Number of classes

    T : int
        Number of profiles

    Returns
    -------
    pHat : ndarray of shape(K, n_profiles)
    """
    pHat = np.zeros((K, T))

    for k in range(K):
        Ik = np.where(y == k)[0]
        mk = len(Ik)
        pHat[k] = np.bincount(XD[Ik], minlength=T)/mk
        #Count number of occurrences of each value in array of non-negative ints.
    return pHat


def compute_pHat_with_cmeans(u, YRTrain, K):
    T = u.shape[0]
    pHat = np.zeros((K, T))
    for k in range(K):
        Ik = np.where(YRTrain == k)[0]
        mk = Ik.size
        for t in range(T):
            if mk > 0:  # 确保分母不为零
                pHat[k, t] = np.sum(u[t, Ik]) / mk
    return pHat


def delta_proba_U(U, pHat, pi, L, methode='before', temperature=0):
    '''
    Parameters
    ----------
    U : Array

    pHat : Array of floats
        Probability estimate of observing the features profile.
    pi : Array of floats
        Real class proportions.
    L : Array
        Loss function.

    Returns
    -------
    Yhat : Vector
        Predicted labels.
    '''

    def softmin_with_temperature(X, temperature=1.0, axis=1):
        X = -X
        X_max = np.max(X, axis=axis, keepdims=True)
        X_adj = X - X_max

        # 计算带温度参数的softmax
        exp_X_adj = np.exp(X_adj / temperature)
        softmax_output = exp_X_adj / np.sum(exp_X_adj, axis=axis, keepdims=True)

        return softmax_output

    lambd = U.T @ ((pi.T * L).T @ pHat).T

    if methode == 'softmin':
        prob = softmin_with_temperature(lambd, temperature)

    elif methode == 'argmin':
        prob = np.zeros_like(lambd)
        rows = np.arange(lambd.shape[0])
        cols = np.argmin(lambd, axis=1)
        prob[rows, cols] = 1

    elif methode == 'proportion':
        prob = 1 - np.divide(lambd, np.sum(lambd, axis=1)[:, np.newaxis])

    elif methode == 'before':
        prob = 1 - np.divide(lambd, np.sum(lambd, axis=1)[:, np.newaxis])

    elif methode == 'after':
        prob_init = 1 - np.divide(lambd, np.sum(lambd, axis=1)[:, np.newaxis])
        index = np.argmax(prob_init, axis=1)
        prob = np.zeros_like(prob_init)
        prob[np.arange(index.shape[0]), index] = 1
    return prob


def compute_conditional_risk(y_true: np.ndarray, y_pred: np.ndarray, K: int, L: np.ndarray):
    '''
    Function to compute the class-conditional risks.
    Parameters
    ----------
    YR : DataFrame
        Real labels.
    Yhat : Array
        Predicted labels.
    K : int
        Number of classes.
    L : Array
        Loss Function.

    Returns
    -------
    R : Array of floats
        Conditional risks.
    confmat : Matrix
        Confusion matrix.
    '''
    Labels=[i for i in range(K)]
    confmat=confusion_matrix(np.array(y_true),np.array(y_pred),normalize='true',labels=Labels)
    R=np.sum(np.multiply(L, confmat),axis=1)

   # Is only the confuns 
    
    return R, confmat


def compute_global_risk(R, pi):
    """
    Parameters
    ----------
    R : ndarray of shape (K,)
        Conditional risk
    pi : ndarray of shape (K,)
        Proportion of classes

    Returns
    -------
    r : float
        Global risk.
    """

    r = np.sum(R * pi)

    return r


def predict_profile_label(pi, pHat, L):
    lambd = (pi.reshape(-1, 1) * L).T @ pHat
    lbar = np.argmin(lambd, axis=0)
    return lbar

def proj_simplex_Condat(K, pi):
    """
    This function is inspired from the article: L.Condat, "Fast projection onto the simplex and the 
    ball", Mathematical Programming, vol.158, no.1, pp. 575-585, 2016.
    Parameters
    ----------
    K : int
        Number of classes.
    pi : Array of floats
        Vector to project onto the simplex.

    Returns
    -------
    piProj : List of floats
        Priors projected onto the simplex.

    """

    linK = np.linspace(1, K, K)
    piProj = np.maximum(pi - np.max(((np.cumsum(np.sort(pi)[::-1]) - 1) / (linK[:]))), 0)
    piProj = piProj / np.sum(piProj)
    return piProj

def graph_convergence(V_iter):
    '''
    Parameters
    ----------
    V_iter : List
        List of value of V at each iteration n.

    Returns
    -------
    Plot
        Plot of V_pibar.

    '''

    figConv = plt.figure(figsize=(8, 4))
    plt_conv = figConv.add_subplot(1, 1, 1)
    V = V_iter.copy()
    V.insert(0, np.min(V))
    font = {'weight': 'normal', 'size': 16}
    plt_conv.plot(V, label='V(pi(n))')
    plt_conv.set_xscale('log')
    plt_conv.set_ylim(np.min(V), np.max(V) + 0.01)
    plt_conv.set_xlim(10 ** 0)
    plt_conv.set_xlabel('Interation n', fontdict=font)
    plt_conv.set_title('Maximization of V over U', fontdict=font)
    plt_conv.grid(True)
    plt_conv.grid(which='minor', axis='x', ls='-.')
    plt_conv.legend(loc=2, shadow=True)

def num2cell(a):
    if type(a) is np.ndarray:
        return [num2cell(x) for x in a]
    else:
        return a

def proj_onto_polyhedral_set(pi, Box, K) :
    '''
    Parameters
    ----------
    pi : Array of floats
        Vector to project onto the box-constrained simplex.
    Box : Array
        {'none', matrix} : Box-constraint on the priors.
    K : int
        Number of classes.

    Returns
    -------
    piStar : Array of floats
            Priors projected onto the box-constrained simplex.

    '''
    
    # Verification of constraints
    for i in range(K) :
        for j in range(2) :
            if Box[i,j] < 0 :
                Box[i,j] = 0
            if Box[i,j] > 1 :
                Box[i,j] = 1

    # Generate matrix G:
    U = np.concatenate((np.eye(K), -np.eye(K), np.ones((1,K)), -np.ones((1,K))))            
    eta = Box[:,1].tolist() + (-Box[:,0]).tolist() + [1] + [-1]

    n = U.shape[0]
    
    G = np.zeros((n,n))
    for i in range(n) :
        for j in range(n) :
            G[i,j] = np.vdot(U[i,:],U[j,:])
    
    
    # Generate subsets of {1,...,n}:
    M = (2**n)-1
    I = num2cell(np.zeros((1,M)))
    
    i = 0
    for l in range(n) :
        T = list(combinations(list(range(n)), l+1))
        for p in range(i,i+len(T)) :
            I[0][p] = T[p-i]
        i = i+len(T)
            
        
    # Algorithm    
        
    for m in range(M) :
        Im = I[0][m]
 
        Gmm = np.zeros((len(Im), len(Im)))
        ligne = 0
        for i in Im :
            colonne = 0
            for j in Im :
                Gmm[ligne,colonne] = G[i,j]
                colonne += 1
            ligne +=1
        

        if np.linalg.det(Gmm)!=0 :
            
            nu = np.zeros((2*K+2,1))
            w = np.zeros((len(Im),1))
            for i in range(len(Im)) :
                w[i] = np.vdot(pi,U[Im[i],:]) - eta[Im[i]]
            
            S = np.linalg.solve(Gmm,w) 
            
            for e in range(len(S)) :
                nu[Im[e]] = S[e]
            
            
            if np.any(nu<-10**(-10)) == False  :
                A = G.dot(nu)
                z = np.zeros((1,2*K+2))
                for j in range(2*K+2) :
                    z[0][j] = np.vdot(pi,U[j,:]) - eta[j] - A[j]
                    
                    
                if np.all(z<=10**(-10)) == True :
                    pi_new = pi
                    for i in range(2*K+2) :
                        pi_new = pi_new - nu[i]*U[i,:]

    piStar = pi_new

    # Remove noisy small calculus errors:
    piStar = piStar/piStar.sum()
    
    return piStar

def proj_onto_U(pi, Box, K) :
    '''
    Parameters
    ----------
    pi : Array of floats
        Vector to project onto the box-constrained simplex..
    Box : Matrix
        {'none', matrix} : Box-constraint on the priors.
    K : int
        Number of classes.

    Returns
    -------
    pi_new : Array of floats
            Priors projected onto the box-constrained simplex.

    '''
    
    check_U = 0
    if pi.sum() ==1 :
        for k in range(K) :
            if (pi[0][k] >= Box[k,0]) & (pi[0][k] <= Box[k,1]) :
                check_U = check_U + 1
    
    if check_U == K :
        pi_new = pi

      
    if check_U < K :
        pi_new = proj_onto_polyhedral_set(pi, Box, K)
    
    return pi_new



def compute_piStar(pHat, y_train, K, L, T, N, optionPlot, Box):
    """
    Parameters
    ----------
    pHat : Array of floats
        Probability estimate of observing the features profile in each class.
    y_train : Dataframe
        Real labels of the training set.
    K : int
        Number of classes.
    L : Array
        Loss Function.
    T : int
        Number of discrete profiles.
    N : int
        Number of iterations in the projected subgradient algorithm.
    optionPlot : int {0,1}
        1 plots figure,   0: does not plot figure.
    Box : Array
        {'none', matrix} : Box-constraints on the priors.

    Returns
    -------
    piStar : Array of floats
        Least favorable priors.
    rStar : float
        Global risks.
    RStar : Array of float
        Conditional risks.
    V_iter : Array
        Values of the V function at each iteration.
    stockpi : Array
        Values of pi at each iteration.

    """
    # IF BOX-CONSTRAINT == NONE (PROJECTION ONTO THE SIMPLEX)
    if Box is None:
        pi = compute_pi(y_train, K).reshape(1, -1)
        rStar = 0
        piStar = pi
        RStar = 0

        V_iter = []
        stockpi = np.zeros((K, N))

        for n in range(1, N + 1):
            # Compute subgradient R at point pi (see equation (21) in the paper)
            lambd = np.dot(L, pi.T * pHat)
            R = np.zeros((1, K))

            mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
            R[0,:] = mu_k
            stockpi[:,n-1] = pi[0,:]

            r = compute_global_risk(R, pi)
            V_iter.append(r)
            if r > rStar:
                rStar = r
                piStar = pi
                RStar = R
                # Update pi for iteration n+1
            gamma = 1 / n
            eta = np.maximum(float(1), np.linalg.norm(R))
            w = pi + (gamma / eta) * R
            pi = proj_simplex_Condat(K, w)

        # Check if pi_N == piStar
        lambd = np.dot(L, pi.T * pHat)
        R = np.zeros((1, K))

        mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
        R[0,:] = mu_k
        stockpi[:,n-1] = pi[0,:]

        r = compute_global_risk(R, pi)
        if r > rStar:
            rStar = r
            piStar = pi
            RStar = R

        if optionPlot == 1:
            graph_convergence(V_iter)

    # IF BOX-CONSTRAINT
    if Box is not None:
        pi = compute_pi(y_train, K).reshape(1, -1)
        rStar = 0
        piStar = pi
        RStar = 0

        V_iter = []
        stockpi = np.zeros((K, N))

        for n in range(1, N + 1):
            # Compute subgradient R at point pi (see equation (21) in the paper)
            lambd = np.dot(L, pi.T * pHat)
            R = np.zeros((1, K))

            mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
            R[0,:] = mu_k
            stockpi[:,n-1] = pi[0,:]

            r = compute_global_risk(R, pi)
            V_iter.append(r)
            if r > rStar:
                rStar = r
                piStar = pi
                RStar = R
                # Update pi for iteration n+1
            gamma = 1 / n
            eta = np.maximum(float(1), np.linalg.norm(R))
            w = pi + (gamma / eta) * R
            pi = proj_onto_U(w, Box, K)

        # Check if pi_N == piStar
        lambd = np.dot(L, pi.T * pHat)
        R = np.zeros((1, K))

        mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
        R[0,:] = mu_k
        stockpi[:,n-1] = pi[0,:]
            
        r = compute_global_risk(R, pi)
        if r > rStar:
            rStar = r
            piStar = pi
            RStar = R

        if optionPlot == 1:
            graph_convergence(V_iter)

    return piStar, rStar, RStar, V_iter, stockpi
















