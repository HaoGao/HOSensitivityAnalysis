import tensorflow as tf
import tensorflow_probability as tfp
tfb=tfp.bijectors
tfd=tfp.distributions
import gpflow
import pickle
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from gpflow.utilities import set_trainable
from gpflow.config import default_float
from gpflow.mean_functions import MeanFunction
from gpflow.base import Parameter
dtype=tf.float64

class Quadratic(MeanFunction):
    """
    y_i = A x_i^2 + B x_i + C
    """
    def __init__(self, A=None, B=None,C=None):
        """
        If X has N rows and D columns, and Y is intended to have Q columns,
        then A,B must be [D, Q], C must be a vector of length Q.
        """
        MeanFunction.__init__(self)
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        B = np.ones((1, 1), dtype=default_float()) if B is None else B
        C = np.zeros(1, dtype=default_float()) if C is None else C
        self.A = Parameter(np.atleast_2d(A))
        self.B = Parameter(np.atleast_2d(B))
        self.C = Parameter(C)
    def __call__(self, X):
        return tf.tensordot(tf.square(X), self.A, [[-1], [0]]) + tf.tensordot(X, self.B, [[-1], [0]])+self.C

#GPmodel class creates an efficient form of GPflow model where matrix inversions are done in advance
def broadcast_matmul(A, B):
    return tf.reduce_sum(A[..., tf.newaxis] * B[..., tf.newaxis, :, :],
                         axis=-2)

def SEkernel(x,X,l,s,MO=0):
    if MO:
        x=tf.expand_dims(x,0)
        x=tf.expand_dims(x/l,2)
        X=tf.expand_dims(X,0)
        X=tf.expand_dims(X/l,1)
        sub=tf.square(x-X)
        r2=tf.reduce_sum(sub,axis=-1)
        return s * tf.exp(-0.5 * r2) #output is M x k x n
    else:
        x=tf.expand_dims(x/l,1)
        X=tf.expand_dims(X/l,0)
        sub=tf.square(x-X)
        r2=tf.reduce_sum(sub,axis=-1)
        return s * tf.exp(-0.5 * r2)

def SEKernelMaker(l,s,MO=0):
    if MO:
        def SEkernel(x,X):
            x=tf.expand_dims(x,0)
            x=tf.expand_dims(x/l,2)
            X=tf.expand_dims(X,0)
            X=tf.expand_dims(X/l,1)
            sub=tf.square(x-X)
            r2=tf.reduce_sum(sub,axis=-1)
            return tf.expand_dims(s,-1) * tf.exp(-0.5 * r2) #output is M x k x n
    else:
        def SEkernel(x,X):
            x=tf.expand_dims(x/l,1)
            X=tf.expand_dims(X/l,0)
            sub=tf.square(x-X)
            r2=tf.reduce_sum(sub,axis=-1)
            return s * tf.exp(-0.5 * r2)
    return SEkernel

def ArcCosineKernelMaker(w,b,v,MO=0):
    if MO:
        def _weighted_product(X, X2=None):
            if X2 is None:
                return tf.reduce_sum(w * tf.square(X), axis=2) + b
            return (
                broadcast_matmul((w * X), tf.transpose(X2,[0,2,1])) + b[:,:,None]
            )
        def ArcCosKernel(x,X):
            x=tf.expand_dims(x,0)
            X=tf.expand_dims(X,0)
            denom=tf.sqrt(_weighted_product(x))
            x2denom=tf.sqrt(_weighted_product(X))
            numer=_weighted_product(x,X)
            costhet=numer/denom[:,:,None]/x2denom[:,None,:]
            jitter=1e-15
            theta=tf.math.acos(jitter+(1-2*jitter)*costhet)
            J=tf.math.sin(theta)+(np.pi-theta)*tf.math.cos(theta)
            return v[:,:,None]*(1/np.pi)*J*denom[:,:,None]*x2denom[:,None,:]
    else:
        kernel=gpflow.kernels.ArcCosine(weight_variances=w,bias_variance=b,variance=v,order=1)
        def ArcCosKernel(x,X):
            return kernel.K(x,X)
    return ArcCosKernel

class MOGPmodel():
    def __init__(self,data,KernelMaker,lens,sigvar,likvar,only_mean=1):
        self.X = data[0]
        self.Y = data[1]
        self.sigma2 = likvar
        self.kernels = [KernelMaker(lens[i,0],sigvar[i,0],0) for i in range(len(lens))]
        # self.kernels = [kernel(Wvar[i],Bvar[i,0],Svar[i,0],0) for i in range(len(Wvar))]
        # self.kernMO = kernel(Wvar,Bvar,Svar,1)
        self.kernMO = KernelMaker(lens,sigvar,1)
        self.only_mean=only_mean
        self.mf = lambda x: mf.__call__(x)

    def coeff_creation(self):
        coeffmu=[]
        coeffvar=[]
        for i in range(25):
            kern=self.kernels[i]
            # kern=self.kernels
            K=kern(self.X,self.X)+self.sigma2*tf.eye(self.X.shape[0],self.X.shape[0],dtype=tf.float64)
            L=tf.linalg.cholesky(K)
            trm1=tf.linalg.triangular_solve(L,self.Y[:,i].reshape((self.Y.shape[0],1)),lower=True)
            coeffmu.append(tf.linalg.triangular_solve(tf.transpose(L),trm1,lower=False))
            if not self.only_mean:
                trm2=tf.linalg.triangular_solve(L,tf.eye(self.X.shape[0],self.X.shape[0],dtype=tf.float64),lower=True)
                coeffvar=tf.linalg.triangular_solve(tf.transpose(L),trm2,lower=False)
        self.coeffmu=np.array(coeffmu)
    def predict(self,Xtst):
        Kstar=self.kernMO(Xtst,self.X)
        if self.only_mean: #if we only care about predictive mean then do not evaluate variance
            mu=tf.squeeze(broadcast_matmul(Kstar,self.coeffmu))#+meanstar
            return tf.transpose(mu)
        else: #if we do care about predictive variance...
            Kstarstar=self.kernMO(Xtst,Xtst)
            mu=tf.matmul(Kstar,self.coeffmu)#+meanstar
            var=tf.linalg.diag_part(Kstarstar)
            var-=tf.matmul(tf.matmul(Kstar,self.coeffvar),tf.transpose(Kstar))
            var+=self.sigma2
            return mu,var

class GPmodel():
    def __init__(self,data,kernel,lens,sigvar,likvar,only_mean=1):
        self.X = data[0]
        self.Y = data[1]
        self.sigma2 = likvar
        self.kern = lambda x,X: kernel(x,X,lens,sigvar)
        self.only_mean=only_mean

    def coeff_creation(self):
        # K=self.kern.K(self.X,self.X)+self.sigma2*tf.eye(self.X.shape[0],self.X.shape[0],dtype=tf.float64)
        K=self.kern(self.X,self.X)+self.sigma2*tf.eye(self.X.shape[0],self.X.shape[0],dtype=tf.float64)
        L=tf.linalg.cholesky(K)
        trm1=tf.linalg.triangular_solve(L,self.Y,lower=True)
        self.coeffmu=tf.linalg.triangular_solve(tf.transpose(L),trm1,lower=False)
        if not self.only_mean:
            trm2=tf.linalg.triangular_solve(L,tf.eye(self.X.shape[0],self.X.shape[0],dtype=tf.float64),lower=True)
            self.coeffvar=tf.linalg.triangular_solve(tf.transpose(L),trm2,lower=False)
    def predict(self,Xtst):
        Kstar=self.kern(Xtst,self.X)
        if self.only_mean: #if we only care about predictive mean then do no evaluate variance
            return tf.matmul(Kstar,self.coeffmu)#+meanstar
        else: #if we do care about predictive variance...
            Kstarstar=self.kern(Xtst,Xtst)
            mu=tf.matmul(Kstar,self.coeffmu)#+meanstar
            var=tf.linalg.diag_part(Kstarstar)
            var-=tf.matmul(tf.matmul(Kstar,self.coeffvar),tf.transpose(Kstar))
            var+=self.sigma2
            return mu,var
#

def CreateModels(Xtrain,Ytrain,sav_dir,OnlyMean):
    models={}
    Xtr=np.array(Xtrain)
    Ytr=np.array(Ytrain)
    for i in range(1):
        pars=np.loadtxt(sav_dir+'/GPhyps'+str(i)+'.txt')
        wvar=pars[:5]
        biasvar=pars[5]
        sigvar=pars[6]
        likvar=np.float64([0.00001])
        models[str(i)]=GPmodel((Xtr,Ytr[:,i].reshape((Xtr.shape[0],1))),ArcCosineKernelMaker,wvar,biasvar,sigvar,likvar,OnlyMean)
        models[str(i)].coeff_creation()
    return models['0'].coeffmu

def CreateMOModels(Xtrain,Ytrain,sav_dir,OnlyMean):
    models={}
    Xtr=np.array(Xtrain)
    Ytr=np.array(Ytrain)
    lens=[]
    sigvar=[]
    for i in range(25):
        pars=pickle.load(open(sav_dir+'/GPhyps'+str(i)+'.p','rb'))
        lens.append(pars['pars'][0].reshape((1,1,5)))
        sigvar.append(pars['pars'][1].reshape((1,1)))
    lens=tf.concat(lens,axis=0)
    sigvar=tf.concat(sigvar,axis=0)
    # biasvar=tf.concat(biasvar,axis=0)
    likvar=np.float64([0.00001])
    model=MOGPmodel((Xtr,Ytr),SEKernelMaker,lens,sigvar,likvar,OnlyMean)
    model.coeff_creation()
    return model


def loss(x,Ytst,models,EvalLoss=1,WithUnc=0):
    loss=0.
    if WithUnc:
        for i in range(25):
            mu,var=models[str(i)].predict(x)
            loss+=0.5*tf.square(tf.gather(Ytst,[i],axis=1)-mu)/var
            loss-=tf.math.log(2*np.pi*var)
    else:
        loss=0.
        for i in range(25):
            pred=models[str(i)].predict(x)
            if EvalLoss==1:
                loss-=tf.reduce_sum(tf.square(tf.gather(Ytst,[i],axis=1)-pred))
                return -loss
            else:
                if i==0:
                    loss=pred.numpy().reshape([x.shape[0],1])
                else:
                    loss=np.concatenate([loss,pred.numpy().reshape([x.shape[0],1])],axis=1)
    return loss

def prediction(x,models,mx,sx):
    loss=models.predict((x-mx)/sx)
    return loss
def LogLik(x,Ytst,models,mx,sx,stds,OnlyMean):
    loglik=0.
    if OnlyMean:
        pred=models.predict((x-mx)/sx)
        loglik=tf.reduce_sum(tfd.Normal(pred,stds).log_prob(Ytst))
    else:
        for i in range(25):
            mu,var=models[str(i)].predict((x[0]-mx)/sx)
            loglik+=tf.reduce_sum(tfd.Normal(mu,tf.math.sqrt(x[1]+var)).log_prob(tf.gather(Ytst,[i],axis=1)))
    return loglik

def SumofSquaresMaker(models,fixed):
    def ssfun(x,data):
        x=np.concatenate([x.reshape((1,4)),fixed],axis=1)
        preds=np.zeros((25,1))
        for i in range(25):
            pred=models[str(i)].predict(x).numpy().reshape((1))
            preds[i,0]=pred
        return np.sum(np.square(pred-data.ydata[0]))
    return ssfun

def PriorMaker(min,max):
    def prior(a,b,af,bf):
        apr=tfd.Uniform(np.float64(min[0]),np.float64(max[0])).log_prob(a)
        afpr=tfd.Uniform(np.float64(min[2]),np.float64(max[2])).log_prob(af)
        bpr=tfd.Uniform(np.float64(min[1]),np.float64(max[1])).log_prob(b)
        bfpr=tfd.Uniform(np.float64(min[3]),np.float64(max[3])).log_prob(bf)
        # sigpr=tfd.InverseGamma(np.float64(0.001),np.float64(0.001)).log_prob(sig)
        return tf.reduce_sum(apr+afpr+bpr+bfpr)
    return prior


def target(a,b,af,bf,Ytst,fixed,mx,sx,stds,models,prior,OnlyMean):
    x=tf.concat([tf.math.log(a),tf.math.log(b),tf.math.log(af),tf.math.log(bf),fixed],axis=1)
    lik=LogLik(x,Ytst,models,mx,sx,stds,OnlyMean)
    # pr=prior(a,b,af,bf)
    return tf.reduce_sum(lik)

def BijectionMaker(min,max,log):
    bij=[]
    if log:
        for i in range(4):
            bij.append(tfb.Chain([tfb.Log(),tfb.Sigmoid(low=min[i],high=max[i])]))
    else:
        for i in range(4):
            bij.append(tfb.Shift(tf.cast(min[i],tf.float64))(tfb.Scale(tf.cast((max[i])-(min[i]),tf.float64))( tfb.Reciprocal()(
                tfb.Shift(tf.cast(1.,tf.float64))(
                  tfb.Exp()(
                    tfb.Scale(tf.cast(-1.,tf.float64))))))))
    # bij.append(tfb.Exp())
    return bij

def step_size_setter_fn(kernel_results, new_step_size):
    pars=kernel_results.inner_results
    pars=pars._replace(
            step_size=new_step_size)
    return kernel_results._replace(
            inner_results=pars)


def step_size_getter_fn(kernel_results):
    return [tf.cast(ss, dtype) for ss in kernel_results.inner_results.step_size]


def log_accept_prob_getter_fn(kernel_results):
    return kernel_results.inner_results.log_accept_ratio

def NutsAdaptiveKernel(y,fixed,models,nburn,pr,BIJ,mx,sx,stds,OnlyMean):
    FinalTarget=lambda a,b,af,bf: target(a,b,af,bf,y,fixed,mx,sx,stds,models,pr,OnlyMean)
    nuts=tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn=FinalTarget,
    step_size=tf.cast(0.01,tf.float64),max_tree_depth=11)
    TransNuts= tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=nuts,
    bijector=BIJ)
    return TransNuts
#     return tfp.mcmc.DualAveragingStepSizeAdaptation(
#     TransNuts,
#     num_adaptation_steps=int(nburn*0.8),
#     step_size_getter_fn=step_size_getter_fn,
#     step_size_setter_fn=step_size_setter_fn,
#     log_accept_prob_getter_fn=log_accept_prob_getter_fn,
#     target_accept_prob=tf.cast(.9, dtype),
#     decay_rate=tf.cast(.75, dtype)
# )

IniState=[]
for i in range(4):
    IniState.append(tf.Variable(np.float64([[1]])))
# IniState.append(tf.Variable(np.float64(0.1).reshape((1,1))))
