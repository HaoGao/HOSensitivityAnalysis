# import utils
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from GPModelMaker import *
import pickle
import sys
import pymc3 as pm
gewk=pm.geweke
Test=0
if not Test:
    PressureVal=int(sys.argv[1])
    NRun=int(sys.argv[2])
    # if NRun<2:
    # RunInds=[4]
    # else:
    RunInds=list(range(NRun*20,(NRun+1)*20))
# RunInds=[0]
OnlyMean=1
if OnlyMean:
    SavD='/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/Results/LOG2000TrainLowNoise'
else:
    SavD='/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/Results/NUTS/WithVar'

if Test: #PLOT PREDICTIONS
    Xtrain=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/Inputs.txt',delimiter=',')
    Xtrain2=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/Inputs2.txt',delimiter=',')
    Xtrain=np.vstack([Xtrain,Xtrain2])
    Ytrain=[]
    for i in range(1,16):
        y=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/Sims'+str(i)+'.txt',delimiter=',')
        Ytrain.append(y)
    Ytrain=np.vstack(Ytrain)

    indna=[x[0] for x in np.argwhere(np.isnan(Ytrain[:,0]))]

    Xtrain=np.delete(Xtrain,indna,axis=0)
    Ytrain=np.delete(Ytrain,indna,axis=0)
    Xtrain=Xtrain[:2000,:]
    Ytrain=Ytrain[:2000,:]

    Xtrain[:,:4]=np.log(Xtrain[:,:4])

    Xvals=[]
    Yvals=[]
    for p in [5,10,15,20,25]:
        XVal=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/TstInputs.txt')
        print(XVal.shape)
        XVal=np.hstack([XVal,p*np.ones([XVal.shape[0],1])])
        YVal=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/TestSims'+str(p)+'.txt',delimiter=',')
        indna=[x[0] for x in np.argwhere(np.isnan(YVal[:,0]))]
        XVal=np.delete(XVal,indna,axis=0)
        YVal=np.delete(YVal,indna,axis=0)
        Xvals.append(XVal)
        Yvals.append(YVal)
    Xvals=np.vstack(Xvals)
    Yvals=np.vstack(Yvals)
    Xvals[:,:4]=np.log(Xvals[:,:4])
    sav_dir='/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/models/LogEmuls2000Train'
    mx=np.mean(Xtrain,axis=0)
    sx=np.std(Xtrain,axis=0)
    my=np.mean(Ytrain,axis=0)
    sy=np.std(Ytrain,axis=0)

    Xtrain-=mx
    Xtrain/=sx
    Ytrain-=my
    Ytrain/=sy

    Yvals-=my
    Yvals/=sy
    # Xvals-=mx
    # Xvals/=sx

    mdls=CreateMOModels(Xtrain,Ytrain,sav_dir,OnlyMean)
    preds=prediction(Xvals,mdls,mx,sx)
    preds=preds.numpy()
    fig, axs = plt.subplots(5,5, figsize=(15, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .15, wspace=.15,left=0.01,right=0.99,bottom=0.05,top=0.97)
    axs = axs.ravel()
    for i in range(25):
        axs[i].scatter(preds[:,i],Yvals[:,i])
    plt.savefig('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/EmulatorTesting/pred.png')
    plt.close()
elif PredDists:
    Xtrain=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/Inputs.txt',delimiter=',')
    Xtrain2=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/Inputs2.txt',delimiter=',')
    Xtrain=np.vstack([Xtrain,Xtrain2])
    Ytrain=[]
    for i in range(1,16):
        y=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/Sims'+str(i)+'.txt',delimiter=',')
        Ytrain.append(y)
    Ytrain=np.vstack(Ytrain)

    indna=[x[0] for x in np.argwhere(np.isnan(Ytrain[:,0]))]

    Xtrain=np.delete(Xtrain,indna,axis=0)
    Ytrain=np.delete(Ytrain,indna,axis=0)
    Xtrain=Xtrain[:2000,:]
    Ytrain=Ytrain[:2000,:]

    Xtrain[:,:4]=np.log(Xtrain[:,:4])

    Xtest=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/TstInputs.txt')
    Ytest=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/TestSims'+str(PressureVal)+'.txt',delimiter=',')
    indna=[x[0] for x in np.argwhere(np.isnan(Ytest[:,0]))]
    Ytest=np.delete(Ytest,indna,axis=0)
    Xtest=np.delete(Xtest,indna,axis=0)
    std=[5.,0.03]
    np.random.seed(25)
    noise=np.concatenate([std[0]*np.ones([Ytest.shape[0],1]), std[1]*np.ones([Ytest.shape[0],24])],axis=1)
    stds=np.concatenate([std[0]*np.ones((1,1)),std[1]*np.ones((1,24))],axis=1)
    Ytest+=np.random.normal(0,noise)

    sav_dir='/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/models/LogEmuls2000Train'
    mx=np.mean(Xtrain,axis=0)
    sx=np.std(Xtrain,axis=0)
    my=np.mean(Ytrain,axis=0)
    sy=np.std(Ytrain,axis=0)

    Xtrain-=mx
    Xtrain/=sx
    Ytrain-=my
    Ytrain/=sy

    Ytest-=my
    Ytest/=sy
    stds/=sy

    min=np.float64([0.1,0.1,0.1,0.1])
    max=np.float64([10,10,10,10])

    mdls=CreateMOModels(Xtrain,Ytrain,sav_dir,OnlyMean)
    SavD='/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/Results/LOG2000TrainLowNoise'
    PressureVal=5
    for TestInd in RunInds:
        pickle.load(open( SavD+'/Pressure'+str(PressureVal)+'SAMPLES'+str(TestInd)+'.p', "rb" ))

else: #DO INFERENCE
    Xtrain=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/Inputs.txt',delimiter=',')
    Xtrain2=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/Inputs2.txt',delimiter=',')
    Xtrain=np.vstack([Xtrain,Xtrain2])
    Ytrain=[]
    for i in range(1,16):
        y=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/Sims'+str(i)+'.txt',delimiter=',')
        Ytrain.append(y)
    Ytrain=np.vstack(Ytrain)

    indna=[x[0] for x in np.argwhere(np.isnan(Ytrain[:,0]))]

    Xtrain=np.delete(Xtrain,indna,axis=0)
    Ytrain=np.delete(Ytrain,indna,axis=0)
    Xtrain=Xtrain[:2000,:]
    Ytrain=Ytrain[:2000,:]

    Xtrain[:,:4]=np.log(Xtrain[:,:4])

    Xtest=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/TstInputs.txt')
    Ytest=np.genfromtxt('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/data/LogSims/TestSims'+str(PressureVal)+'.txt',delimiter=',')
    indna=[x[0] for x in np.argwhere(np.isnan(Ytest[:,0]))]
    Ytest=np.delete(Ytest,indna,axis=0)
    Xtest=np.delete(Xtest,indna,axis=0)
    # std=[2.,0.01]
    std=[5.,0.03]
    np.random.seed(25)
    noise=np.concatenate([std[0]*np.ones([Ytest.shape[0],1]), std[1]*np.ones([Ytest.shape[0],24])],axis=1)
    stds=np.concatenate([std[0]*np.ones((1,1)),std[1]*np.ones((1,24))],axis=1)
    Ytest+=np.random.normal(0,noise)

    sav_dir='/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/models/LogEmuls2000Train'
    mx=np.mean(Xtrain,axis=0)
    sx=np.std(Xtrain,axis=0)
    my=np.mean(Ytrain,axis=0)
    sy=np.std(Ytrain,axis=0)

    Xtrain-=mx
    Xtrain/=sx
    Ytrain-=my
    Ytrain/=sy

    Ytest-=my
    Ytest/=sy
    stds/=sy

    min=np.float64([0.1,0.1,0.1,0.1])
    max=np.float64([10,10,10,10])


    mdls=CreateMOModels(Xtrain,Ytrain,sav_dir,OnlyMean)
    bijector=BijectionMaker(min,max,0)
    PriorProbFn=PriorMaker(min,max)

    @tf.function(experimental_compile=True)
    def RunNuts(kern,y,fixedpars,init_state,nburn,nsamp):
        kern=kern(y,fixedpars,mdls,nburn,PriorProbFn,bijector,mx,sx,stds,OnlyMean)
        return tfp.mcmc.sample_chain(
        num_results=nsamp+nburn,
        num_burnin_steps=0,
        current_state=init_state,
        kernel=kern
        )
    IQRs=[]

    # nburn=5000
    nburn=1000
    nsamp=1000
    for TestInd in RunInds:
        CHAINS=[]
        for nstart in range(5):
            FixedPars=np.float64(PressureVal).reshape((1,1))
            for i in range(4):
                IniState[i].assign(np.random.uniform(0.1,10,[1,1]))
            states,extra=RunNuts(NutsAdaptiveKernel,Ytest[TestInd,:].reshape((1,25)),FixedPars,IniState,nburn,nsamp)

            ndiv=np.sum(extra.inner_results.has_divergence.numpy())
            # print(ndiv)
            # print(extra.inner_results..has_divergence.numpy())
            if ndiv>0:
                # wherediv=[i for i, x in enumerate(extra.inner_results.inner_results.has_divergence.numpy()) if x]
                wherediv_=[i for i, x in enumerate(extra.inner_results.has_divergence.numpy()) if x]
                wherediv=[x for x in wherediv if x>nburn]
                ndiv=len(wherediv)
                print(wherediv)
            else:
                wherediv=[]

            Samples=np.concatenate([x.numpy().reshape([nburn+nsamp,x.shape[2]]) for x in states],axis=1)
            CHAINS.append(Samples)
            ess=tfp.mcmc.effective_sample_size(Samples[nburn:,:])
            print(ess)
            if ndiv>0:
                fig, ax = plt.subplots(4,1,figsize=(15,4))
                for plotID in range(4):
                    ax[plotID].plot(Samples[nburn:,plotID])
                    for j in wherediv:
                        if j>nburn:
                            ax[plotID].scatter(j-nburn,Samples[j,plotID],c='red',s=10,zorder=5)
                plt.tight_layout()
                fig.savefig('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/Results/Divergent/Pressure'+str(PressureVal)+'SAMPLES'+str(TestInd)+str(nstart)+'.png')
                plt.close()
            if TestInd%5==0:
                fig, ax = plt.subplots(4,1,figsize=(15,4))
                for plotID in range(4):
                    ax[plotID].plot(Samples[nburn:,plotID])
                plt.tight_layout()
                fig.savefig('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/Results/Checking/Pressure'+str(PressureVal)+'SAMPLES'+str(TestInd)+str(nstart)+'.png')
                plt.close()
            if np.min(ess.numpy())<100:
                fig, ax = plt.subplots(4,1,figsize=(15,4))
                for plotID in range(4):
                    ax[plotID].plot(Samples[nburn:,plotID])
                ax[0].set_title(str(np.min(ess.numpy())))
                plt.tight_layout()
                fig.savefig('/xlwork4/2026068l/PhD/projects/NewParameterization/ForDavid/Results/BadESS/Pressure'+str(PressureVal)+'SAMPLES'+str(TestInd)+str(nstart)+'.png')
                plt.close()
            Samples=np.array(Samples[nburn::5,:])
            print('***Subject:'+str(TestInd)+'-Run:'+str(nstart)+'***')
        CHAINS=np.array(CHAINS)
        psrf=tfp.mcmc.potential_scale_reduction(np.transpose(CHAINS,[1,0,2])).numpy()
        print(psrf)
        CHAINS=CHAINS[:,::10,:]
        print('*********Subject:'+str(TestInd)+' FINISHED!*********')
        MCMCStats={'samples':CHAINS,'ESS':ess.numpy(),'psrf':psrf,'Ndivergent':ndiv,'WhereDivergent':wherediv} #saving samples and some diagnostics
        pickle.dump( MCMCStats, open( SavD+'/Pressure'+str(PressureVal)+'SAMPLES'+str(TestInd)+'.p', "wb" ) )
