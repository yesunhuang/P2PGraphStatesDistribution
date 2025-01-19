'''
Name: FixedTypeGraphWithRandomNetwork_ND_N_E_01
Description: For the parallel simulation of the random graph with fixed network 
on cluster across the nodes with all methods.
Email: yesunhuang@uchicago.edu
OpenSource: https://github.com/yesunhuang
Msg: for the cluster simulation
Author: YesunHuang
Date: 2024-06-16 19:08:43
'''

#%% import all the stuffs
import networkx as nx
import numpy as np
import pandas as pd
import yaml
import os

from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster 

try:
    import src.P2PTask as p2pT
    import src.P2PSimulators as p2pS
except:
    import P2PTask as p2pT
    import P2PSimulators as p2pS


#%% global Parameters
IF_LOAD_SEED=True
currentPath=os.getcwd()

#%% set up cluster
CONFIG_FILE_NAME='dask_slurm_config.yaml'
SCALE=10
JOB_NUM=100
MAX_SHOT=200
MAX_LOAD=1
if __name__=='__main__':
    try:
        with open(CONFIG_FILE_NAME) as f:
            config=yaml.safe_load(f)
        print(config)
        cluster=SLURMCluster(**config)
        cluster.scale(jobs=JOB_NUM)
        print(cluster.job_script())
    except:
        #run local cluster
        print('Failed to load the configuration file, running local cluster')
        cluster=LocalCluster(n_workers=SCALE*JOB_NUM,threads_per_worker=2)
    

    client=Client(cluster)
    print('Cluster setup completed')
    print('Dashboard:',client.dashboard_link)

    #upload local module to the cluster
    if currentPath[-3:]=='src':
        client.upload_file(os.path.join(currentPath,'P2PTask.py'))
        client.upload_file(os.path.join(currentPath,'P2PSimulators.py'))
        client.upload_file(os.path.join(currentPath,'P2PAlgorithms.py'))
        client.upload_file(os.path.join(currentPath,'P2PSTAlgorithms.py'))
    elif currentPath[-3:]=='ion':
        client.upload_file(os.path.join(currentPath,'src','P2PTask.py'))
        client.upload_file(os.path.join(currentPath,'src','P2PSimulators.py'))
        client.upload_file(os.path.join(currentPath,'src','P2PAlgorithms.py'))
        client.upload_file(os.path.join(currentPath,'src','P2PSTAlgorithms.py'))
    elif currentPath[-3:]=='Sim':
        client.upload_file(os.path.join(currentPath,'..','src','P2PTask.py'))
        client.upload_file(os.path.join(currentPath,'..','src','P2PSimulators.py'))
        client.upload_file(os.path.join(currentPath,'..','src','P2PAlgorithms.py'))
        client.upload_file(os.path.join(currentPath,'..','src','P2PSTAlgorithms.py'))
    else:
        print(currentPath)
        raise ValueError('Please run the code in the src or main directory')


#%% data Path
EXP_NAME='FixedTypeGraphWithRandomNetwork_ND_N_E_01'
if currentPath[-3:]=='src' or currentPath[-3:]=='Sim':
    DATA_PATH=os.path.join(currentPath,'..','data')
elif currentPath[-3:]=='ion':
    DATA_PATH=os.path.join(currentPath,'data')
else:
    raise ValueError('Please run the code in the src or main directory')
EXP_PATH=os.path.join(DATA_PATH,EXP_NAME)
if not os.path.exists(EXP_PATH):
    os.makedirs(EXP_PATH)

#%% simulation methods
EXP_METHODS=['MGST','P2PGST','P2PGST-Max',\
             'ST-P2PGST-memShot','ST-P2PGST-memShot-Factor']
             

#%% save the seed
if __name__=='__main__':
    seedFileName='globalSeed.txt'
    if IF_LOAD_SEED:
        globalSeed=int(np.loadtxt(os.path.join(EXP_PATH,seedFileName)))
    else:
        globalSeed=np.random.randint(0,100000)
        np.savetxt(os.path.join(EXP_PATH,seedFileName),np.array([globalSeed]))
    globalRng=np.random.default_rng(globalSeed)

#%% generate and store the network
if __name__=='__main__':
    ## parameters
    nodes=50
    channels=1
    def pc(x):
        return np.exp(-0.5*x)
    networkParams=(0.6,0.2)
    while True:
        P2PTask=p2pT.P2PTask(nodes,pc=pc,channels=channels,topology='Waxman',\
                        seed=globalSeed,Params=networkParams,ifFullControllable=True)
        if nx.is_connected(P2PTask.network_graph):
            break
        else:
            globalSeed+=1

    ## print info about the graph
    print('Is the network connected:',nx.is_connected(P2PTask.network_graph))
    print('Average channel width:',np.mean([P2PTask.network_graph[u][v]['width'] \
                                        for u,v in P2PTask.network_graph.edges()]))
    print('Average channel probability:',np.mean([P2PTask.network_graph[u][v]['prob'] \
                                        for u,v in P2PTask.network_graph.edges()]))

#%% Generate the graph state tasks
if __name__=='__main__':
    groupNum=10
    graphNums=1000
    batchNum=SCALE*JOB_NUM
    batchSize=int(graphNums/batchNum)
    vertexGroupNums=np.linspace(10,4*nodes,groupNum,dtype=int)
    graphSeeds=globalRng.integers(0,100000,(groupNum,batchNum))

    ## generate the tasks
    data={}
    tasks=[]
    averageVertexNum=np.full((groupNum,batchNum),0)
    averageDegree=np.full((groupNum,batchNum),0)
    for i in range(0,groupNum):
        tasksGroup=[]
        vertexNum=vertexGroupNums[i]
        for j in range(0,batchNum):
            while True:
                task=p2pT.P2PTask(nodes,pc=pc,channels=channels,topology='Waxman',\
                            seed=int(graphSeeds[i,j]),Params=networkParams,ifFullControllable=True,Tracking=(i,j))
                task.pc=1
                if nx.is_connected(task.network_graph):
                    break
                else:
                    graphSeeds[i,j]=int(globalRng.integers(0,100000))
            #randomize the seed for latter simulation
            graphStates=task.generate_distribution_task([vertexNum]*batchSize,topology='star')
            tasksGroup.append(task)
            for k in range(0,batchSize):
                data[(i,j,k)]={}
                data[(i,j,k)]['groupNum']=i
                data[(i,j,k)]['batchNum']=j
                data[(i,j,k)]['batchIndex']=k
                data[(i,j,k)]['AvgVertexNum']=np.mean([len(list(g.nodes())) for g in graphStates])
                data[(i,j,k)]['AvgProb']=np.mean([task.network_graph[u][v]['prob']\
                                                    for u,v in task.network_graph.edges()])
                data[(i,j,k)]['AvgWidth']=np.mean([task.network_graph[u][v]['width']\
                                                    for u,v in task.network_graph.edges()])
        tasks.append(tasksGroup)

    data=pd.DataFrame(data).T
    print('Generation Completed.')
  ## save the seeds and params
    networkSeedFileName='networkSeeds.npy'
    np.save(os.path.join(EXP_PATH,networkSeedFileName),graphSeeds)

#%% simulation helper function
def run_single_batch_simulation(task:p2pT.P2PTask, method:str='MGST',**kargs):
    options={}
    options['maxShot']=MAX_SHOT
    options['maxLoad']=MAX_LOAD
    if 'NoRecovery' in kargs: options['NoRecovery']=kargs['NoRecovery']
    if 'STRecovery' in kargs: 
        options['STRecovery']=kargs['STRecovery']
        if 'memProb' in kargs: options['memProb']=kargs['memProb']
        else: options['memProb']=1.0
    if 'saveAll' in kargs: options['saveAll']=kargs['saveAll']
    match method:
        case 'MGST': simulator=p2pS.MGSTSimulator(task,task.seed)
        case 'P2PGST': simulator=p2pS.P2PGSTSimulator(task,task.seed)
        case 'P2PGST-Max': 
            simulator=p2pS.P2PGSTSimulator(task,task.seed)
            options['MemoryStrategy']='Maximum'
        case 'ST-P2PGST-memShot':
            simulator=p2pS.STP2PGSTSimulator(task,task.seed)
            options['memShot']=True
            if 'memCostCoeff' in kargs: options['memCostCoeff']=kargs['memCostCoeff']
            else:options['memCostCoeff']=1.0
        case 'ST-P2PGST-memShot-Factor':
            simulator=p2pS.STP2PGSTSimulator(task,task.seed)
            if 'memShot' in kargs: options['memShot']=kargs['memShot']
            else: options['memShot']=1.0
            if 'memCostCoeff' in kargs: options['memCostCoeff']=kargs['memCostCoeff']
            else: options['memCostCoeff']=1.0
        case _: raise ValueError('Invalid method')
    taskResults, _=simulator.run_simulation(**options)
    for t in taskResults:
        t.graph['recBellUsed']=simulator.additionalResourceCount['recBellUsed']
        t.graph['recMemUsed']=simulator.additionalResourceCount['recMemUsed']

    return [t.graph.copy() for t in taskResults]
        

#run and save simulation results for a particular method
def simu(method:str):
    def run_simu(task):
        res=run_single_batch_simulation(task,method=method)
        return res
    print('Start '+method+' Simulation.')
    methodData=data.copy()
    for i in range(0,groupNum):
        tasksGroup=tasks[i]
        future=client.map(run_simu,tasksGroup,\
                          key=method+'_Group-'+str(i))
        results=client.gather(future)
        for j in range(0,len(results)):
            for k in range(0,len(results[j])):
                for key in results[j][k].keys():
                    methodData.loc[(i,j,k),key]=results[j][k][key]
        print('Group Completed:',i)
        #save the data
        methodData.to_csv(os.path.join(EXP_PATH,method+'_simulationResults.csv'),index=False)


#%% start simulation
if __name__=='__main__':
    for method in EXP_METHODS:
        simu(method)
        print(method+' Completed.')


#%% close the cluster
if __name__=='__main__':
    client.close()
    cluster.close()
    
