'''
Name: P2PSimulators.py
Description: For the implementation of the simulators for the P2P entanglement distribution.
Email: yesunhuang@uchicago.edu
OpenSource: https://github.com/yesunhuang
LastEditors: yesunhuang yesunhuang@uchicago.edu
Msg: This is the main file for implementing all the simulators being used for P2P entanglement distribution.
Author: YesunHuang, XiangyuRen
Date: 2024-11-21 02:55:43
'''

#from memory_flow import*
from typing import*
import networkx as nx
import numpy as np
import P2PAlgorithms as p2pA
import P2PTask as p2pT
import P2PSTAlgorithms as p2pSTA

class Simulator:
    '''
    description: the basic class for simulating.
    '''
    def __init__(self,network:p2pT.P2PTask,seed:int=0):
        '''
        description: The constructor of the simulator.

        param {p2pT.P2PTask} network: The network to be simulated.

        param {int} seed: The random seed.
        '''
        self.network=network
        self.seed=seed
        self.rng=np.random.default_rng(seed)
        self.additionalResourceCount={'recMemUsed':0,'recBellUsed':0}

    def add_additional_resources(self,residueNetwork: nx.Graph, \
                                 savedChannels: List[List[Dict[str,Any]]],\
                                 **kargs)->nx.Graph:
        '''
        description: Add the additional resources to the residue network.

        param {nx.Graph} residueNetwork: The residue network.

        param {List[List[Dict[str,Any]]} savedChannels: The saved channels.

        return {nx.Graph} : The residue network with additional resources.
        '''
        if 'ps' in residueNetwork.graph:
            ps=residueNetwork.graph['ps']
        else:
            ps=1.0

        #clean up the saved channels from last shot
        saveChannelNodeList=[node for node in residueNetwork.nodes() if 'saveChannel' in residueNetwork.nodes[node]\
                                and residueNetwork.nodes[node]['saveChannel']]
        residueNetwork.remove_nodes_from(saveChannelNodeList)

        #add the saved channels for current shot
        saveChannelIndex=np.max(list(residueNetwork.nodes()))+1
        for savedChannel in savedChannels:
            endPoints=savedChannel['endPoints']
            switchNum=savedChannel['switchNum']
            ifSuccess=(self.rng.binomial(switchNum,ps)==switchNum)
            #add the saved channel
            if ifSuccess:
                u,v=endPoints
                residueNetwork.add_node(saveChannelIndex,saveChannel=True,memory=0,moccu=0)
                residueNetwork.add_edge(u,saveChannelIndex,prob=1.0,width=1,occu=0,distance=0)
                residueNetwork.add_edge(v,saveChannelIndex,prob=1.0,width=1,occu=0,distance=0)
            saveChannelIndex+=1
        
        return residueNetwork
            
    
    def cal_path_bell_pairs(self,path:List[int],network:nx.Graph)->int:
        '''
        description: Calculate the number of bell pairs for the path.

        param {List[int]} path: The path to be calculated.

        param {nx.Graph} residueNetwork: The residue network.

        return {int} : The number of bell pairs for the path.
        '''
        bellPairs=len(path)-1
        for i in range(len(path)):
            node=network.nodes[path[i]]
            if 'saveChannel' in node and node['saveChannel']:
                bellPairs-=2
            if 'isAncilla' in node and node['isAncilla']:
                bellPairs-=1
        return bellPairs

    def cal_path_switch(self,path:List[int],network:nx.Graph)->int:
        '''
        description: Calculate the number of switches for the path.

        param {List[int]} path: The path to be calculated.

        param {nx.Graph} network: The residue network.

        return {int} : The number of switches for the path.
        '''
        newPaths=p2pA.remove_ancilla(path,network)
        return max(len(newPaths)-2,0)

    
    def shot_plan(self,residueNetwork:nx.Graph,tasks:List[nx.Graph],network:p2pT.P2PTask|None=None, **kargs)\
        ->Tuple[List[List[int]],nx.Graph,List[nx.Graph]]:
        '''
        description: The shot plan for the simulator.

        param {nx.Graph} residueNetwork: The residue network after last shot.

        param {List[nx.Graph]} tasks: The tasks to be simulated.

        param {p2pT.P2PTask} network: The network to be simulated.

        param {kargs} : The additional arguments.

        return {Tuple[List[List[int]],nx.Graph,List[nx.Graph]]} : The main paths, the residue network and the dependent graph.
        '''
        raise NotImplementedError
    
    def run_one_shot(self, residueNetwork:nx.Graph,tasks:List[nx.Graph], network:p2pT.P2PTask|None=None, **kargs)\
        ->Tuple[nx.Graph, List[nx.Graph]]:
        '''
        description: Run one shot of the simulation.

        param {nx.Graph} residueNetwork: The residue network before this shot.

        param {List[nx.Graph]} tasks: The task to be simulated.

        param {p2pT.P2PTask} network: The network to be simulated.

        param {kargs} : The additional arguments.

        return {Tuple[nx.Graph, List[nx.Graph]]} : The network after the shot and the residue tasks.

        '''
        raise NotImplementedError
    
    def run_simulation(self, network:p2pT.P2PTask|None=None, \
                       maxLoad:int=5, maxShot:int|None=None, \
                       maxMemoryShots:int|None=None, **kargs)->Tuple[List[nx.Graph],int]:
        '''
        description: Run the simulation.

        param {p2pT.P2PTask} network: The network to be simulated.

        param {int} taskPerShot: The number of executed tasks per shot.

        param {int} maxLoad: The maximum number of task being buffered.

        param {int} maxMemoryShots: The maximum number of shots for a task.

        param {kargs} : The additional arguments.

        return {List[nx.Graph]} : The simulation results.
        '''
        def push_task(currentExecutionIndex):
            while currentExecutionIndex<len(network.network_task) and len(residueTasks)<maxLoad:
                task=network.network_task[currentExecutionIndex].copy()
                task.graph['binding']=currentExecutionIndex
                task.graph['ifSuccess']=False
                task.graph['shots']=0
                task.graph['CumulativeMem']=0
                task.graph['BellConsumed']=0
                task.graph['deliverable']=True
                activeNodes=[task.nodes[i]['node'] for i in task.nodes()]
                for i in range(len(activeNodes)):
                    residueNetwork.nodes[activeNodes[i]]['active'].append(task.graph['binding'])
                currentExecutionIndex+=1
                if len(task.nodes())==0:
                    task.graph['ifSuccess']=True
                    task.graph['finishedShot']=0
                    pop_task(task)
                else:
                    residueTasks.append(task)
            return currentExecutionIndex

        def pop_task(task:nx.Graph):
            bindTask=network.network_task[task.graph['binding']]
            for i in bindTask.nodes():
                residueNetwork.nodes[bindTask.nodes[i]['node']]['active'].remove(task.graph['binding'])
            taskResults[task.graph['binding']]=task

        if network is None:
            network=self.network
        residueNetwork=network.network_graph.copy()
        for node in residueNetwork.nodes():
            residueNetwork.nodes[node]['active']=[]
        currentExecutionIndex=0
        residueTasks=[]
        graph_init=nx.Graph(ifSuccess=False,shots=0,CumulativeMem=0,BellConsumed=0)
        taskResults=[graph_init.copy() for i in range(len(network.network_task))]

        #initialized the residue tasks
        currentExecutionIndex=push_task(currentExecutionIndex)

        currentShot=0

        if 'simulationBar' in kargs:
            lastExecutionIndex=-1

        while len(residueTasks)>0:
            if maxShot is not None:
                if currentShot>=maxShot:
                    break
            currentShot+=1
            residueNetwork,residueTasks=self.run_one_shot(residueNetwork,residueTasks,network,**kargs)
            if 'recMemUsed' in residueNetwork.graph:
                self.additionalResourceCount['recMemUsed']+=residueNetwork.graph['recMemUsed']
            if 'recBellUsed' in residueNetwork.graph:
                self.additionalResourceCount['recBellUsed']+=residueNetwork.graph['recBellUsed']
            newResidueTasks=[]
            for task in residueTasks:
                if task.graph['ifSuccess']:
                    task.graph['finishedShot']=currentShot
                    pop_task(task)
                    continue
                if not task.graph['deliverable']:
                    task.graph['finishedShot']=currentShot
                    pop_task(task)
                    continue
                if maxMemoryShots is not None:
                    if task.graph['shots']>=maxMemoryShots:
                        pop_task(task)
                        continue
                newResidueTasks.append(task)
            residueTasks=newResidueTasks
            currentExecutionIndex=push_task(currentExecutionIndex)
            if 'simulationBar' in kargs:
                if (currentExecutionIndex+1)%kargs['simulationBar']==0:
                    if currentExecutionIndex!=lastExecutionIndex:
                        print('Completed:',currentExecutionIndex+1,'/',len(network.network_task))
                        lastExecutionIndex=currentExecutionIndex
        
        return taskResults,currentShot
    
    def ST_path_simulator(self, mainPaths:List[List[int]], host:int,\
                          residueNetwork:nx.Graph,**kargs)\
                            ->Tuple[List[bool],List[int],\
                            Dict[int,List[Tuple[int,List[int],float]]],\
                            List[List[Dict[str,Any]]]]:
        '''
        description: simulate the path for the network and return if the path is successful, 
            and store the bell pair for the paths with low success probability.

        param {List[List[int]]} mainPaths: The main paths.

        param {host} host: The maximum number of hosts.

        param {nx.Graph} residueNetwork: The network after removing all the resources used by shot plan.

        return {Tuple[List[bool],List[int],Dict[int,List[Tuple[int,List[int],float]]],nx.Graph]} : 
        The success of the paths before switching, the number of switch required for the path, recoveryPaths,
        and the saved channels
        '''
        def build_path_graph(decision, mainPath_sim, recoveryPath, \
                             recStartBreakNodes, recEndBreakNodes)->nx.Graph:
            def add_decision_edge(currentNodeTuple,connectedNode,\
                                  recoveryIndex,recoveryPath,pathGraph):
                if connectedNode is not None:
                    if recoveryIndex is not None:
                        if connectedNode == recoveryPath[recoveryIndex][1][0]:
                            recoveryPathEndNode=recoveryPath[recoveryIndex][1][-1]
                        else:
                            recoveryPathEndNode=recoveryPath[recoveryIndex][1][0]

                        pathGraph.add_edge(currentNodeTuple,(recoveryPathEndNode,recoveryIndex,'RPN'),type='DEC')
                    else:
                        mainPathChannelNode=None
                        if (connectedNode,currentNodeTuple[0],'MPC') in pathGraph.nodes:
                            mainPathChannelNode=(connectedNode,currentNodeTuple[0],'MPC')
                        elif (currentNodeTuple[0],connectedNode,'MPC') in pathGraph.nodes:
                            mainPathChannelNode=(currentNodeTuple[0],connectedNode,'MPC')
                        if mainPathChannelNode is not None:
                            pathGraph.add_edge(currentNodeTuple,mainPathChannelNode,type='DEC')
                return pathGraph
            pathGraph=nx.Graph()
            #build main path channel nodes
            for i in range(len(mainPath_sim)-1):
                if mainPath_sim[i][1]:
                    pathGraph.add_node((mainPath_sim[i][0],
                                        mainPath_sim[i+1][0],'MPC'))
            #build recovery path channel nodes
            for i in range(len(recoveryPath)):
                recPath=recoveryPath[i][1]
                recStartBreakNode=recStartBreakNodes[i]
                recEndBreakNode=recEndBreakNodes[i]
                #forward path
                if recStartBreakNode==-1:
                    recStartBreakNode=len(recPath)-1
                pathGraph.add_node((recPath[0],i,'RPN'))
                for j in range(1,recStartBreakNode+1):
                    pathGraph.add_node((recPath[j],i,'RPN'))
                    pathGraph.add_edge((recPath[j],i,'RPN'),(recPath[j-1],i,'RPN'),type='RPC')
                #backward path
                if recEndBreakNode!=-1:
                    pathGraph.add_node((recPath[recEndBreakNode],i,'RPN'))
                    for j in range(recEndBreakNode+1,len(recPath)):
                        pathGraph.add_node((recPath[j],i,'RPN'))
                        pathGraph.add_edge((recPath[j],i,'RPN'),(recPath[j-1],i,'RPN'),type='RPC')
            #build main paths node as well as decision edges.
            for i in range(len(mainPath_sim)):
                currentNode=mainPath_sim[i][0]
                currentNodeTuple=(currentNode,i,'MPN')
                pathGraph.add_node(currentNodeTuple)
                if ('save' in decision[currentNode] and decision[currentNode]['save'])\
                or 'savedAll' in kargs and kargs['savedAll']:
                    pathGraph.nodes[currentNodeTuple]['save']=True
                else:
                    pathGraph.nodes[currentNodeTuple]['save']=False
                node_left,rec_left=decision[currentNode]['left']
                pathGraph=add_decision_edge(currentNodeTuple,node_left,rec_left,recoveryPath,pathGraph)
                node_right,rec_right=decision[currentNode]['right']
                pathGraph=add_decision_edge(currentNodeTuple,node_right,rec_right,recoveryPath,pathGraph)
            return pathGraph
        
        def chase_the_path_graph(recoveryPath,pathGraph)\
            ->Tuple[Tuple[int,int],List[int]]:
            endPoints=[node for node in pathGraph.nodes if pathGraph.degree(node)<=1]
            involved=False

            currentNode=endPoints[0]
            path=[]
            endPointNodes=[{'Node':None,'Order':None,'Index':None},\
                           {'Node':None,'Order':None,'Index':None}]
            pathGraph_bk=pathGraph.copy()
            orderCount=0
            while len(pathGraph_bk.nodes)>0:
                degree=pathGraph.degree(currentNode)
                match currentNode[2]:
                    case 'RPN':
                        recPath=recoveryPath[currentNode[1]][1]
                        recEndPoint=[recPath[0],recPath[-1]]
                        if currentNode[0] not in recEndPoint:
                            path.append(currentNode[0])
                        elif degree==1:
                            path.append(None)
                    case 'MPC':
                        if degree==1:
                            path.append(None)
                    case 'MPN':
                        path.append(currentNode[0])
                        involved=True
                        if pathGraph_bk.nodes[currentNode]['save']:
                            if endPointNodes[0]['Node'] is None or\
                                currentNode[1]<endPointNodes[0]['Index']:
                                endPointNodes[0]['Node']=currentNode[0]
                                endPointNodes[0]['Order']=orderCount
                                endPointNodes[0]['Index']=currentNode[1]
                            elif endPointNodes[1]['Node'] is None or\
                                currentNode[1]>endPointNodes[1]['Index']:
                                endPointNodes[1]['Node']=currentNode[0]
                                endPointNodes[1]['Order']=orderCount
                                endPointNodes[1]['Index']=currentNode[1]
                nextNode=list(pathGraph_bk.neighbors(currentNode))

                if len(nextNode)>0:
                    nextNode=nextNode[0]
                pathGraph_bk.remove_node(currentNode)
                currentNode=nextNode
                orderCount+=1
            #fix the order of the path
            if endPointNodes[0]['Node'] is not None and endPointNodes[1]['Node'] is not None:
                if endPointNodes[0]['Order']>endPointNodes[1]['Order']:
                    path=path[::-1]
            return (endPointNodes[0]['Node'],endPointNodes[1]['Node']),path,involved
        
        if 'saveAll' in kargs:
            saveAll=kargs['saveAll']
        else:
            saveAll=False
        if 'memProb' in kargs:
            memProb=kargs['memProb']
        else:
            memProb=0.8

        recMemUsed=0
        recBellUsed=0

        success=np.full(len(mainPaths),False)
        switchNum=np.full(len(mainPaths),0)
        savedChannels=[]
        network_bk=residueNetwork.copy()
        mainPathsSim=[]
        decisions=[]
        for u,v in network_bk.edges:
            if 'real_capacity' not in network_bk[u][v]:
                network_bk[u][v]['real_capacity']=\
                        self.rng.binomial(network_bk[u][v]['width'],\
                                          network_bk[u][v]['prob'])
            network_bk[u][v]['real_occu']=0
            #reset the occupancy
            network_bk[u][v]['occu']=0

        #find the recovery paths
        recoveryPaths=p2pA.find_recovery_path(residueNetwork,mainPaths,metric=p2pA.ET_metric,host=host)

        for j in range(len(mainPaths)):
            path=mainPaths[j]
            mainPath_sim=[]
            for i in range(len(path)-1):
                if network_bk[path[i]][path[i+1]]['real_capacity']-network_bk[path[i]][path[i+1]]['real_occu']>0:
                   mainPath_sim.append((path[i],True))
                   network_bk[path[i]][path[i+1]]['real_occu']+=1
                else:
                    mainPath_sim.append((path[i],False))
            mainPath_sim.append((path[-1],True))
            mainPathsSim.append(mainPath_sim)
            network_bk=p2pA.remove_paths(network_bk,path)
            if kargs['STRecovery']:
                decision=p2pSTA.path_recovery_ST_EUM(network_bk,mainPath_sim,recoveryPaths[j],host,memProb=memProb,saveAll=saveAll)
            else:
                decision=p2pA.path_recovery_EUM(network_bk,mainPath_sim,recoveryPaths[j],host)
            decisions.append(decision)
            #consume the memory for the saved nodes
            for i in range(len(path)):
                node=path[i]
                if 'save' in decision[node] and decision[node]['save']:
                    network_bk.nodes[node]['moccu']+=1
                    recMemUsed+=1

        for i in range(len(mainPathsSim)):
            source,target=mainPaths[i][0],mainPaths[i][-1]
            recForwardPath=[]
            recBackwardPath=[]
            singleSavedChannels=[]
            recoveryPath=recoveryPaths[i]
            recSuccess=np.full(len(recoveryPath),True)
            recStartBreakNode=np.full(len(recoveryPath),-1)
            recEndBreakNode=np.full(len(recoveryPath),-1)
            for j in range(len(recoveryPath)):
                _,path,_=recoveryPath[j]
                #calculate the bell pairs
                recBellUsed+=self.cal_path_bell_pairs(path,residueNetwork)
                for k in range(len(path)-1):
                    network_bk[path[k]][path[k+1]]['occu']+=1
                    if network_bk[path[k]][path[k+1]]['real_capacity']-network_bk[path[k]][path[k+1]]['real_occu']>0:
                        network_bk[path[k]][path[k+1]]['real_occu']+=1
                    else:
                        recSuccess[j]=False
                        if recStartBreakNode[j]==-1:
                            recStartBreakNode[j]=k
                        if k+1>recEndBreakNode[j]:
                            recEndBreakNode[j]=k+1

            #clean up the saved channels from last shot
            mainPathSimClean=[]
            for node,sectionSuccess in mainPathsSim[i]:
                if 'saveChannel' in residueNetwork.nodes[node]\
                    and residueNetwork.nodes[node]['saveChannel']:
                    continue
                mainPathSimClean.append((node,sectionSuccess))
            pathGraph=build_path_graph(decisions[i],mainPathsSim[i],recoveryPath,recStartBreakNode,recEndBreakNode)

            #finding the connected components
            rawGraphComponents=[pathGraph.subgraph(c) for c in nx.connected_components(pathGraph)]
            #finding the endpoints and reconstruct the path for each components
            for rawGraphComponent in rawGraphComponents:
                endPoints, constructedPath, involved=chase_the_path_graph(recoveryPath,rawGraphComponent)
                if len(constructedPath)==0 or not involved:
                    continue
                if constructedPath[-1]==source or constructedPath[0]==target:
                    constructedPath=constructedPath[::-1]
                if constructedPath[0]==source:
                    if target==constructedPath[-1]:
                        success[i]=True
                    recForwardPath=constructedPath
                elif constructedPath[-1]==target:
                    recBackwardPath=constructedPath
                    success[i]=False
                if endPoints[0] is not None and endPoints[1] is not None and not success[i]:
                    savedChannelDict={'endPoints':endPoints,'path':constructedPath,\
                                      'switchNum':self.cal_path_switch(constructedPath,residueNetwork)}
                    singleSavedChannels.append(savedChannelDict)
                    
            switchNum[i]=self.cal_path_switch(recForwardPath,residueNetwork)
            switchNum[i]+=self.cal_path_switch(recBackwardPath,residueNetwork)

            savedChannels+=singleSavedChannels

            #add recovery resource usage information
            network_bk.graph['recMemUsed']=recMemUsed
            network_bk.graph['recBellUsed']=recBellUsed

        return success.tolist(), switchNum.tolist(), recoveryPaths, savedChannels, network_bk
    
    def path_simulator(self, mainPaths:List[List[int]], host:int,\
                       residueNetwork:nx.Graph,**kargs)\
                       ->Tuple[List[bool],List[int],\
                        Dict[int,List[Tuple[int,List[int],float]]],\
                        List[List[Dict[str,Any]]]]:
        '''
        description: simulate the path for the network and return if the path is successful. 

        param {List[List[int]]} mainPaths: The main paths.

        param {host} host: The maximum number of hosts.

        param {nx.Graph} residueNetwork: The network after removing all the resources used by shot plan.

        return {Tuple[List[bool],List[int],Dict[int,List[Tuple[int,List[int],float]]],nx.Graph]} : 
        The success of the paths before switching, the number of switch required for the path, recoveryPaths,
        and the saved channels.
        '''
        if 'STRecovery' in kargs and \
            ('NoRecovery' not in kargs or not kargs['NoRecovery']):
            return self.ST_path_simulator(mainPaths,host,residueNetwork,**kargs)
        
        recBellUsed=0

        success=np.full(len(mainPaths),True)
        switchNum=np.full(len(mainPaths),0)
        network_bk=residueNetwork.copy()
        mainPaths_sim=[]
        decisions=[]
        recs=[]
        for u,v in network_bk.edges:
            if 'real_capacity' not in network_bk[u][v]:
                network_bk[u][v]['real_capacity']=\
                        self.rng.binomial(network_bk[u][v]['width'],\
                                          network_bk[u][v]['prob'])
            network_bk[u][v]['real_occu']=0
            #reset the occupancy
            network_bk[u][v]['occu']=0

        #find the recovery paths
        if 'NoRecovery' not in kargs or not kargs['NoRecovery']:
            recoveryPaths=p2pA.find_recovery_path(residueNetwork,mainPaths,metric=p2pA.ET_metric,host=host)
        else:
            recoveryPaths={i:[] for i in range(len(mainPaths))}

        for j in range(len(mainPaths)):
            path=mainPaths[j]
            mainPath_sim=[]
            for i in range(len(path)-1):
                if network_bk[path[i]][path[i+1]]['real_capacity']-network_bk[path[i]][path[i+1]]['real_occu']>0:
                   mainPath_sim.append((path[i],True))
                   network_bk[path[i]][path[i+1]]['real_occu']+=1
                else:
                    mainPath_sim.append((path[i],False))
            mainPath_sim.append((path[-1],True))
            mainPaths_sim.append(mainPath_sim)
            network_bk=p2pA.remove_paths(network_bk,path)
            decision=p2pA.path_recovery_EUM(network_bk,mainPath_sim,recoveryPaths[j],host)
            decisions.append(decision)

        for i in range(len(mainPaths_sim)):
            recoveryPath=recoveryPaths[i]
            recSuccess=np.full(len(recoveryPath),True)
            recStartBreakNode=np.full(len(recoveryPath),-1)
            recEndBreakNode=np.full(len(recoveryPath),-1)
            for j in range(len(recoveryPath)):
                _,path,_=recoveryPath[j]
                #calculate the bell pairs
                recBellUsed+=self.cal_path_bell_pairs(path,residueNetwork)
                for k in range(len(path)-1):
                    network_bk[path[k]][path[k+1]]['occu']+=1
                    if network_bk[path[k]][path[k+1]]['real_capacity']-network_bk[path[k]][path[k+1]]['real_occu']>0:
                        network_bk[path[k]][path[k+1]]['real_occu']+=1
                    else:
                        recSuccess[j]=False
                        if recStartBreakNode[j]==-1:
                            recStartBreakNode[j]=k
                        if k+1>recEndBreakNode[j]:
                            recEndBreakNode[j]=k+1

            pathComponents=[]
            recoveryPathIndexes=[]
            mainPathRaw=[n[0] for n in mainPaths_sim[i]]
            path, recoveryPathIndex,ifSuccess=\
                p2pA.chase_the_path(mainPathRaw[0],mainPathRaw[-1],decisions[i])
            pathComponents=[path]
            recoveryPathIndexes=[recoveryPathIndex]
            decision=decisions[i]
            if not ifSuccess:
                decisionReversed={n:{'left':decision[n]['right'],'right':decision[n]['left']}\
                         for n in decision.keys()}
                reversedPath,reversedRecoveryPathIndex,_=\
                    p2pA.chase_the_path(mainPathRaw[-1],mainPathRaw[0],decisionReversed)
                recoveryPathIndexes.append(reversedRecoveryPathIndex[::-1])
                pathComponents.append(reversedPath[::-1])
            
            success[i]=ifSuccess
            recs.append((pathComponents,recoveryPathIndexes,ifSuccess))

            if ifSuccess:        
                forwardPath=pathComponents[0]
                forwardRecoveryIndex=recoveryPathIndexes[0]
                backwardPath=pathComponents[0][::-1]
                backwardRecoveryIndex=recoveryPathIndexes[0][::-1]
            else:
                forwardRecoveryIndex=recoveryPathIndexes[0]
                backwardRecoveryIndex=recoveryPathIndexes[-1]
                backwardRecoveryIndex=backwardRecoveryIndex[::-1]
                forwardPath=pathComponents[0]
                backwardPath=pathComponents[-1]
                backwardPath=backwardPath[::-1]

            #reconstruct the path
            recForwardPath=[]
            recStartIndex=0
            recEndIndex=len(forwardPath)
            for k in forwardRecoveryIndex:
                currentRecoveryPath=recoveryPath[k][1]
                start=currentRecoveryPath[0]
                end=currentRecoveryPath[-1]
                indexOfStart=forwardPath.index(start)
                indexOfEnd=forwardPath.index(end)
                recForwardPath+=forwardPath[recStartIndex:indexOfStart+1]
                if recSuccess[k]:
                    recForwardPath+=currentRecoveryPath[1:-1]
                    recStartIndex=indexOfEnd
                else:
                    recForwardPath+=currentRecoveryPath[1:recStartBreakNode[k]+1]
                    recStartIndex=indexOfEnd
                    recEndIndex=indexOfEnd
                    success[i]=False
                    break
            recForwardPath+=forwardPath[recStartIndex:recEndIndex]
            
            recBackwardPath=[]
            if not success[i]:
                recStartIndex=0
                recEndIndex=len(backwardPath)
                for k in backwardRecoveryIndex:
                    currentRecoveryPath=recoveryPath[k][1]
                    start=currentRecoveryPath[-1]
                    end=currentRecoveryPath[0]
                    indexOfStart=backwardPath.index(start)
                    indexOfEnd=backwardPath.index(end)
                    recBackwardPath+=backwardPath[recStartIndex:indexOfStart+1]
                    if recSuccess[k]:
                        recBackwardPath+=currentRecoveryPath[-2:0:-1]
                        recStartIndex=indexOfEnd
                    else:
                        recBackwardPath+=currentRecoveryPath[-2:recEndBreakNode[k]-1:-1]
                        recStartIndex=indexOfEnd
                        recEndIndex=indexOfEnd
                        break
                recBackwardPath+=backwardPath[recStartIndex:recEndIndex]

            switchNum[i]=self.cal_path_switch(recForwardPath,residueNetwork)
            if not success[i]:
                switchNum[i]+=self.cal_path_switch(recBackwardPath,residueNetwork)

            #add recovery resource usage information
            network_bk.graph['recBellUsed']=recBellUsed
            network_bk.graph['recMemUsed']=0
            
        return success.tolist(), switchNum.tolist(), recoveryPaths, [], network_bk
    
class MGSTSimulator(Simulator):
    '''
    description: The simulator for the Modified-GST algorithm.
    '''
    def __init__(self,network:p2pT.P2PTask,seed:int=0):
        '''
        description: The constructor of the simulator.

        param {p2pT.P2PTask} network: The network to be simulated.

        param {int} seed: The random seed.
        '''
        super().__init__(network,seed)

    def path_simulator(self, mainPaths:List[List[int]], host:int,\
                       residueNetwork:nx.Graph,**kargs)\
                       ->Tuple[List[bool],List[int],\
                        Dict[int,List[Tuple[int,List[int],float]]],nx.Graph]:
        return super().path_simulator(mainPaths, host, residueNetwork, **kargs)
    
    def add_additional_resources(self,residueNetwork: nx.Graph, \
                                 savedChannels: List[List[Dict[str,Any]]],\
                                 **kargs)->nx.Graph:
        return super().add_additional_resources(residueNetwork, savedChannels, **kargs)

    def shot_plan(self,residueNetwork:nx.Graph,tasks:List[nx.Graph],\
                  network:p2pT.P2PTask|None=None,**kargs) \
        -> Tuple[List[Tuple[List[int],float]],nx.Graph,List[nx.Graph]]:
        '''
        description: Run one shot of the simulation, the detailed implementation of the shot plan for the base class.

        param {nx.Graph} residueNetwork: The residue network before this shot.

        param {List[nx.Graph]} tasks: The task to be simulated.

        param {p2pT.P2PTask} network: The network to be simulated.

        return {Tuple[List[Tuple[List[int],float]],nx.Graph,List[nx.Graph]]} : The main paths, the residue network and the dependent graph.

        '''

        if network is None:
            network=self.network
        
        if 'ifReset' in kargs:
            ifReset=kargs['ifReset']
        else:
            ifReset=True

        if 'minCost' in kargs:
            minCost=kargs['minCost']
        else:
            minCost=False

        if 'memShot' in kargs:
            memShot=kargs['memShot']
        else:
            memShot=False

        residueNetwork_bk=residueNetwork.copy()
        mainPaths=[]
        dependentGraphs=[]

        for i in range(len(tasks)):
            task=tasks[i]

            dependentGraph=nx.Graph()
            dependentGraph.graph['involved']=False
            dependentGraph.add_nodes_from(task.nodes())

            for node in dependentGraph.nodes():
                dependentGraph.nodes[node]['dependent']=None
                dependentGraph.nodes[node]['InShot']=False
                dependentGraph.nodes[node]['node']=task.nodes[node]['node']

            #determine if there is enough memory
            maxMemory=network.get_maximum_node_memory()
            if maxMemory is not None:
                if maxMemory<len(task.nodes()):
                    dependentGraphs.append(dependentGraph)
                    continue
                
            controllable=[]
            if not network.ifFullControllable:
                for node in residueNetwork.nodes():
                    if 'controllable' in residueNetwork.nodes[node]:
                        if not residueNetwork.nodes[node]['controllable']:
                            if task.graph['binding'] not in residueNetwork.nodes[node]['active']:
                                continue
                    else:
                        if task.graph['binding'] not in residueNetwork.nodes[node]['active']:
                            continue   
                    controllable.append(node)
            else:
                controllable=None

            rawPaths,tasks[i]=p2pA.find_MGST_paths(task,residueNetwork_bk,controllable,\
                                                   memoryConstraint=True,ifReset=ifReset,\
                                                   minCost=minCost,memShot=memShot)

            #build dependent graph
            newIndex=len(mainPaths)
            if len(rawPaths[0])>0:
                dependentGraph.graph['involved']=True

                #consume the root memory for the first time
                if  'ifGenerated' in tasks[i].graph \
                    and tasks[i].graph['ifGenerated']:
                        pass
                else:
                    root=tasks[i].graph['source']
                    residueNetwork_bk.nodes[root]['moccu']+=len(tasks[i].nodes())

                j=0
                for t in rawPaths[0].keys():
                    mainPaths.append(rawPaths[0][t])
                    dependentGraph.nodes[t]['dependent']=newIndex+j
                    dependentGraph.nodes[t]['InShot']=True

                    p2pA.remove_paths(residueNetwork_bk,rawPaths[0][t][0])
                    
                    #consume memory for terminal nodes
                    if isinstance(tasks[i].nodes[t]['node'],list):
                        terminalNode=tasks[i].nodes[t]['node'][0]
                    elif isinstance(tasks[i].nodes[t]['node'],int):
                        terminalNode=tasks[i].nodes[t]['node']
                    
                    if terminalNode!=tasks[i].graph['source']:                 
                        residueNetwork_bk.nodes[terminalNode]['moccu']+=1
                    
                    j+=1

            dependentGraphs.append(dependentGraph)

        return mainPaths,residueNetwork_bk,dependentGraphs
    
    def run_one_shot(self, residueNetwork:nx.Graph,tasks:List[nx.Graph], network:p2pT.P2PTask|None=None,\
                     **kargs)->Tuple[nx.Graph, List[nx.Graph]]:
        '''
        description: Run one shot of the simulation.

        param {nx.Graph} residueNetwork: The residue network before this shot.

        param {List[nx.Graph]} tasks: The task to be simulated.

        param {p2pT.P2PTask} network: The network to be simulated.

        return {Tuple[nx.Graph, List[nx.Graph]]} : The network after the shot and the residue tasks.

        '''
        def empty_memory(task:nx.Graph,residueNetwork:nx.Graph):
            source=task.graph['source']
            residueNetwork.nodes[source]['moccu']-=len(task.nodes())

            for v in network.network_task[task.graph['binding']].nodes():
                if v not in task.nodes():
                    terminalNode=network.network_task[task.graph['binding']].nodes[v]['node']
                    if isinstance(terminalNode,list):
                        terminalNode=terminalNode[0]
                    residueNetwork.nodes[terminalNode]['moccu']-=1
            return residueNetwork
        
        mainPaths,newResidueNetwork,dependentGraphs=self.shot_plan(residueNetwork,tasks,network,**kargs)

        if network is None:
            network=self.network
        if 'ps' in residueNetwork.graph:
            ps=residueNetwork.graph['ps']
        else:
            ps=1.0

        mainPaths_sim=[mainPath for mainPath,_ in mainPaths]
        pass
        success, switchNums, recoveryPaths, savedChannels, newResidueNetwork=\
            self.path_simulator(mainPaths_sim,network.host,newResidueNetwork,**kargs)

        for i in range(len(tasks)):
            totalSwitchNum=0

            maxMemory=network.get_maximum_node_memory()
            if maxMemory is not None:
                if network.get_maximum_node_memory()<len(tasks[i].nodes()):
                    tasks[i].graph['ifSuccess']=False
                    tasks[i].graph['deliverable']=False                
                    continue

            if 'ifGenerated' in tasks[i].graph:
                if not tasks[i].graph['ifGenerated']:
                    totalSwitchNum+=len(tasks[i].edges())
            else:
                tasks[i].graph['ifGenerated']=False
                totalSwitchNum+=len(tasks[i].edges())

            #This task is not involved in the current shot
            if (not dependentGraphs[i].graph['involved']) and (not tasks[i].graph['ifGenerated']):
                continue
            
            if 'shots' in tasks[i].graph:
                tasks[i].graph['shots']+=1
            else:
                tasks[i].graph['shots']=1
                    
            nodeList=list(tasks[i].nodes())
            newMemoryNodes=[]
            successVertexes=[]
            for v in nodeList:
                if not dependentGraphs[i].nodes[v]['InShot']:
                    continue
                else:
                    pathIndex=dependentGraphs[i].nodes[v]['dependent']
                    if len(p2pA.remove_ancilla(mainPaths_sim[pathIndex],residueNetwork))>1:
                        totalSwitchNum+=(1+switchNums[pathIndex])
                        
                if 'BellConsumed' in tasks[i].graph:
                    tasks[i].graph['BellConsumed']+=\
                        self.cal_path_bell_pairs(mainPaths_sim[dependentGraphs[i].nodes[v]['dependent']],residueNetwork)
                else:
                    tasks[i].graph['BellConsumed']=\
                        self.cal_path_bell_pairs(mainPaths_sim[dependentGraphs[i].nodes[v]['dependent']],residueNetwork)
                    
                if success[dependentGraphs[i].nodes[v]['dependent']]:
                    if 'binding' in tasks[i].graph:
                        terminalNode=network.network_task[tasks[i].graph['binding']].nodes[v]['node']
                    else:
                        terminalNode=tasks[i].nodes[v]['node']
                    if isinstance(terminalNode,list):
                        terminalNode=terminalNode[0]
                    if terminalNode != tasks[i].graph['source']:
                        newMemoryNodes.append(terminalNode)
                    successVertexes.append(v)
                    
            #calculate the cumulative memory
            if 'binding' in tasks[i].graph:
                basicMemory=len(network.network_task[tasks[i].graph['binding']].nodes())
            else:
                basicMemory=len(tasks[i].nodes())
            if 'CumulativeMem' in tasks[i].graph:
                tasks[i].graph['CumulativeMem']+=(basicMemory+len(newMemoryNodes))
            else:
                tasks[i].graph['CumulativeMem']=(basicMemory+len(newMemoryNodes))

            
            if totalSwitchNum>0:
                ifSuccess=(self.rng.binomial(totalSwitchNum,ps)==totalSwitchNum)
                #start from scratch
                if not ifSuccess:

                    #empty the memory
                    if tasks[i].graph['ifGenerated']:
                        if 'binding' in tasks[i].graph:
                            residueNetwork=empty_memory(tasks[i],residueNetwork)

                    currentCumulativeMem=tasks[i].graph['CumulativeMem']
                    currentBellConsumed=tasks[i].graph['BellConsumed']
                    currentShots=tasks[i].graph['shots']
                    if 'binding' in tasks[i].graph:
                        binding=tasks[i].graph['binding']
                    else:
                        binding=None
                    tasks[i]=network.network_task[binding].copy()
                    tasks[i].graph['CumulativeMem']=currentCumulativeMem
                    tasks[i].graph['BellConsumed']=currentBellConsumed
                    tasks[i].graph['shots']=currentShots
                    if binding is not None:
                        tasks[i].graph['binding']=binding
                    tasks[i].graph['ifSuccess']=False
                    tasks[i].graph['ifGenerated']=False
                    tasks[i].graph['deliverable']=True
                
                else:
                    if not tasks[i].graph['ifGenerated']:
                        tasks[i].graph['ifGenerated']=True

                        #consume the root memory for the first time
                        root=tasks[i].graph['source']
                        residueNetwork.nodes[root]['moccu']+=len(tasks[i].nodes())
                            
                    #consume the memory for the terminal nodes
                    for node in newMemoryNodes:
                        residueNetwork.nodes[node]['moccu']+=1

                        root=tasks[i].graph['source']
                        residueNetwork.nodes[root]['moccu']-=1
                    
                    for v in successVertexes:
                        tasks[i].remove_node(v)
                
            if len(tasks[i].nodes())==0:
                tasks[i].graph['ifSuccess']=True
                
                #clean the memory
                if 'binding' in tasks[i].graph:
                    residueNetwork=empty_memory(tasks[i],residueNetwork)

        residueNetwork.graph['recMemUsed']=newResidueNetwork.graph['recMemUsed']
        residueNetwork.graph['recBellUsed']=newResidueNetwork.graph['recBellUsed']

        residueNetwork=self.add_additional_resources(residueNetwork,savedChannels,**kargs)
            
        return residueNetwork,tasks            
                
    
    def run_simulation(self, network:p2pT.P2PTask|None=None, \
                       maxLoad:int=5, maxShot:int|None=None, \
                       maxMemoryShots:int|None=None, **kargs) -> Tuple[List[nx.Graph],int]|Tuple[List[nx.Graph],int,List[nx.Graph]]:
        return super().run_simulation(network,maxLoad,maxShot,maxMemoryShots,**kargs)
        
class P2PGSTSimulator(Simulator):
    '''
    description: The simulator for the P2P Graph State Distribution algorithm.
    '''
    def __init__(self,network:p2pT.P2PTask,seed:int=0):
        '''
        description: The constructor of the simulator.

        param {p2pT.P2PTask} network: The network to be simulated.

        param {int} seed: The random seed.
        '''
        super().__init__(network,seed)

    def path_simulator(self, mainPaths:List[List[int]], host:int,\
                       residueNetwork:nx.Graph,**kargs)\
                       ->Tuple[List[bool],List[int],\
                        Dict[int,List[Tuple[int,List[int],float]]],nx.Graph]:
        return super().path_simulator(mainPaths, host, residueNetwork, **kargs)
    
    def add_additional_resources(self,residueNetwork: nx.Graph, \
                                 savedChannels: List[List[Dict[str,Any]]],\
                                 **kargs)->nx.Graph:
        return super().add_additional_resources(residueNetwork, savedChannels, **kargs)
    
    def run_simulation(self, network: p2pT.P2PTask |None= None, maxLoad: int = 5, \
                       maxShot: int|None = None, maxMemoryShots: int|None = None, **kargs) \
                        -> Tuple[List[nx.Graph],int]|Tuple[List[nx.Graph],int,List[nx.Graph]]:
        return super().run_simulation(network, maxLoad, maxShot, maxMemoryShots, **kargs)
    
    def shot_plan(self, residueNetwork: nx.Graph, tasks: List[nx.Graph], network: p2pT.P2PTask|None = None,**kargs) \
        -> Tuple[List[Tuple[List[int],float]],nx.Graph,List[nx.Graph]]:
        '''
        description: Run one shot of the simulation, the detailed implementation of the shot plan for the base class.

        param {nx.Graph} residueNetwork: The residue network before this shot.

        param {List[nx.Graph]} tasks: The task to be simulated.

        param {p2pT.P2PTask} network: The network to be simulated.

        return {Tuple[List[Tuple[List[int],float]],nx.Graph,List[nx.Graph]]} : The main paths, the residue network and the dependent graph.
        '''
        def get_original_node(task: nx.Graph, vertex: int)->int:
            if 'binding' in task.graph:
                originalNode=network.network_task[tasks[i].graph['binding']].nodes[vertex]['node']
            else:
                originalNode=tasks[i].nodes[vertex]['node']
            if isinstance(originalNode,list):
                originalNode=originalNode[0]
            return originalNode

        residueNetwork_bk=residueNetwork.copy()
        mainPaths=[]
        dependentGraphs=[]

        if network is None:
            network=self.network

        if 'MemoryStrategy' not in kargs:
            kargs['MemoryStrategy']='Standard'
        
        if 'ifReset' in kargs:
            ifReset=kargs['ifReset']
        else:
            ifReset=True

        index=0
        for i in range(len(tasks)):
            task=tasks[i]
            controllable=None
            if not network.ifFullControllable:
                controllable=[]
                for node in residueNetwork_bk.nodes():
                    if 'controllable' in residueNetwork_bk.nodes[node]:
                        if not residueNetwork_bk.nodes[node]['controllable']:
                            if task.graph['binding'] not in residueNetwork_bk.nodes[node]['active']:
                                continue
                    else:
                        if task.graph['binding'] not in residueNetwork_bk.nodes[node]['active']:
                            continue
                    controllable.append(node)

           #Modifying submitted task to get rid of the nodes that are not possible to be distributed in current shot
            submittedTask=task.copy()
            for vertex in task.nodes():
                if isinstance(task.nodes[vertex]['node'],int):
                    task.nodes[vertex]['node']=[task.nodes[vertex]['node']]
                originalNode=get_original_node(task,vertex)
                if not 'ifGenerated' in task.nodes[vertex]\
                    or (not task.nodes[vertex]['ifGenerated']):

                    node=task.nodes[vertex]['node'][0]
                    if residueNetwork_bk.nodes[node]['memory'] is None or\
                        (residueNetwork_bk.nodes[node]['moccu']<residueNetwork_bk.nodes[node]['memory']):
                        residueNetwork_bk.nodes[node]['moccu']+=1
                    else:
                        submittedTask.remove_node(vertex)
            if len(submittedTask.nodes())==0:
                dependentGraphs.append(nx.Graph())
                continue
            
            taskMainPaths,edgeDependentDict,VRM=p2pA.find_P2P_GST_paths(submittedTask,residueNetwork_bk,\
                                                                        controllable=controllable,ifReset=ifReset)
            
            #update the MainPaths
            taskMainPathsShot=taskMainPaths[0]
            mainPath=[(path['path'].copy(),path['cost']) for path in taskMainPathsShot]

            #build the dependent graph
            dependentGraph=nx.Graph()
            for vertex in VRM.keys():
                dependentGraph.add_node(vertex)
                dependentGraph.nodes[vertex]['node']={}
                originalNode=get_original_node(task,vertex)
                for node in VRM[vertex].keys():
                    dependentGraph.nodes[vertex]['node'][node]=[]
                    if node==originalNode:
                        keep=True
                    else:
                        keep=False
                    for qubit in VRM[vertex][node]:
                        if not qubit['ifFixed']:
                            continue
                        if qubit['shot']==0:
                            qubitDict={'prev':qubit['prev']}
                            if qubit['mainPathDict'] is not None:
                                qubitDict['mainPathIndex']=index+\
                                taskMainPathsShot.index(qubit['mainPathDict'])
                            else:
                                qubitDict['mainPathIndex']=None
                            dependentGraph.nodes[vertex]['node'][node].append(qubitDict)
                        if qubit['shot']>0 and len(dependentGraph.nodes[vertex]['node'][node])>0:
                            keep=True
                    if len(dependentGraph.nodes[vertex]['node'][node])==0:
                        del dependentGraph.nodes[vertex]['node'][node]
                    else:
                        if len(dependentGraph.nodes[vertex]['node'][node])>0:
                            dependentGraph.nodes[vertex]['node'][node][0]['keep']=keep
                    
                    #deal with the memory
                    if kargs['MemoryStrategy']=='Standard' or 'Maximum':
                        if node==originalNode:
                            continue
                        if node in dependentGraph.nodes[vertex]['node']\
                            and dependentGraph.nodes[vertex]['node'][node][0]['keep']:
                            if residueNetwork_bk.nodes[node]['memory'] is None\
                                or residueNetwork_bk.nodes[node]['memory']-residueNetwork_bk.nodes[node]['moccu']>0:
                                residueNetwork_bk.nodes[node]['moccu']+=1
                            else:
                                #abandon the qubit
                                dependentGraph.nodes[vertex]['node'][node][0]['keep']=False
                    elif kargs['MemoryStrategy']!='Minimum':
                        raise ValueError('Unsupported Memory Strategy!')

            #scan over the redundant qubits
            if kargs['MemoryStrategy']=='Maximum':
                for vertex in VRM.keys():
                    originalNode=get_original_node(task,vertex)
                    for node in VRM[vertex].keys():
                        if node==originalNode:
                            continue
                        if node in dependentGraph.nodes[vertex]['node']\
                            and not dependentGraph.nodes[vertex]['node'][node][0]['keep']:
                            if residueNetwork_bk.nodes[node]['memory'] is None\
                                or residueNetwork_bk.nodes[node]['memory']-residueNetwork_bk.nodes[node]['moccu']>0:
                                residueNetwork_bk.nodes[node]['moccu']+=1
                            

            for edge in edgeDependentDict.keys():
                if edge[0]>edge[1]:
                    continue
                if len(edgeDependentDict[edge])==0:
                    continue

                dependentGraph.add_edge(edge[0],edge[1])
                if edgeDependentDict[edge]['shot']==0:
                    pathUV=edgeDependentDict[edge]['pathDict']['VU']
                    edgeDict={'directNode':edgeDependentDict[edge]['directNode']}
                    edgeDict['pathUV']=pathUV
                    edgeDict['pathIndex']=index+taskMainPathsShot.index(edgeDependentDict[edge]['pathDict'])
                    dependentGraph[edge[0]][edge[1]]['dependent']=edgeDict
                else:
                    dependentGraph[edge[0]][edge[1]]['dependent']=None
            dependentGraphs.append(dependentGraph)

            #update the residue network
            index+=len(mainPath)

            for path in mainPath:
                p2pA.remove_paths(residueNetwork_bk,path[0])
            mainPaths+=mainPath

        return mainPaths,residueNetwork_bk,dependentGraphs
    
    def run_one_shot(self, residueNetwork: nx.Graph, tasks: List[nx.Graph], network: p2pT.P2PTask|None = None,\
                     **kargs) -> Tuple[nx.Graph, List[nx.Graph]]:
        '''
        description: Run one shot of the simulation.

        param {nx.Graph} residueNetwork: The residue network before this shot.

        param {List[nx.Graph]} tasks: The task to be simulated.

        param {p2pT.P2PTask} network: The network to be simulated.

        return {Tuple[nx.Graph, List[nx.Graph]]} : The network after the shot and the residue tasks.

        '''
        def if_qubit_success(currentQubit:Dict[str,any],\
                             vertex:int,node:int,\
                             dependentGraph:nx.Graph,\
                             totalSwitchNum, \
                             mainPaths_sim:list[int],success:List[bool])->Tuple[bool,int]:
            if currentQubit['reached']!=None:
                return currentQubit['reached'], totalSwitchNum

            requiredPath=mainPaths_sim[currentQubit['mainPathIndex']]
            if node==currentQubit['prev']:
                totalSwitchNum+=1
                prevQubit=dependentGraph.nodes[vertex]['node'][node][0]
                currentQubit['reached'],totalSwitchNum=if_qubit_success(prevQubit,\
                                vertex,node,dependentGraph, totalSwitchNum,mainPaths_sim,success)
            else:
                if not success[currentQubit['mainPathIndex']]:
                    currentQubit['reached']=False
                    return False,totalSwitchNum
                else:
                    prevNode=currentQubit['prev']
                    

                    prevQubit=None
                    for qubit in dependentGraph.nodes[vertex]['node'][prevNode]:
                        if qubit['mainPathIndex']==currentQubit['mainPathIndex']:
                            prevQubit=qubit
                            break

                    currentQubit['reached'],totalSwitchNum=if_qubit_success(prevQubit,\
                                            vertex,prevNode,dependentGraph, totalSwitchNum,mainPaths_sim,success)
            return currentQubit['reached'],totalSwitchNum
            
        def get_original_node(task: nx.Graph, vertex: int)->int:
            if 'binding' in task.graph:
                originalNode=network.network_task[tasks[i].graph['binding']].nodes[vertex]['node']
            else:
                originalNode=tasks[i].nodes[vertex]['node']
            if isinstance(originalNode,list):
                originalNode=originalNode[0]
            return originalNode
        
        def empty_memory(task:nx.Graph,residueNetwork:nx.Graph)->nx.Graph:
            for vertex in task.nodes():
                originalNode=get_original_node(task,vertex)
                if 'ifGenerated' in task.nodes[vertex]\
                    and task.nodes[vertex]['ifGenerated']:
                    residueNetwork.nodes[originalNode]['moccu']-=1

            return residueNetwork
        
        if 'ps' in residueNetwork.graph:
            ps=residueNetwork.graph['ps']
        else:
            ps=1.0

        if 'MemoryStrategy' not in kargs:
            kargs['MemoryStrategy']='Standard'
        
        if network is None:
            network=self.network

        mainPaths,newResidueNetwork,dependentGraphs=self.shot_plan(residueNetwork,tasks,network,**kargs)

        mainPaths_sim=[mainPath for mainPath,_ in mainPaths]
        success, switchNum, _, saveChannels, newResidueNetwork=\
            self.path_simulator(mainPaths_sim,network.host,newResidueNetwork,**kargs)

        residueNetwork_mbk=residueNetwork.copy()
        for i in range(len(tasks)):
            if len(dependentGraphs[i].nodes())==0:
                continue

            #update the task
            if 'shots' in tasks[i].graph:
                tasks[i].graph['shots']+=1
            else:
                tasks[i].graph['shots']=1
            
            totalSwitchNum=0
            
            #initialize the qubit simulation results
            for vertex in dependentGraphs[i].nodes():
                for node in dependentGraphs[i].nodes[vertex]['node'].keys():
                    for qubit in dependentGraphs[i].nodes[vertex]['node'][node]:
                        if qubit['mainPathIndex'] is None:
                            qubit['reached']=True
                        else:
                            qubit['reached']=None
            
            minPathIndex=len(mainPaths_sim)
            maxPathIndex=0
            #compute the qubit simulation results
            for vertex in dependentGraphs[i].nodes():
                for node in dependentGraphs[i].nodes[vertex]['node'].keys():
                    for qubit in dependentGraphs[i].nodes[vertex]['node'][node]:
        
                        currentQubit=qubit

                        if qubit['mainPathIndex'] is not None:
                            if qubit['mainPathIndex']<minPathIndex:
                                minPathIndex=qubit['mainPathIndex']
                            if qubit['mainPathIndex']>maxPathIndex:
                                maxPathIndex=qubit['mainPathIndex']
                        if qubit['reached']==None:
                           qubit['reached'],totalSwitchNum=if_qubit_success(currentQubit,\
                                    vertex,node,dependentGraphs[i],totalSwitchNum,mainPaths_sim,success)

            #update bell pairs 
            BellConsumed=0
            for pathIndex in range(minPathIndex,maxPathIndex+1):
                BellConsumed+=self.cal_path_bell_pairs(mainPaths_sim[pathIndex],residueNetwork)
                totalSwitchNum+=switchNum[pathIndex]
            if 'BellConsumed' in tasks[i].graph:
                tasks[i].graph['BellConsumed']+=BellConsumed
            else:
                tasks[i].graph['BellConsumed']=BellConsumed

            #compute the edge simulation results
            for edge in dependentGraphs[i].edges():
                dependentGraphs[i].edges[edge]['success']=False
                if dependentGraphs[i].edges[edge]['dependent'] is None:
                    continue
                pathIndex=dependentGraphs[i].edges[edge]['dependent']['pathIndex']
                if not success[pathIndex]:
                    continue
                else:
                    start=mainPaths_sim[pathIndex][0]
                    end=mainPaths_sim[pathIndex][-1]
                    startQubit=None
                    endQubit=None
                    UV=dependentGraphs[i].edges[edge]['dependent']['pathUV']
                    if start==end:
                        startQubit=dependentGraphs[i].nodes[UV[0]]['node'][start][0]
                        endQubit=dependentGraphs[i].nodes[UV[-1]]['node'][end][0]
                        # When the qubits are in the same node, we need to reimburse the switch number.
                        totalSwitchNum-=1
                    else:
                        for qubit in dependentGraphs[i].nodes[UV[0]]['node'][start]:
                            if qubit['mainPathIndex']==pathIndex:
                                startQubit=qubit
                                break
                        for qubit in dependentGraphs[i].nodes[UV[-1]]['node'][end]:
                            if qubit['mainPathIndex']==pathIndex:
                                endQubit=qubit
                                break


                    if startQubit['reached'] and endQubit['reached']:
                        tasks[i].remove_edge(edge[0],edge[1])
                        dependentGraphs[i].edges[edge]['success']=True
                
                    
            ifSuccess=self.rng.binomial(totalSwitchNum,ps)==totalSwitchNum
            
            #update the task assignment and cumulative memory
            match kargs['MemoryStrategy']:
                case 'Minimum':
                    if 'CumulativeMem' in tasks[i].graph:
                        tasks[i].graph['CumulativeMem']+=len(dependentGraphs[i].nodes())
                    else:
                        tasks[i].graph['CumulativeMem']=len(dependentGraphs[i].nodes())

                    for vertex in dependentGraphs[i].nodes():
                        originalNode=get_original_node(tasks[i],vertex)
                        if 'ifGenerated' not in tasks[i].nodes[vertex] or\
                            not tasks[i].nodes[vertex]['ifGenerated']:
                            residueNetwork.nodes[originalNode]['moccu']+=1
                            residueNetwork_mbk.nodes[originalNode]['moccu']+=1
                            tasks[i].nodes[vertex]['ifGenerated']=True

                case 'Standard':
                    currentMemory=0
                    for vertex in dependentGraphs[i].nodes():
                        originalNode=get_original_node(tasks[i],vertex)
                        tasks[i].nodes[vertex]['node']=[originalNode]
                        for node in dependentGraphs[i].nodes[vertex]['node']:
                            if node==originalNode:
                                currentMemory+=1
                                if 'ifGenerated' not in tasks[i].nodes[vertex] or\
                                    not tasks[i].nodes[vertex]['ifGenerated']:
                                    residueNetwork.nodes[originalNode]['moccu']+=1
                                    residueNetwork_mbk.nodes[originalNode]['moccu']+=1
                                    tasks[i].nodes[vertex]['ifGenerated']=True

                            elif dependentGraphs[i].nodes[vertex]['node'][node][0]['keep']:
                                currentMemory+=1
                                residueNetwork_mbk.nodes[node]['moccu']+=1

                                if dependentGraphs[i].nodes[vertex]['node'][node][0]['reached']:
                                    if isinstance(tasks[i].nodes[vertex]['node'],list):
                                        tasks[i].nodes[vertex]['node'].append(node)
                                    else:
                                        tasks[i].nodes[vertex]['node']=[tasks[i].nodes[vertex]['node'],node]
                    
                    if 'CumulativeMem' in tasks[i].graph:
                        tasks[i].graph['CumulativeMem']+=currentMemory
                    else:
                        tasks[i].graph['CumulativeMem']=currentMemory

                case 'Maximum':
                    #need to consider memory constraint
                    #first, scan over the required memory
                    currentMemory=0
                    for vertex in dependentGraphs[i].nodes():
                        originalNode=get_original_node(tasks[i],vertex)
                        tasks[i].nodes[vertex]['node']=[originalNode]
                        for node in dependentGraphs[i].nodes[vertex]['node']:
                            if node==originalNode:
                                currentMemory+=1
                                if 'ifGenerated' not in tasks[i].nodes[vertex] or\
                                    not tasks[i].nodes[vertex]['ifGenerated']:
                                    residueNetwork.nodes[originalNode]['moccu']+=1
                                    residueNetwork_mbk.nodes[originalNode]['moccu']+=1
                                    tasks[i].nodes[vertex]['ifGenerated']=True

                            elif dependentGraphs[i].nodes[vertex]['node'][node][0]['keep']:
                                currentMemory+=1
                                residueNetwork_mbk.nodes[node]['moccu']+=1

                                if dependentGraphs[i].nodes[vertex]['node'][node][0]['reached']:
                                    if isinstance(tasks[i].nodes[vertex]['node'],list):
                                        tasks[i].nodes[vertex]['node'].append(node)
                                    else:
                                        tasks[i].nodes[vertex]['node']=[tasks[i].nodes[vertex]['node'],node]

                    #scan over the redundant memory
                    for vertex in dependentGraphs[i].nodes():
                        originalNode=get_original_node(tasks[i],vertex)
                        for node in dependentGraphs[i].nodes[vertex]['node']:
                            if node==originalNode:
                                continue
                            if not dependentGraphs[i].nodes[vertex]['node'][node][0]['keep']:
                                if  residueNetwork_mbk.nodes[node]['memory'] is None \
                                    or residueNetwork_mbk.nodes[node]['moccu']<residueNetwork_mbk.nodes[node]['memory']:
                                    currentMemory+=1
                                    residueNetwork_mbk.nodes[node]['moccu']+=1
                                    if dependentGraphs[i].nodes[vertex]['node'][node][0]['reached'] and\
                                        nx.degree(tasks[i],vertex)>0:
                                        if isinstance(tasks[i].nodes[vertex]['node'],list):
                                            tasks[i].nodes[vertex]['node'].append(node)
                                        else:
                                            tasks[i].nodes[vertex]['node']=[tasks[i].nodes[vertex]['node'],node]
                                            
                    if 'CumulativeMem' in tasks[i].graph:
                        tasks[i].graph['CumulativeMem']+=currentMemory
                    else:
                        tasks[i].graph['CumulativeMem']=currentMemory

            #start from scratch
            if not ifSuccess:
                empty_memory(tasks[i],residueNetwork)
                currentCumulativeMem=tasks[i].graph['CumulativeMem']
                currentBellConsumed=tasks[i].graph['BellConsumed']
                currentShots=tasks[i].graph['shots']
                binding=tasks[i].graph['binding']
                tasks[i]=network.network_task[binding].copy()
                tasks[i].graph['CumulativeMem']=currentCumulativeMem
                tasks[i].graph['BellConsumed']=currentBellConsumed
                tasks[i].graph['shots']=currentShots
                tasks[i].graph['binding']=binding
                tasks[i].graph['ifSuccess']=False
                tasks[i].graph['deliverable']=True
                continue

            #update the task status
            if len(tasks[i].edges())==0:
                tasks[i].graph['ifSuccess']=True
                empty_memory(tasks[i],residueNetwork)
        #add the additional resources
        residueNetwork.graph['recMemUsed']=newResidueNetwork.graph['recMemUsed']
        residueNetwork.graph['recBellUsed']=newResidueNetwork.graph['recBellUsed']
        residueNetwork=self.add_additional_resources(residueNetwork,saveChannels,**kargs)
        return residueNetwork,tasks
    
class STP2PGSTSimulator(P2PGSTSimulator):
    '''
    description: The simulator for the ST-P2P Graph State Distribution algorithm.
    '''
    def __init__(self,network:p2pT.P2PTask,seed:int=0):
        '''
        description: The constructor of the simulator.

        param {p2pT.P2PTask} network: The network to be simulated.

        param {int} seed: The random seed.
        '''
        super().__init__(network,seed)

    def path_simulator(self, mainPaths:List[List[int]], host:int,\
                       residueNetwork:nx.Graph,**kargs)\
                       ->Tuple[List[bool],List[int],\
                        Dict[int,List[Tuple[int,List[int],float]]],nx.Graph]:
        return super().path_simulator(mainPaths, host, residueNetwork, **kargs)
    
    def add_additional_resources(self,residueNetwork: nx.Graph, \
                                 savedChannels: List[List[Dict[str,Any]]],\
                                 **kargs)->nx.Graph:
        return super().add_additional_resources(residueNetwork, savedChannels, **kargs)
    
    def run_simulation(self, network:p2pT.P2PTask=None, maxLoad:int=5, maxShot:int=None, maxMemoryShots:int=None, **kargs) \
        -> Tuple[List[nx.Graph],int]:
        return super().run_simulation(network,maxLoad,maxShot,maxMemoryShots,**kargs)
    
    def shot_plan(self, residueNetwork: nx.Graph, tasks: List[nx.Graph], network: p2pT.P2PTask|None=None,**kargs) \
        -> Tuple[List[Tuple[List[int],float]],nx.Graph,List[nx.Graph]]:
        '''
        description: Run one shot of the simulation, the detailed implementation of the shot plan for the base class.

        param {nx.Graph} residueNetwork: The residue network before this shot. 
        Warning: The residueNetwork will be modified to record the memory occupation.

        param {List[nx.Graph]} tasks: The task to be simulated.

        param {p2pT.P2PTask} network: The network to be simulated.

        return {Tuple[List[Tuple[List[int],float]],nx.Graph,List[nx.Graph]]} : The main paths, the residue network and the dependent graph.
        '''
        def consume_memory(residueNetwork: nx.Graph, node: int, memory: int):
            residueNetwork.nodes[node]['moccu']+=memory

            return residueNetwork
        def append_qubit_dict(dependentGraph:nx.Graph, vertex:int, node:int, qubitDict:Dict[str,any], \
                              ifFirst:bool=False)->Tuple[nx.Graph,bool]:
            if node not in dependentGraph.nodes[vertex]['simuNode']:
                dependentGraph.nodes[vertex]['simuNode'][node]=[qubitDict]
                return dependentGraph,qubitDict['keep']
            else:
                #determine whether the memory shall be consumed or not
                consumed=qubitDict['keep'] and \
                    (not dependentGraph.nodes[vertex]['simuNode'][node][0]['keep'])
                if not ifFirst:
                    dependentGraph.nodes[vertex]['simuNode'][node].append(qubitDict)
                    if consumed:
                        dependentGraph.nodes[vertex]['simuNode'][node][0]['keep']=True
                else:
                    #reset the memory configuration
                    if not qubitDict['keep'] and \
                        dependentGraph.nodes[vertex]['simuNode'][node][0]['keep']:
                        qubitDict['keep']=True
                    dependentGraph.nodes[vertex]['simuNode'][node].insert(0,qubitDict)
            return dependentGraph, consumed

        #initialize
        residueNetwork_bk=residueNetwork.copy()
        mainPaths=[]
        dependentGraphs=[]
        if network is None:
            network=self.network

        if 'MemoryStrategy' not in kargs:
            kargs['MemoryStrategy']='Standard'
        
        #loop over the tasks
        index=-1
        for i in range(len(tasks)):
            task=tasks[i]
            
            #collect the controllable nodes
            controllable=None
            if not network.ifFullControllable:
                controllable=[]
                for virtualVertex in residueNetwork_bk.nodes():
                    if 'controllable' in residueNetwork_bk.nodes[virtualVertex]:
                        if not residueNetwork_bk.nodes[virtualVertex]['controllable']:
                            if task.graph['binding'] not in residueNetwork_bk.nodes[virtualVertex]['active']:
                                continue
                    else:
                        if task.graph['binding'] not in residueNetwork_bk.nodes[virtualVertex]['active']:
                            continue
                    controllable.append(virtualVertex)
            
            #get the shot plan
            submittedTask=task.copy()
            ifFirstTask=(i==0)
            res, n_shot=p2pSTA.memflow_bin_search(submittedTask,residueNetwork_bk,\
                                                controllable=controllable,ifFirst=ifFirstTask,\
                                                **kargs)

            
            
            #build the dependent graph
            dependentGraph=nx.Graph()
            dependentGraph.add_nodes_from(task.nodes(data=True))
            for virtualVertex in dependentGraph.nodes():
                dependentGraph.nodes[virtualVertex]['simuNode']={}

            #expected shot
            dependentGraph.graph['n_shot']=n_shot
            ##precalculate the memory
            dependentGraph.graph['cumulativeMem']=0
            ##deal with non-injective mapping
            dependentGraph.graph['addSwitch']=0

            if res is None:
                dependentGraph.graph['pathIndexRange']=(index+1,index)
                dependentGraphs.append(dependentGraph)
                continue

            taskMainPaths,VRM=res

            orderCount=0
            mainPath=[]
            ## search through the main paths
            for j in range(len(taskMainPaths)):
                #get the task main path in the desired order
                mainPathRaw={'path':taskMainPaths[j]['path'].copy(),\
                             'cost':taskMainPaths[j]['cost'],\
                             'UV':taskMainPaths[j]['UV']}
                pathUV=mainPathRaw['UV']

                if len(mainPathRaw['path'])==1:
                    memNode=mainPathRaw['path'][0]
                    currentShot,_=p2pSTA.get_node_shot(residueNetwork_bk,memNode)
                    if currentShot==n_shot and n_shot==1:
                        dependentGraph.graph['addSwitch']+=1
                
                reversedFlow=(('reversedFlow' in taskMainPaths[j]) \
                            and taskMainPaths[j]['reversedFlow'])
                #determine the flow direction
                flowQubit=None
                for qubit in VRM[pathUV[0]][mainPathRaw['path'][0]]:
                    if qubit['mainPathDict'] is taskMainPaths[j]:
                        flowQubit=qubit
                        break
                if flowQubit['prev']==mainPathRaw['path'][-1] \
                    and len(mainPathRaw['path'])>1:
                    mainPathRaw['path']=mainPathRaw['path'][::-1]
                    reversedFlow=True
 
                partialPaths,_=p2pSTA.decompose_mainPathRaw(mainPathRaw,residueNetwork_bk)
                pathUV=mainPathRaw['UV']
                for k in range(len(partialPaths)):
                    partialPath,saveStatus=partialPaths[k]
                    index+=1
                    orderCount+=1
                    if len(partialPath)>1:
                        #for the left node
                        if saveStatus[0]:
                            previousNode=-1
                        else:
                            previousNode=partialPath[0]
                        connectNode=((pathUV[0]!=pathUV[1]) and (k==len(partialPaths)-1))
                        if connectNode:
                            qubitPathUV=(pathUV[0],pathUV[1])
                        else:
                            qubitPathUV=(pathUV[0],pathUV[0])
                        #add the left qubit information
                        leftQubitDict={'prev':previousNode,'mainPathIndex':index,'keep':saveStatus[0],\
                                             'order':orderCount,'pathUV':qubitPathUV,'reversedFlow':reversedFlow}
                        dependentGraph,consumed=append_qubit_dict(dependentGraph, pathUV[0], partialPath[0], leftQubitDict)
                        if consumed:
                            residueNetwork_bk=consume_memory(residueNetwork_bk, partialPath[0], 1)
                            dependentGraph.graph['cumulativeMem']+=1
                        #for the right node
                        ##determine its vertex information
                        qubitReversedFlow=reversedFlow
                        if connectNode:
                            addVertex=pathUV[1]
                            qubitReversedFlow=True
                            #reset the previous node
                            if saveStatus[1]:
                                previousNode=-1
                            else:
                                previousNode=partialPath[-1]
                            #add edge dependent information
                            edgeDependentDict={'mainPathIndex':index,'order':orderCount,'pathUV':(pathUV[0],pathUV[1])}
                            if (pathUV[0],pathUV[1]) not in dependentGraph.edges():
                                dependentGraph.add_edge(pathUV[0],pathUV[1])
                            dependentGraph[pathUV[0]][pathUV[1]]['dependent']=edgeDependentDict
                        else:
                            addVertex=pathUV[0]
                            previousNode=partialPath[0]
                        #add the right qubit information
                        rightQubitDict={'prev':previousNode,'mainPathIndex':index,'keep':saveStatus[1],\
                                            'order':orderCount,'pathUV':qubitPathUV, 'reversedFlow':qubitReversedFlow}
                        dependentGraph,consumed=append_qubit_dict(dependentGraph, addVertex, partialPath[-1], rightQubitDict)
                        if consumed:
                            residueNetwork_bk=consume_memory(residueNetwork_bk, partialPath[-1], 1)
                            dependentGraph.graph['cumulativeMem']+=1
                    else:
                        #determine whether it is left node or right node
                        ifLeft=None
                        connectNode=((pathUV[0]!=pathUV[1]) and (k==len(partialPaths)-1))
                        match saveStatus:
                            case [True,False]:
                                ifLeft=True
                            case [False,False]:
                                if not connectNode:
                                    ifLeft=True
                            case [False,True]:
                                ifLeft=False

                        if connectNode:
                            #there two nodes to be added
                            #for the left node
                            if saveStatus[0]:
                                previousNode=-1
                            else:
                                previousNode=partialPath[0]
                            #add the left qubit information
                            leftQubitDict={'prev':previousNode,'mainPathIndex':index,'keep':saveStatus[0],\
                                             'order':orderCount,'pathUV':(pathUV[0],pathUV[1]),'reversedFlow':reversedFlow}
                            dependentGraph,consumed=append_qubit_dict(dependentGraph, pathUV[0], partialPath[0], leftQubitDict)
                            if consumed:
                                residueNetwork_bk=consume_memory(residueNetwork_bk, partialPath[0], 1)
                                dependentGraph.graph['cumulativeMem']+=1
                            #for the right node
                            if saveStatus[1]:
                                previousNode=-1
                            else:
                                previousNode=partialPath[0]
                            #add the right qubit information
                            rightQubitDict={'prev':previousNode,'mainPathIndex':index,'keep':saveStatus[1],\
                                             'order':orderCount,'pathUV':(pathUV[0],pathUV[1]),'reversedFlow':not reversedFlow}
                            dependentGraph,consumed=append_qubit_dict(dependentGraph, pathUV[1], partialPath[0], rightQubitDict)
                            if consumed:
                                residueNetwork_bk=consume_memory(residueNetwork_bk, partialPath[0], 1)
                                dependentGraph.graph['cumulativeMem']+=1
                            #add edge dependent information
                            edgeDependentDict={'mainPathIndex':index,'order':orderCount,'pathUV':(pathUV[0],pathUV[1])}
                            dependentGraph.add_edge(pathUV[0],pathUV[1])
                            dependentGraph[pathUV[0]][pathUV[1]]['dependent']=edgeDependentDict
                        else:
                            #there is only one node to be added
                            if ifLeft:
                                if saveStatus[0]:
                                    previousNode=-1
                                else:
                                    previousNode=partialPath[0]
                                #add the qubit information
                                qubitDict={'prev':previousNode,'mainPathIndex':index,'keep':saveStatus[0],\
                                           'order':orderCount,'pathUV':(pathUV[0],pathUV[0]),'reversedFlow':reversedFlow}
                                dependentGraph,consumed=append_qubit_dict(dependentGraph, pathUV[0], partialPath[0], qubitDict)
                                if consumed:
                                    residueNetwork_bk=consume_memory(residueNetwork_bk, partialPath[0], 1)
                                    dependentGraph.graph['cumulativeMem']+=1
                            else:
                                previousNode=partialPath[0]
                                #add the qubit information
                                qubitDict={'prev':previousNode,'mainPathIndex':index,'keep':saveStatus[1],\
                                           'order':orderCount,'pathUV':(pathUV[0],pathUV[0]),'reversedFlow':reversedFlow}
                                dependentGraph,consumed=append_qubit_dict(dependentGraph, pathUV[0], partialPath[0], qubitDict)
                                if consumed:
                                    residueNetwork_bk=consume_memory(residueNetwork_bk, partialPath[0], 1)
                                    dependentGraph.graph['cumulativeMem']+=1

                        
                #deal with the case where there is no partial paths                
                if len(partialPaths)==0:
                    orderCount+=1
                    if pathUV[0]!=pathUV[1]:
                        dependentGraph.add_edge(pathUV[0],pathUV[1])
                        edgeDependentDict={'mainPathIndex':None,'order':orderCount,'pathUV':pathUV}
                        dependentGraph[pathUV[0]][pathUV[1]]['dependent']=edgeDependentDict
                
                #update the main paths
                for path,_ in partialPaths:
                    mainPath.append((path,p2pA.ET_metric(residueNetwork_bk,path)))
                    #update the residue network
                    p2pA.remove_paths(residueNetwork_bk,path)

            #add the missing qubits from virtual nodes
            for virtualVertex in dependentGraph.nodes():
                if 'ifVirtual' in dependentGraph.nodes[virtualVertex] \
                    and dependentGraph.nodes[virtualVertex]['ifVirtual']:
                    #get the previous node of the virtual vertex
                    preNode=VRM[virtualVertex]['preNode']
                    sourceVertex=dependentGraph.nodes[virtualVertex]['sourceVertex']
                    #This virtual vertex is attached to a real vertex
                    if preNode is not None:
                        _,previousNode=p2pSTA.get_node_shot(residueNetwork_bk, preNode)

                        order=dependentGraph.nodes[sourceVertex]['simuNode'][preNode][0]['order']
                        motherQubitDict={'prev':previousNode,'virtualNode':virtualVertex,\
                                 'mainPathIndex':None,'keep':False,'order':order,'pathUV':None,\
                                 'reversedFlow':VRM[virtualVertex]['reversedFlow']}
                        #initialize the mother dict for the virtual vertex
                        for node in dependentGraph.nodes[virtualVertex]['node']:
                            #judging whether the qubits is the first qubit in VRM
                            currentDict=motherQubitDict.copy()
                            if VRM[sourceVertex][node][0]['mainPathDict'] is None:
                                    
                                    dependentGraph,_=append_qubit_dict(dependentGraph, sourceVertex, \
                                                                     node, currentDict, ifFirst=True)
                            elif node==preNode:
                                dependentGraph,_=append_qubit_dict(dependentGraph, sourceVertex, node, currentDict)

                            if node==preNode:
                                if dependentGraph.nodes[sourceVertex]['simuNode'][node][0]['keep']:
                                    continue
                                else:
                                    if residueNetwork_bk.nodes[node]['memory'] is None or\
                                        residueNetwork_bk.nodes[node]['moccu']<residueNetwork_bk.nodes[node]['memory']:
                                        dependentGraph.nodes[sourceVertex]['simuNode'][node][0]['keep']=True
                                        residueNetwork_bk=consume_memory(residueNetwork_bk, node, 1)
                                        dependentGraph.graph['cumulativeMem']+=1
                
            #update the main paths and the dependent graphs
            ## assign the range of path indexes
            dependentGraph.graph['pathIndexRange']=(index-len(mainPath)+1,index)
            mainPaths+=mainPath
            dependentGraphs.append(dependentGraph)

        return mainPaths,residueNetwork_bk,dependentGraphs
        
    
    def run_one_shot(self, residueNetwork: nx.Graph, tasks: p2pA.List[nx.Graph], network: p2pT.P2PTask | None = None, **kargs) -> p2pA.Tuple[nx.Graph | p2pA.List[nx.Graph]]:
        '''
        description: Run one shot of the simulation.

        param {nx.Graph} residueNetwork: The residue network before this shot.

        param {List[nx.Graph]} tasks: The task to be simulated.

        param {p2pT.P2PTask} network: The network to be simulated.

        return {Tuple[nx.Graph, List[nx.Graph]]} : The network after the shot and the residue tasks.

        '''
        def qubit_merge_with_ancestor(currentQubit:Dict[str,any],\
                            vertex:int,node:int,\
                            dependentGraph:nx.Graph,\
                            totalSwitchNum, \
                            mainPaths_sim:list[int],\
                            success:List[bool])->Tuple[Dict[str,any],int]:
            #check the ancestor
            if 'isAncestor' in currentQubit and currentQubit['isAncestor']:
                return currentQubit,totalSwitchNum
            if 'reached' in currentQubit and currentQubit['reached']:
                return currentQubit['ancestor'],totalSwitchNum
            currentQubit['reached']=True
            #retrieve information of this qubit
            mainPathIndex=currentQubit['mainPathIndex']
            prevNode=currentQubit['prev']
            ifSave=dependentGraph.nodes[vertex]['simuNode'][node][0]['keep']
            #initialize qubit information
            currentQubit['isAncestor']=False
            currentQubit['ancestor']=None
            currentQubit['node']=[node]
            currentQubit['edges']=[]
            inheritEdge=False
            #check whether the qubit belongs to a virtual node and inherits the edge
            if 'virtualNode' in currentQubit:
                virtualVertex=currentQubit['virtualNode']
                #inherits the edges
                if prevNode==node:
                    currentQubit['edges']=dependentGraph.nodes[virtualVertex]['edges'].copy()
                    #virtual node requires more switch
                    totalSwitchNum+=1
                    #require to inherit the edge
                    inheritEdge=True
            currentQubit['sourceVertex']=vertex
            #determine whether it is a left node or a right node
            if node==currentQubit['prev'] or prevNode==-1:
                leftNode=True 
                totalSwitchNum+=1
            else:
                leftNode=False
            #check whether the qubit is ancestor itself
            if prevNode==-1:
                if ifSave and success[mainPathIndex]:
                    currentQubit['isAncestor']=True
                    currentQubit['ancestor']=currentQubit
                    return currentQubit,totalSwitchNum 
                else:
                    return None,totalSwitchNum
            else:
                #find the ancestor qubit
                ancestorQubit,totalSwitchNum=qubit_merge_with_ancestor(\
                    dependentGraph.nodes[vertex]['simuNode'][prevNode][0],\
                    vertex,prevNode,dependentGraph,totalSwitchNum,\
                    mainPaths_sim,success)
                #if the ancestor fails
                if ancestorQubit is None:
                    #left node situation
                    if leftNode:
                        #save and the current qubit becomes the ancestor
                        if ifSave and (mainPathIndex is None or success[mainPathIndex]):
                            currentQubit['isAncestor']=True
                            currentQubit['ancestor']=currentQubit
                            currentQubit['reversedFlow']=\
                                dependentGraph.nodes[vertex]['simuNode'][node][0]['reversedFlow']
                            return currentQubit,totalSwitchNum
                        #not save and the current qubit is removed
                        else:
                            return None,totalSwitchNum
                    #right node situation
                    elif ifSave and (currentQubit is dependentGraph.nodes[vertex]['simuNode'][node][0]):
                        currentQubit['isAncestor']=True
                        currentQubit['ancestor']=currentQubit
                        currentQubit['reversedFlow']=\
                                dependentGraph.nodes[vertex]['simuNode'][node][0]['reversedFlow']
                        return currentQubit,totalSwitchNum
                    else:
                        return None,totalSwitchNum
                else:
                    if (mainPathIndex is None) or success[mainPathIndex]:
                        #left node situation
                        if leftNode:
                            #inherit the ancestor
                            #update the ancestor information
                            currentQubit['ancestor']=ancestorQubit
                            currentQubit['node']=ancestorQubit['node']
                            if inheritEdge:
                                ancestorQubit['edges']+=currentQubit['edges']
                            currentQubit['edges']=ancestorQubit['edges']
                            return ancestorQubit,totalSwitchNum
                        #right node situation
                        else:
                            #update the ancestor information
                            currentQubit['ancestor']=ancestorQubit
                            ancestorQubit['edges']+=currentQubit['edges']
                            if ifSave:
                                for node in currentQubit['node']:
                                    if node not in ancestorQubit['node']:
                                        ancestorQubit['node'].append(node)
                            currentQubit['node']=ancestorQubit['node']
                            currentQubit['edges']=ancestorQubit['edges']
                            return ancestorQubit,totalSwitchNum
                    #right node situation (the first right node)
                    elif not leftNode and ifSave\
                        and (currentQubit is dependentGraph.nodes[vertex]['simuNode'][node][0]):
                        currentQubit['isAncestor']=True
                        currentQubit['ancestor']=currentQubit
                        currentQubit['reversedFlow']=\
                                dependentGraph.nodes[vertex]['simuNode'][node][0]['reversedFlow']
                        return currentQubit,totalSwitchNum
                        
            return currentQubit['ancestor'],totalSwitchNum

        if 'ps' in residueNetwork.graph:
            ps=residueNetwork.graph['ps']
        else:
            ps=1.0
        
        if network is None:
            network=self.network

        mainPaths,newResidueNetwork,dependentGraphs=self.shot_plan(residueNetwork,tasks,network,**kargs)
        mainPaths_sim=[mainPath for mainPath,_ in mainPaths]
        success, switchNum, _, saveChannels,newResidueNetwork=\
            self.path_simulator(mainPaths_sim,network.host,newResidueNetwork,**kargs)
        
        
        for i in range(len(tasks)):
            #skip the task that is not involved in this shot
            ## get the path index range
            pathIndexRange=dependentGraphs[i].graph['pathIndexRange']
            if pathIndexRange[1]-pathIndexRange[0]+1==0 and\
                ('ifGenerated' not in tasks[i].graph\
                or not tasks[i].graph['ifGenerated']):
                     continue
            #update the task resources
            if 'binding' not in tasks[i].graph:
                newTask=tasks[i].copy()
            else:
                newTask=network.network_task[tasks[i].graph['binding']].copy()
                newTask.graph=tasks[i].graph.copy()
            if 'shots' in tasks[i].graph:
                newTask.graph['shots']=tasks[i].graph['shots']+1
            else:
                newTask.graph['shots']=1
            ##update the cumulative 
            if 'CumulativeMem' in tasks[i].graph:
                newTask.graph['CumulativeMem']=tasks[i].graph['CumulativeMem']+\
                    dependentGraphs[i].graph['cumulativeMem']
            else:
                newTask.graph['CumulativeMem']=dependentGraphs[i].graph['cumulativeMem']
            ##update the bell pairs
            allPathSuccess=True
            BellConsumed=0
            totalSwitchNum=0
            if pathIndexRange[1]-pathIndexRange[0]+1>0:
                for pathIndex in range(pathIndexRange[0],pathIndexRange[1]+1):
                    currentBellConsumed=self.cal_path_bell_pairs(mainPaths_sim[pathIndex],residueNetwork)
                    BellConsumed+=currentBellConsumed
                    totalSwitchNum+=switchNum[pathIndex]
                    #reimburse the single point path
                    if currentBellConsumed==0:
                        totalSwitchNum-=1
                    if not success[pathIndex]:
                        allPathSuccess=False
            if 'BellConsumed' in tasks[i].graph:
                newTask.graph['BellConsumed']=tasks[i].graph['BellConsumed']+BellConsumed
            else:
                newTask.graph['BellConsumed']=BellConsumed

            #compute the qubit simulation results and merge connected components
            for vertex in dependentGraphs[i].nodes():
                for node in dependentGraphs[i].nodes[vertex]['simuNode'].keys():
                    for qubit in dependentGraphs[i].nodes[vertex]['simuNode'][node]:
                        currentQubit=qubit
                        if 'reached' not in qubit or qubit['reached'] is False:
                            _, totalSwitchNum=qubit_merge_with_ancestor(\
                                                currentQubit,vertex,node,\
                                                dependentGraphs[i],totalSwitchNum,\
                                                mainPaths_sim,success)
                            

            #compute the edge simulation results
            for edge in dependentGraphs[i].edges():
                pathIndex=dependentGraphs[i].edges[edge]['dependent']['mainPathIndex']
                pathUV=dependentGraphs[i].edges[edge]['dependent']['pathUV']
                if pathIndex is None:
                    continue
                if not success[pathIndex]:
                    pass
                else:
                    start=mainPaths_sim[pathIndex][0]
                    end=mainPaths_sim[pathIndex][-1]
                    startQubit=None
                    endQubit=None
                    for qubit in dependentGraphs[i].nodes[pathUV[0]]['simuNode'][start]:
                        if qubit['mainPathIndex']==pathIndex:
                            startQubit=qubit
                            break
                    for qubit in dependentGraphs[i].nodes[pathUV[1]]['simuNode'][end]:
                        if qubit['mainPathIndex']==pathIndex:
                            endQubit=qubit
                            break
                    startAncestor=startQubit['ancestor']
                    endAncestor=endQubit['ancestor']
                    if startAncestor is not None and endAncestor is not None:
                        startAncestor['edges'].append(edge)
                        endAncestor['edges'].append(edge)

            
            #update the task status
            if dependentGraphs[i].graph['n_shot']==1 and allPathSuccess:
                newTask.graph['ifSuccess']=True
                #add the switch for executing at the final moment.
                totalSwitchNum+=dependentGraphs[i].graph['addSwitch']
            
            #start from scratch if switch fails
            ifSuccess=self.rng.binomial(totalSwitchNum,ps)==totalSwitchNum

            if not ifSuccess:
                newTask.graph['ifSuccess']=False
                newTask.graph['deliverable']=True
                newTask.graph['ifGenerated']=False
                tasks[i]=newTask
                continue

            newTask.graph['ifGenerated']=True

            #update task nodes and edges
            ##assign the order for the true edges
            for edge in newTask.edges():
                if edge in dependentGraphs[i].edges():
                    newTask.edges[edge]['order']=dependentGraphs[i].edges[edge]['dependent']['order']
                    newTask.edges[edge]['pathUV']=dependentGraphs[i].edges[edge]['dependent']['pathUV']
                else:
                    newTask.edges[edge]['order']=None
            for vertex in newTask.nodes():
                newTask.nodes[vertex]['isVirtual']=False
            
            clearEdgesCandidate={}
            ##add virtual qubits and virtual edges
            virtualVertexIndex=max(newTask.nodes())+1
            for vertex in dependentGraphs[i].nodes():
                virtualVertexCandidate=[]
                for node in dependentGraphs[i].nodes[vertex]['simuNode'].keys():
                    for qubit in dependentGraphs[i].nodes[vertex]['simuNode'][node]:
                        if not qubit['isAncestor']:
                            continue
                        #remove useless virtual node
                        if len(qubit['node'])==1 and len(qubit['edges'])==0:
                            continue
                        #add virtual node candidate
                        virtualVertexDict={}
                        virtualVertexDict['node']=qubit['node'].copy()
                        virtualVertexDict['edges']=qubit['edges'].copy()
                        virtualVertexDict['reversedFlow']=qubit['reversedFlow']
                        virtualVertexDict['memNode']=node
                        virtualVertexDict['order']=qubit['order']
                        virtualVertexCandidate.append(virtualVertexDict)
                            
                #add virtual node
                for virtualVertexDict in virtualVertexCandidate:
                    newTask.add_node(virtualVertexIndex)
                    newTask.nodes[virtualVertexIndex]['memNode']=virtualVertexDict['memNode']
                    newTask.nodes[virtualVertexIndex]['node']=virtualVertexDict['node'].copy()
                    newTask.nodes[virtualVertexIndex]['edges']=virtualVertexDict['edges'].copy()
                    newTask.nodes[virtualVertexIndex]['ifVirtual']=True
                    newTask.nodes[virtualVertexIndex]['reversedFlow']=virtualVertexDict['reversedFlow']
                    newTask.nodes[virtualVertexIndex]['order']=virtualVertexDict['order']
                    newTask.nodes[virtualVertexIndex]['sourceVertex']=vertex
                    #add virtual edge
                    newTask.add_edge(virtualVertexIndex,vertex)
                    newTask.edges[virtualVertexIndex,vertex]['ifVirtual']=True
                    newTask.edges[virtualVertexIndex,vertex]['order']=virtualVertexDict['order']
                    if virtualVertexDict['reversedFlow']:
                        newTask.edges[virtualVertexIndex,vertex]['pathUV']=(virtualVertexIndex,vertex)
                    else:
                        newTask.edges[virtualVertexIndex,vertex]['pathUV']=(vertex,virtualVertexIndex)

                    #add clear edges candidate
                    for edge in virtualVertexDict['edges']:
                        clearEdge=None
                        if edge in clearEdgesCandidate:
                            clearEdge=edge
                        elif (edge[1],edge[0]) in clearEdgesCandidate:
                            clearEdge=(edge[1],edge[0])
                        if clearEdge is None:
                            clearEdgesCandidate[edge]={edge[0]:None,edge[1]:None}
                            clearEdgesCandidate[edge][vertex]=virtualVertexIndex
                        else:
                            clearEdgesCandidate[clearEdge][vertex]=virtualVertexIndex

                    virtualVertexIndex+=1
            
            #clear the edges
            for edge in clearEdgesCandidate.keys():
                #edge establishes successfully
                if clearEdgesCandidate[edge][edge[0]] is not None and clearEdgesCandidate[edge][edge[1]] is not None:
                    newTask.remove_edge(edge[0],edge[1])
                #one of the vertexes lost
                else:
                    if clearEdgesCandidate[edge][edge[0]] is not None:
                        virtualVertex=clearEdgesCandidate[edge][edge[0]]
                        clearEdge=None
                        if edge in newTask.nodes[virtualVertex]['edges']:
                            clearEdge=edge
                        elif (edge[1],edge[0]) in newTask.nodes[virtualVertex]['edges']:
                            clearEdge=(edge[1],edge[0])
                        newTask.nodes[virtualVertex]['edges'].remove(clearEdge)
                    if clearEdgesCandidate[edge][edge[1]] is not None:
                        virtualVertex=clearEdgesCandidate[edge][edge[1]]
                        clearEdge=None
                        if edge in newTask.nodes[virtualVertex]['edges']:
                            clearEdge=edge
                        elif (edge[1],edge[0]) in newTask.nodes[virtualVertex]['edges']:
                            clearEdge=(edge[1],edge[0])
                        newTask.nodes[virtualVertex]['edges'].remove(clearEdge)
                        
            tasks[i]=newTask

        #add the additional resources
        residueNetwork.graph['recMemUsed']=newResidueNetwork.graph['recMemUsed']
        residueNetwork.graph['recBellUsed']=newResidueNetwork.graph['recBellUsed']
        residueNetwork=self.add_additional_resources(residueNetwork,saveChannels,**kargs)

        return residueNetwork,tasks
                