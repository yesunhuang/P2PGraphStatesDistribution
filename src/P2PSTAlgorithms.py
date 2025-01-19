'''
Name: P2PSTAlgorithms.py
Description: For the implementation of the spacetime algorithms
Email: yesunhuang@uchicago.edu
OpenSource: https://github.com/yesunhuang
LastEditors: yesunhuang yesunhuang@uchicago.edu
Msg: This is the main file for implementing all the spacetime algorithms being used for P2P entanglement distribution.
Author: YesunHuang, XiangyuRen
Date: 2024-11-21 02:55:43
'''
import networkx as nx
import numpy as np
import P2PAlgorithms as p2pA
from scipy.special import comb
from typing import*
from heapq import*

def get_node_shot(network:nx.Graph, memoryNode: int) -> Tuple[int, int]:
    '''
    description: get the shot and the original node from the memory node.

    param {nx.Graph} network: The original network.

    return {Tuple[int, int]}: The shot and the original node.
    '''
    n_node = network.graph['n_node']
    return memoryNode // n_node, memoryNode % n_node

def get_memory_node(network:nx.Graph, shot: int) -> int:
    '''
    description: get the memory node from the shot and the original node.

    param {nx.Graph} network: The input graph.

    param {int} shot: The input shot.

    return {int}: The memory node.
    '''
    n_node = network.graph['n_node']
    def node_convert(x):
        return shot * n_node + x
    return node_convert

def construct_memory_flow(g: nx.Graph, n_shot: int, \
                          ifReset: bool=False, memCostCoeff: float=0.4) -> nx.Graph:
    '''
    description: construct the graph of memory flow from the origin network

    param {nx.Graph} g: The input network graph.

    param {int} n_shot: The number of shots in memory flow.

    param {bool} ifReset: If reset the graph occupation.

    param {float} memCostCoeff: The memory cost coefficient.

    return nx.Graph: the generated memory flow
    '''
    #take care of non-consecutive node index
    if 'n_node' not in g.graph:
        n_nodes=np.max(list(g.nodes))+1
        g.graph['n_node'] = n_nodes
    else:
        n_nodes = g.graph['n_node']
    
    g_reset=g.copy()
    #delete the saved channels
    savedChannelsNode=[node for node in g.nodes \
                       if 'saveChannel' in g.nodes[node] and g.nodes[node]['saveChannel']]
    g_reset.remove_nodes_from(savedChannelsNode)
    if ifReset:
        for edge in g_reset.edges:
            g_reset.edges[edge]['occu']=0
        for node in g_reset.nodes:
            g_reset.nodes[node]['moccu']=0

    memory_graph_list = []
    mem_edge_list = []
    for i in range(0,n_shot):
        mem_edge = []
        #collect all the memory links
        for node in g.nodes:
            mem_edge.append((get_memory_node(g,i)(node),\
                             get_memory_node(g,i+1)(node)))
        mem_edge_list.append(mem_edge)
        if  i>0:
            cg = g_reset.copy()
        else:
            cg = g.copy()
        cg = nx.relabel_nodes(cg, get_memory_node(g,i))
        memory_graph_list.append(cg)

    #take care of the ancilla shot
    cg_anc = nx.Graph()
    cg_anc.add_nodes_from(g.nodes(data=True))
    cg_anc = nx.relabel_nodes(cg_anc, get_memory_node(g,n_shot))
    memory_graph_list.append(cg_anc)

    memory_flow_graph = nx.union_all(memory_graph_list)

    #mem_edge_list.reverse()
    for i in range(0,n_shot):
        for (u,v) in mem_edge_list[i]:
            _,originalU=get_node_shot(g,u)

            if g.nodes[originalU]['memory'] is not None:
                width=g.nodes[originalU]['memory']
            else:
                width=np.inf
            if ifReset and i>0:
                if originalU in g_reset.nodes:
                    occu=g_reset.nodes[originalU]['moccu']
            else:
                occu=g.nodes[originalU]['moccu']
            memory_flow_graph.add_edge(u,v, width=width, occu=occu, prob=memCostCoeff)
    
    memory_flow_graph.graph['n_node'] = n_nodes
    memory_flow_graph.graph['n_shot'] = n_shot
    return memory_flow_graph


def ET_metric_flow(graph: nx.Graph, path: List[int], \
                   endpointsProbs:Tuple[Dict[int,float],Dict[int,float]]=None,\
                   memShot: bool|float=False, returnTrueLength: bool=False) -> float|Tuple[float,int]:
    '''
    description: The expectation time of constructing at least one entanglement link on the path.

    param {nx} graph: the input graph.

    param {List[int]} path: the input path.

    param {Tuple[Dict[int,float],Dict[int,float]]} endpointsProbs: the probability of the endpoints.

    param {bool|float} memShot: whether enable the memory shot metric and the factor for the memory shot.

    param {bool} returnTrueLength: whether return the true length of the path.

    return {float|Tuple[float,int]}: the cost and the true length if returnTrueLength is True.
    '''
    P_t=1
    n_node = graph.graph['n_node']

    factor=None
    degradedFactor=None
    if isinstance(memShot,float):
        factor=memShot
        memShot=True

    if 'ps' in graph.graph:
        ps=graph.graph['ps']
    else:
        ps=1.0

    if endpointsProbs is not None:
        sourceCost,targetCost=endpointsProbs
        if path[0] in sourceCost:
            P_t*=sourceCost[path[0]]
        if path[-1] in targetCost:
            P_t*=targetCost[path[-1]]

    n_memory_path = 0

    for i in range(0,len(path)-1):
        u,v = path[i],path[i+1]
        
        currentFactor=None
        p_i=graph.edges[u,v]['prob']
        #deal with the memory links
        if u%n_node == v%n_node:
            #take care of switch policy
            P_t *= (p_i*ps)
            n_memory_path += 1
            continue
        
        if not memShot:
            k = graph.edges[u,v]['occu']+1
            width = graph.edges[u,v]['width']
        else:
            currentShot = u//n_node
            k = graph.edges[u,v]['occu']+1
            width = graph.edges[u,v]['width']
            if factor is None:
                degradedFactor=np.power(p_i,currentShot*graph.edges[u,v]['width'])
            else:
                currentFactor=factor*(currentShot+1)


        p_one = 0
        if p_i == 1:
            p_one = 1
        else:
            for j in range(k,width+1):
                p_one += np.power(1-p_i,width-j) * np.power(p_i,j) * comb(width,j)
        if currentFactor is not None:
            p_one=np.power(p_one,currentFactor)
        elif degradedFactor is not None:
            p_one*=degradedFactor
        P_t *= p_one

    #calculate switch number
    path=p2pA.remove_ancilla(path,graph)
    trueLength=len(path)
    if trueLength>2:
            P_t*=np.power(ps,trueLength-2)
    else:
        if trueLength==1:
            P_t/=ps
    
    if P_t==0:
        cost=np.inf
    else:
        cost=1/P_t
    
    if returnTrueLength:
        return cost, trueLength
    else:
        return cost
    
def getAssignNode(node:int,graphState: nx.Graph)->int|None:
    '''
    description: Get the assigned node for the non-virtual node.

    param {int} node: The input node.

    param {nx.Graph} graphState: The input graph state.

    return {int|None}: The assigned node.
    '''
    if 'ifVirtual' not in graphState.nodes[node] or \
        not graphState.nodes[node]['ifVirtual']:
        if isinstance(graphState.nodes[node]['node'],list):
            return graphState.nodes[node]['node'][0]
        else:
            return graphState.nodes[node]['node']
    else:
        return None

def split_path(node:int, vertex:int,\
               VRM:Dict[int,Dict[int,Dict[str,Any]]],\
               mainPaths:Dict[int,Tuple[List[int],float,int]],\
               switchProb: float)->int:
    '''
    description: Split the path at the node.
    
    param {int} node: The input node.

    param {int} vertex: The input vertex.

    param {Dict[int,Dict[int,Dict[str,Any]]]} VRM: The input VRM.

    param {Dict[int,Tuple[List[int],float,int]]} mainPaths: The input main paths.

    param {float} switchProb: The input switch probability.

    return {int}: The current mainPaths length.
    '''
    pathDict=VRM[vertex][node][0]['mainPathDict']
    if pathDict is None:
        return len(mainPaths)-1
    currentIndex=mainPaths.index(pathDict)
    pathForSplit, cost, vu=pathDict['path'],pathDict['cost'],pathDict['UV']
    reversedFlow=('reversedFlow' in pathDict and pathDict['reversedFlow'])
    (v,u) = vu

    if v==vertex:
        keep=v
        out=u
    else:
        keep=u
        out=v

    index=pathForSplit.index(node)
        
    prev=VRM[vertex][node][0]['prev']

    if prev==pathForSplit[0]:
        partialPath=pathForSplit[:index+1]
        otherPartialPath=pathForSplit[index:]
        newPartialUV=(keep,vertex)
        newOtherPartialUV=(vertex,out)
    else:
        partialPath=pathForSplit[index:]
        otherPartialPath=pathForSplit[:index+1]
        newPartialUV=(vertex,keep)
        newOtherPartialUV=(out,vertex)

    partialPathCost=switchProb/VRM[vertex][node][0]['prob']

    if len(partialPath)==1:
        #no need to split
        return len(mainPaths)-1
            
    #for the section of keep
    for node_i in partialPath:
        if node_i in VRM[keep]:
            for qubit in VRM[keep][node_i]:
                if qubit['mainPathDict'] is pathDict:
                   if qubit['ifFixed']==False:
                       qubit['ifFixed']=True
                       break
                
        if node_i in VRM[out]:
            if node_i == node:
                continue
            newQubitList=[]
            for qubit in VRM[out][node_i]:
                if qubit['mainPathDict'] is pathDict:
                   if qubit['ifFixed']==False:
                        continue
                newQubitList.append(qubit)
            if len(newQubitList)==0:
                del VRM[out][node_i]
            else:
                VRM[out][node_i]=newQubitList
            
    if len(otherPartialPath)==1 and keep==out:
        #no need to split
        return len(mainPaths)-1
            
    #an additional switch probability is taken into account
    pathDict['path']=partialPath
    pathDict['cost']=partialPathCost
    pathDict['UV']=newPartialUV
    otherPartialPathCost=cost/partialPathCost
    otherPartialPathDict={'path':otherPartialPath,'cost':otherPartialPathCost,\
                          'UV':newOtherPartialUV,'reversedFlow':reversedFlow}
    #insert the new partial path after the current path
    mainPaths.insert(currentIndex+1,otherPartialPathDict)

    #for the section of out
    for node_i in otherPartialPath:
        if keep!=out and (node_i in VRM[out]):
            for qubit in VRM[out][node_i]:
                if qubit['mainPathDict'] is pathDict:
                    qubit['mainPathDict']=otherPartialPathDict
                    break

        if node_i in VRM[keep]:
            if node_i == node:
                qubitDict={'prob':switchProb,'ifFixed':True,'prev':node,'mainPathDict':otherPartialPathDict}
                VRM[vertex][node].append(qubitDict)
                continue
            for qubit in VRM[keep][node_i]:
                if qubit['mainPathDict'] is pathDict:
                    qubit['prev']=node
                    qubit['mainPathDict']=otherPartialPathDict
                    #modify the probability
                    qubit['prob']=qubit['prob']/VRM[vertex][node][0]['prob']
                    break

    return len(mainPaths)-1

def update_VRM_memflow(VRM:Dict[int,Dict[int,Dict[str,Any]]],\
               network:nx.Graph,\
               mainPaths:List[Dict[str,Any]], \
               controllable:List|None=None,
               memShot: bool|float=False)\
               ->Tuple[Dict[int,Dict[int,Dict[str,Any]]],\
                        Dict[Tuple[int,int],Dict[str,Any]],\
                        Dict[int,Tuple[List[int],float,int]],\
                        int]:
    '''
    description: (memory flow version) Update the Vertex Reaching Map (VRM) for the network.

    param {Dict[int,Dict[int,Dict[str,Any]]]} VRM: The input VRM.

    param {nx.Graph} network: The input network.

    param {List[Dict[str,Any]]} mainPaths: The input main paths.

    param {List} controllable: The controllable nodes.

    param {bool} memShot: whether to punish higher shots

    return {Tuple[Dict[int,Dict[int,Dict[str,Any]]],\
                    Dict[int,Tuple[List[int],float,int]]]} : The updated VRM and main paths.
    '''
            

    controllableNodes=[]
    currentPath=mainPaths[-1]
    path,UV=currentPath['path'],currentPath['UV']
    

    if 'ps' in network.graph:
        switchProb=network.graph['ps']
    else:
        switchProb=1

    v,u=UV

    if controllable is None:
        controllableNodes = path
    else:
        controllableNodes = [n for n in path \
                             if get_node_shot(network,n)[1] in controllable]
    
    if v==u:
        split_path(path[0],v,VRM,mainPaths,switchProb)
        for node in controllableNodes:
            partialPath=path[:path.index(node)+1]
            partialCost,trueLength=ET_metric_flow(network,partialPath,memShot=memShot,returnTrueLength=True)
            #reimburse the cost for the switch probability
            if trueLength==1:
                qubitDict={'prob':switchProb**2/partialCost,'ifFixed':True,'prev':path[0],'mainPathDict':currentPath}
            else:
                qubitDict={'prob':switchProb/partialCost,'ifFixed':True,'prev':path[0],'mainPathDict':currentPath}
            if node not in VRM[v]:
                VRM[v][node]=[]
            VRM[v][node].append(qubitDict)
        return VRM, mainPaths

    split_path(path[0],v,VRM,mainPaths,switchProb)
    split_path(path[-1],u,VRM,mainPaths,switchProb)
    
    for node in controllableNodes:
        partialPath=path[:path.index(node)+1]
        partialCost,trueLength=ET_metric_flow(network,partialPath,memShot=memShot,returnTrueLength=True)
        #reimburse the cost for the switch probability
        if trueLength==1:
            qubitDict={'prob':switchProb**2/partialCost,'ifFixed':True,'prev':path[0],'mainPathDict':currentPath}
        else:
            qubitDict={'prob':switchProb/partialCost,'ifFixed':False,'prev':path[0],'mainPathDict':currentPath}
        if node not in VRM[v]:
            VRM[v][node]=[]
        VRM[v][node].append(qubitDict)

        partialPath=path[path.index(node):]
        partialCost,trueLength=ET_metric_flow(network,partialPath,memShot=memShot,returnTrueLength=True)
        if trueLength==1:
            qubitDict={'prob':switchProb**2/partialCost,'ifFixed':True,'prev':path[-1],'mainPathDict':currentPath}
        else:
            qubitDict={'prob':switchProb/partialCost,'ifFixed':False,'prev':path[-1],'mainPathDict':currentPath}
        if node not in VRM[u]:
            VRM[u][node]=[]
        VRM[u][node].append(qubitDict)
            
    return VRM, mainPaths 

def decompose_mainPathRaw(mainPathRaw: Dict[str,Any],network:nx.Graph)\
    -> Tuple[List[Tuple[List[int],List[bool]]],bool]:
    '''
    description: Decompose the main path raw into partial paths.

    param {Dict[str,Any]} mainPathRaw: The input main path raw.

    param {nx.Graph} network: The input network.

    return {Tuple[List[Tuple[List[int],List[bool]]],bool]}: The partial paths and whether Bell pair is used.
    '''

    rawSTPath=mainPathRaw['path']
    partialPaths=[]
    currentPartialPath=[]
    currentSaveStatus=[False,False]
    BellUsed=False
    for i in range(len(rawSTPath)):
        currentShot,currentNode=get_node_shot(network, rawSTPath[i])
        #nothing to do
        if currentShot>0:
            continue
        currentPartialPath.append(currentNode)
        #get previous node
        if i>0:
            preShot,_=get_node_shot(network, rawSTPath[i-1])
        else:
            preShot=0
        #require left memory
        if preShot>0:
            currentSaveStatus[0]=True
        #get the next node
        if i<len(rawSTPath)-1:
            nextShot,_=get_node_shot(network, rawSTPath[i+1])
        else:
            nextShot=0
        #require right memory
        if nextShot>0:
            currentSaveStatus[1]=True
            #save the current partial path
            partialPaths.append((currentPartialPath.copy(),currentSaveStatus.copy()))
            if len(currentPartialPath)>1:
                BellUsed=True
            currentPartialPath=[]
            currentSaveStatus=[False,False]
    #save the last partial path
    if len(currentPartialPath)>0:
        partialPaths.append((currentPartialPath.copy(),currentSaveStatus.copy()))
        if len(currentPartialPath)>1:
            BellUsed=True
    return partialPaths, BellUsed

def collapse_res(taskMainPaths:Dict[int,Tuple[List[int],float,int]],\
                  VRM:Dict[int,Dict[int,Dict[str,Any]]],\
                  network:nx.Graph)->None:
    '''
    description: Collapse the memory flow graph if no bell pair used in current shot.

    param {Dict[int,Tuple[List[int],float,int]]} taskMainPaths: The input task main paths.

    param {Dict[int,Dict[int,Dict[str,Any]]]} VRM: The input VRM.

    param {nx.Graph} network: The input network.

    return {None}
    '''

    #remove redundant node in taskMainPaths
    for taskMainPath in taskMainPaths:
        path,UV=taskMainPath['path'],taskMainPath['UV']
        newPath=[]
        for node in path:
            currentShot,currentNode=get_node_shot(network,node)
            if currentShot>0:
                newNode=get_memory_node(network,currentShot-1)(currentNode)
                newPath.append(newNode)
            else:
                if len(path)==1:
                    newPath.append(currentNode)
                else:
                    continue
        taskMainPath['path']=newPath
    #collapse the VRM
    for vertex in VRM.keys():
        nodeList=list(VRM[vertex].keys())
        newVertexDict={}
        for node in nodeList:
            #deal with preserved node
            if isinstance(node,str):
                newVertexDict[node]=VRM[vertex][node]
                continue
            currentShot,currentNode=get_node_shot(network,node)
            if currentShot==0:
                continue
            for qubit in VRM[vertex][node]:
                prevMemNode=qubit['prev']
                if prevMemNode is not None:
                    prevShot,prevNode=get_node_shot(network,prevMemNode)
                    if prevShot>0:
                        newPrevMemNode=get_memory_node(network,prevShot-1)(prevNode)
                    else:
                        newPrevMemNode=prevMemNode
                    qubit['prev']=newPrevMemNode
            newMemNode=get_memory_node(network,currentShot-1)(currentNode)
            newVertexDict[newMemNode]=VRM[vertex][node]
        for node in nodeList:
            #deal with the node that shall be deleted
            if isinstance(node,str):
                continue
            currentShot,currentNode=get_node_shot(network,node)
            if currentShot>0:
                continue
            for qubit in VRM[vertex][node]:
                prevMemNode=qubit['prev']
                if prevMemNode is None:
                    continue
                prevShot,prevNode=get_node_shot(network,prevMemNode)
                if prevShot>0:
                    newPrevMemNode=get_memory_node(network,prevShot-1)(prevNode)
                else:
                    newPrevMemNode=prevMemNode
                qubit['prev']=newPrevMemNode
                mainPath=qubit['mainPathDict']
                #deal with the qubits merged from virtual vertex
                if mainPath is None:
                    if node not in newVertexDict:
                        newVertexDict[node]=[]
                    newVertexDict[node].append(qubit)
                    continue
                path,UV=mainPath['path'],mainPath['UV']
                #single switch to perform edge
                if len(path)==1 and UV[0]!=UV[1]:
                    if node not in newVertexDict:
                        newVertexDict[node]=[]
                    #append it to the zero shot node
                    newVertexDict[node].append(qubit)
                else:
                    continue
        VRM[vertex]=newVertexDict
    return taskMainPaths,VRM

def memflow_bin_search(graphState:nx.Graph, network:nx.Graph|None, \
                       controllable:List=None,ifFirst:bool=True,**kargs)\
    ->Tuple[Dict[int,Tuple[List[int],float,Tuple[int,int],int]],\
            Dict[Tuple[int,int],Dict[str,Any]],\
            Dict[int,Dict[int,Dict[str,Any]]]]:
    '''
    description: Find the P2P paths in the memory-flow model with fewest number of shots used.

    param {nx.Graph} graphState: The input graph state.
 
    param {nx.Graph} network: The input network.

    param {List} controllable: The controllable nodes.

    param {bool} ifFirst: if this is the first task to use the network    

    return {Tuple[Dict[int,Tuple[List[int],float,Tuple[int,int],int]],
            Dict[Tuple[int,int],Dict[str,Any]],
            Dict[int,Dict[int,Dict[str,Any]]]]}: 
            The main paths:{shots:[(path,cost,(u,v))]}
            The edge dependent dictionary:{(u,v):(shot,pathIndex,(directNode,directNode))}
            The vertex Reaching Map:{vertex:{Node:[if fixed,shot,mainPathDict, prob]}}
    '''
    n_node = len(graphState.nodes)
    lShots = 0
    rShots = None
    rShotsMax = n_node
    shots=1
    validRes=None
    if 'memCostCoeff' not in kargs:
        kargs['memCostCoeff']=0.5
    if 'ifReset' not in kargs:
        #kargs['ifReset']=False
        kargs['ifReset']=True
    
    while rShots is None or \
        (lShots<rShots and  shots>=1):
        network_memflow = construct_memory_flow(network, shots,ifReset=kargs['ifReset'],memCostCoeff=kargs['memCostCoeff'])

        res = find_P2P_memflow_paths(graphState, network_memflow, controllable, **kargs)
        if res is None:
            lShots = shots+1
            if rShots is None:
                shots = shots*2
                if shots>rShotsMax:
                    rShots=rShotsMax
                    shots=rShotsMax
            else:
                if shots>rShotsMax:
                    rShots=rShotsMax
                shots=shots+int(np.ceil((rShots-shots)/2))
        else:
            rShots = shots
            shots=shots-int(np.ceil((shots-lShots)/2))
    shots=rShots
    network_memflow = construct_memory_flow(network, shots,ifReset=kargs['ifReset'],memCostCoeff=kargs['memCostCoeff'])
    validRes=find_P2P_memflow_paths(graphState, network_memflow, controllable, **kargs)
    if validRes is not None:
        if shots>1:
            requiredCollapsed=True
        else:
            requiredCollapsed=False
        taskMainPaths,VRM=validRes
        while requiredCollapsed:
            for mainPathRaw in taskMainPaths:
                _,BellUsed=decompose_mainPathRaw(mainPathRaw,network)
                if BellUsed:
                    requiredCollapsed=False
                    break
            if requiredCollapsed:
                #collapse the graph
                if ifFirst:
                    collapse_res(taskMainPaths,VRM,network_memflow)
                    shots-=1
                else:
                    return None, shots

    return validRes, shots

def find_P2P_memflow_paths(graphState:nx.Graph, network:nx.graph, \
                           controllable:List=None,memShot: bool|float=False,**kargs)\
    ->Tuple[Dict[int,Tuple[List[int],float,Tuple[int,int],int]],\
            Dict[Tuple[int,int],Dict[str,Any]],\
            Dict[int,Dict[int,Dict[str,Any]]]]:
    '''
    description: Find the P2P paths to distribute the graph state, don't use the ancilla nodes!

    param {nx.Graph} graphState: The input graph state.
 
    param {nx.Graph} network: The input network.

    param {List} controllable: The controllable nodes.

    param {bool|float} memShot: whether to punish higher shots

    return {Tuple[Dict[int,Tuple[List[int],float,Tuple[int,int],int]],
            Dict[Tuple[int,int],Dict[str,Any]],
            Dict[int,Dict[int,Dict[str,Any]]]]}: 
            The main paths:{shots:[(path,cost,(u,v))]}
            The edge dependent dictionary:{(u,v):(shot,pathIndex,(directNode,directNode))}
            The vertex Reaching Map:{vertex:{Node:[if fixed,shot,mainPathDict, prob]}}
    '''     
    def get_exact_prob(vertex:int)-> Dict[int,float]:
        def get_current_node_prob(node:int)->float:
            if VRM[vertex][node][0]['prev']==None \
                and VRM[vertex][node][0]['mainPathDict']==None:
                return 1.0
            else:
                if costDict[node] is not None:
                    return costDict[node]
                previousNode=VRM[vertex][node][0]['prev']
                costDict[node]=get_current_node_prob(previousNode)*VRM[vertex][node][0]['prob']
                return costDict[node]
                
        costDict={node:None for node in VRM[vertex].keys() if not isinstance(node,str)}
        if 'ifVirtual' in graphState.nodes[vertex] \
            and graphState.nodes[vertex]['ifVirtual']:
            for node in costDict.keys():
                costDict[node]=VRM[vertex][node][0]['prob']
            return costDict
        for node in costDict.keys():
            if costDict[node] is None:
                costDict[node]=get_current_node_prob(node)
        return costDict
    
    def free_memory(vertex:int, \
                    network:nx.Graph, graphState:nx.Graph,\
                    ifTemporal: bool=True)->nx.Graph:
        if 'ifVirtual' in graphState.nodes[vertex] and graphState.nodes[vertex]['ifVirtual']:
            originalNode=graphState.nodes[vertex]['memNode']
        else:
            originalNode=getAssignNode(vertex,graphState)
        if 'reserved' in graphState.nodes[vertex]:
            for shot in graphState.nodes[vertex]['reserved']:
                u,v=get_memory_node(network,shot)(originalNode),\
                    get_memory_node(network,shot+1)(originalNode)
                network.edges[u,v]['occu']-=1
        if not ifTemporal:
            del graphState.nodes[vertex]['reserved']
            
        return network
    
    def reserve_memory(vertex:int, network:nx.Graph, graphState:nx.Graph)->nx.Graph:
        if 'ifVirtual' in graphState.nodes[vertex] and graphState.nodes[vertex]['ifVirtual']:
            originalNode=graphState.nodes[vertex]['memNode']
            #the memory has been served by its source vertex
            sourceVertex=graphState.nodes[vertex]['sourceVertex']
            sourceVertexAssignedNode=getAssignNode(sourceVertex,graphState)
            if sourceVertexAssignedNode==originalNode:
                return network
        else:
            originalNode=getAssignNode(vertex,graphState)
        if 'reserved' not in graphState.nodes[vertex]:
            graphState.nodes[vertex]['reserved']=[]
            for shot in range(0,network.graph['n_shot']):
                u,v=get_memory_node(network,shot)(originalNode),\
                    get_memory_node(network,shot+1)(originalNode)
                if network.edges[u,v]['width']>network.edges[u,v]['occu']:
                    network.edges[u,v]['occu']+=1
                    graphState.nodes[vertex]['reserved'].append(shot)
                else:
                    break
        else:
            for shot in graphState.nodes[vertex]['reserved']:
                u,v=get_memory_node(network,shot)(originalNode),\
                    get_memory_node(network,shot+1)(originalNode)
                network.edges[u,v]['occu']+=1
        return network
    
    def add_path(u: int,v:int,path: List[int],\
                VRM:Dict[int,Dict[int,Dict[str,Any]]],\
                network:nx.Graph,graphState:nx.Graph,\
                mainPaths:List[Dict[str,Any]], \
                memShot:bool,controllable:List|None=None)-> None:
        
            
        def add_path_and_clear_resource(minPath:Dict[str,Any])\
                                        ->None:
            mainPaths.append(minPath)
            update_VRM_memflow(VRM,network,mainPaths,\
                               controllable=controllable,\
                               memShot=memShot)
            p2pA.remove_paths(network,minPath['path'])
            return None
        
        if 'ps' in network.graph:
            switchProb=network.graph['ps']
        else:
            switchProb=1
        
        minPath={'path':path,'cost':ET_metric_flow(network,path,memShot=memShot),'UV':(u,v)}
        #deal with the reserved memory
        UAssigned=getAssignNode(u,graphState)
        VAssigned=getAssignNode(v,graphState)
        #get the virtual node dictionary
        virtualNodeDict=None
        reversedFlow=False
        if UAssigned is None:
            virtualNodeDict=graphState.nodes[u]
            UMemNode=virtualNodeDict['memNode']
            reversedFlow=True
        else:
            UMemNode=UAssigned
        if VAssigned is None:
            virtualNodeDict=graphState.nodes[v]
            VMemNode=virtualNodeDict['memNode']
        else:
            VMemNode=VAssigned
        lSplitIndex=0
        rSplitIndex=len(path)-1
        for i in range(0,len(path)-1):
            shotN1,originalN1=get_node_shot(network,path[i])
            shotN2,originalN2=get_node_shot(network,path[i+1])
            #see if it is the memory link
            if shotN1==shotN2:
                continue
            else:
                recordShot=min(shotN1,shotN2)
                #update left split index
                if originalN1==UMemNode:
                    lSplitIndex=i+1
                    #free the memory
                    if 'reserved' in graphState.nodes[u]\
                        and recordShot in graphState.nodes[u]['reserved']:
                        graphState.nodes[u]['reserved'].remove(recordShot)
                        
                #update right split index
                if originalN2==VMemNode:
                    if rSplitIndex==len(path)-1:
                        rSplitIndex=i
                    #free the memory
                    if 'reserved' in graphState.nodes[v]\
                        and recordShot in graphState.nodes[v]['reserved']:
                        graphState.nodes[v]['reserved'].remove(recordShot)

        #update the VRM
        if u==v \
            or (VAssigned is None or UAssigned is None):
            #no need to split
            if virtualNodeDict is None:
                add_path_and_clear_resource(minPath)
            #deal with the virtual node
            else:
                if not reversedFlow:
                    minPath['UV']=(u,u)
                    virtualVertex=v
                    realVertex=u
                else:
                    path=path[::-1]
                    minPath['path']=path
                    minPath['UV']=(v,v)
                    virtualVertex=u
                    realVertex=v
                    minPath['reversedFlow']=True
                VRM[virtualVertex]['preNode']=path[-1]
                VRM[virtualVertex]['reversedFlow']=reversedFlow
                add_path_and_clear_resource(minPath)
                #expanding the virtual node
                for node in VRM[virtualVertex].keys():
                    if isinstance(node,str):
                        continue
                    if node==path[-1]:
                        continue
                    #deal with the switch strategy.
                    newQubit={'prob':switchProb**2,'ifFixed':True,'prev':path[-1],'mainPathDict':None}
                    if node not in VRM[realVertex]:
                        VRM[realVertex][node]=[]
                    VRM[realVertex][node].append(newQubit)
                              
        else:
            if not lSplitIndex<=rSplitIndex:
                print('Finite precision effect warning!')
                lSplitIndex=0
        
            #split the left path
            if lSplitIndex>0:
                pathUMemory={'path':path[:lSplitIndex+1],\
                             'cost':ET_metric_flow(network,path[:lSplitIndex+1],memShot=memShot),\
                             'UV':(u,u)}
                add_path_and_clear_resource(pathUMemory)
            
            #split the right path
            if rSplitIndex<len(path)-1:
                pathVMemory={'path':path[rSplitIndex:][::-1],\
                             'cost':ET_metric_flow(network,path[rSplitIndex:][::-1],memShot=memShot),\
                             'UV':(v,v),'reversedFlow':True}
                add_path_and_clear_resource(pathVMemory)

            pathUV={'path':path[lSplitIndex:rSplitIndex+1],\
                    'cost':ET_metric_flow(network,path[lSplitIndex:rSplitIndex+1],memShot=memShot),\
                    'UV':(u,v)}
            add_path_and_clear_resource(pathUV)
            
        graphState.remove_edge(u,v)
        return None
    
    def lock_back_memory(vertex:int, graphState:nx.Graph, \
                         network:nx.Graph)->nx.Graph:
        #only release the memory for the virtual node
        if 'ifVirtual' in graphState.nodes[vertex] \
            and graphState.nodes[vertex]['ifVirtual']:
            if nx.degree(graphState,vertex)==0:
                if 'reserved' in graphState.nodes[vertex]:
                    graphState.nodes[vertex]['reserved']=[]
                return network
                
        network=reserve_memory(vertex,network,graphState)
        return network

    mainPaths=[]
    network_bk=network.copy()
    graphState_bk=graphState.copy()

    #initialize the VRM and reserve the memory
    VRM={}
    for v in graphState.nodes:
        VRM[v]={}
        #virtual vertex
        if 'ifVirtual' in graphState.nodes[v] and graphState.nodes[v]['ifVirtual']:
            VRM[v]['preNode']=None
            for node in graphState.nodes[v]['node']:
                VRM[v][node]=[]
                qubitDict={'prob':1,'ifFixed':True,'prev':None,'mainPathDict':None}
                VRM[v][node].append(qubitDict)
        #real vertex
        else:
            if isinstance(graphState.nodes[v]['node'],list):
                node=graphState.nodes[v]['node'][0]
            else:
                node=graphState.nodes[v]['node']
                graphState_bk.nodes[v]['node']=[node]
            ancilla = node + network.graph['n_node']*network.graph['n_shot']
            VRM[v][ancilla]=[]
            qubitDict={'prob':1,'ifFixed':True,'prev':None,'mainPathDict':None}
            VRM[v][ancilla].append(qubitDict)
        #reserve memory for the vertex
        network_bk=reserve_memory(v,network_bk,graphState_bk)
    
    #create priority heap for the degrees of vertexes
    degreeHeap=[]
    for v in graphState_bk.nodes:
        if graphState_bk.degree[v]>0 and\
            ('ifVirtual' not in graphState_bk.nodes[v] or\
             not graphState_bk.nodes[v]['ifVirtual']):
            degreeHeap.append(v)
    
    def get_degree(v:int)->int:
        return -graphState_bk.degree[v]
    
    degreeHeap_bk=degreeHeap.copy()

    #implement the virtual nodes first
    while len(degreeHeap_bk)>0:
        degreeHeap_bk.sort(key=get_degree)
        realVertex=degreeHeap_bk.pop(0)
        virtualNeighbors=[virtualVertex for virtualVertex \
                          in graphState_bk.neighbors(realVertex) \
                            if 'ifVirtual' in graphState_bk.nodes[virtualVertex]\
                            and graphState_bk.nodes[virtualVertex]['ifVirtual']]
        network_bk=free_memory(realVertex,network_bk,graphState_bk)
        def get_virtual_order(v:int)->int:
            return -(len(graphState_bk.nodes[v]['edges'])+len(graphState_bk.nodes[v]['node']))
        virtualNeighbors.sort(key=get_virtual_order)
        for virtualVertexIndex in range(len(virtualNeighbors)):

            sourceCost=get_exact_prob(realVertex)
            sources=list(sourceCost.keys())
            
            virtualVertex=virtualNeighbors[virtualVertexIndex]
            targetCost=get_exact_prob(virtualVertex)
            targets=list(targetCost.keys())
            def get_metric(network:nx.Graph,path:List[int])->float:
                return ET_metric_flow(network,path,(targetCost,sourceCost),memShot=memShot)
                
            network_bk=free_memory(virtualVertex,network_bk,graphState_bk)
            #use more first shot paths
            path,_=p2pA.find_path_ED(network_bk,targets,sources,single=True,metric=get_metric)
            if len(path)==0:
                if len(graphState_bk.nodes[virtualVertex]['edges'])>0:
                    return None
                else:
                    graphState_bk.remove_edge(realVertex,virtualVertex)
            else:
                add_path(virtualVertex,realVertex,path,VRM,network_bk,graphState_bk,mainPaths,\
                         memShot=memShot,controllable=controllable)
                #lock back the memory
                network_bk=lock_back_memory(virtualVertex,graphState_bk,network_bk)
        #lock back the memory
        network_bk=lock_back_memory(realVertex,graphState_bk,network_bk)

    #implement the rest of the edges
    while len(degreeHeap)>0:
        degreeHeap.sort(key=get_degree)
        u=degreeHeap.pop(0)
        neighbors=list(graphState_bk.neighbors(u))
        network_bk=free_memory(u,network_bk,graphState_bk)

        #sort the neighbors by the degree
        neighbors.sort(key=get_degree)
        for neighborIndex in range(len(neighbors)):
            sourceCost=get_exact_prob(u)
            sources=list(sourceCost.keys())
                               
            v=neighbors[neighborIndex]
            targetCost=get_exact_prob(v)
            targets=list(targetCost.keys())
            def get_metric(network:nx.Graph,path:List[int])->float:
                return ET_metric_flow(network,path,(sourceCost,targetCost),memShot=memShot)
                    
            network_bk=free_memory(v,network_bk,graphState_bk)
                
            path,_=p2pA.find_path_ED(network_bk,sources,targets,single=True,metric=get_metric)
                
            if len(path)==0:
                return None
            
            add_path(u,v,path,VRM,network_bk,graphState_bk,mainPaths,\
                     memShot=memShot,controllable=controllable)
            #lock back the memory
            network_bk=lock_back_memory(v,graphState_bk,network_bk)
        #delete the reserved list
        network_bk=lock_back_memory(u,graphState_bk,network_bk)

    return mainPaths,VRM

def path_recovery_ST_EUM(network:nx.Graph,mainPath:List[Tuple[int,bool]],\
                      recoveryPaths:List[Tuple[int,List[int],float]],h:int,\
                      memProb: float=0.8,saveAll: bool=False)->Dict[str,Any]:
    '''
    description: Fix the path with the main path and recovery path with the spacetime expected union method.

    param {nx.Graph} network: The input network.

    param {List[Tuple[int,bool]]} mainPath: The main path.

    param {List[Tuple[int,List[int],float]]} recoveryPaths: The recovery path.

    param {int} h: The maximum host.

    param {float} threshold: The threshold for memory link flow.

    param {bool} saveAll: Whether to save all the switched qubits.

    return {Dict[str,Any]} : The decision for each node along the path.
    '''
    def simple_metric(network:nx.Graph,path:List[int])->float:
        totalCost=0
        for i in range(len(path)-1):
            u,v=path[i],path[i+1]
            cost=network[u][v]['cost']
            totalCost+=cost
        return totalCost

    def add_decision(currentNode:int, addingNode:int)\
        -> Tuple[int|None,int|None]:
        if addingNode is None:
            return None,None
        if addingNode<=originalMaxIndex:
            return addingNode,None
        else:
            recIndex=pathGraph.nodes[addingNode]['recIndex']
            #determine the adding node is the left or right node
            if recoveryPaths[recIndex][1][0]==currentNode:
                return recoveryPaths[recIndex][1][-1],recIndex
            elif recoveryPaths[recIndex][1][-1]==currentNode:
                return recoveryPaths[recIndex][1][0],recIndex

    def cal_expected_cost(prob,width,occu,ps)->float:
        expCost=0
        calGraph=nx.Graph()
        calGraph.add_edge(0,1,prob=prob,width=width,occu=occu)
        pAcc=0
        for i in range(0,occu):
            calGraph[0][1]['occu']=i
            realCapacity=occu-i
            pi=np.power(1-prob,width-realCapacity)*np.power(prob,realCapacity)\
                *comb(width,realCapacity)
            pAcc+=pi
            expCost+=pi*(np.log(p2pA.ET_metric(calGraph,[0,1])/ps))
        resProb=1-pAcc
        calGraph[0][1]['occu']=0
        expCost+=resProb*np.log(p2pA.ET_metric(calGraph,[0,1])/ps)
        return expCost
    decision={}
    pathGraph=nx.Graph()
    originalMaxIndex=np.max(list(network.nodes))
    #save the room for the second shot nodes
    recIndex=originalMaxIndex+1
    if 'ps' in network.graph:
        pathGraph.graph['ps']=network.graph['ps']
    else:
        pathGraph.graph['ps']=1
    lowestProb=1
    #build the path graph
    ## recovery paths
    for i in range(len(recoveryPaths)):
        _,recoveryPath,cost=recoveryPaths[i]
        if cost==np.inf:
            continue
        #compensate the cost for ancilla nodes
        pathGraph.add_edge(recoveryPath[0],recIndex,\
                          prob=1/(cost*pathGraph.graph['ps']),width=1,\
                          distance=1,occu=0,index=i,mCost=0)
        pathGraph.edges[recoveryPath[0],recIndex]['cost']=\
                np.log(cost/pathGraph.graph['ps'])
        pathGraph.add_edge(recIndex,recoveryPath[-1],\
                          prob=1,width=1,\
                          distance=1,occu=0,index=i,mCost=0)
        pathGraph.edges[recIndex,recoveryPath[-1]]['cost']=0
        pathGraph.nodes[recIndex]['recIndex']=i
        pathGraph.nodes[recIndex]['prob']=1/(cost)
        if pathGraph.nodes[recIndex]['prob']<lowestProb:
            lowestProb=pathGraph.nodes[recIndex]['prob']
        recIndex+=1
    pathGraph.graph['n_node']=recIndex
    ## main paths
    for i in range(len(mainPath)-1):
        u=mainPath[i][0]
        v=mainPath[i+1][0]
        pathGraph.add_edge(u,v,\
                          prob=network[u][v]['prob'],\
                          width=network[u][v]['width'],\
                          distance=network[u][v]['distance'],\
                          occu=network[u][v]['occu']-1,mCost=0)
        realProb=1/ET_metric_flow(pathGraph,[u,v])
        pathGraph[u][v]['prob']=realProb
        pathGraph[u][v]['cost']=-np.log(realProb*pathGraph.graph['ps'])
        pathGraph[u][v]['width']=1
        if realProb<lowestProb:
            lowestProb=realProb
        #add second shot nodes
        if ('saveChannel' not in network.nodes[u] or not network.nodes[u]['saveChannel'])\
            and ('saveChannel' not in network.nodes[v] or not network.nodes[v]['saveChannel']):
            secondShotU=get_memory_node(pathGraph,1)(u)
            secondShotV=get_memory_node(pathGraph,1)(v)
            pathGraph.add_edge(secondShotU,secondShotV,\
                              prob=network[u][v]['prob'],
                              width=network[u][v]['width'],\
                              distance=network[u][v]['distance'],occu=0,mCost=0)
            realProb=lowestProb/ET_metric_flow(pathGraph,[secondShotU,secondShotV])
            pathGraph[secondShotU][secondShotV]['prob']=realProb
            pathGraph[secondShotU][secondShotV]['cost']=\
                cal_expected_cost(network[u][v]['prob'],network[u][v]['width'],network[u][v]['occu'],pathGraph.graph['ps'])
            pathGraph[secondShotU][secondShotV]['width']=1
    #build memory paths
    for i in range(len(mainPath)):
        node=mainPath[i][0]
        if network.nodes[node]['memory'] is None or\
            network.nodes[node]['moccu']+1<=network.nodes[node]['memory']:
            secondShotNode=get_memory_node(pathGraph,1)(node)
            pathGraph.add_edge(node,secondShotNode,\
                              prob=1,width=1,distance=0,occu=0,cost=-np.log(memProb))
    if len(mainPath)==1:
        decision[mainPath[0][0]]={'left':(None,None),'right':(None,None),'save':False}
        return decision
    #make decision for each node
    for i in range(len(mainPath)):
        localGraph=pathGraph.copy()
        begin=np.max([0,i-h])
        end=np.min([len(mainPath)-1,i+h])
        currentNode=mainPath[i][0]
        allPass=True
        decision[currentNode]={'left':(None,None),'right':(None,None),'save':False}
        for j in range(begin,end):
            if mainPath[j][1]==False:
                allPass=False
                localGraph.remove_edge(mainPath[j][0],mainPath[j+1][0])
            else:
                localGraph[mainPath[j][0]][mainPath[j+1][0]]['prob']=1
                localGraph[mainPath[j][0]][mainPath[j+1][0]]['cost']=-np.log(pathGraph.graph['ps'])
            secondShotU=get_memory_node(localGraph,1)(mainPath[j][0])
            secondShotV=get_memory_node(localGraph,1)(mainPath[j+1][0])
            #skip the saved channels
            if not localGraph.has_edge(secondShotU,secondShotV):
                continue
            residueOccu=max(network[mainPath[j][0]][mainPath[j+1][0]]['occu']-\
                        network[mainPath[j][0]][mainPath[j+1][0]]['real_capacity']-1,0)
            localGraph[secondShotU][secondShotV]['occu']=residueOccu
            localGraph[secondShotU][secondShotV]['width']=network[mainPath[j][0]][mainPath[j+1][0]]['width']
            realProb=localGraph[secondShotU][secondShotV]['prob']
            localGraph[secondShotU][secondShotV]['prob']=network[mainPath[j][0]][mainPath[j+1][0]]['prob']
            localGraph[secondShotU][secondShotV]['cost']=\
            np.log(ET_metric_flow(localGraph,[secondShotU,secondShotV])/pathGraph.graph['ps'])
            localGraph[secondShotU][secondShotV]['prob']=realProb
            localGraph[secondShotU][secondShotV]['width']=1

        newPath,_=p2pA.find_middle_point_path(localGraph,\
                                              mainPath[0][0],currentNode,mainPath[-1][0])
        if len(newPath)==0:
            continue

        #add decision
        #determine the save status first
        secondShotNode=get_memory_node(localGraph,1)(currentNode)
        if localGraph.has_edge(currentNode,secondShotNode):
            if saveAll:
                decision[currentNode]['save']=True
            else:
                for j in range(len(newPath)-1):
                    edge=(newPath[j],newPath[j+1])
                    shotU,_=get_node_shot(localGraph,newPath[j])
                    shotV,_=get_node_shot(localGraph,newPath[j+1])
                    if shotU==0 and shotV==0:
                        localGraph.edges[edge]['preserve']=True
                edges_bk=list(localGraph.edges)
                for edge in edges_bk:
                    shotU,_=get_node_shot(localGraph,edge[0])
                    shotV,_=get_node_shot(localGraph,edge[1])
                    if shotU==0 and shotV==0 and \
                        ('preserve' not in localGraph.edges[edge] \
                        or not localGraph.edges[edge]['preserve']):
                        localGraph.remove_edge(edge[0],edge[1])
            
                #calculate the two shot path
                sourceNode=get_memory_node(localGraph,1)(mainPath[0][0])
                targetNode=get_memory_node(localGraph,1)(mainPath[-1][0])
                twoShotPath,_=p2pA.find_path_ED(localGraph,sourceNode,targetNode,single=True,metric=simple_metric)
                if secondShotNode in twoShotPath:
                    secondShotNodeIndex=twoShotPath.index(secondShotNode)
                    if currentNode in twoShotPath:
                        currentNodeIndex=twoShotPath.index(currentNode)
                        if np.abs(currentNodeIndex-secondShotNodeIndex)==1:
                            decision[currentNode]['save']=True
        
        #add the switch decision
        if allPass:
            newPath=[node for node in mainPath]
            if i==0:
                decision[currentNode]['left']=(None,None)
                decision[currentNode]['right']=(newPath[i+1][0],None)
            elif i==len(mainPath)-1:
                decision[currentNode]['left']=(newPath[i-1][0],None)
                decision[currentNode]['right']=(None,None)
            else:
                decision[currentNode]['left']=(newPath[i-1][0],None)
                decision[currentNode]['right']=(newPath[i+1][0],None)
        elif len(newPath)!=0:
            index=newPath.index(currentNode)
            prev=None
            next=None
            containMemory=False
            if i==0:
                next=newPath[1]
            elif i==len(mainPath)-1:
                prev=newPath[-2]
            else:
                prev=newPath[index-1]
                next=newPath[index+1]
            if prev is not None:
                prevShot,_=get_node_shot(pathGraph,prev)
                if prevShot>0:
                    prev=None
                    containMemory=True
            if next is not None:
                nextShot,_=get_node_shot(pathGraph,next)
                if nextShot>0:
                    next=None
                    containMemory=True
            if containMemory and not decision[currentNode]['save']:
                continue
            decision[currentNode]['left']=add_decision(currentNode,prev)
            decision[currentNode]['right']=add_decision(currentNode,next)
    return decision