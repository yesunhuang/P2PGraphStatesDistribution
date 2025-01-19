'''
Name: P2PAlgorithms.py
Description: This is the main file for implementing all the algorithms being used for P2P entanglement distribution.
Email: yesunhuang@uchicago.edu
OpenSource: https://github.com/yesunhuang
LastEditors: yesunhuang yesunhuang@uchicago.edu
Msg: 
Author: YesunHuang, XiangyuRen
Date: 2024-11-20 21:29:12
'''

import networkx as nx
import numpy as np
from scipy.special import comb
from typing import*
from heapq import*

def build_unit_width_graph(graph:nx.Graph)->nx.Graph:
    '''
    description: Build the unit width graph from the input graph.

    param {nx.Graph} graph: The input graph.

    return {nx.Graph} : The unit width graph.
    '''
    graph_unit=graph.copy()
    graph_bk=graph.copy()
    nodesIndex=np.max(list(graph.nodes))
    graph_unit.graph['maxNodeIndex']=nodesIndex
    graph_unit.remove_edges_from(list(graph.edges))
    for edge in graph.edges:
        #check whether it is a save channel
        saveNode,trueNode=None,None
        if 'saveChannel' in graph.nodes[edge[0]] and graph.nodes[edge[0]]['saveChannel']:
            saveNode=edge[0]
            trueNode=edge[1]
        elif 'saveChannel' in graph.nodes[edge[1]] and graph.nodes[edge[1]]['saveChannel']:
            saveNode=edge[1]
            trueNode=edge[0]
        if saveNode is not None:
            graph_unit.add_edge(saveNode,trueNode,prob=1,width=1,distance=1,occu=0)
            if saveNode not in graph_unit.nodes:
                graph_unit.add_node(saveNode)
                for key in graph.nodes[saveNode].keys():
                    graph_unit.nodes[saveNode][key]=graph.nodes[saveNode][key]
            continue
        for occu in range(1,graph.edges[edge]['width']+1):
            nodesIndex+=1
            graph_unit.add_node(nodesIndex)
            graph_unit.nodes[nodesIndex]['edgeInfo']={'edge':edge,'occu':occu}
            graph_bk.edges[edge]['occu']=occu-1
            edgeProb=1/ET_metric(graph_bk,[edge[0],edge[1]])
            shotFactor=np.power(graph.edges[edge]['prob'],graph.edges[edge]['width'])
            graph_unit.add_edge(edge[0],nodesIndex,prob=edgeProb,\
                                width=1,distance=1,occu=0,shotFactor=shotFactor)
            graph_unit.add_edge(nodesIndex,edge[1],prob=edgeProb,width=1,\
                                distance=1,occu=0,shotFactor=shotFactor)
            if graph.edges[edge]['occu']>=occu:
                graph_unit.edges[edge[0],nodesIndex]['occu']=1
                graph_unit.edges[nodesIndex,edge[1]]['occu']=1
    return graph_unit

def collapse_unit_flow(flowDict:Dict,graph_dec:nx.DiGraph)->nx.DiGraph:
    '''
    description: Collapse the flow dictionary into the flow graph.

    param {Dict} flowDict: The input flow dictionary.

    param {nx.DiGraph} graph_dec: The input graph.

    return {nx.DiGraph} : The flow graph.
    '''
    flowGraph=graph_dec.copy()
    nodeList=list(graph_dec.nodes)
    for node in nodeList:
        if 'edgeInfo' in graph_dec.nodes[node]:
            edgeInfo=graph_dec.nodes[node]['edgeInfo']
            edge=edgeInfo['edge']
            shot=graph_dec.nodes[node]['shot']
            edge_k=(edge[0]+shot*graph_dec.graph['maxIndex'],edge[1]+shot*graph_dec.graph['maxIndex'])
            flow_l=flowDict[edge_k[0]][node]
            flow_re_l=flowDict[node][edge_k[0]]
            flow_r=flowDict[node][edge_k[1]]
            flow_re_r=flowDict[edge_k[1]][node]
            if flow_l>0:
                flow=flow_l
            else:
                flow=-flow_re_l
            if edge_k not in flowGraph.edges:
                flowGraph.add_edge(edge_k[0],edge_k[1],flow=flow)
                flowGraph.add_edge(edge_k[1],edge_k[0],flow=-flow)
            else:
                flowGraph.edges[edge_k[0],edge_k[1]]['flow']+=flow
                flowGraph.edges[edge_k[1],edge_k[0]]['flow']-=flow

            flowGraph.remove_node(node)

    #add the flow for other edges
    for edge_k in flowGraph.edges:
        if 'flow' not in flowGraph.edges[edge_k]:
            flowGraph.edges[edge_k]['flow']=flowDict[edge_k[0]][edge_k[1]]

    return flowGraph
            

def remove_ancilla(path:List[int], network:nx.Graph)->List[int]:
    '''
    description: Remove the ancilla nodes from the path.

    param {List[int]} path: The input path.

    param {nx.Graph} network: The input network.

    return {List[int]} : The path without ancilla nodes.
    '''
    path_bk=[]
    for i in range(len(path)):
        if path[i] is not None:
            node=network.nodes[path[i]]
            if 'isAncilla' in node:
                if node['isAncilla']:
                    continue
            #remove the nodes corresponding to saved channels
            if 'saveChannel' in node \
                and node['saveChannel']:
                    continue
        path_bk.append(path[i]) 
    return path_bk

def find_path_ED(graph: nx.Graph, source: int|List[int], targets: int|List[int], single:bool=False, \
                    capacity:str='width', occupy:str='occu',
                    metric: Callable[[nx.Graph,List[int]],float]|None=None) \
                    -> Tuple[List[int], float]|Tuple[Dict[int,Tuple[List[int],float]],np.ndarray]:
    '''
    description: The extended dijkstra algorithm for finding the optimal path from source to targets with momentum recording

    param {nx.Graph} : The input graph.

    param {int|List[int]} source: The source nodes.

    param {List[int]} target: target nodes

    param {bool} single: whether to find the single optimal path or all paths.

    param {callable} metric: a function that takes path as input and returns a float.

    return {Tuple[List[int], float]|Dict[int,Tuple[List[int],float],np.ndarray]} : 
    The optimal paths and the cost of the paths as well as branching number if single is False.
    '''
    def construct_path(v: int, calBranch:bool=False) -> List[int]:
        '''
        description: Construct the path from the previous node list.

        param {int} v: The target node.

        param {bool} fixMomentum: whether to find the branching number.

        return {List[int]} : The path from source to target.
        '''
        path=[]
        if not visited[v]:
            return path
        while v != -1:
            path.append(v)
            if calBranch:
                branch[v]+=1
            v=prev[v]
        return path[::-1]
    
    if not isinstance(source,Iterable):
        source=[source]
    if not isinstance(targets,Iterable):
        targets=[targets]
    # initialize the distance and the previous node
    maxIndex=np.max(list(graph.nodes))+1
    visited=np.full(maxIndex,False,dtype=bool)
    # initialize the cost value
    cost=np.full(maxIndex,np.inf,dtype=float)
    # initialize the branch
    branch=np.full(maxIndex,0,dtype=int)
    # initialize the previous nodes and width for the path
    prev=np.full(maxIndex,-1,dtype=int)
    # initialize the heap
    heap=[]

    #initialize the source
    for s in source:
        cost[s]=metric(graph,[s])
        heappush(heap,(cost[s],s))


    n=len(targets)

    while len(heap)>0 and n>0:
        # pop the node with the smallest distance
        _,u=heappop(heap)
        if visited[u]:
            continue
        visited[u]=True
        if single and (u in targets):
            return construct_path(u,True),cost[u]
        # update the expectation value
        for v in graph.neighbors(u):
            if visited[v] or v in source or\
                (graph[u][v][capacity]-graph[u][v][occupy])<=0:
                continue
            if metric is None:
                if cost[v]>cost[u]+1:
                    cost[v]=cost[u]+1
                    prev[v]=u
                    heappush(heap,(cost[v],v))
            else:
                path=construct_path(u)+[v]
                newCost=metric(graph,path)
                if cost[v]>newCost:
                    cost[v]=newCost
                    prev[v]=u
                    heappush(heap,(cost[v],v))
    paths={v:(construct_path(v,True),cost[v]) for v in targets}
    if single:
        return [],np.inf
    else:
        return paths,branch

def remove_paths(graph: nx.Graph, path: List[int],\
                 parallel:int=1,
                 capacity:str='width',occupy:str="occu", \
                 ifAll:bool=False) -> nx.Graph:
    '''
    description: Remove the paths resources from the graph.

    param {nx.Graph} graph: The input graph.

    param {List[int]} path: paths to be removed

    param {int} parallel: the number of parallel paths. if negative, then the path resources will be added.

    param {str} capacity: name of the capacity.

    param {str} occupy: name of the occupy.

    param {bool} ifAll: whether to remove all the resources.

    return {nx.Graph} : The graph after removing the edges.
    '''
    for i in range(len(path)-1):
        u,v=path[i],path[i+1]
        if not ifAll:
            graph.edges[u,v][occupy]+=parallel
        else:
            graph.edges[u,v][occupy]=graph.edges[u,v][capacity]
    return graph

def find_recovery_path(graph: nx.Graph, mainPath:List[List[int]], \
                       metric: Callable[[nx.Graph,List[int]],float]|None=None,host:int=1)\
    ->Dict[int,List[Tuple[int,List[int],float]]]:
    '''
    description: find the recovery path for each main paths. Note that the graph will be modified after paths are found.

    param {nx.Graph} graph: the input graph.

    param {List} mainPath: the input main paths

    param {Callable[[nx.Graph,List[int]],float]} metric: the metric for the path.

    param {int} host: the maximum host node

    return {Dict[int,List[Tuple[int,List[int],float]]]}: the recovery paths for each main path. [Index of main path, [host, recovery path, cost]]
    ''' 

    recoveryPaths={}
    graph_bbk=graph.copy()
    for i in range(len(mainPath)):
        graph_bk=graph_bbk.copy()
        remove_paths(graph_bk,mainPath[i],ifAll=True)
        path=mainPath[i]
        recoveryPaths[i]=[]
        for h in range(1,host+1):
            for j in range(len(path)-h):
                u,v=path[j],path[j+h]
                recoveryPath, cost=find_path_ED(graph_bk,u,[v],single=True,metric=metric)
                if len(recoveryPath)==0:
                    continue
                else:
                    recoveryPaths[i].append((h,recoveryPath,cost))
                    remove_paths(graph_bbk,recoveryPath)
                    remove_paths(graph_bk,recoveryPath)
    return recoveryPaths

def ET_metric(graph: nx.Graph, path: List[int], endpointsProbs:Tuple[Dict[int,float],Dict[int,float]]|None=None) -> float:
    '''
    description: The expectation time of constructing at least one entanglement link on the path.

    param {nx.Graph} graph: the input graph.

    param {List[int]} path: the input path.

    param {Tuple[Dict[int,float],Dict[int,float]]} endpointsProbs: the probability of the endpoints.

    return {float}: the cost of the path.
    '''
    P_t=1
    
    if endpointsProbs is not None:
        sourceCost,targetCost=endpointsProbs
        if path[0] in sourceCost:
            P_t*=sourceCost[path[0]]
        if path[-1] in targetCost:
            P_t*=targetCost[path[-1]]

    #path=remove_ancilla(path,graph)
    for i in range(0,len(path)-1):
        u,v=path[i],path[i+1]
        
        p_i=graph.edges[u,v]['prob']
        k=graph.edges[u,v]['occu']+1
        p_one=0
        if p_i==1:
            p_one=1
        else:
            for j in range(k,graph.edges[u,v]['width']+1):
                p_one+=np.power(1-p_i,graph.edges[u,v]['width']-j)*np.power(p_i,j)*comb(graph.edges[u,v]['width'],j)
        P_t*=p_one

    if 'ps' in graph.graph:
        ps=graph.graph['ps']
    else:
        ps=1.0

    path=remove_ancilla(path,graph)

    if len(path)>2:
        P_t*=np.power(ps,len(path)-2)
    else:
        if len(path)==1:
            P_t/=ps
    
    if P_t==0:
        return np.inf
    else:
        return 1/P_t

def decorate_multi_target_maxflow(graph: nx.Graph, targets:List[int])->Tuple[nx.Graph,int]:
    '''
    description: Decorate the graph with the multi-target maxflow.

    param {nx.Graph} graph: the input graph.

    param {List[int]} targets: the input target nodes.

    return {Tuple[nx.Graph,int]}: the graph after decoration and the ancilla node
    '''
    graph_dec=graph.copy()
    if 'pos' in graph.nodes[0]:
        graph_dec.add_node(len(graph.nodes),pos=(0,0),isAncilla=True,father=None)
    else:
        graph_dec.add_node(len(graph.nodes),isAncilla=True,father=None)
    for t in targets:
        graph_dec.add_edge(t,len(graph.nodes),prob=1,width=1,res=1,distance=0)
    return graph_dec, len(graph.nodes)




def decorate_multi_target_disjoint(graph:nx.Graph, source:int, targets:List[int]|Dict[int,List[int]],\
                                   shots:int=1,ifReset: bool=True, minCost: bool=False, \
                                   memShot: bool|float=False,regular: int=100)->Tuple[nx.DiGraph,int,int]:
    '''
    description: Decorate the graph for the multi-target disjoint paths finding under k shots.

    param {nx.Graph} graph: the input graph.

    param {int} source: the input source node.

    param {List[int]} targets: the input target nodes.

    param {int} shots: the number of shots.

    param {bool} ifReset: whether to reset the flow of the edges for higher shots.

    param {bool} minCost: whether to find the minimum cost.

    param {bool|float} memShot: whether to punish the higher shots.

    param {int} regular: the regular factor for the cost.

    return {Tuple[nx.DiGraph,int,int]}: the graph after decoration and the ancilla node and index virtual target node and virtual root node.
    '''
    graph_dec=nx.DiGraph()
    maxIndex=np.max(list(graph.nodes))+1
    graph_dec.graph['maxIndex']=maxIndex

    if 'ps' in graph.graph:
        ps=graph.graph['ps']
    else:
        ps=1.0

    targets_bk={}
    if isinstance(targets,List):
        for i in range(len(targets)):
            targets_bk[i]=[targets[i]]
        maxTargetIndex=len(targets)
    else:
        targets_bk=targets
        if len(targets)>0:
            maxTargetIndex=np.max(list(targets.keys()))+1
        else:
            maxTargetIndex=0
    
    sourceIndex=shots*maxIndex+maxTargetIndex+1
    targetIndex=shots*maxIndex+maxTargetIndex+2

    #add the secondary target nodes
    for t in targets_bk.keys():
        graph_dec.add_edge(t+shots*maxIndex,targetIndex,prob=1,width=1,distance=0)
        if minCost:
            graph_dec.edges[t+shots*maxIndex,targetIndex]['cost']=0
    #add edges for each shots
    for k in range(shots):
        #add edge from the source node to the source node of the k-th shot
        graph_dec.add_edge(sourceIndex,source+k*maxIndex,prob=1,width=len(targets),distance=0)
        if minCost:
            graph_dec.edges[sourceIndex,source+k*maxIndex]['cost']=0

        #add edge from the target node of the k-th shot secondary target nodes
        for t in targets_bk.keys():
            for node in targets_bk[t]:
                graph_dec.add_edge(node+k*maxIndex,t+shots*maxIndex,prob=1,width=1,distance=0)
                if minCost:
                    graph_dec.edges[node+k*maxIndex,t+shots*maxIndex]['cost']=0

        #add nodes for the original graph
        for u in graph.nodes:
            if 'saveChannel' in graph.nodes[u] and graph.nodes[u]['saveChannel']:
                continue
            uk=u+k*maxIndex
            graph_dec.add_node(uk)
            for key in graph.nodes[u].keys():
                graph_dec.nodes[uk][key]=graph.nodes[u][key]
            graph_dec.nodes[uk]['shot']=k

        #add edges for the original graph
        for u,v in graph.edges:
            #skip the save channels for higher shots.
            if k>0:
                if 'saveChannel' in graph.nodes[u] and graph.nodes[u]['saveChannel']:
                    continue
                if 'saveChannel' in graph.nodes[v] and graph.nodes[v]['saveChannel']:
                    continue
            uk,vk=u+k*maxIndex,v+k*maxIndex
            graph_dec.add_edge(uk,vk,prob=graph.edges[u,v]['prob'],width=graph.edges[u,v]['width']-graph.edges[u,v]['occu'])
            graph_dec.add_edge(vk,uk,prob=graph.edges[u,v]['prob'],width=graph.edges[u,v]['width']-graph.edges[u,v]['occu'])
            if minCost:
                if isinstance(memShot,bool) and memShot:
                    if 'shotFactor' in graph.edges[u,v]:
                        factor=np.power(graph.edges[u,v]['shotFactor'],k)
                    else:
                        factor=1
                    graph_dec.edges[uk,vk]['cost']=max(int(min(-regular*np.log(graph.edges[u,v]['prob']*ps*factor),1E4*regular)),1)
                    graph_dec.edges[vk,uk]['cost']=max(int(min(-regular*np.log(graph.edges[u,v]['prob']*ps*factor),1E4*regular)),1)
                elif isinstance(memShot,float):
                    graph_dec.edges[uk,vk]['cost']=max(int(min(-regular*((k+1)*memShot*np.log(graph.edges[u,v]['prob'])+np.log(ps)),1E4*regular)),1)
                    graph_dec.edges[vk,uk]['cost']=max(int(min(-regular*((k+1)*memShot*np.log(graph.edges[u,v]['prob'])+np.log(ps)),1E4*regular)),1)
                else:
                    graph_dec.edges[uk,vk]['cost']=max(int(min(-regular*np.log(graph.edges[u,v]['prob']*ps),1E4*regular)),1)
                    graph_dec.edges[vk,uk]['cost']=max(int(min(-regular*np.log(graph.edges[u,v]['prob']*ps),1E4*regular)),1)
            if ifReset and k>0:
                graph_dec.edges[uk,vk]['width']=graph.edges[u,v]['width']
                graph_dec.edges[vk,uk]['width']=graph.edges[u,v]['width']
            
    return graph_dec, sourceIndex, targetIndex

def maxflow_decomposition(graph:nx.Graph,flowGraph: nx.DiGraph, source: int, targets:List[int], virtualNode:int=None,\
                          metric:Callable[[nx.Graph,List[int]],float]|None=None)->Dict[int,Tuple[List[int],float]]:
    '''
    description: The multi-target maxflow algorithm for finding the optimal flows and the cost from source to targets with momentum recording

    param {nx.Graph} graph: The input graph.
    
    param {nx.DiGraph} flowGraph: The flow graph.

    param {int} source: The source nodes.

    param {List[int]} target: target nodes

    param {int} virtualNode: the virtual node for the flow graph.

    param {Callable[[nx.Graph,List[int]],float} metric: the metric for the path.

    return {Dict[int,Tuple[List[int],float]]} : 
    The optimal flows path decomposition and the digraph for representing the flow.
    '''
    targets_bk=targets.copy()
    flowGraph_bk=flowGraph.copy()
    paths={}
    if virtualNode is None:
        virtualNode=len(flowGraph.nodes)-1
    for u,v in flowGraph_bk.edges:
        if (u==virtualNode) or (v==virtualNode):
            flowGraph_bk.edges[u,v]['prob']=1
            flowGraph_bk.edges[u,v]['width']=0
            flowGraph_bk.edges[u,v]['flow']=0
            flowGraph_bk.edges[u,v]['occu']=0
            flowGraph_bk.edges[u,v]['distance']=1
        else:
            flowGraph_bk.edges[u,v]['prob']=graph.edges[u,v]['prob']
            flowGraph_bk.edges[u,v]['width']=graph.edges[u,v]['width']
            flowGraph_bk.edges[u,v]['distance']=graph.edges[u,v]['distance']
            flowGraph_bk.edges[u,v]['occu']=0
    while len(targets_bk)>0:
        path, cost=find_path_ED(flowGraph_bk,source,targets_bk,single=True,metric=metric, capacity='flow')
        if len(path)==0:
            break
        paths[path[-1]]=path,cost
        flowGraph_bk=remove_paths(flowGraph_bk,path,occupy='occu',capacity='flow')
        targets_bk.remove(path[-1])
    return paths

def find_multi_targets_disjoint(graph:nx.Graph, source:int, targets:List[int]|Dict[int,List[int]],\
                                metric:Callable[[nx.Graph,List[int]],float]|None=None,\
                                shots:int|None=None, decompose:bool=True, \
                                ifReset: bool=True, minCost:bool=False,\
                                memShot: bool|float=False)\
                                ->Dict[int,Dict[int,List[Tuple[List[int],float]]]]|\
                                Tuple[Dict[int,Dict[int,List[Tuple[List[int],float]]]],int]|int:
    '''
    description:find the minimum shots and the paths to distribute entanglement to all targets. 
    This method will not modify the original graph. 

    param {nx} graph: the input graph.

    param {int} source: the input source node.

    param {List[int]|Dict[int,List[int]]} targets: the input target nodes or the dictionary of the target nodes.

    param {Callable[[nx.Graph,List[int]],float]} metric: the metric for the path.

    param {int} shots: the maximum shots.

    param {bool} decompose: whether to decompose the paths.

    param {bool} ifReset: whether to reset the flow of the edges.

    param {bool} minCost: whether to find the minimum cost.

    param {bool|float} memShot: whether to punish the higher shots.

    return {Dict[int,Dict[int,List[Tuple[List[int],float]]]]|Tuple[Dict[int,Dict[int,List[Tuple[List[int],float]]]],int]|int} :
        {shots: {target:(path,cost)}} + integer cost if minCost or shot if not decompose
    '''
    #backup the targets
    targets_bk={}
    if isinstance(targets,dict):
        for k in targets.keys():
            targets_bk[k]=targets[k].copy()
        if len(targets)>0:
            maxTargetIndex=np.max(list(targets.keys()))+1
        else:
            maxTargetIndex=0
    else:
        for i in range(len(targets)):
            targets_bk[i]=[targets[i]]
        maxTargetIndex=len(targets)
    
    if minCost:
        graph_bk=build_unit_width_graph(graph)
    else:
        graph_bk=graph.copy()

    if shots is None:
        shots=1
        lShots=0
        rShots=None
        rShotsMax=len(targets)
        while rShots is None or \
        (lShots<rShots and  shots>=1):
            graph_dec, sourceIndex, targetIndex=decorate_multi_target_disjoint(\
                graph_bk,source,targets_bk,shots,ifReset,minCost=minCost,memShot=memShot)
            if not minCost:
                graph_flow=nx.algorithms.flow.shortest_augmenting_path(graph_dec,sourceIndex,targetIndex, capacity='width')
            else:
                graph_flow=nx.algorithms.flow.max_flow_min_cost(graph_dec,sourceIndex,targetIndex, capacity='width', weight='cost')
                flowValue=sum((graph_flow[u][targetIndex] for u in graph_dec.predecessors(targetIndex)))
                graph_flow=collapse_unit_flow(graph_flow,graph_dec)
                graph_flow.graph['flow_value']=flowValue
            if graph_flow.graph['flow_value']<len(targets_bk.keys()):
                lShots=shots+1
                if rShots is None:
                    shots=shots*2
                    if shots>rShotsMax:
                        shots=rShotsMax
                        rShots=rShotsMax
                else:
                    if shots>rShotsMax:
                        rShots=rShotsMax
                    shots=shots+int(np.ceil((rShots-shots)/2))
            else:
                rShots=shots
                shots=shots-int(np.ceil((shots-lShots)/2))

        shots=rShots

    if not decompose:
        return shots

    graph_dec, sourceIndex, targetIndex=decorate_multi_target_disjoint(graph_bk,source,targets_bk,shots,ifReset,\
                                                                       minCost=minCost,memShot=memShot)
    if not minCost:
        graph_flow=nx.algorithms.flow.shortest_augmenting_path(graph_dec,sourceIndex,targetIndex, capacity='width')
    else:
        graph_flow=nx.algorithms.flow.max_flow_min_cost(graph_dec,sourceIndex,targetIndex, capacity='width', weight='cost')
        costOfMinCost=nx.algorithms.flow.cost_of_flow(graph_dec,graph_flow,weight='cost')
        flowValue=sum((graph_flow[u][targetIndex] for u in graph_dec.predecessors(targetIndex)))
        graph_flow=collapse_unit_flow(graph_flow,graph_dec)
        graph_flow.graph['flow_value']=flowValue
        

    paths={}
    maxIndex=np.max(list(graph_bk.nodes))+1
    graph_bk=graph.copy()

    lt=shots*maxIndex
    rt=shots*maxIndex+maxTargetIndex
    for k in range(shots):
        graph_flow_s=nx.DiGraph()
        l=k*maxIndex
        r=(k+1)*maxIndex
        paths[k]={}

        #construct the subgraph for the k-th shot
        for u in graph_flow.nodes:
            if u<l or u>=r:
                continue
            graph_flow_s.add_node(u-l)
            if 'isAncilla' in graph_bk.nodes[u-l]:
                graph_flow_s.nodes[u-l]['isAncilla']=graph_bk.nodes[u-l]['isAncilla']

        targets_s=[]
        targets_map={}
        
        for u,v in graph_flow.edges:
            #add reachable targets in this shot
            if v>=lt and v<rt:
                if u>=l and u<r: 
                    if graph_flow.edges[u,v]['flow']>0:
                        for i in range(graph_flow.edges[u,v]['flow']):
                            targets_s.append(u-l)
                            if (u-l) in targets_map:
                                targets_map[u-l].append(v-lt)
                            else:
                                targets_map[u-l]=[v-lt]
                        #targets_s.append(u-l)
            if u<l or u>=r or v<l or v>=r:
                continue
            #add the edges
            graph_flow_s.add_edge(u-l,v-l,prob=graph_bk.edges[u-l,v-l]['prob'],\
                                  width=graph_bk.edges[u-l,v-l]['width'],\
                                  distance=graph_bk.edges[u-l,v-l]['distance'],\
                                  flow=graph_flow.edges[u,v]['flow'],\
                                  occu=graph_bk.edges[u-l,v-l]['occu'],\
                                  flowOccu=0)
            if k>0 and ifReset:
                graph_flow_s.edges[u-l,v-l]['occu']=0
        graph_flow_s.graph['ps']=graph_bk.graph['ps']

        while len(targets_s)>0:
            path, cost=find_path_ED(graph_flow_s,source,targets_s,single=True,\
                                    metric=metric, capacity='flow', occupy='flowOccu')
            if len(path)==0:
                break
            mapIndex=targets_map[path[-1]].pop()
            paths[k][mapIndex]=(path,cost)
            graph_flow_s=remove_paths(graph_flow_s,path,occupy='flowOccu',capacity='flow')
            graph_flow_s=remove_paths(graph_flow_s,path,occupy='occu',capacity='width')
            targets_s.remove(path[-1])
            del targets_bk[mapIndex]
    if not minCost:
        return paths
    else:
        return paths, costOfMinCost
                
def find_middle_point_path(graph:nx.Graph, source:int, middle:int, target:int, regular:int=100)->Tuple[List[int],float]:
    '''
    description: Find the path with minimum cost from source to target with the middle point.

    param {nx.Graph} graph: The input graph.

    param {int} source: The source node.

    param {int} middle: The middle node.

    param {int} target: The target node.

    param {int} regular: The regular factor for the cost.

    return {Tuple[List[int],float]} : The path and the cost.
    '''
    #turn the graph into a directed graph
    graph_dec=nx.DiGraph()
    ps=graph.graph['ps']
    maxIndex=np.max(list(graph.nodes))
    virtualTarget=maxIndex+1
    graph_dec.add_nodes_from(graph.nodes)
    graph_dec.add_edge(source,virtualTarget,cost=0,width=1)
    graph_dec.add_edge(target,virtualTarget,cost=0,width=1)
    for u,v in graph.edges:
        rawCost=-np.log(graph.edges[u,v]['prob']*ps)
        cost=np.ceil(rawCost*regular)
        graph_dec.add_edge(u,v,cost=cost,width=graph.edges[u,v]['width'])
        graph_dec.add_edge(v,u,cost=cost,width=graph.edges[u,v]['width'])

    #find the max flow min cost
    flow=nx.algorithms.max_flow_min_cost(graph_dec,middle,virtualTarget,capacity='width',weight='cost')
    #turn the flow into path
    if flow[source][virtualTarget]==0 or flow[target][virtualTarget]==0:
        return [],np.inf

    flowGraph=nx.DiGraph()
    for u in flow.keys():
        for v in flow[u].keys():
            flowGraph.add_edge(u,v,flow=flow[u][v])
    paths=maxflow_decomposition(graph,flowGraph,middle,\
                                [source,target],virtualNode=virtualTarget,\
                                metric=ET_metric)
    path=paths[source][0][-1:0:-1]
    path+=paths[target][0]
    cost=ET_metric(graph,path)
    return path,cost
    

def chase_the_path(source:int,target:int,\
                   decision: Dict[int,Dict[str,Tuple[int|None,int|None]]])\
                    ->Tuple[List[int],List[int],bool]:
    '''
    description: Chase the path from source to target.

    param {int} source: The source node.

    param {int} target: The target node.

    param {Dict[int,Dict[str,Tuple[int|None,int|None]]]} decision: The decision for each node.

    return {Tuple[List[int],List[int],bool]} : The path, the recovery path index and whether the path is successful.
    '''
    path=[source]
    recoveryPathIndex=[]
    sourceRight=decision[source]['right'][0]
    if sourceRight is None:
        if target==source:
            return path,recoveryPathIndex,True
        else:
            return path,recoveryPathIndex,False
    else:
        nextNode=sourceRight
    while path[-1]!=target:
        current=nextNode
        nextNode=decision[current]['right'][0]
        prevNode=decision[current]['left'][0]  
        if prevNode is None: 
            return path,recoveryPathIndex,False
        elif prevNode!=path[-1]:
            #take care of additional switch
            path.append(None)
            return path,recoveryPathIndex,False
        else:
            path.append(current)
            if decision[current]['left'][1] is not None:
                recoveryPathIndex.append(decision[current]['left'][1])
        if nextNode is None and current!=target:
            return path,recoveryPathIndex,False
    
    return path,recoveryPathIndex,True

def path_recovery_EUM(network:nx.Graph,mainPath:List[Tuple[int,bool]],\
                      recoveryPaths:List[Tuple[int,List[int],float]],h:int)->Dict[str,Any]:
    '''
    description: Fix the path with the main path and recovery path with the expected union method.

    param {nx.Graph} network: The input network.

    param {List[Tuple[int,bool]]} mainPath: The main path.

    param {List[Tuple[int,List[int],float]]} recoveryPaths: The recovery paths.

    param {int} h: The maximum host.

    return {Dict[str,Any]} : The decision for each node along the path.
    '''
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

    pathGraph=nx.Graph()
    originalMaxIndex=np.max(list(network.nodes))+1
    maxIndex=originalMaxIndex+1
    if 'ps' in network.graph:
        pathGraph.graph['ps']=network.graph['ps']
    else:
        pathGraph.graph['ps']=1.0
    #build the path graph
    for i in range(len(recoveryPaths)):
        _,recoveryPath,cost=recoveryPaths[i]
        if cost==np.inf:
            continue
        #compensate the cost for ancilla nodes
        pathGraph.add_edge(recoveryPath[0],maxIndex,\
                          prob=1/(cost*network.graph['ps']),width=1,\
                          distance=1,occu=0,index=i)
        pathGraph.add_edge(maxIndex,recoveryPath[-1],\
                          prob=1,width=1,\
                          distance=1,occu=0,index=i)
        pathGraph.nodes[maxIndex]['recIndex']=i
        pathGraph.nodes[maxIndex]['prob']=1/(cost*network.graph['ps'])
        maxIndex+=1

    for i in range(len(mainPath)-1):
        u=mainPath[i][0]
        v=mainPath[i+1][0]
        pathGraph.add_edge(u,v,\
                          prob=network[u][v]['prob'],\
                          width=network[u][v]['width'],\
                          distance=network[u][v]['distance'],\
                          occu=network[u][v]['occu']-1)
        realProb=1/ET_metric(pathGraph,[u,v])
        pathGraph[u][v]['prob']=realProb
        pathGraph[u][v]['width']=1
    decision={}
    if len(mainPath)==1:
        decision[mainPath[0][0]]={'left':(None,None),'right':(None,None)}
        return decision
    #make decision for each node
    for i in range(len(mainPath)):
        localGraph=pathGraph.copy()
        begin=np.max([0,i-h])
        end=np.min([len(mainPath)-1,i+h])
        currentNode=mainPath[i][0]
        allPass=True
        decision[currentNode]={'left':(None,None),'right':(None,None)}
        for j in range(begin,end):
            if mainPath[j][1]==False:
                allPass=False
                localGraph.remove_edge(mainPath[j][0],mainPath[j+1][0])
            else:
                localGraph[mainPath[j][0]][mainPath[j+1][0]]['prob']=1
        newPath,_=find_middle_point_path(localGraph,\
                                mainPath[0][0],mainPath[i][0],mainPath[-1][0])
        if allPass:
            newPath=[node for node in mainPath]
            if i==0:
                decision[currentNode]={'left':(None,None),'right':(mainPath[i+1][0],None)}
            elif i==len(mainPath)-1:
                decision[currentNode]={'left':(mainPath[i-1][0],None),'right':(None,None)}
            else:
                decision[currentNode]={'left':(mainPath[i-1][0],None),'right':(mainPath[i+1][0],None)}
        elif len(newPath)!=0:
            index=newPath.index(currentNode)
            prev=None
            next=None
            if i==0:
                next=newPath[1]
            elif i==len(mainPath)-1:
                prev=newPath[-2]
            else:
                prev=newPath[index-1]
                next=newPath[index+1]
            decision[currentNode]={'left':add_decision(currentNode,prev),\
                         'right':add_decision(currentNode,next)}
    return decision

def update_VRM(VRM:Dict[int,Dict[int,Dict[str,Any]]],\
               edgeDependent:Dict[Tuple[int,int],Dict[str,Any]],\
               network:nx.Graph,\
               mainPaths:Dict[int,Tuple[List[int],float,int]], \
               k:int, controllable:List|None=None)\
               ->Tuple[Dict[int,Dict[int,Dict[str,Any]]],\
                        Dict[Tuple[int,int],Dict[str,Any]],\
                        Dict[int,Tuple[List[int],float,int]]]:
    '''
    description: Update the Vertex Reaching Map (VRM) for the network.

    param {Dict[int,Dict[int,Dict[str,Any]]]} VRM: The input VRM.

    param {Dict[Tuple[int,int],Dict[str,Any]]} edgeDependent: The input edge dependent.

    param {nx.Graph} network: The input network.

    param {Dict[int,Tuple[List[int],float,int]]} mainPaths: The input main paths.

    param {int} k: current shot.

    param {List} controllable: The controllable nodes.

    return {Tuple[Dict[int,Dict[int,Dict[str,Any]]],\
                    Dict[Tuple[int,int],Dict[str,Any]],\
                    Dict[int,Tuple[List[int],float,int]]]} : The updated VRM, edge dependent, main paths.
    '''

    
    def split_path(node:int, vertex:int)->int:
        pathDict=VRM[vertex][node][0]['mainPathDict']
        kSplit=VRM[vertex][node][0]['shot']
        if pathDict is None:
            return len(mainPaths[kSplit])-1
        currentIndex=mainPaths[kSplit].index(pathDict)
        pathForSplit, cost, vu=pathDict['path'], pathDict['cost'], pathDict['VU']

        if vu[0]==vertex:
            keep=vu[0]
            out=vu[1]
        else:
            keep=vu[1]
            out=vu[0]

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

        partialPathCost=1/VRM[vertex][node][0]['prob']

        if len(partialPath)==1:
            #no need to split
            return len(mainPaths[kSplit])-1
            
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
            return len(mainPaths[kSplit])-1
            
        #an additional switch probability is taken into account
        pathDict['path']=partialPath
        pathDict['cost']=partialPathCost
        pathDict['VU']=newPartialUV
        otherPartialPathCost=cost/partialPathCost
        otherPartialPathDict={'path':otherPartialPath,'cost':otherPartialPathCost,\
                              'VU':newOtherPartialUV}
        
        #insert the new partial path after the current path
        mainPaths[kSplit].insert(currentIndex+1,otherPartialPathDict)

        #for the section of out
        for node_i in otherPartialPath:
            if keep!=out and (node_i in VRM[out]):
                for qubit in VRM[out][node_i]:
                    if qubit['mainPathDict'] is pathDict:
                        qubit['mainPathDict']=otherPartialPathDict
                        break

            if node_i in VRM[keep]:
                if node_i == node:
                    qubitDict={'prob':switchProb/partialPathCost,'shot':kSplit,'ifFixed':True,'prev':node,\
                                'mainPathDict':otherPartialPathDict}
                    VRM[vertex][node].append(qubitDict)
                    continue
                for qubit in VRM[keep][node_i]:
                    if qubit['mainPathDict'] is pathDict:
                        qubit['prev']=node
                        qubit['mainPathDict']=otherPartialPathDict
                        break

        if (vu[0],vu[1]) in edgeDependent:
            if edgeDependent[(vu[0],vu[1])]['pathDict'] is pathDict:
                edgeDependent[(vu[0],vu[1])]['pathDict']=otherPartialPathDict
                edgeDependent[(vu[1],vu[0])]['pathDict']=otherPartialPathDict

        return len(mainPaths[kSplit])-1
            

    controllableNodes=[]
    currentPath=mainPaths[k][-1]
    path,VU=currentPath['path'],currentPath['VU']

    if 'ps' in network.graph:
        switchProb=network.graph['ps']
    else:
        switchProb=1

    v,u=VU

    for node in path:
        if controllable is None:
            controllableNodes.append(node)
        else:
            if node in controllable:
                controllableNodes.append(node)
    
    if v==u:
        source_cost={path[0]:VRM[v][path[0]][0]['prob']}
        split_path(path[0],v)
        for node in controllableNodes:
            partialPath=path[:path.index(node)+1]
            partialCost=ET_metric(network,partialPath,(source_cost,{}))
            #reimburse the cost for the switch probability
            if len(partialPath)==1:
                qubitDict={'prob':switchProb**2/partialCost,'shot':k,'ifFixed':True,'prev':path[0],\
                           'mainPathDict':currentPath}
            else:
                qubitDict={'prob':switchProb/partialCost,'shot':k,'ifFixed':False,'prev':path[0],\
                           'mainPathDict':currentPath}
            if node not in VRM[v]:
                VRM[v][node]=[qubitDict]
            else:
                VRM[v][node].append(qubitDict)
        return VRM, edgeDependent, mainPaths

    source_cost={path[0]:VRM[v][path[0]][0]['prob']}
    target_cost={path[-1]:VRM[u][path[-1]][0]['prob']}

    split_path(path[0],v)
    split_path(path[-1],u)
    
    for node in controllableNodes:
        partialPath=path[:path.index(node)+1]
        partialCost=ET_metric(network,partialPath,(source_cost,{}))
        #reimburse the cost for the switch probability
        if len(partialPath)==1:
            qubitDict={'prob':switchProb**2/partialCost,'shot':k,'ifFixed':True,'prev':path[0],\
                       'mainPathDict':currentPath}
        else:
            qubitDict={'prob':switchProb/partialCost,'shot':k,'ifFixed':False,'prev':path[0],\
                       'mainPathDict':currentPath}
            
        if node not in VRM[v]:
            VRM[v][node]=[qubitDict]
        else:
            VRM[v][node].append(qubitDict)

        partialPath=path[path.index(node):]
        partialCost=ET_metric(network,partialPath,({},target_cost))
        if len(partialPath)==1:
            qubitDict={'prob':switchProb**2/partialCost,'shot':k,'ifFixed':True,'prev':path[-1],\
                       'mainPathDict':currentPath}
        else:
            qubitDict={'prob':switchProb/partialCost,'shot':k,'ifFixed':False,'prev':path[-1],\
                       'mainPathDict':currentPath}
        if node not in VRM[u]:
            VRM[u][node]=[qubitDict]
        else:
            VRM[u][node].append(qubitDict)
            

    return VRM, edgeDependent, mainPaths

def find_P2P_GST_paths(graphState:nx.Graph, network:nx.Graph, maxShot:int|None=None, \
                       controllable:List|None=None,ifReset:bool=True)\
    ->Tuple[Dict[int,Tuple[List[int],float,Tuple[int,int],int]],\
            Dict[Tuple[int,int],Dict[str,Any]],\
            Dict[int,Dict[int,Dict[str,Any]]]]:
    '''
    description: Find the P2P paths to distribute the graph state, don't use the ancilla nodes!

    param {nx.Graph} graphState: The input graph state.

    param {nx.Graph} network: The input network.

    param {int} maxShot: The maximum shots.

    param {controllable} controllable: The controllable nodes.

    param {ifReset} ifReset: Whether to reset the occupation of the bell pairs at each shot.

    return {Tuple[Dict[int,Tuple[List[int],float,Tuple[int,int],int]],
            Dict[Tuple[int,int],Dict[str,Any]],
            Dict[int,Dict[int,Dict[str,Any]]]]}: 
            The main paths:{shots:[(path,cost,(u,v))]}
            The edge dependent dictionary:{(u,v):(shot,pathIndex,(directNode,directNode))}
            The vertex Reaching Map:{vertex:{Node:[if fixed,shot,mainPathDict, prob]}}
    '''
    #initialize the vertex reaching map    
    VRM={}
    if 'ps' in network.graph:
        switchProb=network.graph['ps']
    else:
        switchProb=1
    for v in graphState.nodes:
        VRM[v]={}
        if isinstance(graphState.nodes[v]['node'],int):
            graphState.nodes[v]['node']=[graphState.nodes[v]['node']]

        for node in graphState.nodes[v]['node']:
            VRM[v][node]=[]
            qubitDict={'prob':1,'shot':0,'ifFixed':True,'prev':None,\
                       'mainPathDict':None}
            VRM[v][node].append(qubitDict)

    graphState_bk=graphState.copy()

    #get the list of saved channels node
    savedChannelsNode=[node for node in network.nodes \
                       if 'saveChannel' in network.nodes[node] \
                        and network.nodes[node]['saveChannel']]

    if maxShot is None:
        maxShot=len(graphState.nodes)

    #initialize edge dependent dictionary
    edgeDependent={}
    for u,v in graphState.edges:
        edgeDependent[(u,v)]={}
        edgeDependent[(v,u)]={}

    #iterate the shots
    mainPaths={}
    k=0

    def get_degree(v:int)->int:
        return -graphState_bk.degree[v]
    
    while len(graphState_bk.edges)>0 and k<maxShot:
        #find the vertex with the maximum degree
        mainPaths[k]=[]
        network_bk=network.copy()

        if k>0:
            #delete the saved channels
            network_bk.remove_nodes_from(savedChannelsNode)
            if ifReset:
                for u,v in network_bk.edges:network_bk[u][v]['occu']=0

        #create priority heap for the degrees of vertexes
        degreeHeap=[]

        nodeList=list(graphState_bk.nodes)
        for v in nodeList:
            if graphState_bk.degree[v]>0:
                degreeHeap.append(v)
        
        degreeHeap.sort(key=get_degree)

        while len(degreeHeap)>0:
            v=degreeHeap.pop(0)
            neighbors=list(graphState_bk.neighbors(v))

            #sort the neighbors by the degree
            def get_degree(x:int)->int:
                return -graphState_bk.degree[x]
            
            neighbors.sort(key=get_degree)

            for neighborIndex in range(len(neighbors)):

                source_cost={node:VRM[v][node][0]['prob'] for node in VRM[v].keys()}
                sources=list(source_cost.keys())


                u=neighbors[neighborIndex]
                target_cost={node:VRM[u][node][0]['prob'] for node in VRM[u].keys()}
                targets=list(target_cost.keys())

                def get_metric(network:nx.Graph,path:List[int])->float:
                    return ET_metric(network,path,(source_cost,target_cost))
                
                path,cost=find_path_ED(network_bk,sources,targets,single=True,metric=get_metric)
                minPath={'path':path,'cost':cost/(switchProb**2),'VU':(v,u)}
                if len(path)==0:
                    continue

                mainPaths[k].append(minPath)
                
                edgeDependent[(v,u)]={'shot':k,'pathDict':minPath,'directNode':(path[0],path[-1])}
                edgeDependent[(u,v)]={'shot':k,'pathDict':minPath,'directNode':(path[-1],path[0])}

                VRM,edgeDependent,mainPaths=update_VRM(VRM,edgeDependent,network_bk,mainPaths,k,controllable)
                remove_paths(network_bk,minPath['path'])
                graphState_bk.remove_edge(v,u)

                degreeHeap.sort(key=get_degree)

        k+=1

    return mainPaths,edgeDependent,VRM

def find_MGST_paths(graphState:nx.Graph,network:nx.Graph, controllable:List|None=None, \
                    memoryConstraint: bool=False,ifReset:bool=True, \
                    minCost: bool=False,memShot: bool|float=False)\
    ->Tuple[Dict[int,Dict[int,List[Tuple[List[int],float]]]],nx.Graph]:
    '''
    description: Find the MGST paths to distribute the graph state, don't use the ancilla nodes!

    param {nx.Graph} graphState: The input graph state.

    param {nx.Graph} network: The input network.

    param {List} controllable: The controllable nodes.

    param {bool} memoryConstraint: The memory constraint.

    param {bool} ifReset: Whether to reset the occupation of the bell pairs at each shot.

    param {bool} minCost: Whether to find the minimum cost.

    param {bool|float} memShot: Whether to punish the higher shots.

    return {Tuple[Dict[int,Dict[int,List[Tuple[List[int],float]]]],nx.Graph}:
            The main paths:{shots:{target:[(path,cost)]}}
            The graphState with source information.
    '''
    def compute_cost(paths:Dict[int,Dict[int,List[Tuple[List[int],float]]]])->float:
        newP=1
        for k in paths.keys():
            for t in paths[k]:
                newP*=1/paths[k][t][1]
                if applyInitialCost:
                    newP*=graphState.nodes[t]['node'][paths[k][t][0][-1]]
        return 1/newP
    
    targets={}

    applyInitialCost=False
    for v in graphState.nodes():
        if isinstance(graphState.nodes[v]['node'],int):
            #some node is unreachable in current shot
            if memoryConstraint:
                if 'memory' in network.nodes[graphState.nodes[v]['node']]:
                    if network.nodes[graphState.nodes[v]['node']]['memory'] is not None:
                        if network.nodes[graphState.nodes[v]['node']]['memory']\
                            -network.nodes[graphState.nodes[v]['node']]['moccu']<1:
                            continue
            targets[v]=[graphState.nodes[v]['node']]
        elif isinstance(graphState.nodes[v]['node'],list):
            #some node is unreachable in current shot
            if memoryConstraint:
                if 'memory' in network.nodes[graphState.nodes[v]['node'][0]]:
                    if network.nodes[graphState.nodes[v]['node'][0]]['memory'] is not None:
                        if network.nodes[graphState.nodes[v]['node'][0]]['memory']\
                            -network.nodes[graphState.nodes[v]['node'][0]]['moccu']\
                                <len(graphState.nodes[v]['node']):
                            continue
            targets[v]=graphState.nodes[v]['node']
        elif isinstance(graphState.nodes[v]['node'],dict):
            targets[v]=list(graphState.nodes[v]['node'].keys())
            applyInitialCost=True
    
    #initialize
    shot=len(targets)
    cost=1E5
    delivered=0
    rawPaths={0:{}}

    if 'source' in graphState.graph:
        if minCost:
            rawPaths,_=find_multi_targets_disjoint(network,graphState.graph['source'],targets,\
                                            metric=ET_metric,decompose=True,\
                                            ifReset=ifReset,minCost=minCost,memShot=memShot)
        else:
            rawPaths=find_multi_targets_disjoint(network,graphState.graph['source'],targets,\
                                            metric=ET_metric,decompose=True,\
                                            ifReset=ifReset,memShot=memShot)
        return rawPaths,graphState
    else:
        for node in network.nodes():
            #skip uncontrollable nodes
            if controllable is not None:
                if node not in controllable:
                    continue

            #skip the nodes that with not enough memory
            if memoryConstraint:
                if 'memory' in network.nodes[node]:
                    if network.nodes[node]['memory'] is not None:
                        if network.nodes[node]['memory']-network.nodes[node]['moccu']<len(graphState.nodes):
                            continue

            newShot=find_multi_targets_disjoint(network,node,targets,decompose=False,\
                                                ifReset=ifReset,minCost=minCost,memShot=memShot)
            if newShot<shot:
                shot=newShot
                if minCost:
                    rawPaths,cost=find_multi_targets_disjoint(network,node,targets,\
                            shots=newShot,metric=ET_metric,decompose=True,ifReset=ifReset,minCost=minCost,memShot=memShot)
                else:
                    rawPaths=find_multi_targets_disjoint(network,node,targets,\
                            shots=newShot,metric=ET_metric,decompose=True,ifReset=ifReset,memShot=memShot)
                    cost=compute_cost(rawPaths)
                graphState.graph['source']=node
                delivered=len(rawPaths[0].keys())
            elif newShot==shot:
                if minCost:
                    newRawPaths,newCost=find_multi_targets_disjoint(network,node,targets,\
                            shots=newShot,metric=ET_metric,decompose=True,ifReset=ifReset,minCost=minCost,memShot=memShot)
                else:
                    newRawPaths=find_multi_targets_disjoint(network,node,targets,\
                            shots=newShot,metric=ET_metric,decompose=True,ifReset=ifReset,memShot=memShot)
                    newCost=compute_cost(newRawPaths)
                newDelivered=len(newRawPaths[0].keys())
                if newDelivered>delivered:
                    rawPaths=newRawPaths
                    cost=newCost
                    delivered=newDelivered
                    graphState.graph['source']=node
                elif newDelivered==delivered:
                    if newCost<cost:
                        rawPaths=newRawPaths
                        cost=newCost
                        graphState.graph['source']=node

    return rawPaths,graphState    

