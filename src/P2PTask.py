'''
Name: P2PTask.py
Description: This is a file for generating the P2P task for distribution. 
Including generating the network topology as well as the graph state to be distributed.
Email: yesunhuang@uchicago.edu
OpenSource: https://github.com/yesunhuang
LastEditors: yesunhuang yesunhuang@uchicago.edu
Msg: 
Author: YesunHuang
Date: 2024-11-20 21:44:40
'''

import networkx as nx
import numpy as np
from typing import*


class P2PTask:
    
    def __init__(self,n:int,pc:float|Callable[[float],float], \
                 channels:int, ps:float=1.0, \
                 memory:int|None=None,\
                 host:int=2, seed:int|None=None, \
                 ifFullControllable:bool=False,topology:str="Waxman",**kargs)->None:         
        '''
        description: generate the P2P network.

        param {*} self:

        param {int} n: the number of nodes in the network.

        param {float or Callable} pc: the probability of connection between two nodes. It can be either a function of distance or a constant.

        param {int} channels: the average number of channels(width) for each edge.

        param {float} ps: the probability of CZ gate.

        param {int} memory: the average long-term memory of each node.

        param {int} seed: the random seed for generating the network.

        param {int} host: the number of maximum hosts can be communicated without using long term memory.

        param {ifFullControllable} ifFullControllable: whether all the nodes in the network can perform CZ gates.

        param {str} topology: the topology of the network, including Waxman, Barabasi-Albert, etc.

        param {object} kargs: other parameters for generating the network.

        return {*}
        '''

        self.n = n
        self.pc = pc
        self.channels = channels
        self.ps=ps
        self.network_task=[]
        self.memory=memory
        self.host=host
        self.ifFullControllable=ifFullControllable

        if seed is not None:
            self.seed=seed
        else:
            self.seed = np.random.randint(0,10000)

        if "Tracking" in kargs:
            self.tracking=kargs["Tracking"]
        else:
            self.tracking=None

        if "Params" in kargs:
            self.params=kargs["Params"]
        else:
            self.params=None
            print("The parameters for generating the network is not provided.")

        match topology:
            case "Waxman":
                self.network_graph=self.generate_waxman_topology(self.n,self.params,self.seed)
            case "Barabasi-Albert":
                self.network_graph=self.generate_barabasi_albert_topology(self.n,self.params,self.seed)
            case "Radius-Complete":
                self.network_graph=self.generate_radius_complete_topology(self.n,self.params,self.seed)
            case "Cell":
                self.network_graph=self.generate_cell_topology(self.n,self.seed)
            case _:
                raise ValueError("The topology is not supported.")

        self.network_graph=self.weighted_network(self.pc,self.channels)
        self.network_backbone_nodes=self.network_graph.nodes()
        self.network_graph.graph['ps']=self.ps
    
    def generate_cell_topology(self,n:int,seed:int|None=None)->nx.Graph:
        '''
        description: generate the network topology based on cell network

        param {*} self:

        param {int} n: number of nodes

        param {int} seed: the random seed for generating the network.

        return {nx.Graph}: the network topology
        '''
        numberOfCells=int((n-1)//7)
        basicCell=nx.cycle_graph(4)
        nodeList=list(basicCell.nodes())
        for node in nodeList:
            basicCell.add_node(node+4)
            basicCell.add_edge(node,node+4)
        np.random.seed(seed)
        #merge multiple cells into a network with their leaves connected
        for i in range(0,numberOfCells):
            if i==0:
                g=basicCell.copy()
            else:
                selectionList=[node for node in g.nodes() if nx.degree(g,node)==1]
                nodeIndex=int(np.random.choice(selectionList))
                basicCellCopy=basicCell.copy()
                #remove the node with largest index from basicCellCopy
                basicCellCopy.remove_node(7)
                g=nx.disjoint_union(g,basicCellCopy)
                connectedNodeOnCell=len(g.nodes())-5
                g.add_edge(nodeIndex,connectedNodeOnCell)
        self.avgDegree=self.calAvgDegree(g)
        return g

    def generate_radius_complete_topology(self,n:int,params:Tuple[int,float],seed:int|None=None)->nx.Graph:
        '''
        description: generate the network topology based on radius complete graph

        param {*} self:

        param {int} n: number of nodes

        param {float} params: the radius of the complete graph

        param {int} seed: the random seed for generating the network.

        return {nx.Graph}: the network topology
        '''
        dim,rad=params
        g=nx.random_geometric_graph(n,rad,dim=dim,seed=seed)
        self.avgDegree=self.calAvgDegree(g)
        return g
    
    def generate_barabasi_albert_topology(self,n:int,params:int,seed:int|None=None)->nx.Graph:
        '''
        description: generate the network topology based on barabasi albert model

        param {*} self:

        param {int} n: number of nodes

        param {int} params: the number of edges to attach from a new node to existing nodes

        param {int} seed: the random seed for generating the network.

        return {nx.Graph}: the network topology
        '''
        g=nx.barabasi_albert_graph(n,params,seed=seed)
        self.avgDegree=self.calAvgDegree(g)
        return g
    
    def calAvgDegree(self,g:nx.Graph|None=None)->float:
        '''
        description: calculate the average degree of the network

        param {*} self:

        param {nx} g: given network

        return {float}: the average degree of the network
        '''
        if g is None:
            g=self.network_graph
        return (np.array(nx.degree(g))[:,1]).mean()
  
    def generate_waxman_topology(self,n:int,params:Tuple[float,float],seed:int|None=None)->nx.Graph:
        '''
        description: generate the network topology based on waxman model

        param {*} self:

        param {int} n: the number of nodes in the network.

        param {Tuple[float,float]} params: params for waxman network

        param {int} seed: the random seed for generating the network.

        return {nx.Graph}: the network topology
        '''

        alpha,beta=params
        g=nx.waxman_graph(n,alpha,beta,seed=seed)
        self.avgDegree=self.calAvgDegree(g)
        return g

    def weighted_network(self, pc:float|Callable[[float],float], channels:int, seed:int|None=None)->nx.Graph:
        '''
        description: return the weighted network based on the network topology 

        param {*} self:

        param {float or Callable} pc: the probability of connection between two nodes. It can be either a function of distance or a constant.

        param {int} channels:

        return {*}
        '''
        #generate the weighted network based on the network topology
        if seed is None:
            seed=self.seed
        np.random.seed(seed)

        #width of the channels
        width={(u,v):np.random.poisson(channels)+1 for u,v in self.network_graph.edges()}
        occu={(u,v):0 for u,v in self.network_graph.edges()}
        nx.set_edge_attributes(self.network_graph,width,"width")
        nx.set_edge_attributes(self.network_graph,occu,"occu")

        #memory of the nodes
        if self.memory is not None:
            memory={v:np.random.poisson(self.memory) for v in self.network_graph.nodes()}
            occu={v:0 for v in self.network_graph.nodes()}
            nx.set_node_attributes(self.network_graph,memory,"memory")
            nx.set_node_attributes(self.network_graph,occu,"moccu")
        else:
            memory={v:None for v in self.network_graph.nodes()}
            occu={v:0 for v in self.network_graph.nodes()}
            nx.set_node_attributes(self.network_graph,memory,"memory")
            nx.set_node_attributes(self.network_graph,occu,"moccu")
            
        if "pos" in self.network_graph.nodes[0]:
            distance={(u,v):np.linalg.norm(np.array(self.network_graph.nodes[u]["pos"])\
                                           -np.array(self.network_graph.nodes[v]["pos"])) for u,v in self.network_graph.edges()}
        else:
            distance={(u,v):1 for u,v in self.network_graph.edges()}
        nx.set_edge_attributes(self.network_graph,distance,"distance")
        if callable(pc):
            prob={(u,v):pc(self.network_graph[u][v]["distance"]) for u,v in self.network_graph.edges()}
        else:
            prob={(u,v):pc for u,v in self.network_graph.edges()}
        nx.set_edge_attributes(self.network_graph,prob,"prob")

        return self.network_graph
    
    def generate_distribution_task(self, ns:List[int]=[10], graph_states:List[nx.Graph]|None=None,\
                                    topology:str="erdos",seed:List[int]|None=None,\
                                    memoryConstraint: str='default', **kargs)->List[nx.Graph]:
        '''
        description: generate the distribution task based on the graph states

        param {*} self:

        param {List[int]} ns: the number of vertexes for each graph.

        param {List[nx.graph]} graph_states: the graph states to be distributed

        param {str} topology: the topology of the network, including erdos, regular, etc.

        param {List[int]} seed: the random seed for generating the graph states.

        param {str} memoryConstraint: the strategy for assigning the nodes in the network fulfilling the memory constraint.

        param {object} kargs: other parameters required by generator for generating the graph state.

        return {List[nx.graph]}: the distribution task
        '''
        def set_nodes(g: nx.Graph,network: nx.Graph, memoryConstraint: str, seed: int)->nx.Graph:
            '''
            description: set the assigned nodes for the graph in the network

            param {*} g: the graph state

            param {*} network: the network

            param {str} memoryConstraint: the memory constraint for the network

            param {int} seed: the random seed for generating the network

            return {*}
            '''
            np.random.seed(seed)
            match memoryConstraint:
                case "default":
                    #The original version
                    for node in network.nodes():
                        if network.nodes[node]['memory'] is not None:
                            raise ValueError("The memory constraint is not supported by the network.")
                    assignment={v:np.random.randint(0,len(self.network_backbone_nodes)) for v in g.nodes()}
                    nx.set_node_attributes(g,assignment,"node")

                case "memory":
                    #considering the memory constraint
                    network_bk=network.copy()
                    for node in network_bk.nodes():
                        network_bk.nodes[node]["moccu"]=0
                    for v in g.nodes():
                        availableNodes=[]
                        for u in self.network_backbone_nodes:
                            if network_bk.nodes[u]['memory'] is None:
                                availableNodes.append(u)
                            elif network_bk.nodes[u]["memory"]-network_bk.nodes[u]["moccu"]>0:
                                availableNodes.append(u)
                        if len(availableNodes)==0:
                            raise ValueError("The memory constraint is not satisfiable.")
                        node=int(np.random.choice(np.asarray(availableNodes)))
                        g.nodes[v]["node"]=node
                        if network_bk.nodes[node]["memory"] is not None:
                            network_bk.nodes[node]["moccu"]+=1

                case "flexible":
                    #increase the memory for the assigned nodes
                    network_bk=network.copy()
                    for node in network_bk.nodes():
                        network_bk.nodes[node]["moccu"]=0
                    assignment={v:int(np.random.choice(np.asarray(self.network_backbone_nodes))) for v in g.nodes()}
                    nx.set_node_attributes(g,assignment,"node")
                    for v in assignment:
                        network_bk.nodes[assignment[v]]["moccu"]+=1
                        if network_bk.nodes[assignment[v]]["memory"] is not None:
                            if network_bk.nodes[assignment[v]]["memory"]<network_bk.nodes[assignment[v]]["moccu"]:
                                network_bk.nodes[assignment[v]]["memory"]=network_bk.nodes[assignment[v]]["moccu"]
                                network.nodes[assignment[v]]["memory"]=network_bk.nodes[assignment[v]]["memory"]
                case _:
                    raise ValueError("The memory constraint is not supported.")
            return g
        
        if "Params" in kargs:
            params=kargs["Params"]
        else:
            params=None
        if seed is None:
            seed=[int(np.random.randint(0,10000))]*len(ns)
        if graph_states is None:
            graph_states=[]
            i=0
            for n in ns:
                match topology:
                    case "erdos":
                        if params is None:
                            raise ValueError("The parameters for generating the graph state is not provided.")
                        g= nx.erdos_renyi_graph(n,params[i],int(seed[i]))
                    case "regular":
                        if params is None:
                            raise ValueError("The parameters for generating the graph state is not provided.")
                        g= nx.random_regular_graph(params[i],n,int(seed[i]))
                    case "star":
                        g= nx.star_graph(n-1)
                    case "wheel":
                        g= nx.wheel_graph(n-1)
                    case "grid":
                        g= nx.grid_2d_graph(int(np.sqrt(n)),int(np.sqrt(n)))
                        g=nx.convert_node_labels_to_integers(g)
                    case "powerlaw_tree":
                        if params is None:
                            raise ValueError("The parameters for generating the graph state is not provided.")
                        g= nx.random_powerlaw_tree(n,gamma=params[i],seed=int(seed[i]))
                    case "full_rary_tree":
                        if params is None:
                            raise ValueError("The parameters for generating the graph state is not provided.")
                        g= nx.full_rary_tree(params[i],n)
                    case "prufer_tree":
                        if params is None:
                            raise ValueError("The parameters for generating the graph state is not provided.")
                        g=nx.random_labeled_tree(n,seed=int(seed[i]))
                    case _:
                        raise ValueError("The topology is not supported.")
                #remove degree 0 nodes
                junkNodes=list(nx.isolates(g))
                g.remove_nodes_from(junkNodes)
                #assign the graph state to the network
                g=set_nodes(g,self.network_graph,memoryConstraint,seed[i])
                graph_states.append(g)
                i+=1
        else:
            i=0
            for g in graph_states:
                if "node" not in g.nodes[0]:
                    g=set_nodes(g,self.network_graph,memoryConstraint,seed[i])
                i+=1
        self.network_task=self.network_task+graph_states
        return graph_states
    
    def set_controllable_nodes(self,controllableNodes:List[int])->None:
        '''
        description: set the controllable nodes in the network

        param {*} self:

        param {List[int]} controllableNodes: the controllable nodes in the network

        return {*}
        '''
        if self.ifFullControllable:
            print("The network is fully controllable.")
        else:
            for v in self.network_graph.nodes():
                if v in controllableNodes:
                    self.network_graph.nodes[v]["controllable"]=True
                else:
                    self.network_graph.nodes[v]["controllable"]=False

    def get_average_channel_width(self)->float:
        '''
        description: get the average channel width in the network

        param {*} self:

        return {float}: the average channel width
        '''
        return (np.array(list(nx.get_edge_attributes(self.network_graph,"width").values()))).mean()
    
    def get_average_channel_prob(self)->float:
        '''
        description: get the average channel probability in the network

        param {*} self:

        return {float}: the average channel probability
        '''
        return (np.array(list(nx.get_edge_attributes(self.network_graph,"prob").values()))).mean()
    
    def get_average_node_memory(self)->float:
        '''
        description: get the average node memory in the network

        param {*} self:

        return {float}: the average node memory
        '''
        totalMemory=0
        for v in self.network_backbone_nodes:
            if self.network_graph.nodes[v]['memory'] is None:
                return None
            totalMemory+=self.network_graph.nodes[v]['memory']
        return totalMemory/len(self.network_backbone_nodes)
    
    def get_maximum_node_memory(self)->int|None:
        '''
        description: get the maximum node memory in the network

        param {*} self:

        return {int}: the maximum node memory
        '''
        maxMemory=0
        for v in self.network_backbone_nodes:
            if self.network_graph.nodes[v]['memory'] is None:
                return None
            if self.network_graph.nodes[v]['memory']>maxMemory:
                maxMemory=self.network_graph.nodes[v]['memory']
        return maxMemory

        