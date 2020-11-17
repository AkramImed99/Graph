#!/usr/bin/env python
# coding: utf-8

# # TP4 - plus courts chemins dans un graphe orienté

# ## Rappel de l'algorithme
# 
# Pour rappel, l'algorithme de Dijkstra peut s'implémenter avec une structure de données disposant de 3 opérations :
# - insérer un élément 
# - extraire l'élément minimum
# - diminuer un élément
# 
# Un élément doit ici être entendu comme un couple constitué d'un sommet d'un graphe et de la distance associée (par rapport à un sommet de départ).
# 
# On peut alors avoir une expression de l'algorithme qui est :
# ```text
# Entrée : un graphe G, un sommet de départ
# Sortie : les distances les plus courtes entre le sommet de départ et chacun des sommets de G
# 
#     initialiser la structure de données SD
#     créer un tableau T qui conservera les distances
#     tant qu'il reste des sommets à traiter dans SD faire
#         nmin, dmin = extraire l'élément minimum de SD
#         enregistrer dans T la distance dmin pour le sommet nmin       
#         pour chaque voisin du sommet nmin faire
#             mettre à jour si besoin la distance associée au voisin dans la SD           
#     renvoyer T
# ```
# 
# On propose dans ce TP de tester une structures de données basée sur des tableaux ou des listes pour implémenter l'algorithme. En complément, une seconde partie du TP est à votre disposition pour tester d'autres structures de données.

# ## Préliminaires

# ### Graphes orientés en NetworkX

# La page de la [documentation de NetworkX](https://networkx.github.io/documentation/stable/reference/classes/digraph.html) sur les graphes orientés, ci-dessous un exemple.

import networkx as nx
import matplotlib.pyplot as plt
import math

##fonction pour dessiner une graphe
def dessine_graphe(g):
    nx.draw(g,with_labels=True,node_color='lightgrey',node_size=600,font_weight='bold')
    plt.show()

## exple
def exemple_de_graphe_oriente():
    g = nx.DiGraph()
    l = [(0,1,5),(0,2,7),(0,3,4),(0,4,2),(1,4,2),(3,4,3),(3,6,4),(3,5,7),(3,2,9),(4,6,7),(6,5,12),(5,2,5)]
    for a,b,w in l:
        g.add_edge(a,b,weight=w)
    return g

# La fonction NetworkX permettant de calculer les plus courts chemins avec l'algorithme de Dijkstra est [shortest_path_length](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path_length.html#networkx.algorithms.shortest_paths.generic.shortest_path_length). La tester sur ce graphe.
def test_de_la_fonction_networkx(g, start):
    return nx.shortest_path_length(g,source=start,method='dijkstra')

#exemple
#>>> g=exemple_de_graphe_oriente()
#>>> test_de_la_fonction_networkx(g,0)
#{0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2}

# ### La structure de données pour implanter l'algorithme
# 
# On propose ci-dessous la définition d'une interface pour implanter la structure de données permettant de mettre en œuvre l'algorithme de Dijkstra tel que décrit ci-dessus.
class DSD (object):
    """
    DSD = Data Structure for Dijkstra algorithm
    
    Cette structure de données implémente les trois opérations nécessaires à la mise en
    œuvre de l'algorithme de Dijkstra.
    
    
    """
    def __init__ (self, graph, start):
        """
        Initialise la structure de données avec une distance de 0 associée au sommet 
        `start` du `graph` et l'infini pour les autres.
        """
        assert(start in graph.nodes())
        self.__graph = graph
        self.__start = start
        distance =dict()
        for n in self.__graph.nodes :
            if n == self.__start :
                d=0
            else :
                d=math.inf
            distance[n]=d
            
                 
    
    
    def insert (self, node, distance):
        """
        Ajoute le sommet `node` à la strucuture de données en y associant la distance
        `distance`.
        """
        self.__graph.add_node(node,weight=distance)
        
    def extract_min (self):
        """
        Retourne un couple (sommet,distance) correspondant à la distance minimale stockée
        dans la structure de données. Par effet de bord, cet élément est retiré de la 
        structure de données.
        
        Si la structure de données est vide produit une erreur.
        """

        pass
    
    def decrease (self, node, distance):
        """
        Met à jour la distance associée au sommet `node`.
        Cette mise à jour n'est réalisée que si `distance`est inférieure à la valeur 
        associée à `node`dans la structure de donnée.
        Sinon ne fait rien.
        """
        pass
    

    def is_empty (self):
        pass


# ## Implémentation de l'algorithme

# Proposer une implantation en Python de l'algorithme de Dijkstra utilisant cette interface (bien sûr, il ne sera pas possible de tester tant qu'on n'a pas au moins une réalisation de cette interface).
def plus_court_chemin (graph, start, dsdo):
    """
    Calcule le plus court chemin dans `graph` entre le sommet de départ `start` et 
    tous les autres sommets accessibles.
    
    Parametres
    ----------
    graph: networkX.DiGraph ou networkx.Graph
        le graphe
    start: int
        le sommet de départ
    dsdo: Class
        le nom d'une classe qui hérite de DSD
        
    Retourne
    --------
    dict
        un dictionnaire des distances entre `start` et les autres sommets
    """
    
    struct=dsdo(graph,start)
    distance={}
    
    while(struct.is_empty()==False):
        #node qui a la distance est min ## node , distance 
        nmin,dmin= struct.extract_min()
        distance[nmin]=dmin
        try:
            l=list(graph.predecessors(nmin))
            t=list(graph.successors(nmin))
            for i in l:
                t.append(i)
        except AttributeError :
            t=graph.neighbors(nmin)
        for voisin in t:
            #mettre a jour distance vois
            try:
                poids=graph[voisin][nmin]['weight']
            except KeyError :
                poids=graph[nmin][voisin]['weight']
            
            struct.decrease(voisin,distance[nmin]+poids)
          
    return distance


    
    
    
# ### Implémentation avec un tableau
# 
# On propose de débuter par une implémentation de la structure de données qui utilise un tableau (ou une liste en Python), comme décrit dans le polycopié.
# 
# Implanter une classe `DSDArray`.
class DSDArray (DSD):
    
    def __init__(self,graph,start):
        super().__init__(graph,start)
        distance=[]
        for i in list(graph.nodes()):
            distance.append((i,1000))
        distance[start]=(start,0)
        self.__distance=distance
        
    def insert(self, node, distance):
        self.__distance[node]=(node,distance)
        
    def extract_min(self):
        distance=self.__distance
        if(self.is_empty()):
            raise MyError('la liste distance est vide ')
            
        dmin=1000
        sommet=None
        for candidat in distance:
            if candidat[1]<dmin:
                dmin=candidat[1]
                sommet=candidat[0]
                
        if (sommet,dmin) in distance:
            distance.remove((sommet,dmin))
        return (sommet,dmin)
        
    def decrease(self, node, distance):
        distance_liste=self.__distance
        if(self.is_empty()):
            pass
        else:
            #trouver l'indice de node dans distance_liste
            indice_node=0
            for indice in range(len(distance_liste)):
                if distance_liste[indice][0]==node:
                    indice_node=indice
            if(distance_liste[indice_node][1]==1000):
                distance_liste[indice_node]=(node,distance)
            else:
                if(distance_liste[indice_node][1]>distance):
                    distance_liste[indice_node]=(node,distance)
            
            
            
    def is_empty(self):
        return self.__distance==[]
    
    def __str__(self):
        return "Implementation de l'interface avec un tableau"

#Definition de la classe MyError pour declencher une erreur si notre liste est vide
class MyError(Exception):
    def __init__(self,value):
        self.value=value
    def __str__(self):
        return repr(self.value)
    
def test_plus_court_chemin_DSDArray(g, start):
    return plus_court_chemin (g, start, DSDArray)


# ### Observation du temps de calcul

# #### Génération de graphes connexes aléatoires
# 
# La génération de graphes orientés aléatoires fortement connexes (pour s'assurer qu'on trouvrea un chemin entre tout paire de sommets) n'est pas très facile. On propose donc ici de revnir à des graphes non orientés et de s'appuyer sur la proposition ci-dessous pour avoir un générateur de graphes aléatoires dont la densité en nombre d'arêtes peut être ajustée.
# 
# https://stackoverflow.com/questions/61958360/how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx

# In[7]:


from itertools import combinations, groupby
import networkx as nx
import random

def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is connected
    """
    assert p > 0 and p <= 1
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = random.randint(1,100)
    return G


# Et voici des exemples d'utilisation.
def exemple_de_generation_de_graphe_aleatoire1():
    G = gnp_random_connected_graph(10,0.01)
    return G

def exemple_de_generation_de_graphe_aleatoire2():
    G = gnp_random_connected_graph(10,0.1)
    return G

def exemple_de_generation_de_graphe_aleatoire3():
    G = gnp_random_connected_graph(10,0.5)
    return G



# #### Mesure du temps de calcul
# 
# On propose d'utiliser [timeit](https://docs.python.org/fr/3.8/library/timeit.html#timeit-examples) pour mesurer le temps d'exécution. Cela peut-être réalisé dans le Notebook comme montré en exemple ci-dessous mais aussi via la ligne de commande.

import timeit

def test_Dijkstra_networkX(order,p):
    G = gnp_random_connected_graph(order,p) 
    nx.shortest_path_length(G, source=0,weight='weight')




# Créer une fonction test similaire à celle ci-dessus pour votre implémentation basée sur la structure de données avec tableaux.

def test(order,p):
    G = gnp_random_connected_graph(order,p)
    plus_court_chemin (G, 0, DSDArray)
    




# #### Observation de l'évolution du temps de calcul pour une densité d'arêtes grandissante pour un ordre fixé.
# 
# Modififier les fonctions `test_Dijkstra_networkX` et `test` pour qu'elles prennent en second paramètre la probabilité d'arête. 
# 
# Puis effectuer des tests et observer la différence de temps de calcul entre des graphes peu denses en nombre d'arêtes et d'autres plus denses.
#exemple 1
def temps_execution_Dijkstra_networkx1():
    timeit.Timer('test_Dijkstra_networkX(200,0.1)', "from __main__ import test_Dijkstra_networkX")
    return timeit.timeit(number=100)


def temps_execution_test1():
    timeit.Timer('test(200,0.1)', "from __main__ import test")
    return timeit.timeit(number=100)

#exemple2
def temps_execution_Dijkstra_networkx1():
    timeit.Timer('test_Dijkstra_networkX(200,0.5)', "from __main__ import test_Dijkstra_networkX")
    return timeit.timeit(number=100)


def temps_execution_test1():
    timeit.Timer('test(200,0.5)', "from __main__ import test")
    return timeit.timeit(number=100)

#exemple3
def temps_execution_Dijkstra_networkx1():
    timeit.Timer('test_Dijkstra_networkX(200,1)', "from __main__ import test_Dijkstra_networkX")
    return timeit.timeit(number=100)


def temps_execution_test1():
    timeit.Timer('test(200,1)', "from __main__ import test")
    return timeit.timeit(number=100)
#premierement avec la fonction qui implemente notre algorithme avec un tableau ,le temps d'execution est
    #superieure a celui donner comme exemple
#aussi ,plus les arêtes du graphes sont denses plus le temps d'execution est plus eleve

# ## Implémentation NetworkX
# 
# Quelle structure de données est utilisée par NetworkX ? (on pourra regarder le code de la fonction [shortest_path_length](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path_length.html#networkx.algorithms.shortest_paths.generic.shortest_path_length).

if __name__ == "__main__":
    # programmation de l'algorithme
    g = exemple_de_graphe_oriente()
    #dessine_graphe(g)
    res = test_de_la_fonction_networkx(g,0)
    print(res)
    res = test_plus_court_chemin_DSDArray(g,0)
    print(res)

    # comparaison des temps
    g1 = exemple_de_generation_de_graphe_aleatoire1()
    dessine_graphe(g1)

    temps1 = temps_execution_Dijkstra_networkx1()
    print(temps1)
    
    temps1bis=temps_execution_test1()
    print(temps1bis)

    temps2 = temps_execution_Dijkstra_networkx1()
    print(temps2)
    
    temps2bis=temps_execution_test1()
    print(temps2bis)
    
    temps3 = temps_execution_Dijkstra_networkx1()
    print(temps3)
    
    temps3bis=temps_execution_test1()
    print(temps3bis)

