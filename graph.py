from collections import defaultdict, deque
from heapq import heappush, heappop

class Graph(object):

    def __init__(self, directed=False):
        self.edges = defaultdict(dict)
        self.nodes = set()
        self.directed = directed

    def add_edge(self, node1, node2, weight = 1):
        self.edges[node1][node2] = weight
        if not self.directed:
            self.edges[node2][node1] = weight
        self.nodes.add(node1)
        self.nodes.add(node2)

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self.edges))

    def __repr__(self):
        return str(self)


    def cyclicUtilDirected(self, node, color):
        color[node] = "gray"
        adjacent_nodes = self.edges[node]

        for adjacent_node in adjacent_nodes:
            if color[adjacent_node] == "gray":
                return True

            if color[adjacent_node] == "white":
                if self.cyclicUtilDirected(adjacent_node, color):
                    return True

        color[node] = "black"
        return False

    def is_cyclicDirected(self):
        color = dict()
        for node in self.nodes:
            color[node] = "white"

        for node in self.nodes:
            if color[node] == "white" and self.cyclicUtilDirected(node, color):
                return True

        return False

    def cyclicUtilUnDirected(self, node, visited, parent=None):
        visited.add(node)

        adjacent_nodes = self.edges[node]
        for adjacent_node in adjacent_nodes:
            if adjacent_node not in visited:
                if self.cyclicUtilUnDirected(adjacent_node, visited, node):
                    return True
            elif parent != None and adjacent_node != parent:
                return True

        return False

    def is_cyclicUndirected(self):
        visited = set()

        for node in self.nodes:
            if node not in visited and self.cyclicUtilUnDirected(node, visited):
                return True

        return False

    def isCyclic(self):
        if self.directed:
            return self.is_cyclicDirected()
        else:
            return self.is_cyclicUndirected()


    def dfsUtil(self, node, visited, result):
        visited.add(node)
        result.append(node)

        adjacent_nodes = self.edges[node]
        for adjacent_node in adjacent_nodes:
            if adjacent_node not in visited:
                self.dfsUtil(adjacent_node, visited, result)

    def dfs(self):
        visited = set()
        result = list()
        for node in self.nodes:
            if node not in visited:
                self.dfsUtil(node, visited, result)

        return result


    def bfsUtil(self, node, visited, queue, result):
        visited.add(node)
        queue.append(node)

        while queue:

            current = queue.popleft()
            result.append(current)
            adjacent_nodes = self.edges[current]
            for adjacent_node in adjacent_nodes:
                if adjacent_node not in visited:
                    visited.add(adjacent_node)
                    queue.append(adjacent_node)

    def bfs(self):
        visited = set()
        queue = deque()

        result = list()
        for node in self.nodes:
            if node not in visited:
                self.bfsUtil(node, visited, queue, result)

        return result

    def topologicalSortUtil(self, node, visited, stack):
        visited.add(node)
        adjacent_nodes = self.edges[node]

        for adjacent_node in adjacent_nodes:
            if adjacent_node not in visited:
                self.topologicalSortUtil(adjacent_node, visited, stack)

        stack.append(node)

    def topologicalSort(self):
        stack = list()
        visited = set()

        for node in self.nodes:
            if node not in visited:
                self.topologicalSortUtil(node, visited, stack)

        return stack[::-1]

    def dijkstra(self, start, end):
        visited = set()
        path = list()
        min_heap = [(0, start, path)]

        while min_heap:
            cost, node, path = heappop(min_heap)
            if node not in visited:
                visited.add(node)
                path = list(path)
                path.append(node)

                if node == end:
                    return cost, path

                for adjacent_node, weight in self.edges[node].items():
                    if adjacent_node not in visited:
                        heappush(min_heap, (cost+weight, adjacent_node, path))

        return float("inf")


    def prim(self):
        result = []
        node = list(self.nodes)[0]
        visited = set()
        visited.add(node)
        min_heap = []
        for adjacent_node, weight in self.edges[node].items():
                heappush(min_heap, (weight, node, adjacent_node))

        while min_heap:
            weight, node1, node2 = heappop(min_heap)
            if node2 not in visited:
                visited.add(node2)
                result.append((node1, node2, weight))

                for adjacent_node, weight in self.edges[node2].items():
                    if adjacent_node not in visited:
                        heappush(min_heap, (weight, node2, adjacent_node))
        return result



    def coloring(self):
        result = {}
        nodes = list(self.nodes)
        result[nodes[0]] = 0
        available = [False]*len(nodes)

        for node in nodes[1:]:

            for adjacent_node in self.edges[node]:
                if adjacent_node in result:
                    available[result[adjacent_node]] = True

            for color in xrange(len(nodes)):
                if not available[color]:
                    break

            result[node] = color

            for adjacent_node in self.edges[node]:
                if adjacent_node in result:
                    available[result[adjacent_node]] = False

        return result




#############################################################################
### Add Edges and Print

edges = [('A', 'B'), ('B', 'C'), ('B', 'D'),
                   ('C', 'D'), ('E', 'F'), ('F', 'C')]

graph = Graph(True)
for node1, node2 in edges:
    graph.add_edge(node1, node2)

print graph


#############################################################################




#############################################################################
### Toplogical Sort - Build System

class Order(object):

    def __init__(self, name):
        self.orderName = name

    def __str__(self):
        return '{}'.format(self.orderName)

    def __repr__(self):
        return str(self)

a = Order("a");
b = Order("b");
c = Order("c");
d = Order("d");
e = Order("e");
f = Order("f");

packageDependecies = set()
#a depends on b
packageDependecies.add((b,a));
packageDependecies.add((c,b));
packageDependecies.add((d,c));
packageDependecies.add((f,c));
packageDependecies.add((e,d));


graph = Graph(True)
for node1, node2 in packageDependecies:
    graph.add_edge(node1, node2)

print graph.topologicalSort()

#############################################################################


#############################################################################
### Cycle Detection Directed Graph

edges = [('A', 'B'), ('B', 'C'), ('B', 'D'),
                   ('C', 'D'), ('E', 'F'), ('C', 'A')]

graph = Graph(True)
for node1, node2 in edges:
    graph.add_edge(node1, node2)

print graph.isCyclic()


#############################################################################



#############################################################################
### Cycle Detection UnDirected Graph

edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('G', 'A')]

graph = Graph()
for node1, node2 in edges:
    graph.add_edge(node1, node2)

print graph.isCyclic()


#############################################################################



#############################################################################
### DFS of Graph

edges = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A'), ('C', 'D'), ('D', 'D')]

graph = Graph(True)
for node1, node2 in edges:
    graph.add_edge(node1, node2)

print graph.dfs()


#############################################################################


#############################################################################
### BFS of Graph

edges = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A'), ('C', 'D'), ('D', 'D')]

graph = Graph(True)
for node1, node2 in edges:
    graph.add_edge(node1, node2)

print graph.bfs()


#############################################################################


#############################################################################
### Dijsktra of Graph

edges = [('A', 'B', 1), ('A', 'C', 2), ('B', 'C', 8), ('C', 'A', 3), ('C', 'D',4), ('D', 'D',0)]

graph = Graph(True)
for node1, node2, weight in edges:
    graph.add_edge(node1, node2, weight)

print graph.dijkstra('A','D')


#############################################################################




#############################################################################
### Prim's Minimum Spanning Tree of Graph

edges = [('A', 'B', 10), ('A', 'C', 2), ('B', 'C', 4), ('C', 'B', 3), ('C', 'D',4), ('D', 'D',0)]

graph = Graph(True)
for node1, node2, weight in edges:
    graph.add_edge(node1, node2, weight)

print graph.prim()


#############################################################################


#############################################################################
### Graph Coloring NP-complete

edges = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D')]

graph = Graph()
for node1, node2 in edges:
    graph.add_edge(node1, node2)

print graph.coloring()


#############################################################################
