"""

Data Structs for Auto Design

author: Moji Shi
date: 2024-04-08

"""

import copy

class Graph(object):

    def __init__(self, graph_dict=None):
        """ initializes a graph object 
            If no dictionary or None is given, 
            an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self._graph_dict = graph_dict

    def edges(self, vertice):
        """ returns a list of all the edges of a vertice"""
        return self._graph_dict[vertice]
        
    def all_vertices(self):
        """ returns the vertices of a graph as a set """
        return set(self._graph_dict.keys())

    def all_edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self._graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        edge = set(edge)
        vertex1, vertex2 = tuple(edge)
        for x, y in [(vertex1, vertex2), (vertex2, vertex1)]:
            if x in self._graph_dict:
                self._graph_dict[x].add(y)
            else:
                self._graph_dict[x] = [y]

    def __generate_edges(self):
        """ A static method generating the edges of the 
            graph "graph". Edges are represented as sets 
            with one (a loop back to the vertex) or two 
            vertices 
        """
        edges = []
        for vertex in self._graph_dict:
            for neighbour in self._graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges
    
    def __iter__(self):
        self._iter_obj = iter(self._graph_dict)
        return self._iter_obj
    
    def __next__(self):
        """ allows us to iterate over the vertices """
        return next(self._iter_obj)

    def __str__(self):
        res = "vertices: "
        for k in self._graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res


class TreeNode(object):

    def __init__(self, val=None):
        self.val = val
        self.children = []

    def add_child(self, new_node):
        self.children.append(new_node)

    def get_all_children(self):
        all_children = []
        all_connectors = []

        for child in self.children:
            all_children.append(child)
            all_connectors.append([self, child])

            all_children.extend(child.get_all_children()[0])
            all_connectors.extend(child.get_all_children()[1])

        return all_children, all_connectors

    def find_child(self, func):
        if func(self.val):
            return self
        else:
            for child in self.children:
                if child.find_child(func):
                    return child
        return None

    def find_children(self, func, val):
        """ find all children that satisfy the function"""
        children = []
        if func(self.val):
            children.append(self)
        for child in self.children:
            children.extend(child.find_children(func, val))
        return children

    def __str__(self):
        return str(self.val)
    
    def __repr__(self):
        return str(self.val)
    



# if __name__ == '__main__':
    
#     root = TreeNode('root')
#     child1 = TreeNode('child1')
#     child2 = TreeNode('child2')
#     root.add_child(child1)
#     root.add_child(child2)

#     child1.add_child(TreeNode('grandchild1'))
#     child1.add_child(TreeNode('grandchild2'))
#     child2.add_child(TreeNode('grandchild3'))

#     all_children, all_connectors = root.get_all_children()

#     print(all_children)
#     print(all_connectors)






