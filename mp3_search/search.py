# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP3. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
# Note that if you want to test one of your search methods, please make sure to return a blank list
#  for the other search methods otherwise the grader will not crash.
import math
from collections import deque
import heapq


class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances = {
            (i, j): abs(i[0] - j[0]) + abs(i[1] - j[1])
            for i, j in self.cross(objectives)
        }

    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root

    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a)
        rb = self.resolve(b)
        if ra == rb:
            return False
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    source_row, source_col = maze.start
    dest_row, dest_col = maze.waypoints[0]

    parent = {}
    distance = {}
    queue = deque([])
    queue.append((source_row, source_col))
    distance[(source_row, source_col)] = 0
    parent[(source_row, source_col)] = (-1, -1)
    while len(queue) > 0:
        row, col = queue.popleft()
        if (row, col) == (dest_row, dest_col):
            break
        child = maze.neighbors(row, col)
        for (child_row, child_col) in child:
            if maze.navigable(child_row, child_col) and (child_row, child_col) not in distance:
                distance[(child_row, child_col)] = distance[(row, col)] + 1
                parent[(child_row, child_col)] = (row, col)
                queue.append((child_row, child_col))

    resulting_path = []
    # print(f"source = {source_row, source_col} and destination is = {dest_row, dest_col}")
    while (dest_row, dest_col) != (-1, -1):
        resulting_path.append((dest_row, dest_col))
        dest_row, dest_col = parent[(dest_row, dest_col)]
    resulting_path.reverse()
    # print(f"the path is {resulting_path}")

    return resulting_path


def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    source_row, source_col = maze.start
    dest_row, dest_col = maze.waypoints[0]
    visited = {}
    parent = {}
    distance = {}
    dist = abs(source_row - dest_row) + abs(source_col - dest_col)
    li = [(dist + 0, 0, source_row, source_col)]
    heapq.heapify(li)
    distance[(source_row, source_col)] = dist + 0
    parent[(source_row, source_col)] = (-1, -1)

    own_navigable = {}
    own_neighbor = {}
    while len(li) > 0:
        _, old_dist, row, col = heapq.heappop(li)
        if (row, col) in visited:
            continue
        else:
            visited[(row, col)] = True

        if (row, col) == (dest_row, dest_col):
            break
        if (row, col) not in own_neighbor:
            own_neighbor[(row, col)] = maze.neighbors(row, col)

        child = own_neighbor[(row, col)]
        for (child_row, child_col) in child:
            new_dist = old_dist + 1
            new_heur_dist = abs(dest_row - child_row) + abs(dest_col - child_col) + new_dist
            if(child_row, child_col) not in own_navigable:
                own_navigable[(child_row, child_col)] = maze.navigable(child_row, child_col)
            if own_navigable[(child_row, child_col)] and \
                    ((child_row, child_col) not in distance or distance[(child_row, child_col)] > new_heur_dist):
                distance[(child_row, child_col)] = new_heur_dist
                parent[(child_row, child_col)] = (row, col)
                heapq.heappush(li, (new_heur_dist, new_dist, child_row, child_col))

    resulting_path = []
    # print(f"source = {source_row, source_col} and destination is = {dest_row, dest_col}")
    while (dest_row, dest_col) != (-1, -1):
        resulting_path.append((dest_row, dest_col))
        dest_row, dest_col = parent[(dest_row, dest_col)]
    resulting_path.reverse()
    # print(f"the path is {resulting_path}")

    return resulting_path


def shortest_dist_amongst_nodes(row, col, dest):
    if len(dest) == 0:
        return 0
    min_value = 1e11
    for (r, c) in dest:
        d = abs(row - r) + abs(col - c)
        if min_value > d:
            min_value = d
    return min_value


def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    source_row, source_col = maze.start
    dest_list = maze.waypoints

    N = len(dest_list)
    # given a bitmask of destinations --> what's the MST cost
    bitmask_dest = {}
    dest_row = None
    dest_col = None
    # given the destination spot - (di,dj) what's the index of it
    destination_map = {}
    for i in range(N):
        destination_map[dest_list[i]] = i
    for i in range(1 << N):
        dest = []
        for j in range(N):
            if (i & (1 << j)) == (1 << j):
                dest.append(dest_list[j])
        mst_wt = MST(dest).compute_mst_weight()
        bitmask_dest[i] = mst_wt, dest
        # print(f"points {dest} --> {mst_wt}")

    resulting_path = []
    visited = {}
    distance = {}
    parent = {}
    mask = (1 << N) - 1
    dist, destinations = bitmask_dest[mask]
    connection_wt = shortest_dist_amongst_nodes(source_row, source_col, destinations)
    dist += connection_wt
    distance[(source_row, source_col, mask)] = dist + 0
    parent[(source_row, source_col, mask)] = (-1, -1, -1)
    li = [(dist + 0, 0, mask, source_row, source_col)]
    heapq.heapify(li)
    # own_navigable = {}
    # own_neighbor = {}
    while len(li) > 0:
        _, old_dist, mask, row, col = heapq.heappop(li)
        # print(f"in loop {old_dist} {mask} {row} {col}")
        if mask == 0:
            dest_row = row
            dest_col = col
            dest_mask = 0

            break
        if (row, col, mask) in visited:
            continue
        else:
            visited[(row, col, mask)] = True
        # if (row, col) not in own_neighbor:
        #     own_neighbor[(row, col)] = maze.neighbors(row, col)
        # child = own_neighbor[(row, col)]
        child = maze.neighbors(row, col)
        for (child_row, child_col) in child:
            new_dist = old_dist + 1
            # if (child_row, child_col) not in own_navigable:
            #     own_navigable[(child_row, child_col)] = maze.navigable(child_row, child_col)
            # if own_navigable[(child_row, child_col)]:
            if maze.navigable(child_row, child_col):
                new_mask = mask
                if (child_row, child_col) in destination_map:
                    bit_position = destination_map[(child_row, child_col)]
                    if ((1 << bit_position) & mask) != 0:
                        new_mask = mask - (1 << bit_position)
                    # print(f"old and the new mask is {mask} {new_mask} {bit_position}")
                dist, destinations = bitmask_dest[new_mask]
                dist += shortest_dist_amongst_nodes(child_row, child_col, destinations)
                new_heur_dist = new_dist + dist
                if ((child_row, child_col, new_mask) not in distance or distance[(child_row, child_col, new_mask)] > new_heur_dist):
                    distance[(child_row, child_col, new_mask)] = new_heur_dist
                    parent[(child_row, child_col, new_mask)] = (row, col, mask)
                    heapq.heappush(li, (new_heur_dist, new_dist, new_mask, child_row, child_col))

    while (dest_row, dest_col, dest_mask) != (-1, -1, -1):
        resulting_path.append((dest_row, dest_col))
        dest_row, dest_col, dest_mask = parent[(dest_row, dest_col, dest_mask)]
    resulting_path.reverse()

    return resulting_path


def fast(maze):
    """
    Runs suboptimal search algorithm for extra credit/part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # return astar_multiple(maze)
    return []
