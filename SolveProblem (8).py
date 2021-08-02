import time
import operator
from PIL import Image

class h_node():
    def __init__(self,prev=None,position=None):
        self.prev = prev
        self.position = position
        self.f = 0
        self.g = 0
        self.h = 0
    
    def __eq__(self, other):
        return self.position == other.position
        
def Read_Grid(n,mazeNum) :

    grid = [[0]*n for i in range(n)]

    with open(".\maze_"+mazeNum+".txt") as f:
        for line in f:
            num = line.split()
            grid[int(num[0])][int(num[1])] = int(num[2])  
      
    return grid

def Read_Problem(path):

    c = 0
    n = 0
    g0 = 0
    g1 = 0
    e0 = 0
    e1 = 0
    alg = 0
    mazeNum = 0

    with open(path) as f:
        for line in f:
            num = line.split()
            if c == 0:
                n = int(num[0])
            
            if c == 1:
                g0 = int(num[0])
                g1 = int(num[1])
            
            if c == 2:
                e0 = int(num[0])
                e1 = int(num[1])
            
            if c == 3:
                alg = int(num[0])
            
            if c == 4:
                mazeNum = num[0]

            c = c+1

    grid = Read_Grid(n,mazeNum)

    if grid[g0][g1] != 1:
        grid[g0][g1] = 5

    if grid[e0][e1] != 1:
        grid[e0][e1] = 10

    return n,g0,g1,e0,e1,alg,mazeNum,grid

def find_edges(v,grid,n,visited,e):

    edges = []

    x,y = v

    if visited[x][y]==1:
        return edges
    if((x-1,y) == e and visited[x-1][y] !=1) or ((x+1,y) == e and visited[x+1][y] !=1) or ((x,y+1) == e and visited[x][y+1] !=1) or ((x,y-1) == e and  visited[x][y-1] !=1):
        edges.append(e)
    if y+1 < n and grid[x][y+1] == 0 and visited[x][y+1] !=1:
        edges.append((x,y+1))
    if y-1 >= 0 and grid[x][y-1] == 0 and visited[x][y-1] !=1:
        edges.append((x,y-1))
    if x+1 < n and grid[x+1][y] == 0 and visited[x+1][y] !=1:
        edges.append((x+1,y))
    if x-1 >= 0 and grid[x-1][y] == 0 and visited[x-1][y] !=1:
        edges.append((x-1,y))

    return edges

def find_edges_h1(node,grid,n,e):

    edges = []

    x,y = node.position

    if y-1 >= 0 and grid[x][y-1] == 0:
        new_node0 = h_node(node,(x,y-1))
        edges.append(new_node0)
    if y+1 < n and grid[x][y+1] == 0:
        new_node1 = h_node(node,(x,y+1))
        edges.append(new_node1)
    if x-1 >= 0 and grid[x-1][y] == 0:
        new_node2 = h_node(node,(x-1,y))
        edges.append(new_node2)
    if x+1 < n and grid[x+1][y] == 0:
        new_node3 = h_node(node,(x+1,y))
        edges.append(new_node3)
    if((x-1,y) == e) or ((x+1,y) == e) or ((x,y+1) == e) or ((x,y-1) == e):
        new_node4 = h_node(node,e)
        edges.append(new_node4)

    return edges

def find_edges_backward(v,grid,n,visited):

    edges = []

    x,y = v

    if visited[x][y]==1:
        return edges
    if y-1 >= 0 and grid[x][y-1] == 0 and visited[x][y-1] !=1:
        edges.append((x,y-1))
    if y+1 < n and grid[x][y+1] == 0 and visited[x][y+1] !=1:
        edges.append((x,y+1))
    if x-1 >= 0 and grid[x-1][y] == 0 and visited[x-1][y] !=1:
        edges.append((x-1,y))
    if x+1 < n and grid[x+1][y] == 0 and visited[x+1][y] !=1:
        edges.append((x+1,y))

    return edges

def find_edges_forward(v,grid,n,visited):

    edges = []

    x,y = v

    if visited[x][y]==1:
        return edges
    if y+1 < n and grid[x][y+1] == 0 and visited[x][y+1] !=1:
        edges.append((x,y+1))
    if y-1 >= 0 and grid[x][y-1] == 0 and visited[x][y-1] !=1:
        edges.append((x,y-1))
    if x+1 < n and grid[x+1][y] == 0 and visited[x+1][y] !=1:
        edges.append((x+1,y))
    if x-1 >= 0 and grid[x-1][y] == 0 and visited[x-1][y] !=1:
        edges.append((x-1,y))
   
    return edges

def find_intercept(path1,path2):

    for x,y in path1:
        for j,k in path2:
            if (x,y-1) == (j,k+1):
                return True
            if (x,y+1) == (j,k-1):
                return True
            if (x-1,y) == (j+1,k):
                return True
            if (x+1,y) == (j-1,k):
                
                return True
    return False

def make_path_h1(node):

    path = []

    ptr = node

    while ptr is not None:
        path.append(ptr.position)
        ptr = ptr.prev
    
    path = path[::-1]

    return path

def calculate_Euclidean(nodes,e):
    euclidean = (((nodes.position[0] - e.position[0]) ** 2) + ((nodes.position[1] - e.position[1]) ** 2)) ** 0.5
    return euclidean

def calculate_Manhattan(nodes,e):
    mht = abs(nodes.position[0] - e.position[0]) + abs(nodes.position[1] - e.position[1])
    return mht

def calculate_Max(nodes):
    max_x_y = max(nodes.position[0],nodes.position[1])
    return max_x_y

def show_grid(g0,g1,e0,e1,mazeNum):
    im1 = Image.open("mazes\maze_"+mazeNum+".png")
    im1.show()
    return

def find_cost(path):

    total_cost=0

    x,y = path[0]

    for t1,t2 in path:
        if x < t1 or x > t1:
            total_cost = total_cost+2
        if y < t2 or y > t2:
            total_cost = total_cost+1
        x,y = t1,t2
    
    return total_cost

def h1_initial_state(node,grid,n,visited):

    edges = []

    x,y = node.position

    if y-1 >= 0 and grid[x][y-1] == 0:
        edges.append((x,y-1))
    if y+1 < n and grid[x][y+1] == 0:
        edges.append((x,y+1))
    if x-1 >= 0 and grid[x-1][y] == 0:
        edges.append((x-1,y))
    if x+1 < n and grid[x+1][y] == 0:
        edges.append((x+1,y))

    return edges  # no path found

def find_path_bfs(n,g0,g1, e0, e1, grid):

    s = (g0,g1)
    e = (e0,e1)

    queue = [(s, [])]  # start point, empty path

    initial_state = []

    visited = [[0]*n for i in range(n)]

    while len(queue) > 0:
        node, path = queue.pop(0)
        path.append(node)
        x,y=node
        
        if node == (e0,e1):
            print("The initial state of the agent is ",initial_state)
            print("\n")
            print(path)
            print("\n")
            return path

        adj_nodes = find_edges(node, grid, n, visited, e)

        visited[x][y] = 1

        if node == s:
            for item in adj_nodes:
                initial_state.append(item)

        for item in adj_nodes:
            x,y = item
            if visited[x][y] == 0:
                queue.append((item, path[:]))


    return None  # no path found

def find_path_bidr(n,g0,g1, e0, e1, grid):

    s = (g0,g1)
    e = (e0,e1)

    Ag_queue = [(s, [])]  # start point, empty path
    End_queue = [(e, [])]  # end point, empty path

    initial_state = []

    visited = [[0]*n for i in range(n)]
    visited2 = [[0]*n for i in range(n)]

    while len(Ag_queue) > 0 or len(End_queue) > 0:

        Ag_node, Ag_path = Ag_queue.pop(0)
        End_node, End_path = End_queue.pop(0)
        Ag_path.append(Ag_node)
        End_path.append(End_node)
        x,y=Ag_node
        x2,y2=End_node

        check = find_intercept(Ag_path,End_path)

        if check==True:
            print("The initial state of the agent is ",initial_state)
            print("\n")
            while len(End_path) > 0:
                Ag_path.append(End_path.pop())
            print(Ag_path)
            print("\n")
            return Ag_path

        if g0 > e1:
            adj_nodes = find_edges_forward(Ag_node, grid, n, visited)
            adj_nodes2 = find_edges_backward(End_node, grid, n, visited2)
        if e0 > g1:
            adj_nodes = find_edges_backward(Ag_node, grid, n, visited)
            adj_nodes2 = find_edges_forward(End_node, grid, n, visited2)
        
        visited[x][y] = 1
        visited2[x2][y2] = 1

        if Ag_node == s:
            for item in adj_nodes:
                initial_state.append(item)
        
        for item3 in adj_nodes:
            x,y = item3
            if visited[x][y] != 1:
                Ag_queue.append((item3, Ag_path[:]))
        
        for item2 in adj_nodes2:
            x,y = item2
            if visited2[x][y] != 1:
                End_queue.append((item2, End_path[:]))

    return None

def find_path_h0(n,g0,g1, e0, e1, grid):

    s = h_node(None,(g0,g1))
    e = h_node(None,(e0,e1))
    end = (e0,e1)

    s.g = s.h = s.f = 0
    e.g = s.h = s.f = 0

    nodes_list = []  # start point

    visited = []

    nodes_list.append(s)

    initial_state = h1_initial_state(s,grid,n,visited)

    print(initial_state)
    print("\n")

    while len(nodes_list) > 0:

        nodes_list = sorted(nodes_list,key=operator.attrgetter("f"))

        top_node = nodes_list.pop(0)

        if top_node.position == end:
                path = make_path_h1(top_node)
                print(path)
                print("\n")
                return path

        edges = find_edges_h1(top_node,grid,n,end)

        for nodes in edges:

            x,y = top_node.position
            x2,y2 = nodes.position

            if x > x2 or x2 > x:
                nodes.g = top_node.g + 2
            if y > y2 or y2 > y:
                nodes.g = top_node.g + 1
            
            nodes.h = (((nodes.position[0] - e.position[0]) ** 2) + ((nodes.position[1] - e.position[1]) ** 2)) ** 0.5

            nodes.f = nodes.g + nodes.h

            if len([visited_node for visited_node in visited if visited_node == nodes]) > 0:
                continue

            if len([i for i in nodes_list if nodes == i and nodes.f >= i.f]) > 0:
                continue
            
            nodes_list.append(nodes)

        visited.append(top_node)

    return None  # no path found

def find_path_h1(n,g0,g1, e0, e1, grid):

    s = h_node(None,(g0,g1))
    e = h_node(None,(e0,e1))
    end = (e0,e1)

    s.g = s.h = s.f = 0
    e.g = s.h = s.f = 0

    nodes_list = []  # start point

    visited = []

    nodes_list.append(s)

    initial_state = h1_initial_state(s,grid,n,visited)

    print(initial_state)
    print("\n")

    while len(nodes_list) > 0:

        nodes_list = sorted(nodes_list,key=operator.attrgetter("f"))

        top_node = nodes_list.pop(0)

        if top_node.position == end:
                path = make_path_h1(top_node)
                print(path)
                print("\n")
                return path

        edges = find_edges_h1(top_node,grid,n,end)

        for nodes in edges:

            x,y = top_node.position
            x2,y2 = nodes.position

            if x > x2 or x2 > x:
                nodes.g = top_node.g + 2
            if y > y2 or y2 > y:
                nodes.g = top_node.g + 1
            
            nodes.h = min(calculate_Euclidean(nodes,e),calculate_Manhattan(nodes,e))

            nodes.f = nodes.g + nodes.h

            if len([visited_node for visited_node in visited if visited_node == nodes]) > 0:
                continue

            if len([i for i in nodes_list if nodes == i and nodes.f >= i.f]) > 0:
                continue
            
            nodes_list.append(nodes)

        visited.append(top_node)

    return None  # no path found

def find_path_h2(n,g0,g1, e0, e1, grid):

    s = h_node(None,(g0,g1))
    e = h_node(None,(e0,e1))
    end = (e0,e1)

    s.g = s.h = s.f = 0
    e.g = s.h = s.f = 0

    nodes_list = []  # start point

    visited = []

    nodes_list.append(s)

    initial_state = h1_initial_state(s,grid,n,visited)

    print(initial_state)
    print("\n")

    while len(nodes_list) > 0:

        nodes_list = sorted(nodes_list,key=operator.attrgetter("f"))

        top_node = nodes_list.pop(0)

        if top_node.position == end:
                path = make_path_h1(top_node)
                print(path)
                print("\n")
                return path

        edges = find_edges_h1(top_node,grid,n,end)

        for nodes in edges:

            x,y = top_node.position
            x2,y2 = nodes.position

            if x > x2 or x2 > x:
                nodes.g = top_node.g + 2
            if y > y2 or y2 > y:
                nodes.g = top_node.g + 1
            
            nodes.h = (calculate_Euclidean(nodes,e) + calculate_Manhattan(nodes,e)) // 2

            nodes.f = nodes.g + nodes.h

            if len([visited_node for visited_node in visited if visited_node == nodes]) > 0:
                continue

            if len([i for i in nodes_list if nodes == i and nodes.f >= i.f]) > 0:
                continue
            
            nodes_list.append(nodes)

        visited.append(top_node)

    return None  # no path found

def Solve_Problem():

    t = time.perf_counter()

    n,g0,g1,e0,e1,alg,mazeNum,grid = Read_Problem(".\problem.txt")

    if grid[e0][e1] == 1:
        print("\n")
        print("The end goal is a 1, please enter a valid end point\n")
        if grid[g0][g1] == 1:
            print("\n")
            print("The start goal is a 1, please enter a valid start goal\n")
            return
        return

    if grid[g0][g1] == 1:
        print("\n")
        print("The start goal is a 1, please enter a valid start goal\n")
        return

    
    print(n,g0,g1,e0,e1,alg,mazeNum) 
    print("\n")

    show_grid(g0,g1,e0,e1,mazeNum) 


    if alg == 0:
        path = find_path_bfs(n, g0, g1, e0, e1, grid)

    if alg == 1:
        path = find_path_bidr(n, g0, g1, e0, e1, grid)

    if alg == 2:
        path = find_path_h0(n,g0, g1,e0,e1,grid)

    if alg == 3:
        path = find_path_h1(n,g0, g1,e0,e1,grid)
    
    if alg == 4:
        path = find_path_h2(n,g0, g1,e0,e1,grid)

    cost = find_cost(path)

    print("The cost is,",cost)
    print("\n")

    end = time.perf_counter()

    print("The time to process is,",end)
    print("\n")

Solve_Problem()
