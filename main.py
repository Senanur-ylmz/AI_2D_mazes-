import matplotlib.pyplot as plt
import heapq
from collections import deque


directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]


class Node:
    def __init__(self, position, parent=None, action=None, cost=0):
        self.position = position
        self.parent = parent
        self.action = action
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


class Maze:
    def __init__(self, grid, start, goals, traps):
        self.grid = grid
        self.start = start
        self.goals = goals
        self.traps = traps

    def neighbors(self, node):
        neighbors = []
        x, y = node.position

        for direction in directions:
            dx, dy = direction
            next_position = (x + dx, y + dy)

            if self.valid_position(next_position, direction):
                if next_position in self.traps:

                    cost = node.cost + 7
                else:
                    cost = node.cost + 1

                neighbors.append(Node(next_position, node, direction, cost))

        return neighbors

    def valid_position(self, position, direction):
        x, y = position
        dx, dy = direction

        current_cell = self.grid[x][y]
        next_position = (x + dx, y + dy)

        if (
            0 <= x < len(self.grid)
            and 0 <= y < len(self.grid[0])
            and not current_cell[self.get_direction_key(dx, dy)]
        ):

            if 0 <= next_position[0] < len(self.grid) and 0 <= next_position[1] < len(self.grid[0]):
                next_cell = self.grid[next_position[0]][next_position[1]]

                if (
                    direction == (1, 0) and current_cell['N']
                    or direction == (-1, 0) and current_cell['S']
                    or direction == (0, 1) and current_cell['W']
                    or direction == (0, -1) and current_cell['E']
                ):
                    return False
            return True

        return False

    def get_direction_key(self, dx, dy):
        if dx == 1:
            return 'S'
        elif dx == -1:
            return 'N'
        elif dy == 1:
            return 'E'
        elif dy == -1:
            return 'W'
        else:
            raise ValueError(f'Invalid direction: ({dx}, {dy})')

    def goal_test(self, node):
        return node.position in self.goals

    def solve(self, strategy):
        if strategy == 'DFS':
            return self.depth_first_search()
        elif strategy == 'BFS':
            return self.breadth_first_search()
        elif strategy == 'IDS':
            return self.iterative_deepening_search()
        elif strategy == 'UCS':
            return self.uniform_cost_search()
        elif strategy == 'GBFS':
            return self.greedy_best_first_search()
        elif strategy == 'A*':
            return self.a_star_search()
        else:
            raise ValueError('Invalid search strategy: {}'.format(strategy))

    def depth_first_search(self, max_iterations=float('inf')):
        frontier = [Node(self.start)]
        explored = set()
        iterations = 0
        while frontier and iterations < max_iterations:
            node = frontier.pop()
            if self.goal_test(node):
                return node, explored
            if node.position not in explored:
                explored.add(node.position)
                frontier.extend(child for child in self.neighbors(
                    node) if child.position not in explored)
            iterations += 1
        return None, explored

    def breadth_first_search(self, max_iterations=float('inf')):
        frontier = deque([Node(self.start)])
        explored = set()
        iterations = 0
        while frontier and iterations < max_iterations:
            node = frontier.popleft()
            if self.goal_test(node):
                return node, explored
            if node.position not in explored:
                explored.add(node.position)
                frontier.extend(child for child in self.neighbors(
                    node) if child.position not in explored)
            iterations += 1
        return None, explored

    def iterative_deepening_search(self):
        for depth in range(1, len(self.grid) * len(self.grid[0])):
            result, _ = self.depth_limited_search(depth)
            if result is not None:
                return result
        return None

    def depth_limited_search(self, limit):
        frontier = [Node(self.start)]
        explored = set()
        while frontier:
            node = frontier.pop()
            if self.goal_test(node):
                return node, explored
            elif node.cost < limit:
                explored.add(node.position)
                frontier.extend(child for child in self.neighbors(
                    node) if child.position not in explored)
        return None, explored

    def uniform_cost_search(self):
        frontier = []
        heapq.heappush(frontier, Node(self.start))
        explored = set()
        while frontier:
            node = heapq.heappop(frontier)
            if self.goal_test(node):
                return node, explored
            explored.add(node.position)
            for child in self.neighbors(node):
                if child.position not in explored:
                    heapq.heappush(frontier, child)
        return None, explored

    def greedy_best_first_search(self):
        frontier = []
        heapq.heappush(frontier, (self.heuristic(
            Node(self.start)), Node(self.start)))
        explored = set()
        while frontier:
            _, node = heapq.heappop(frontier)
            if self.goal_test(node):
                return node, explored
            explored.add(node.position)
            for child in self.neighbors(node):
                if child.position not in explored:
                    heapq.heappush(frontier, (self.heuristic(child), child))
        return None, explored

    def a_star_search(self):
        frontier = []
        heapq.heappush(frontier, (self.heuristic(
            Node(self.start)), Node(self.start)))
        explored = set()
        while frontier:
            _, node = heapq.heappop(frontier)
            if self.goal_test(node):
                return node, explored
            explored.add(node.position)
            for child in self.neighbors(node):
                if child.position not in explored:
                    heapq.heappush(
                        frontier, (child.cost + self.heuristic(child), child))
        return None, explored

    def heuristic(self, node):
        return min(abs(x - node.position[0]) + abs(y - node.position[1]) for x, y in self.goals)


def simulate_agent(maze, solution):
    current_node = solution
    path = []
    while current_node is not None:
        path.append(current_node.position)
        current_node = current_node.parent
    path.reverse()

    for position in path:
        print_maze_with_agent(maze, position)
        input("Press Enter to continue...")


def print_maze_with_agent(maze, agent_position):
    for row in range(len(maze.grid)):
        for col in range(len(maze.grid[0])):
            if (row, col) == agent_position:
                print(' A ', end='')
            elif maze.grid[row][col]['N']:
                print('███', end='')
            else:
                print('   ', end='')
        print()

    print()


def solve_maze(maze, strategy):
    print(f"Using {strategy} strategy:")
    result, explored = maze.solve(strategy)

    if result is not None:
        print("Goal state found:", result.position)
        print("Cost of the solution:", result.cost)
        print("Solution path:")
        print_path(result)
        print("Nodes explored:", len(explored))
        print("Expanded nodes:")
        print_explored_nodes(explored)
    else:
        print("No solution found.")


def print_path(solution):
    path = []
    while solution is not None:
        path.append(solution.position)
        # draw_maze_with_agent(maze, solution.position)
        solution = solution.parent
    path.reverse()
    print(path)


def print_explored_nodes(explored):
    for node in explored:
        print(node)


def read_maze(filename):
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    size = tuple(map(int, lines[0].split()[1:]))
    grid = [[{'N': False, 'E': False, 'S': False, 'W': False}
             for _ in range(size[1])] for _ in range(size[0])]
    start = None
    goals = []
    traps = []

    i = 1
    while i < len(lines):
        line = lines[i]
        if line == 'Walls':
            i += 1
            while i < len(lines) and lines[i] != '"':
                row_or_col, *walls = lines[i].split()
                if not walls:
                    raise ValueError(
                        'Invalid line in Walls section: {}'.format(lines[i]))
                if row_or_col == 'row':
                    row = int(walls[0]) - 1
                    for wall in walls[1:]:
                        if isinstance(wall, str) and len(wall) > 1:
                            grid[row][int(wall[0]) - 1][wall[1]] = True
                        else:
                            grid[row][int(wall) - 1]['E'] = True
                            if int(wall) < size[1]:
                                grid[row][int(wall)]['W'] = True
                else:  # column
                    col = int(walls[0]) - 1
                    for wall in walls[1:]:
                        if isinstance(wall, str) and len(wall) > 1:
                            grid[int(wall[0]) - 1][col][wall[1]] = True
                        else:
                            grid[int(wall) - 1][col]['S'] = True
                            if int(wall) < size[0]:
                                grid[int(wall)][col]['N'] = True
                i += 1
        elif line == 'Traps':
            i += 1
            while i < len(lines) and lines[i] != '"':
                trap = tuple(map(int, lines[i].split()))
                traps.append((trap[0] - 1, trap[1] - 1))
                i += 1
        elif line == 'Start':
            i += 1
            start = tuple(map(int, lines[i].split()))
            start = (start[0] - 1, start[1] - 1)
            i += 1
        elif line == 'Goals':
            i += 1
            while i < len(lines) and lines[i] != '"':
                goal = tuple(map(int, lines[i].split()))
                goals.append((goal[0] - 1, goal[1] - 1))
                i += 1
        else:
            i += 1

    return Maze(grid, start, goals, traps)


def print_maze(maze):
    for row in maze.grid:
        print(' '.join(str(cell) for cell in row))
    print()


def draw_maze_with_agent(maze, agent_position):
    rows, cols = len(maze.grid), len(maze.grid[0])
    cell_size = 1

    for row in range(rows):
        for col in range(cols):
            x1 = col
            y1 = row
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            if maze.grid[row][col]['N']:
                plt.plot([x1, x2], [y2, y2], color='black')
            if maze.grid[row][col]['E']:
                plt.plot([x2, x2], [y1, y2], color='black')
            if maze.grid[row][col]['S']:
                plt.plot([x1, x2], [y1, y1], color='black')
            if maze.grid[row][col]['W']:
                plt.plot([x1, x1], [y1, y2], color='black')

    if maze.start is not None:
        plt.scatter(maze.start[1] + 0.5, maze.start[0] +
                    0.5, color='red', marker='o', label='Start')

    for goal in maze.goals:
        plt.scatter(goal[1] + 0.5, goal[0] + 0.5,
                    color='green', marker='x', label='Goal')

    for trap in maze.traps:
        plt.scatter(trap[1] + 0.5, trap[0] + 0.5,
                    color='gray', marker='s', label='Trap')

    if agent_position is not None:
        plt.scatter(agent_position[1] + 0.5, agent_position[0] + 0.5,
                    color='blue', marker='*', label='Agent')

    plt.title('Maze')
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    plt.show()
    plt.grid(True)


maze = read_maze('maze.txt')

print_maze(maze)
search_strategies = ['DFS', 'BFS', 'UCS', 'GBFS', 'A*']

for strategy in search_strategies:
    solve_maze(maze, strategy)
