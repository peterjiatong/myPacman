"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.core.actions import Actions
from pacai.core.search import heuristic
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent

from pacai.core.directions import Directions
from pacai.core.distance import manhattan, maze
from pacai.student.search import uniformCostSearch
import copy

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """

    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

        # *** Your Code Here ***
        # indicating startingGameState
        self.startingGameState = startingGameState

    def startingState(self):
        init = [False, False, False, False]  # set initial to false for all 4 corners
        # check if there's any corner == starting position, if any, set it to True
        for i in range(len(self.corners)):
            if self.corners[i] == self.startingPosition:
                init[i] = True
        # return starting position and corners status
        return self.startingPosition, init

    def isGoal(self, state):
        # return True if all 4 corners == True
        for i in state[1]:
            if not i:
                return False
        return True

    def successorStates(self, state):
        successors = []

        for action in Directions.CARDINAL:
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # copy down the corner status
                cornerStatus = copy.deepcopy(state[1])
                # set corner to true if we are about to explore it
                for i in range(len(cornerStatus)):
                    if (nextx, nexty) == self.corners[i]:
                        cornerStatus[i] = True
                # append((nextnode, corner status), action, cost == 1) to successors
                successors.append((((nextx, nexty), cornerStatus), action, 1))

        self._numExpanded += 1  # (bug?) node expanded = 0 without this line
        return successors

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

    # *** Your Code Here ***
    corners = problem.corners
    # goals = unvisited corners
    goals = []
    manhattanDis = []
    # add all unvisited corners to goals and their manhattan distance to manhattanDis
    for i in range(len(state[1])):
        if not state[1][i]:
            goals.append(corners[i])
            manhattanDis.append(manhattan(state[0], corners[i]))

    # return default if no more goals available
    if goals is None:
        return heuristic.null(state, problem)

    shortestGoal = -1  # distance to closest corner, -1 if none
    shortestIndex = 0  # keep track of index to find corresponding corner

    # find the shortest distance between current location to goals available
    for i in range(len(manhattanDis)):
        # find the corner with max manhattan distance
        if manhattanDis[i] > shortestGoal:
            shortestGoal = manhattanDis[i]
            shortestIndex = i

    # if none corner find, return current location
    if shortestGoal == -1:
        return maze(state[0], state[0], problem.startingGameState)

    # use manhattan distance + euclidean distance as heuristic
    return maze(state[0], goals[shortestIndex], problem.startingGameState)


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """

    position, foodGrid = state

    # *** Your Code Here ***
    goals = foodGrid.asList()
    # return default if no more food available
    if goals is None:
        return heuristic.null(state, problem)

    # find manhattan distance for all available foods
    manhattanDis = []
    for i in goals:
        manhattanDis.append(manhattan(position, i))

    shortestGoal = -1  # distance to closest corner, -1 if none
    shortestIndex = 0  # keep track of index to find corresponding corner

    # find the shortest distance between current location to goals available
    for i in range(len(manhattanDis)):
        # find the corner with max manhattan distance
        if manhattanDis[i] > shortestGoal:
            shortestGoal = manhattanDis[i]
            shortestIndex = i

    # if none corner find, return current location
    if shortestGoal == -1:
        return maze(state[0], state[0], problem.startingGameState)

    # use manhattan distance as heuristic
    return maze(state[0], goals[shortestIndex], problem.startingGameState)

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                                    (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """

        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        # problem = AnyFoodSearchProblem(gameState)

        # *** Your Code Here ***
        # use uniformCostSearch to find the path
        # node expanded - dfs: 5324, bfs: 350, ucs: 323
        return uniformCostSearch(AnyFoodSearchProblem(gameState))


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start=None):
        super().__init__(gameState, goal=None, start=start)

        # Store the food for later reference.
        self.food = gameState.getFood()

    def isGoal(self, state):
        # since self.food will give a 2d matrix with T/F,
        # this function will return T/F according to the given position
        return self.food[state[0]][state[1]]


class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
