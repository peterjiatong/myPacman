import random
import copy

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan, maze
from pacai.core.directions import Directions


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        # find new ghost state
        newGhostStates = successorGameState.getGhostStates()
        ghostPositions = []
        for i in newGhostStates:
            ghostPositions.append(i.getPosition())

        oldfood = currentGameState.getFood().asList()
        newPosition = successorGameState.getPacmanPosition()

        # find ghost distance and food distance(both in manhattan) at new position
        ghostDistances = []
        foodDistances = []

        for i in ghostPositions:
            ghostDistances.append(manhattan(newPosition, i))

        for i in oldfood:
            foodDistances.append(manhattan(newPosition, i))

        # find closest ghost
        closestGhostDist = min(ghostDistances)
        if closestGhostDist == 0:
            closestGhostDist = 0.001

        # get true distance to closest food to avoid progress loss
        closestFood = copy.deepcopy(newPosition)
        closestDistance = min(foodDistances, default=0)
        # find which food
        for i in range(len(oldfood)):
            if foodDistances[i] == closestDistance:
                closestFood = oldfood[i]

        # get distance
        closestFoodDist = maze(newPosition, closestFood, currentGameState)
        if closestFoodDist == 0:
            closestFoodDist = 0.001

        # return 1 - reciprocal of closest ghost loc + reciprocal of closest food
        return (1 - (1 / closestGhostDist)) + (1 / closestFoodDist)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        numAgents = state.getNumAgents()

        def miniMax(self, state, agent, depth, prev):
            # edge case, return if depth limit reached
            if state.isOver() or depth == self.getTreeDepth():
                return (self.getEvaluationFunction()(state), prev)

            # find max nodes
            def maxNode(self, state, agent, depth):
                maxi = float('-inf')
                action = 'STOP'

                # for loop to find max value
                for i in state.getLegalActions(agent):
                    suc = state.generateSuccessor(agent, i)
                    cost = miniMax(self, suc, agent + 1, depth, i)[0]
                    maxi = max(maxi, cost)
                    if maxi == cost:
                        action = i

                # return pair(max, action)
                return (maxi, action)

            # find min nodes
            def minNode(self, state, agent, depth):
                mini = float('inf')
                action = 'STOP'

                # for loop to find min value
                for i in state.getLegalActions(agent):
                    suc = state.generateSuccessor(agent, i)

                    if agent + 1 == numAgents:
                        cost = miniMax(self, suc, 0, depth + 1, i)[0]
                        mini = min(mini, cost)
                    else:
                        cost = miniMax(self, suc, agent + 1, depth, i)[0]
                        mini = min(mini, cost)

                    if mini == cost:
                        action = i

                # return pair(min, action)
                return (mini, action)

            # run maxValue if max, else run minValue
            if agent == 0:
                return maxNode(self, state, agent, depth)
            else:
                return minNode(self, state, agent, depth)

        # return action
        action = miniMax(self, state, 0, 0, 'STOP')[1]
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        numAgents = state.getNumAgents()

        def miniMax(self, state, agent, depth, prev, alpha, beta):
            # edge case, return if depth limit reached
            if state.isOver() or depth == self.getTreeDepth():
                return (self.getEvaluationFunction()(state), prev)

            # find max nodes
            def maxNode(self, state, agent, depth, alpha, beta):
                maxi = float('-inf')
                action = 'STOP'

                # for loop to find max value
                for i in state.getLegalActions(agent):
                    suc = state.generateSuccessor(agent, i)
                    cost = miniMax(self, suc, agent + 1, depth, i, alpha, beta)[0]
                    maxi = max(maxi, cost)
                    if maxi == cost:
                        action = i

                    # if alpha >= beta, we can ignore the rest
                    alpha = max(alpha, maxi)
                    if alpha >= beta:
                        break

                # return pair(max, action)
                return (maxi, action)

            # find min nodes
            def minNode(self, state, agent, depth, alpha, beta):
                mini = float('inf')
                action = 'STOP'

                # for loop to find min value
                for i in state.getLegalActions(agent):
                    suc = state.generateSuccessor(agent, i)

                    if agent + 1 == numAgents:
                        cost = miniMax(self, suc, 0, depth + 1, i, alpha, beta)[0]
                        mini = min(mini, cost)
                    else:
                        cost = miniMax(self, suc, agent + 1, depth, i, alpha, beta)[0]
                        mini = min(mini, cost)

                    if mini == cost:
                        action = i

                    # if alpha >= beta, we can ignore the rest
                    beta = min(beta, mini)
                    if beta <= alpha:
                        break

                # return pair(min, action)
                return (mini, action)

            # run maxValue if max, else run minValue
            if agent == 0:
                return maxNode(self, state, agent, depth, alpha, beta)
            else:
                return minNode(self, state, agent, depth, alpha, beta)

        # return action, set alpha = -inf, beta = inf, which is the upper and lower limit
        action = miniMax(self, state, 0, 0, 'STOP', float('-inf'), float('inf'))[1]
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        def expectimax(state, depth, agent):
            # edge case, return if depth limit reached
            if state.isLose() or state.isWin() or depth == 0:
                return self.getEvaluationFunction()(state)

            # find expecti max node
            if agent == 0:
                actions = state.getLegalActions()
                actions.remove(Directions.STOP)
                value = float('-inf')
                for i in actions:
                    x = state.generateSuccessor(0, i)
                    for j in range(state.getNumAgents() - 1):
                        y = expectimax(x, depth - 1, j + 1)
                        value = max(value, y)
                return value
            # find chance nodes
            else:
                actions = state.getLegalActions(agent)
                value = 0
                prob = 1.0 / len(actions)
                for i in actions:
                    x = state.generateSuccessor(agent, i)
                    value += prob * expectimax(x, depth, 0)
                return value

        agents = state.getNumAgents()
        actions = state.getLegalActions()
        actions.remove(Directions.STOP)
        depth = self.getTreeDepth()
        bestOP = None
        bestValue = float('-inf')

        for action in actions:
            successor = state.generateSuccessor(0, action)
            old = bestValue
            bestValue = max(bestValue, expectimax(successor, depth, agents - 1))
            if bestValue != old:
                bestOP = action
        return bestOP


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """
    newGhostStates = currentGameState.getGhostStates()
    ghostPositions = []
    for i in newGhostStates:
        ghostPositions.append(i.getPosition())

    oldfood = currentGameState.getFood().asList()
    newPosition = currentGameState.getPacmanPosition()

    # find ghost distance and food distance(both in manhattan) at new position
    ghostDistances = []
    foodDistances = []

    for i in ghostPositions:
        ghostDistances.append(manhattan(newPosition, i))

    for i in oldfood:
        foodDistances.append(manhattan(newPosition, i))

    # find closest ghost
    closestGhostDist = min(ghostDistances)
    if closestGhostDist == 0:
        closestGhostDist = 0.001

    # get true distance to closest food to avoid progress loss
    closestFood = copy.deepcopy(newPosition)
    closestDistance = min(foodDistances, default=0)
    # find which food
    for i in range(len(oldfood)):
        if foodDistances[i] == closestDistance:
            closestFood = oldfood[i]

    # get distance
    closestFoodDist = maze(newPosition, closestFood, currentGameState)
    if closestFoodDist == 0:
        closestFoodDist = 0.001

    # return 1 - reciprocal of closest ghost loc + reciprocal of closest food + score(weighted)
    return ((0.1 * (1 - (1 / closestGhostDist)))
            + (0.8 * (1 / closestFoodDist)) + (0.1 * currentGameState.getScore()))


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
