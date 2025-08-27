import random
from pacai.agents.capture.capture import CaptureAgent

def createTeam(firstIndex, secondIndex, isRed,
               first='pacai.agents.capture.capture.CaptureAgent',
               second='pacai.agents.capture.capture.CaptureAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = aggroAgent
    secondAgent = aggroAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

# A attack agent based on reflex agent
class aggroAgent(CaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    # Choose a action with highest score
    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        scores = [self.evaluate(gameState, a) for a in actions]
        maxScore = max(scores)
        # Find all actions with max score
        bestActions = [action for action, score in zip(actions, scores) if score == maxScore]
        # return a random one if there's multiple
        return random.choice(bestActions)

    def evaluate(self, gameState, action):
        # get successor first
        successor = gameState.generateSuccessor(self.index, action)

        # make feature as a dict
        features = dict()
        features['successorScore'] = self.getScore(successor)
        myPosition = successor.getAgentState(self.index).getPosition()

        foodList = self.getFood(successor).asList()
        # find all foods and get the closest one
        minFoodDis = min([self.getMazeDistance(myPosition, food) for food in foodList])
        features['foodDis'] = minFoodDis

        capsuleList = self.getCapsules(successor)
        # find the closest capsule if there's any exists
        if (len(capsuleList) > 0):
            minCapsuleDis = min([self.getMazeDistance(myPosition, capsule) for capsule
                                 in capsuleList])
            features['capsuleDis'] = minCapsuleDis

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        # if enemy is not pacman, then it must be a ghost
        ghosts = [enemy for enemy in enemies if not enemy.isPacman() and enemy.getPosition()
                  is not None]
        # find all ghosts distance
        ghostsDis = [self.getMazeDistance(myPosition, ghost.getPosition()) for ghost in ghosts]
        # find the distance to the closest ghost
        minGhostDis = min(ghostsDis)
        features['ghostDis'] = minGhostDis

        # make weights as a dict as well
        weights = {
            'successorScore': 100,
            'foodDis': -1,
            'capsuleDis': -0.2,
            'ghostDis': 0.75,
        }

        score = 0
        # evaluate use elements inside weights and feature
        for key in features:
            score += features[key] * weights[key]
        return score


# this agent is similar to the aggroagent, instead there are some defense feature invented
class defenseAgent(CaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        scores = [self.evaluate(gameState, a) for a in actions]
        maxScore = max(scores)
        bestActions = [action for action, score in zip(actions, scores) if score == maxScore]
        return random.choice(bestActions)

    def evaluate(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        features = dict()
        myPosition = successor.getAgentState(self.index).getPosition()

        foodList = self.getFoodYouAreDefending(gameState).asList()
        foodDis = [self.getMazeDistance(myPosition, food) for food in foodList]
        # we want to stay on a point where we can take care of all foods
        center = sum(foodDis) / len(foodDis)
        features['center'] = center

        # we always want our defense ghost on side
        if (not successor.getAgentState(self.index).isPacman()):
            features['onside'] = 1
        else:
            features['onside'] = 0

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        # find all enemy that is pacman
        enemyPacmans = [enemy for enemy in enemies if enemy.isPacman() and enemy.getPosition()
                        is not None]
        enemyPacmansDis = [self.getMazeDistance(myPosition, enemyPacman.getPosition()) for
                           enemyPacman in enemyPacmans]
        # find all enemy that is not pacman
        enemyGhosts = [a for a in enemies if not a.isPacman() and a.getPosition()
                       is not None]
        enemyGhostsDis = [self.getMazeDistance(myPosition, enemyGhost.getPosition())
                          for enemyGhost in enemyGhosts]
        # we want to make invaders as less as possible
        if len(enemyPacmans) > 0:
            features['Invaders'] = len(enemyPacmans)

        # if enemy pacman gets a capsule, we want to avoid being caught
        isScared = successor.getAgentState(self.index).isScared()
        # if there's enemypacman, we want to get as close as possible,
        # else, we can to get as close as possible to any enemy on board
        if (len(enemyPacmansDis) > 0):
            features['invaderDis'] = min(enemyPacmansDis) if not isScared else -10
        else:
            features['enemyDist'] = min(enemyGhostsDis)

        weights = {
            'Invaders': -1000,
            'onside': 100,
            'enemyDist': -10,
            'center': -10,
            'invaderDis': -100,
        }

        score = 0
        for key in features:
            score += features[key] * weights[key]
        return score
