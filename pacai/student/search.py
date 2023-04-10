"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.core.directions import Directions
from pacai.util.stack import Stack
from pacai.util.queue import Queue

#helper function for dfs

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    # corner case
    if problem.isGoal(problem.startingState()):
        return []
    visited = []
    myStack = Stack()
    myStack.push((problem.startingState(), []))

    while myStack:
        currentNode = myStack.pop()
        if currentNode[0] in visited:
            continue
        visited.append(currentNode[0])

        if problem.isGoal(currentNode[0]):
            return currentNode[1]
    
        for nodeToExpand in problem.successorStates(currentNode[0]):
            newResult = currentNode[1] + [nodeToExpand[1]]
            myStack.push((nodeToExpand[0], newResult))


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    if problem.isGoal(problem.startingState()):
        return []
    visited = []
    myStack = Queue()
    myStack.push((problem.startingState(), []))

    while myStack:
        currentNode = myStack.pop()
        if currentNode[0] in visited:
            continue
        visited.append(currentNode[0])

        if problem.isGoal(currentNode[0]):
            return currentNode[1]
    
        for nodeToExpand in problem.successorStates(currentNode[0]):
            newResult = currentNode[1] + [nodeToExpand[1]]
            myStack.push((nodeToExpand[0], newResult))

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    raise NotImplementedError()

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    raise NotImplementedError()
