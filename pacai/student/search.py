"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue
import copy


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
    # keep track of visited nodes
    visited = []
    # stack to store nodes to expand
    myStack = Stack()
    # push the starting node and a queue to keep track of the answer path
    myStack.push((problem.startingState(), Queue()))

    while myStack:
        # pop the top node
        currentNode = myStack.pop()
        # skip if current node has been visited
        if currentNode[0] in visited:
            continue
        # add current node into visited set
        visited.append(currentNode[0])
        # return if we hit the goal
        if problem.isGoal(currentNode[0]):
            result = []
            # pop up the queue
            while currentNode[1]:
                result.append(currentNode[1].pop())
            return result

        # add child nodes of current node into the stack, track result path respectively
        for nodeToExpand in problem.successorStates(currentNode[0]):
            # make a deep copy so current node remains the same
            newResult = copy.deepcopy(currentNode[1])
            newResult.push(nodeToExpand[1])
            myStack.push((nodeToExpand[0], newResult))


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***

    # corner case
    if problem.isGoal(problem.startingState()):
        return []

    # keep track of visited nodes
    visited = []
    # queue up nodes to expand
    myQueue = Queue()
    # push the starting node and a queue to keep track of the answer path
    myQueue.push((problem.startingState(), Queue()))

    while myQueue:
        # pop the top node
        currentNode = myQueue.pop()
        # skip if current node has been visited
        if currentNode[0] in visited:
            continue
        # add current node into visited set
        visited.append(currentNode[0])
        # return if we hit the goal
        if problem.isGoal(currentNode[0]):
            result = []
            # pop up the queue
            while currentNode[1]:
                result.append(currentNode[1].pop())
            return result

        # add child nodes of current node into the stack, track result path respectively
        for nodeToExpand in problem.successorStates(currentNode[0]):
            # make a deep copy so current node remains the same
            newResult = copy.deepcopy(currentNode[1])
            newResult.push(nodeToExpand[1])
            myQueue.push((nodeToExpand[0], newResult))


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***

    # corner case
    if problem.isGoal(problem.startingState()):
        return []

    # keep track of visited nodes
    visited = []
    # use a priorityQueue to store nodes to expand
    myPQ = PriorityQueue()
    # push the starting node and a list to keep track of the answer path
    # (because problem.actionsCost need to iterate)
    # set the first priority to 0 because nothing in the path
    myPQ.push((problem.startingState(), []), 0)

    while myPQ:
        # pop the top node
        currentNode = myPQ.pop()
        # skip if current node has been visited
        if currentNode[0] in visited:
            continue
        # add current node into visited set
        visited.append(currentNode[0])
        # return if we hit the goal
        if problem.isGoal(currentNode[0]):
            return currentNode[1]

        # add child nodes of current node into the stack, track result path respectively
        for nodeToExpand in problem.successorStates(currentNode[0]):
            # make a deep copy so current node remains the same
            newResult = copy.deepcopy(currentNode[1])
            newResult.append(nodeToExpand[1])
            # use problem.actionsCost to get new priority
            myPQ.push((nodeToExpand[0], newResult), problem.actionsCost(newResult))


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***

    # corner case
    if problem.isGoal(problem.startingState()):
        return []

    # keep track of visited nodes
    visited = []
    # use a priorityQueue to store nodes to expand
    myPQ = PriorityQueue()
    # push the starting node and a list to keep track of the answer path
    # (because problem.actionsCost need to iterate)
    # set the first priority to 0 because nothing in the path
    myPQ.push((problem.startingState(), []), 0)

    while myPQ:
        # pop the top node
        currentNode = myPQ.pop()
        # skip if current node has been visited
        if currentNode[0] in visited:
            continue
        # add current node into visited set
        visited.append(currentNode[0])
        # return if we hit the goal
        if problem.isGoal(currentNode[0]):
            return currentNode[1]

        # add child nodes of current node into the stack, track result path respectively
        for nodeToExpand in problem.successorStates(currentNode[0]):
            # make a deep copy so current node remains the same
            newResult = copy.deepcopy(currentNode[1])
            newResult.append(nodeToExpand[1])
            # use problem.actionsCost + Cost to get new priority
            myPQ.push((nodeToExpand[0], newResult),
                      problem.actionsCost(newResult) + heuristic(nodeToExpand[0], problem))
