#!usr/bin/python
# -*- coding: utf-8 -*-

# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # stack???????????????, reached????????????????????????

    stack = util.Stack()
    reached = []
    path = []  # ????????????

    # ??????????????????????????????
    # print("Start:", problem.getStartState())

    stack.push((problem.getStartState(), path))

    # ??????????????????????????????????????????
    while not stack.isEmpty():

        # ??????????????????????????????????????????????????????
        cur_node, path = stack.pop()

        # ???????????????
        if problem.isGoalState(cur_node):
            return path

        # ??????????????????
        if cur_node not in reached:
            reached.append(cur_node)
            cur_successor = problem.getSuccessors(cur_node)

            for successor, action, stepCost in cur_successor:
                if (successor not in reached):
                    stack.push((successor, path + [action]))

    return path
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    closed = []  # store explored nodes
    fringe = util.Queue()
    start_node = [problem.getStartState(), []]  # search node==node's state+direction(i.e. path)
    fringe.push(start_node)
    while 1:
        if fringe.isEmpty():
            break
        [state, path] = fringe.pop()
        if problem.isGoalState(state):  # find the target
            return path
        if state not in closed:  # unexplored
            closed.append(state)
            child_nodes = problem.getSuccessors(state)
            for child_node in child_nodes:  # traverse all legal nodes
                child_state = child_node[0]
                child_action = child_node[1]
                fringe.push([child_state, path + [child_action]])  # add into open chart(i.e. queue)
    return path
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    ## ++ ??????????????????path-cost??????????????????????????????????????????
    frontier = PriorityQueue()
    tempPath = []  # ???????????????????????????????????????
    # tempState = ( problem.getStartState() , tempPath )
    frontier.push(problem.getStartState(), 0)  # ????????????

    visited = []  # ?????????coordinate

    path_current = []  # ???????????????????????????????????????

    pathToCurrent = {}  # ?????????key??????state???value???????????????????????????state???path??????
    pathToCurrent[problem.getStartState()] = (0, [])

    currentState = problem.getStartState()
    tmp = frontier.isEmpty()

    while not frontier.isEmpty():
        currentState = frontier.pop()  # lowest-cost???state
        cost_current = pathToCurrent[currentState][0]
        path_current = pathToCurrent[currentState][1]  # lowest-cost state????????????startstate?????????state???path??????

        if not problem.isGoalState(currentState):  # ????????????????????????

            if currentState not in visited:  # ???????????????closed
                # ???????????????????????????
                visited.append(currentState)
                del pathToCurrent[currentState]

                successors = problem.getSuccessors(currentState)
                # ?????????????????????????????????action
                for child, direction, cost in successors:
                    if child not in visited:  # ????????????????????? ???????????????closed???
                        tempPath = path_current + [direction]
                        # tempState = (child,tempPath)
                        costtaken = cost_current + cost  # ????????????????????????
                        # frontier.push(child,costtaken)
                        # frontier.update(tempState,)
                        # ++ ??????update???????????????frontier?????????????????????child-state?????????????????????cost?????????
                        frontier.update(child, costtaken)
                        if child in pathToCurrent:  # ?????????????????????????????????open???frontier??????
                            if pathToCurrent[child][0] >= costtaken:
                                pathToCurrent[child] = (costtaken, tempPath)
                        else:
                            pathToCurrent[child] = (costtaken, tempPath)
        else:
            break

    if frontier.isEmpty and not problem.isGoalState(currentState):
        return []  # ???frontier?????????open???????????????????????????path????????????path
    else:
        return path_current


# util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    closed = []
    path = []
    fringe = util.PriorityQueue()  # ?????????????????????????????????????????????
    fringe.push((problem.getStartState(), []), 0)
    while 1:
        if fringe.isEmpty():  # ??????????????????
            break
        [state, path] = fringe.pop()
        if problem.isGoalState(state):  # ???????????????????????????
            return path
        if state not in closed:
            closed.append(state)
            for record in problem.getSuccessors(state):  # ??????
                child_state = record[0]
                child_action = record[1]
                new_path = path + [child_action]
                # ?????? f(n)=g(n)+h(n)
                new_cost = heuristic(child_state, problem) + problem.getCostOfActions(new_path)
                fringe.push((child_state, new_path), new_cost)
    return path
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
