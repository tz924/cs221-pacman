from util import manhattanDistance, Counter
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation
        function.

        getAction takes a GameState and returns some Directions.X for some X
        in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function,
        the |gameState| argument
        is an object of GameState class. Following are a few of the helper
        methods that you
        can use to query a GameState object to gather information about the
        present state
        of Pac-Man, the ghosts and the maze.

        gameState.getLegalActions():
            Returns the legal actions for the agent specified. Returns
            Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the
            action.
            Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game


        The GameState class is defined in pacman.py and you might want to
        look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in
                  legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if
                       scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are
        better.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in
                          newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.
      It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (problem 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of
          the following:
          pacman won, pacman lost or there are no legal moves.

          Here are some method calls that might be useful when implementing
          minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.getScore():
            Returns the score corresponding to the current state of the game

          gameState.isWin():
            Returns True if it's a winning state

          gameState.isLose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue

        """

        # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry
        # if you deviate from this)
        maxPlayerIndex = gameState.getNumAgents() - 1
        pacmanIndex = 0
        IsEnd = lambda s: s.isWin() or s.isLose()

        def Vminimax(s, d, agentIndex):
            """
            pacman (a_0)            => argmax Succ(s, a), d
            ghost (a_1 to a_n-1)    => argmax Succ(s, a), d
            ghost (a_n)             => argmax Succ(s, a), d - 1
            Recurrence that mirrors Problem 1
            """
            # IsEnd(s) => Utility(s)
            if IsEnd(s):
                return s.getScore(), Directions.STOP

            # d = 0 => Eval(s)
            if d == 0:
                return self.evaluationFunction(s), Directions.STOP

            legalActions = s.getLegalActions(agentIndex)

            # No legal moves
            if not legalActions:
                return s.getScore(), Directions.STOP

            if agentIndex == pacmanIndex:  # Pacman
                # pacman (a_0) => argmax Succ(s, a), d
                return max([(Vminimax(s.generateSuccessor(agentIndex, action),
                                      d, agentIndex + 1)[0], action)
                            for action in legalActions])
                # [1] since we return s, a pairs [1] => a
            elif agentIndex < maxPlayerIndex:  # Ghost 1 to n-1
                # ghost (a_1 to a_n-1) => argmax Succ(s, a), d
                return min([(Vminimax(s.generateSuccessor(agentIndex, action),
                                      d, agentIndex + 1)[0], action)
                            for action in legalActions])
            elif agentIndex == maxPlayerIndex:  # Ghost n
                # ghost (a_n) => argmax Succ(s, a), d - 1
                return min([(Vminimax(s.generateSuccessor(agentIndex, action),
                                      d - 1, pacmanIndex)[0], action)
                            for action in legalActions])

        return Vminimax(gameState, self.depth, self.index)[1]
        # END_YOUR_CODE


######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (problem 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and
          self.evaluationFunction
        """

        # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry
        # if you deviate from this)
        maxPlayerIndex = gameState.getNumAgents() - 1
        pacmanIndex = 0
        IsEnd = lambda s: s.isWin() or s.isLose()
        MIN_VAL, MAX_VAL = float('-inf'), float('inf')

        def Vminimax(s, d, agentIndex, alpha, beta):
            """
            pacman (a_0)            => argmax Succ(s, a), d
            ghost (a_1 to a_n-1)    => argmax Succ(s, a), d
            ghost (a_n)             => argmax Succ(s, a), d - 1
            Recurrence that mirrors Problem 1
            """
            # IsEnd(s) => Utility(s)
            if IsEnd(s):
                return s.getScore(), Directions.STOP

            # d = 0 => Eval(s)
            if d == 0:
                return self.evaluationFunction(s), Directions.STOP

            legalActions = s.getLegalActions(agentIndex)

            # No legal moves
            if not legalActions:
                return s.getScore(), Directions.STOP

            if agentIndex == pacmanIndex:  # Pacman
                # pacman (a_0) => argmax Succ(s, a), d
                bestScore, bestAction = (MIN_VAL, Directions.STOP)
                for action in legalActions:
                    item = Vminimax(s.generateSuccessor(agentIndex, action),
                                    d, agentIndex + 1, alpha, beta)
                    if not item:
                        return self.evaluationFunction(s), action

                    # Update best
                    score, _ = item
                    if score > bestScore:
                        bestScore, bestAction = score, action

                    # Update beta
                    alpha = max(alpha, bestScore)
                    if beta < bestScore:
                        return bestScore, bestAction
                return bestScore, bestAction

                # [1] since we return s, a pairs [1] => a
            elif agentIndex < maxPlayerIndex:  # Ghost 1 to n-1
                # ghost (a_1 to a_n-1) => argmax Succ(s, a), d
                bestScore, bestAction = (MAX_VAL, Directions.STOP)
                for action in legalActions:
                    item = Vminimax(s.generateSuccessor(agentIndex, action),
                                    d, agentIndex + 1, alpha, beta)

                    if not item:
                        return self.evaluationFunction(s), action

                    # Update best
                    score, _ = item
                    if score < bestScore:
                        bestScore, bestAction = score, action

                    # Update beta
                    beta = min(beta, bestScore)
                    if alpha > bestScore:
                        return bestScore, bestAction
                return bestScore, bestAction

            elif agentIndex == maxPlayerIndex:  # Ghost n
                # ghost (a_n) => argmax Succ(s, a), d - 1
                bestScore, bestAction = (MAX_VAL, Directions.STOP)
                for action in legalActions:
                    item = Vminimax(s.generateSuccessor(agentIndex, action),
                                    d - 1, pacmanIndex, alpha, beta)
                    if not item:
                        return self.evaluationFunction(s), action

                    # Update best
                    score, _ = item
                    if score < bestScore:
                        bestScore, bestAction = score, action

                    # Update beta
                    beta = min(beta, bestScore)
                    if alpha > bestScore:
                        return bestScore, bestAction
                return bestScore, bestAction

        return Vminimax(gameState, self.depth, self.index, MIN_VAL, MAX_VAL)[1]
        # END_YOUR_CODE


######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (problem 3)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and
          self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from
          their
          legal moves.
        """
        # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry
        # if you deviate from this)
        maxPlayerIndex = gameState.getNumAgents() - 1
        pacmanIndex = 0
        IsEnd = lambda s: s.isWin() or s.isLose()

        def Vminimax(s, d, agentIndex):
            """
            pacman (a_0)            => argmax Succ(s, a), d
            ghost (a_1 to a_n-1)    => E[a] Succ(s, a), d
            ghost (a_n)             => E[a] Succ(s, a), d - 1
            Recurrence that mirrors Problem 1
            """
            # IsEnd(s) => Utility(s)
            if IsEnd(s):
                return s.getScore(), Directions.STOP

            # d = 0 => Eval(s)
            if d == 0:
                return self.evaluationFunction(s), Directions.STOP

            legalActions = s.getLegalActions(agentIndex)

            # No legal moves
            if not legalActions:
                return s.getScore(), Directions.STOP

            P_uniform = 1. / len(legalActions)

            if agentIndex == pacmanIndex:  # Pacman
                # pacman (a_0) => argmax Succ(s, a), d
                return max([(Vminimax(s.generateSuccessor(agentIndex, action),
                                      d, agentIndex + 1)[0], action)
                            for action in legalActions])
                # [1] since we return s, a pairs [1] => a
            elif agentIndex < maxPlayerIndex:  # Ghost 1 to n-1
                # ghost (a_1 to a_n-1) => E[a] Succ(s, a), d
                return random.choice([(P_uniform * Vminimax(s.generateSuccessor(
                    agentIndex, action), d, agentIndex + 1)[0], action)
                                      for action in legalActions])
            elif agentIndex == maxPlayerIndex:  # Ghost n
                # ghost (a_n) => E[a] Succ(s, a), d - 1
                return random.choice([(P_uniform * Vminimax(s.generateSuccessor(
                    agentIndex, action), d - 1, pacmanIndex)[0], action)
                                      for action in legalActions])

        return Vminimax(gameState, self.depth, self.index)[1]
        # END_YOUR_CODE


######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
    """
      Your extreme, unstoppable evaluation function (problem 4).

      DESCRIPTION: <write something here so we know what you did>
    """
    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if
    # you deviate from this)
    MIN_UTILITY, MAX_UTILITY = float('-inf'), float('inf')
    if currentGameState.isLose():
        return MIN_UTILITY
    if currentGameState.isWin():
        return MAX_UTILITY

    values, weights = Counter(), Counter()

    # Score
    currentScore = currentGameState.getScore()
    values['score'] = currentScore

    currentFood = currentGameState.getFood().asList()
    n, m = len(currentFood), len(currentFood[0])
    foodsLeft = [(i, j) for i in range(n) for j in range(m)
                 if currentFood[i][j]]
    position = currentGameState.getPacmanPosition()

    # Distance to Closest Food
    distClosestFood = min(manhattanDistance(position, fp) for fp in
                          foodsLeft)
    values['distFood'] = distClosestFood

    # Number of Food
    n_food = currentGameState.getNumFood()
    values['nFood'] = n_food

    # Ghost
    ghostStates = currentGameState.getGhostStates()
    scared, ghosts = [], []
    for ghost in ghostStates:
        if ghost.scaredTimer > 0:
            scared += [ghost]
        else:
            ghosts += [ghost]
    values['distScared'] = min(manhattanDistance(position, gp.getPosition())
                               for gp in scared) if scared else 0
    values['distGhost'] = min(manhattanDistance(position, gp.getPosition())
                              for gp in ghosts) if ghosts else MIN_UTILITY

    # a list of positions (x,y) of the remaining capsules
    capsulesLeft = currentGameState.getCapsules()
    n_capsules = len(capsulesLeft)
    values['nCap'] = n_capsules

    distClosestCapsule = min(manhattanDistance(position, cp)
                             for cp in capsulesLeft) if capsulesLeft else 0
    values['distCap'] = distClosestCapsule

    # walls = currentGameState.getWalls()
    # distWall = min(manhattanDistance(position, wp)
    #                for wp in walls)
    # values['distWall'] = distWall

    weights['score'] = 1
    weights['distFood'] = -0.2
    weights['nFood'] = -4
    weights['distScared'] = 2
    weights['distGhost'] = -0.5
    weights['nCap'] = -15
    weights['distCap'] = 0
    # weights['distWall'] = 0
    return values * weights
    # END_YOUR_CODE


# Abbreviation
better = betterEvaluationFunction
