import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags)]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
# player 0 -- white --> goes first
# player 1 -- black
# side==False if Player 0 should play next.

def minimax_dfs(side, board, flags, depth):
    moveTree = {}
    moves = [move for move in generateMoves(side, board, flags)]
    if depth == 0 or len(moves) == 0:
        return (evaluate(board), [], moveTree)
    elif side == False:
        maxValue = None
        optimalMove = None
        optimalMoveList = None
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            score, newMoveList, newMoveTree = minimax_dfs(newside, newboard, newflags, depth-1)
            encodedMove = encode(*move)
            moveTree[encodedMove] = newMoveTree
            if maxValue == None or maxValue < score:
                maxValue = score
                optimalMove = encodedMove
                optimalMoveList = newMoveList
        return (maxValue, [decode(optimalMove), *optimalMoveList], moveTree)
    else:
        minValue = None
        optimalMove = None
        optimalMoveList = None
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            score, newMoveList, newMoveTree = minimax_dfs(newside, newboard, newflags, depth - 1)
            encodedMove = encode(*move)
            moveTree[encodedMove] = newMoveTree
            if minValue == None or minValue > score:
                minValue = score
                optimalMove = encodedMove
                optimalMoveList = newMoveList
        return (minValue, [decode(optimalMove), *optimalMoveList], moveTree)

def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    return minimax_dfs(side, board, flags, depth)

def alphabeta_dfs(side, board, flags, depth, alpha, beta):
    moveTree = {}
    moves = [move for move in generateMoves(side, board, flags)]
    newAlpha = alpha
    newBeta = beta

    if depth == 0 or len(moves) == 0:
        if side == False:
            newAlpha = evaluate(board)
            return (newAlpha, newBeta, newAlpha, [], moveTree)
        else:
            newBeta = evaluate(board)
            return (newAlpha, newBeta, newBeta, [], moveTree)

    elif side == False:
        maxValue = None
        optimalMove = None
        optimalMoveList = None
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            alpha_, beta_, score, newMoveList, newMoveTree = alphabeta_dfs(newside, newboard, newflags, depth-1, newAlpha, newBeta)
            encodedMove = encode(*move)
            moveTree[encodedMove] = newMoveTree
            if (score is not None) and (maxValue == None or maxValue < score):
                maxValue = score
                optimalMove = encodedMove
                optimalMoveList = newMoveList
            if beta_ is not None and newAlpha < beta_:
                newAlpha = beta_
            if newAlpha >= newBeta:
                # exit right from here
                break
        return (newAlpha, newBeta, maxValue, [decode(optimalMove), *optimalMoveList], moveTree)
    else:
        minValue = None
        optimalMove = None
        optimalMoveList = None
        newAlpha = alpha
        newBeta = newBeta
        for move in moves:
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            alpha_, beta_, score, newMoveList, newMoveTree = alphabeta_dfs(newside, newboard, newflags, depth - 1, newAlpha, newBeta)
            encodedMove = encode(*move)
            moveTree[encodedMove] = newMoveTree
            if (score is not None) and (minValue == None or minValue > score):
                minValue = score
                optimalMove = encodedMove
                optimalMoveList = newMoveList
            if alpha_ is not None and newBeta > alpha_:
                newBeta = alpha_
            if newAlpha >= newBeta:
                # exit right from here
                break
        return (newAlpha, newBeta, minValue, [decode(optimalMove), *optimalMoveList], moveTree)

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    alpha_, beta_, score, moveList, moveTree = alphabeta_dfs(side, board, flags, depth, alpha, beta)
    return (score, moveList, moveTree)

def stochastic_single(side, board, flags, depth, chooser):
    moveList = []
    moveTree = {}
    moves = [move for move in generateMoves(side, board, flags)]
    if depth == 0 or len(moves) == 0:
        return (evaluate(board), moveList, moveTree)
    else:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        newScore, newMoveList, newMoveTree = stochastic_single(newside, newboard, newflags, depth - 1, chooser)
        encodedMove = encode(*move)
        moveTree[encodedMove] = newMoveTree
        moveList = [move, *newMoveList]  # anything can be chosen
        return newScore, moveList, moveTree

def stochastic_breadth(side, board, flags, depth, breadth, chooser):
    moveTree = {}
    moveList = []
    score = 0
    moves = [move for move in generateMoves(side, board, flags)]
    if depth == 0 or len(moves) == 0:
        return (evaluate(board), moveList, moveTree)
    else:
        for iter in range(breadth):
            move = chooser(moves)
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            newScore, newMoveList, newMoveTree = stochastic_single(newside, newboard, newflags, depth-1, chooser)
            encodedMove = encode(*move)
            moveTree[encodedMove] = newMoveTree
            moveList = [move, *newMoveList] # anything can be chosen
            score += newScore
        return (score/breadth ,moveList, moveTree)

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''


    moves = [move for move in generateMoves(side, board, flags)]
    if depth == 0 or len(moves) == 0:
        return (evaluate(board), [], {})
    maxScore = None
    maxMoveList = None
    # maxMoveTree = None
    moveTree = {}
    minScore = None
    minMoveList = None
    # minMoveTree = None
    for move in moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        score, newMoveList, newMoveTree = stochastic_breadth(newside, newboard, newflags, depth - 1, breadth, chooser)
        encodedMove = encode(*move)
        moveTree[encodedMove] = newMoveTree
        if maxScore is None or maxScore < score:
            maxScore = score
            maxMoveList = [move, *newMoveList]
        if minScore is None or minScore > score:
            minScore = score
            minMoveList = [move, *newMoveList]


    if side:
        return minScore, minMoveList, moveTree
    else:
        return maxScore, maxMoveList, moveTree
    # raise NotImplementedError("you need to write this!")
