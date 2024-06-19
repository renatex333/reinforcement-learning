def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

def is_winner(board, player):
    # Verifica se há um vencedor nas linhas, colunas ou diagonais
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def is_board_full(board):
    # Verifica se o tabuleiro está cheio
    return all(board[i][j] != ' ' for i in range(3) for j in range(3))

def evaluate(board):
    # Avalia o estado do tabuleiro
    if is_winner(board, 'X'):
        return 1
    elif is_winner(board, 'O'):
        return -1
    elif is_board_full(board):
        return 0
    return None

def minimax(board, depth, maximizing_player):
    score = evaluate(board)

    if score is not None:
        return score

    if maximizing_player:
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    eval = minimax(board, depth + 1, False)
                    board[i][j] = ' '
                    max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    eval = minimax(board, depth + 1, True)
                    board[i][j] = ' '
                    min_eval = min(min_eval, eval)
        return min_eval

def find_best_move(board):
    best_val = float('-inf')
    best_move = None

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X'
                move_val = minimax(board, 0, False)
                board[i][j] = ' '

                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val

    return best_move

def play_game():
    board = [[' ' for _ in range(3)] for _ in range(3)]

    while True:
        print_board(board)

        player_move = tuple(map(int, input("Digite a sua jogada (linha e coluna separadas por espaço): ").split()))
        if board[player_move[0]][player_move[1]] != ' ':
            print("Jogada inválida. Tente novamente.")
            continue
        else:
            board[player_move[0]][player_move[1]] = 'O'

        if is_winner(board, 'O'):
            print("Você venceu!")
            break
        elif is_board_full(board):
            print("Empate!")
            break

        print("Aguarde enquanto o computador realiza a sua jogada...")
        computer_move = find_best_move(board)
        board[computer_move[0]][computer_move[1]] = 'X'

        if is_winner(board, 'X'):
            print("O computador venceu!")
            break
        elif is_board_full(board):
            print("Empate!")
            break

if __name__ == "__main__":
    play_game()
