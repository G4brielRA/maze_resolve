import cv2
import numpy as np
from collections import deque


def solve_maze(image_path):
    # Carrega imagem
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    # Redimensiona para facilitar processamento (ajuste se necessário)
    resized = cv2.resize(img, (100, 100), interpolation=cv2.INTER_NEAREST)

    # Define máscaras de cor
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])

    lower_green = np.array([0, 200, 0])
    upper_green = np.array([100, 255, 100])

    lower_red = np.array([0, 0, 200])
    upper_red = np.array([100, 100, 255])

    # Gera máscara para caminhos e paredes
    mask_path = cv2.inRange(resized, lower_white, upper_white)
    mask_wall = cv2.inRange(resized, lower_black, upper_black)

    # Detecta início (verde)
    mask_start = cv2.inRange(resized, lower_green, upper_green)
    start_pos = np.column_stack(np.where(mask_start > 0))
    if start_pos.size == 0:
        raise ValueError("Início (verde) não encontrado na imagem.")
    start = tuple(start_pos[0])

    # Detecta fim (vermelho)
    mask_end = cv2.inRange(resized, lower_red, upper_red)
    end_pos = np.column_stack(np.where(mask_end > 0))
    if end_pos.size == 0:
        raise ValueError("Fim (vermelho) não encontrado na imagem.")
    end = tuple(end_pos[0])

    # Gera matriz do labirinto: 1 = caminho, 0 = parede
    maze = (mask_path > 0).astype(np.uint8)

    # Garante que início e fim sejam marcados como caminho
    maze[start] = 1
    maze[end] = 1

    path = bfs(maze, start, end)
    if not path:
        return "Nenhum caminho encontrado."

    # Desenha o caminho na imagem original redimensionada
    for y, x in path:
        resized[y, x] = (255, 0, 255)  # magenta

    # Exibe resultado
    cv2.imshow("Labirinto Resolvido (BFS)", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return "Caminho encontrado com sucesso!"


def bfs(maze, start, end):
    rows, cols = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    parent = {}

    queue = deque([start])
    visited[start] = True

    # cima, baixo, esquerda, direita
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current = queue.popleft()
        if current == end:
            break
        for dy, dx in directions:
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if maze[ny, nx] == 1 and not visited[ny, nx]:
                    visited[ny, nx] = True
                    parent[(ny, nx)] = current
                    queue.append((ny, nx))

    # Reconstrói caminho
    path = []
    node = end
    while node != start:
        path.append(node)
        node = parent.get(node)
        if node is None:
            return []  # caminho não encontrado
    path.append(start)
    path.reverse()
    return path


# Exemplo de uso:
if __name__ == "__main__":
    resultado = solve_maze(r'C:\maze_resolve\maze_original.png')
    print(resultado)
