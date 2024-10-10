import numpy as np  

def train_hebb(patterns, size):  
    """Entrena una red de Hopfield utilizando el aprendizaje de Hebb."""  
    weights = np.zeros((size, size))  
    for pattern in patterns:  
        # Convertir la imagen en un vector de columna binario  
        pattern = pattern.reshape(size, 1)  
        weights += pattern @ pattern.T  
    # Establecer la diagonal en cero  
    np.fill_diagonal(weights, 0)  
    return weights  

def update_pattern(weights, input_pattern, steps=5):  
    """Actualiza el patrón de entrada, para intentar llegar a un patrón conocido."""  
    size = input_pattern.size  
    for _ in range(steps):  
        for i in range(size):  
            potential = np.dot(weights[i], input_pattern.flatten())  
            input_pattern.flatten()[i] = 1 if potential >= 0 else -1  
    return input_pattern  

def print_pattern(pattern, description=""):  
    """Imprime el patrón en un formato binario legible."""  
    print(description)  
    print('\n'.join(''.join('1' if x > 0 else '0' for x in row) for row in pattern))  
    print()  

# Ejemplo de imagen a aprender ("target")  
target_pattern = np.array([  
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],  
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],  
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],  
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],  
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],  
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],  
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],  
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  
]).astype(float) * 2 - 1  

# Crear red de Hopfield y entrenar  
size = target_pattern.size  
weights = train_hebb([target_pattern], size)  

# Introducir una versión distorsionada del patrón  
distorted_pattern = np.array([  
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],  
    [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],  
    [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],  
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0],  
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],  
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  
]).astype(float) * 2 - 1  

print_pattern(distorted_pattern, description="Distorted Pattern:")  

# Recuperar el patrón original usando la red de Hopfield  
recalled_pattern = update_pattern(weights, distorted_pattern.copy())  
print_pattern(recalled_pattern, description="Recalled Pattern:")  
