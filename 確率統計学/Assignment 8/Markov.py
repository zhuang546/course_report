import numpy as np

# english text entropy assuming a first-order Markov source

with open('text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

a = np.zeros((29, 29), dtype=int)

symbols = list("abcdefghijklmnopqrstuvwxyz., ")
char_to_index = {char: idx for idx, char in enumerate(symbols)}

for i in range(1, len(text)):
    a[char_to_index[text[i-1]], char_to_index[text[i]]] += 1
pair_num = np.sum(a)

def p_i_j(i, j):
    return a[i, j] / pair_num

def p_i_j_given_i(i, j):
    return a[i, j] / np.sum(a[i, :]) if np.sum(a[i, :]) > 0 else 0

entropy = 0
for i in range(29):
    for j in range(29):
        if a[i, j] > 0:
            entropy -= p_i_j(i,j) * np.log2(p_i_j_given_i(i, j))

print(f"Entropy of the text: {entropy:.4f} bits per character")