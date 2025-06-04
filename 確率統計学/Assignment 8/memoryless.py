import numpy as np

# english text entropy assuming a memoryless information source

with open('text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

a = np.zeros(29, dtype=int)

symbols = list("abcdefghijklmnopqrstuvwxyz., ")
char_to_index = {char: idx for idx, char in enumerate(symbols)}

for i in range(1, len(text)):
    a[char_to_index[text[i]]] += 1
c_num = np.sum(a)

def p_x(i):
    return a[i] / c_num

entropy = 0
for i in range(29):
    if a[i] > 0:
        entropy -= p_x(i) * np.log2(p_x(i))

print(f"Entropy of the text: {entropy:.4f} bits per character")