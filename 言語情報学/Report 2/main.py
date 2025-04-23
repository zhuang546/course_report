import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re

with open("tom_sawyer.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

words = re.findall(r'\b[a-z]+\b', text)
word_counts = Counter(words)

frequencies = sorted(word_counts.values(), reverse=True)
ranks = np.arange(1, len(frequencies) + 1)

plt.figure(figsize=(8, 6))
plt.loglog(ranks, frequencies, 'k.', markersize=3)

plt.xlabel("rank")
plt.ylabel("frequency")

plt.show()