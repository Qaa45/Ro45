from collections import defaultdict

def mapper(line):
    words = line.strip().split()
    return [(word.lower(), 1) for word in words]

def reducer(pairs):
    word_count = defaultdict(int)
    for word, count in pairs:
        word_count[word] += count
    return word_count

def word_count(filename):
    intermediate = []
    with open(filename, 'r') as f:
        for line in f:
            intermediate.extend(mapper(line))
    final_counts = reducer(intermediate)
    for word, count in sorted(final_counts.items()):
        print(f'{word}: {count}')

word_count('input.txt')
