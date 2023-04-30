import time
from RandomAlgorithms import RandomAlgorithms
import numpy as np
import pandas as pd

seedLCG = int(time.time())
seedXorShift = int(time.time())
size = 500


def randomLcg():
    randMax = 4294967296
    global seedLCG
    seedLCG = 214013 * seedLCG + 2531011
    seedLCG ^= seedLCG >> 15
    return seedLCG % randMax


def randomXorShift():
    global seedXorShift
    seedXorShift ^= seedXorShift << 13
    seedXorShift ^= seedXorShift >> 17
    seedXorShift ^= seedXorShift << 5
    return seedXorShift


def generateSequence(algorithm, mod, length):
    res = []
    match algorithm:
        case RandomAlgorithms.LCG:
            random = randomLcg
        case RandomAlgorithms.XorShift:
            random = randomXorShift
        case _:
            raise Exception("Undefined algorithm")

    for i in range(length):
        res.append(random() % mod)
    return res


def writeSequenceInFile(filename, sequence):
    file = open(filename, 'a')
    file.write(str(sequence)[1:-2] + '\n')
    file.close()


def makeTableOfParams(algorithm):
    sequences = np.genfromtxt(f'assets/{algorithm.name}Sequences.txt', delimiter=", ", dtype=int)
    df = pd.DataFrame(index=[f'Выборка {i}' for i in np.arange(1, 11, 1)])
    df["Mean"] = sequences.mean(axis=1)
    df["Deviation"] = sequences.std(axis=1)
    df["CV"] = df["Deviation"]/df["Mean"]
    return df


def calcChiSquare(sequence):
    global size
    k = int(1 + 3.322 * np.log(size))
    partition = np.linspace(0, 1, k)
    chisquare = 0

    uniformSeq = sequence/10000
    m = pd.Series(uniformSeq)
    frequencies = m.groupby(pd.cut(m, bins=partition)).count()
    for n in frequencies:
        chisquare += (n - size / k) ** 2 / (size / k)

    return chisquare


def writeChiSquareToFile(algorithm):
    file = open(f'assets/{algorithm.name}ChiSquare.txt', 'w')
    sequences = np.genfromtxt(f'assets/{algorithm.name}Sequences.txt', delimiter=", ", dtype=int)
    for sequence in sequences:
        file.write(str(calcChiSquare(sequence)) + '\n')
    file.close()


def calcTime():
    sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    file = open("assets/time.txt", 'a')
    header = f'\t\tLCG\t XorShift Native\n'
    file.write(header)
    for size in sizes:
        startTimeLCG = time.time()
        generateSequence(RandomAlgorithms.LCG, 10000, size)
        timeLCG = '{:.5f}'.format(round(time.time()-startTimeLCG, 5))
        startTimeXorShift = time.time()
        generateSequence(RandomAlgorithms.XorShift, 10000, size)
        timeXorShift = '{:.5f}'.format(round(time.time()-startTimeXorShift, 5))
        startTimeNative = time.time()
        np.random.randint(0, 10000, size)
        timeNative = '{:.5f}'.format(round(time.time()-startTimeNative, 5))
        string = f'{size} {timeLCG} {timeXorShift}  {timeNative}\n'
        file.write(string)
    file.close()


if __name__ == '__main__':
    print(np.random.randint(0, 10000, 1000000))



# п. 3
# Выборка однородная
# Эталоны:
# Mean = 5000
# Deviation = 2886,75134595
# CV = 0,57735026919