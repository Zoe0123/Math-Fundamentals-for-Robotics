# Compute the P A = LDU decomposition and the SVD decomposition for each A examples
import numpy as np
from q1 import LDU

EXAMPES_A = [[[1, 1, 1], [5, 2, 1], [2, 1, 3]], [[1, 1, 1, 1], [0, 3, 3, 0], [1, 0, 2, 0], 
[0, 0, 0, 5], [0, 0, 4, 1]], [[6, 2, 1], [1, 2, 3], [9, 8, 10]]]

def main():
    for A in EXAMPES_A:
        A = np.array(A)
        U, S, VH = np.linalg.svd(A, full_matrices=False)
        print(f"SVD result: U: {U} \n, S: {S} \n, VH: {VH} \n")
        # P, L, D, U = LDU(A)
        # print(f"LDU result: P: {P} \n, L: {L} \n, D: {D} \n, U: {U} \n")

        if np.allclose(A, np.dot(U * S, VH)):
            print("SUCCESS SVD")

if __name__ == "__main__":
    main()


