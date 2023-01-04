# Implement the PA = LDU decomposition algorithm discussed in class. Do so yourself (in other words, do not merely use predefined Gaussian elimination code in MatLab or Python).
# Simplifications: (i) You may assume that the matrix A is square and invertible. (ii) Do not worry about column interchanges, just row interchanges.
# Demonstrate that your implementation works properly on some examples.

from re import I
import numpy as np
# import scipy.linalg as la # for test

TEST_A = [[[1, 0, 5], [2, 1, 6], [3, 4, 0]], [[1, -2, 3], [3, 5, 2], [-1, 3, -4]], [[4, -20, -12], [-8, 45, 44], [20, -105, 79]]]

def LDU(test_A):
    A = test_A.copy()
    n = A.shape[0]
    P, L, D, U = np.identity(n), np.identity(n), np.identity(n), np.identity(n)
    
    for c in range(n-1):
        # # check if swap is needed by finding the max in c_th column
        # pivot = np.argmax(A[c:n, c]) + c

        # if c != pivot:
        #     A[[pivot, c]] = A[[c, pivot]]
        #     P[[pivot, c]] = P[[c, pivot]] 
        #     L = L - np.identity(n)
        #     L[[pivot, c]] = P[[c, pivot]]
        #     L = L + np.identity(n)
        
        # guassian reduction to get L, A, and then D, U
        for r in range(c+1, n):
            L[r, c] = A[r, c] / A[c, c]
            A[r] = A[r] - L[r,c]*A[c]
        # check if pivot is zero, then need to swap
        if A[c+1, c+1] == 0:
            for r in range(c+2, n):
                if A[r, r-1] != 0:
                    P[[c+1,r]], A[[c+1,r]], L[[c+1,r]] = P[[r,c+1]], A[[c+1,r]], L[[r,c+1]]
                    break

        D = np.diag(np.diag(A))

        for r in range(n):
            for c in range(r, n):
                U[r, c]=A[r, c]/D[r, r]

        return P, L, D, U
        

def main():
    for A in TEST_A:
        print(f"A example: {A}")
        P, L, D, U = LDU(np.array(A))
        print(f"LDU decompose result: P: {P} \n, L: {L} \n, D: {D} \n, U: {U} \n")
        #check if very close to A 
        print(f"PLDU:{P.dot(L.dot(D.dot(U)))}") 
        
        # P, L, U = la.lu(np.asarray(A))
        # D = np.diag(np.diag(U))   
        # U /= np.diag(U)[:, None]  
        # print(f"fomula result: P: {P} \n, L: {L} \n, D: {D} \n, U: {U} \n")

if __name__ == "__main__":
    main()