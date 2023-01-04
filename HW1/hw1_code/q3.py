import numpy as np

EXAMPES_A = [[[1, 1, 1], [5, 2, 1], [2, 1, 3]], [[6, 2, 1], [1, 2, 3], [9, 8, 10]], [[6, 2, 1], [1, 2, 3], [9, 8, 10]]]
EXAMPES_b = [[1, 6, 0], [6, -3, -3], [7, 0, -4]]

def main():
    for i, A in enumerate(EXAMPES_A):
        A = np.array(A)
        b = np.array(EXAMPES_b[i]).reshape(3, 1)
        U, S, VH = np.linalg.svd(A, full_matrices=False)
        S1 = 1/S
        S1 = np.diag(S1)
        S1[np.abs(S) < 1e-8] = 0
        x = VH.T @ S1 @ U.T @ b
        print(f"result: {x}")
        # test
        if np.allclose(A@x, b):
            print(f"Success Ax=b example {i}")
        x_null = np.array([2/5, -17/10, 1]).reshape(3, 1)
        if np.allclose(A@(x+x_null), b):
            print(f"Success A(x+x_nll)=b example {i}")

if __name__ == "__main__":
    main()