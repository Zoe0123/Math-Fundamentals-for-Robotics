import numpy as np
import matplotlib.pyplot as plt

def infer_trans_rot(P, Q):
    """
    Infer translation and Rotation
    Input:
    P: p1, . . . , pn are the 3D coordinates of n points located on a rigid body in three-space. 
    Q: q1, . . . , qn are the 3D coordinates of these same points after the body has been translated 
    and rotated by some unknown amount.
    (n is at least 3) 
    
    Returns
    trans: the body’s translation inferred 
    rotat: the body’s rotation inferred
    """
    n = P.shape[0]
    
    # consider both origin as centroid
    CenP, CenQ = np.mean(P, axis=0), np.mean(Q, axis=0)

    # 1. rotation 
    P1, Q1 = P - np.array([CenP]*n), Q - np.array([CenQ]*n)
    # use SVD to infer
    U, S, VH = np.linalg.svd(P1.T@Q1, full_matrices=False)
    
    # the derivation of these two formula is shown in the written report
    rotat = VH.T @ U.T

    #2. translation
    trans = CenQ - CenP @ rotat

    return trans, rotat    

def test_plot(P, Q, Q_inf):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(P[:, 0], P[:, 1], P[:, 2], 'y', label='P')
    ax.plot3D(Q[:, 0], Q[:, 1], Q[:, 2], 'k', label='Q')
    ax.scatter(Q_inf[:, 0], Q_inf[:, 1], Q_inf[:, 2], color='r', label='Q_inffered')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def main():
    # randomly generate 3D points with n = 5 (within interval [0, 1) in this example)
    # P = np.random.random((5, 3))
    # Q = np.random.random((5, 3))
    # P = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Q = np.array([[3, 1, 0], [4, 3, 0], [3, 3, 1]])
    # P = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1], [2, 1, 0]])
    # Q = np.array([[3, 1, 0], [4, 3, 0], [3, 3, 1], [4, 1, 0]])
    P = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1], [2, 1, 0], [2, 1, 1]])
    Q = np.array([[3, 1, 0], [4, 3, 0], [3, 3, 1], [4, 1, 0], [4, 1, 1]])
    # P = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1], [2, 1, 0], [2, 1, 1], [0, 0, 0]])
    # Q = np.array([[3, 1, 0], [4, 3, 0], [3, 3, 1], [4, 1, 0], [4, 1, 1], [3, 3, 0]])
    trans, rotat = infer_trans_rot(P, Q)
    Q_inf = P@rotat + np.tile(trans, P.shape[0]).reshape(P.shape[0], 3)
    print(f"P: {P} \n, Q: {Q} \n, Q_inffered: {Q_inf} \n")
    print(f"trans: {trans} \n, rotat: {rotat} \n")
    test_plot(P, Q, Q_inf)

if __name__ == "__main__":
    main()