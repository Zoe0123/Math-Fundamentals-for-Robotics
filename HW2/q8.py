import numpy as np
import matplotlib.pyplot as plt

def get_w(A, b):
    """
    Let the point defined by b fall into the triangle formed the three corners 
    defined by A, and return the weights w.r.t the three corners 
    """
    u, s, vh = np.linalg.svd(A)
    s_diag = np.diag(s)
    w = vh.T @ (np.linalg.solve(s_diag, u.T@b))
    return w

def read_txt(fpath):
    raw_data = np.loadtxt(fpath,dtype=str,delimiter='\t')
    n = raw_data.shape[0]
    m = 50
    data = np.zeros((n, m))
    for i in range(n):
        row = raw_data[i].split(' ')
        for j in range(50):
            data[i][j] = row[j]
    # paths, each entry (x, y)
    P = np.zeros((n//2, m, 2)) 
    for i in range(0, n, 2):
        P[i//2, :, 0] = data[i]
    for i in range(1, n, 2):
        P[i//2, :, 1] = data[i]
    return P

def pick_3path(P_side, p0):
    p0s_inP = P_side[:, 0, :]
    p0_dist = []
    for i in p0s_inP:
        x0, y0 = i
        dist = np.sqrt((x0-p0[0])**2 + (y0-p0[1])**2 )
        p0_dist.append(dist)
    ind = np.argsort(np.array(p0_dist))[:3]
    # construct A and p, and then find weights
    A, b = np.ones((3, 3)), np.ones((3, 1))
    A[0, :] = p0s_inP[ind, 0]
    print(p0s_inP.shape)
    A[1, :] = p0s_inP[ind, 1]
    b[0], b[1] = p0
    w = get_w(A, b)
    while w[0] < 0 or w[1] < 0 or w[2] < 0:
        nxt = 1
        for i in range(3):
            if w[i] < 0:
                ind_nxt = np.argsort(np.array(p0_dist))[2+nxt]
                ind[i] = ind_nxt
                A[0, i], A[1, i]= p0s_inP[ind_nxt, 0], p0s_inP[ind_nxt, 1]
                nxt += 1
        w = get_w(A, b)
    return A, ind, w
    
def avg_p0dist(ind, P_side, p0):
    p0s = P_side[ind, 0, :]
    avg_dist = 0
    for i in p0s:
        x0, y0 = i
        avg_dist += np.sqrt((x0-p0[0])**2 + (y0-p0[1])**2 )
    return avg_dist/3

def plot(p_selected, p):
    fig, ax = plt.subplots()
    for i in range(3):
        ax.plot(p_selected[i,:,0],p_selected[i,:,1],'g')
    ax.add_artist(plt.Circle((5, 5), 1.5, color='r', fill = False))
    ax.plot(p[:,0],p[:,1],'k')
    plt.xticks(np.arange(0, 13, 2))
    plt.yticks(np.arange(0, 13, 2))
    plt.show()

if __name__ == '__main__':
    P = read_txt("paths.txt")
    P_left, P_right = [], []
    n, m = P.shape[0], P.shape[1]
    for i in range(n):
        if np.sum(P[i, :, 1]) > 5*50:
            P_left.append(P[i, :, :])
        else:
            P_right.append(P[i, :, :])
    P_left, P_right = np.array(P_left), np.array(P_right)
    
    p0s = [[0.8,1.8], [2.2,1.0], [2.7,1.4]]
    for p0 in p0s:
        A_left, ind_left, w_left = pick_3path(P_left, p0)
        A_right, ind_right, w_right = pick_3path(P_right, p0)

        # determine which side is better choice, by avg starting points' distances
        p0_distleft = avg_p0dist(ind_left, P_left, p0)
        p0_distright = avg_p0dist(ind_right, P_right, p0)
        
        if p0_distleft <= p0_distright:
            ind, w = ind_left, w_left
            p_selected =  P_left[ind, :, :]
        else:
            ind, w = ind_right, w_right
            p_selected =  P_right[ind, :, :]

        # construct the path 
        p = w[0] * p_selected[0, :, :] + w[1] * p_selected[1, :, :] + w[2] * p_selected[1, :, :]

        print(p)

        # plot the graph
        plot(p_selected, p)
        

        


    

    
