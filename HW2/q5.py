import cmath
import q1

def get_Px(x, roots):
    return x**5 + x**4 + x**3 + x**2 + x + 1

def get_Px1(x, roots):
    p_x = get_Px(x, [])
    for root in roots:
        p_x = p_x/(x-root)
    return p_x

def muller(x0, x1, x2, fn, roots=[]):
    while abs(x1-x2) > 1e-5: # cannot divided by zero
        p_x0, p_x1, p_x2 = fn(x0, roots), fn(x1, roots), fn(x2, roots)
        
        # update a, b, c
        diff1 = (p_x1 - p_x0)/(x1-x0)
        diff2 = (p_x2 - p_x1)/(x2-x1)
        a = p_x2
        c = (diff2-diff1)/(x2-x0)
        b = diff2 + c * (x2-x1)

        discrim = cmath.sqrt(b**2 - 4*a*c) # for real ands complex roots
        root1 = (2 * a) / (b + discrim)  
        root2 = (2 * a) / (b - discrim)
        
        if abs(b + discrim) > abs(b - discrim):
            root = x2 - root1
        else:
            root = x2 - root2

        x0, x1, x2 = x1, x2, root
    return x2

if __name__ == '__main__':
    root1 = muller(1, 2, 3, get_Px) 
    print(root1)
    root2 = muller(1, 2, 3, get_Px1, roots=[root1]) 
    print(root2)
    root3 = muller(1, 2, 3, get_Px1, roots=[root1, root2]) 
    print(root3)
    root4 = muller(1, 2, 3, get_Px1, roots=[root1, root2, root3]) 
    print(root4)
    root5 = muller(1, 2, 3, get_Px1, roots=[root1, root2, root3, root4]) 
    print(root5)
