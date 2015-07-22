#!/home/maxhutch/anaconda3/bin/python

import numpy as np
"""
app_domain = [8, 8, 16]
sys_domain = [4,4,4,8,2]
"""

from sys import argv, exit
toks = argv[1].split(',')
app_domain_in = np.array([int(toks[0]), int(toks[1]), int(toks[2])], dtype=int)
toks = argv[2].split(',')
sys_domain = np.array([int(toks[0]),int(toks[1]),int(toks[2]),int(toks[3]),int(toks[4])], dtype=int)
toks = argv[3]
mode = int(toks)

from math import log2
nproc = np.prod(sys_domain) * mode
size_local = max(np.prod(app_domain_in) / nproc, 1)
shape_local = np.zeros(3)
log_local = log2(size_local)
for i in range(3):
    shape_local[i] = 2**int(log_local / (3-i))
    log_local = log_local - int(log_local / (3-i))
app_domain = app_domain_in / shape_local


shape_node = np.zeros(3)
log_node = log2(mode)
for i in range(3):
    shape_node[i] = 2**int(log_node / (3-i))
    log_node = log_node - int(log_node / (3-i))
shape_node = np.flipud(shape_node)


sys_domain_d = list(range(5))

def find_pair(domain, target):
    for i in range(len(domain)):
        for j in range(i+1, len(domain)):
            if domain[i]*domain[j] == target:
                return i, j
    return -1, -1

def find_triplet(domain, target):
    for i in range(len(domain)):
        for j in range(i+1, len(domain)):
            for k in range(j+1, len(domain)):
                if domain[i]*domain[j]*domain[k] == target:
                    return i, j, k
    return -1, -1, -1


def get_dims(app_domain, sys_domain):
    dims = []
    #for d in sorted(app_domain):
    for d in app_domain:
        if d in sys_domain:
            ind = sys_domain.index(d)
            dims.append((sys_domain_d[ind],))
            del sys_domain[ind]
            del sys_domain_d[ind]
            continue
        d0, d1 = find_pair(sys_domain, d)
        if d0 >= 0:
            dims.append((sys_domain_d[d0], sys_domain_d[d1]))    
            del sys_domain[d1]
            del sys_domain[d0]
            del sys_domain_d[d1]
            del sys_domain_d[d0]
            continue
        d0, d1, d2 = find_triplet(sys_domain, d)
        if d0 >= 0:
            dims.append((sys_domain_d[d0], sys_domain_d[d1], sys_domain_d[d2]))    
            del sys_domain[d2]
            del sys_domain[d1]
            del sys_domain[d0]
            del sys_domain_d[d2]
            del sys_domain_d[d1]
            del sys_domain_d[d0]
            continue

    return dims

def expand_ind(ind0, trans, domain):
    res = np.zeros(len(domain), dtype=int)
    for i in range(len(ind0)):
        if len(trans[i]) == 1:
            res[trans[i][0]] = ind0[i]
        elif len(trans[i]) == 2:
            j = ind0[i]
            dims = trans[i]
            if j < domain[dims[0]]:
                res[dims[0]] = j
                res[dims[1]] = 0
                continue
            j = j - domain[dims[0]] 
            res[dims[1]] = min((j % (2*domain[dims[1]]-2)), 2*domain[dims[1]]-3 - (j % (2*domain[dims[1]]-2))) + 1
            res[dims[0]] = domain[dims[0]] - 1 - int((j / (domain[dims[1]]-1)))
        elif len(trans[i]) == 3:
            j = ind0[i]
            dims = trans[i]
            res_l = expand_ind([j,], [(0, 1),], [domain[dims[0]], domain[dims[1]]*domain[dims[2]]])
            res[dims[0]] = res_l[0]
            j = res_l[1]
            res_l = expand_ind([j,], [(0, 1),], [domain[dims[1]],domain[dims[2]]])
            res[dims[1]] = res_l[0]
            res[dims[2]] = res_l[1]
    return res
            
dims = get_dims(list(app_domain/shape_node), list(sys_domain))
print("# Application domain: [ {:d}, {:d}, {:d} ]".format(*np.array(app_domain_in, dtype=int)))
print("# Local domain:       [ {:d}, {:d}, {:d} ]".format(*np.array(shape_local, dtype=int)))
print("# Core-al topology:   [ {:d}, {:d}, {:d} ]".format(*np.array(app_domain, dtype=int)))
print("# Nodal domain:       [ {:d}, {:d}, {:d} ]".format(*np.array(shape_node, dtype=int)))
print("# Nodal topology:     [ {:d}, {:d}, {:d} ]".format(*np.array(app_domain/shape_node, dtype=int)))
print("# Network topology:   [ {:d}, {:d}, {:d}, {:d}, {:d} ]".format(*np.array(sys_domain, dtype=int)))
print("# Mode: {:d}".format(mode))
print("# Dim compression:   "+str(dims))
if len(dims) < 3:
    print("Warning: this mesh didn't work.  Consider running on a different partition")
    exit()
#print(sys_domain, app_domain/shape_node, dims, shape_local)

prev = set()
for i in range(int(np.prod(sys_domain)*mode)):
    ix = int(i % app_domain[0])
    iy = int((i / app_domain[0]) % app_domain[1])
    iz = int((i / (app_domain[0] * app_domain[1])) % app_domain[2])
    t = int(ix %  shape_node[0])
    t += int(iy % shape_node[1]) * shape_node[0]
    t += int(iz % shape_node[2]) * shape_node[0] * shape_node[1]

    ix = int(ix / shape_node[0])
    iy = int(iy / shape_node[1])
    iz = int(iz / shape_node[2])
    ind = tuple(list(expand_ind((ix, iy, iz), dims, sys_domain)) + [int(t),])
    print(ind)
    if ind in prev:
        print("Oops, collision", ind)
    else:
        prev.add(ind)


"""

tmp_sys = np.copy(sys_domain)
tmp_app = np.copy(app_domain)

while tmp_app.size > 16:
    dim = np.argmax(tmp_app.shape)


for i in range(8192):
    ix = int(i % app_domain[0])
    iy = int(i / app_domain[0]) % app_domain[1]
    iz = int(i / (app_domain[0]*app_domain[1])) % app_domain[2]
    #t = int((iz % 4)*4 + (iy % 2)*2 + (ix % 2))
    t = int((iz % 4)*4 + (iy % 2)*0 + (ix % 4))
    ix = int(ix / 4)
    iy = int(iy / 1)
    iz = int(iz / 4)

    #ix = int(ix + 8*(iy / 2))
    #a = int(min((ix % 8),7 - (ix % 8)))
    a = ix
    b = int(min((iy % 8),7 - (iy % 8)))
    #b = int(ix / 4)
    c = int(iy / 4)
    #c = int(iy % 4)
    d = int(min((iz % 8),7 - (iz % 8)) )
    e = int(iz / 4)
    print("{:d} {:d} {:d} {:d} {:d} {:d}".format(a,b,c,d,e,t))
    key = (a,b,c,d,e,t)
    if key in prev:
        print("Oops, collision", key)
    else:
        prev.add(key)

"""
