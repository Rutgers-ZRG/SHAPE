import numpy as np
from scipy.optimize import linear_sum_assignment
import rcovdata
# import numba

# @numba.jit()
def readvasp(vp):
    buff = []
    with open(vp) as f:
        for line in f:
            buff.append(line.split())

    lat = np.array(buff[2:5], float) 
    try:
        typt = np.array(buff[5], int)
    except:
        del(buff[5])
        typt = np.array(buff[5], int)
    nat = sum(typt)
    pos = np.array(buff[7:7 + nat], float)
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)
    rxyz = np.dot(pos, lat)
    # rxyz = pos
    return lat, rxyz, types

# @numba.jit()
def read_types(vp):
    buff = []
    with open(vp) as f:
        for line in f:
            buff.append(line.split())
    try:
        typt = np.array(buff[5], int)
    except:
        del(buff[5])
        typt = np.array(buff[5], int)
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)
    return types

# @numba.jit()
def get_gom(lseg, rxyz, alpha, amp):
    # s orbital only lseg == 1
    nat = len(rxyz)    
    if lseg == 1:
        om = np.zeros((nat, nat))
        mamp = np.zeros((nat, nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                t1 = alpha[iat] * alpha[jat]
                t2 = alpha[iat] + alpha[jat]
                om[iat][jat] = np.sqrt(2.0*np.sqrt(t1)/t2)**3 * np.exp(-t1/t2*d2)
                mamp[iat][jat] = amp[iat]*amp[jat]

    else:
        # for both s and p orbitals
        om = np.zeros((4*nat, 4*nat))
        mamp = np.zeros((4*nat, 4*nat))
        for iat in range(nat):
            for jat in range(nat):
                d = rxyz[iat] - rxyz[jat]
                d2 = np.vdot(d, d)
                t1 = alpha[iat] * alpha[jat]
                t2 = alpha[iat] + alpha[jat]
                sij = np.sqrt(2.0*np.sqrt(t1)/t2)**3 * np.exp(-t1/t2*d2)
                om[4*iat][4*jat] = sij
                mamp[4*iat][4*jat] = amp[iat]*amp[jat]
                
                # <s_i | p_j>
                stv = -2.0 * np.sqrt(alpha[jat])*alpha[iat] * sij
                om[4*iat][4*jat+1] = stv * d[0] 
                om[4*iat][4*jat+2] = stv * d[1] 
                om[4*iat][4*jat+3] = stv * d[2]  
                
                mamp[4*iat][4*jat+1] = amp[iat]*amp[jat]
                mamp[4*iat][4*jat+2] = amp[iat]*amp[jat]
                mamp[4*iat][4*jat+3] = amp[iat]*amp[jat]
                # <p_i | s_j> 
                stv = -2.0 * np.sqrt(alpha[iat])*alpha[jat] * sij
                om[4*iat+1][4*jat] = stv * d[0] 
                om[4*iat+2][4*jat] = stv * d[1] 
                om[4*iat+3][4*jat] = stv * d[2] 

                mamp[4*iat+1][4*jat] = amp[iat]*amp[jat]
                mamp[4*iat+2][4*jat] = amp[iat]*amp[jat]
                mamp[4*iat+3][4*jat] = amp[iat]*amp[jat]

                # <p_i | p_j>
                # stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                stv = 2.0 * sqrt(t1)/t2 * sij
                sx = -2.0*t1/t2
                om[4*iat+1][4*jat+1] = stv * (sx * d[0] * d[0] + 1.0) 
                om[4*iat+1][4*jat+2] = stv * (sx * d[1] * d[0]      ) 
                om[4*iat+1][4*jat+3] = stv * (sx * d[2] * d[0]      ) 
                om[4*iat+2][4*jat+1] = stv * (sx * d[0] * d[1]      ) 
                om[4*iat+2][4*jat+2] = stv * (sx * d[1] * d[1] + 1.0) 
                om[4*iat+2][4*jat+3] = stv * (sx * d[2] * d[1]      ) 
                om[4*iat+3][4*jat+1] = stv * (sx * d[0] * d[2]      ) 
                om[4*iat+3][4*jat+2] = stv * (sx * d[1] * d[2]      ) 
                om[4*iat+3][4*jat+3] = stv * (sx * d[2] * d[2] + 1.0) 

                mamp[4*iat+1][4*jat+1] = amp[iat]*amp[jat]
                mamp[4*iat+1][4*jat+2] = amp[iat]*amp[jat]
                mamp[4*iat+1][4*jat+3] = amp[iat]*amp[jat]
                mamp[4*iat+2][4*jat+1] = amp[iat]*amp[jat]
                mamp[4*iat+2][4*jat+2] = amp[iat]*amp[jat]
                mamp[4*iat+2][4*jat+3] = amp[iat]*amp[jat]
                mamp[4*iat+3][4*jat+1] = amp[iat]*amp[jat]
                mamp[4*iat+3][4*jat+2] = amp[iat]*amp[jat]
                mamp[4*iat+3][4*jat+3] = amp[iat]*amp[jat]
    
    # for i in range(len(om)):
    #     for j in range(len(om)):
    #         if abs(om[i][j] - om[j][i]) > 1e-6:
    #             print ("ERROR", i, j, om[i][j], om[j][i])
    if check_symmetric(om*mamp) and check_pos_def(om*mamp):
        return om, mamp
    else:
        raise Exception("Gaussian Overlap Matrix is not symmetric and positive definite!")

# @numba.jit()
def check_symmetric(A, rtol = 1e-05, atol = 1e-08):
    return np.allclose(A, A.T, rtol = rtol, atol = atol)

# @numba.jit()
def check_pos_def(A):
    eps = np.finfo(float).eps
    B = A + eps*np.identity(len(A))
    if np.array_equal(B, B.T):
        try:
            np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

# @numba.jit()
def get_dgom(gom, amp, damp, rxyz, alpha, icenter):
    
    # <s|s>
    nat = len(gom)
    dgom = np.zeros((nat, 3, nat, nat))
    for jat in range(nat):
        for iat in range(nat):
            d = rxyz[iat] - rxyz[jat]
            d2 = np.dot(d, d)
            t1 = alpha[iat]*alpha[jat]
            t2 = alpha[iat] + alpha[jat]
            tt = -2 * t1/t2
            dic = rxyz[iat] - rxyz[icenter]
            djc = rxyz[jat] - rxyz[icenter]

            pij = amp[iat]*amp[jat]
            dipj = damp[iat] * amp[jat]
            djpi = damp[jat] * amp[iat]

            di = pij * tt * gom[iat][jat] * d + dipj * gom[iat][jat] * dic
            dj = -pij * tt * gom[iat][jat] * d + djpi * gom[iat][jat] * djc
            dc = -dipj * gom[iat][jat] * dic - djpi * gom[iat][jat] * djc

            for i in range(3):
                dgom[iat][i][iat][jat] += di[i]
                dgom[jat][i][iat][jat] += dj[i]
                dgom[icenter][i][iat][jat] += dc[i]
    return dgom

# @numba.jit()
def get_fp_nonperiodic(rxyz, znucls):
    rcov = []
    amp = [1.0] * len(rxyz)
    for x in znucls:
        rcov.append(rcovdata.rcovdata[x][2])
    gom = get_gom(1, rxyz, rcov, amp)
    fp = np.linalg.eigvals(gom)
    fp = sorted(fp)
    fp = np.array(fp, float)
    return fp

# @numba.jit()
def get_fpdist_nonperiodic(fp1, fp2):
    d = fp1 - fp2
    return np.sqrt(np.vdot(d, d))

# @numba.jit()
def get_fp(lat, rxyz, types, znucl,
           contract,
           ldfp,
           ntyp,
           nx,
           lmax,
           cutoff):
    if lmax == 0:
        lseg = 1
        l = 1
    else:
        lseg = 4
        l = 2

    #Modified so that now a float is returned and converted into an int
    ixyzf = get_ixyz(lat, cutoff)
    ixyz = int(ixyzf) + 1
    NC = 2
    wc = cutoff / np.sqrt(2.* NC)
    fc = 1.0 / (2.0 * NC * wc**2)
    nat = len(rxyz)
    cutoff2 = cutoff**2

    n_sphere_list = []
    lfp = np.empty((nat,lseg*nx))
    sfp = []
    dfp = np.zeros((nat, nat, 3, lseg*nx))
    for iat in range(nat):
        rxyz_sphere = []
        rcov_sphere = []
        alpha = []
        ind = [0] * (lseg * nx)
        indori = []
        amp = []
        damp = []
        xi, yi, zi = rxyz[iat]
        n_sphere = 0
        for jat in range(nat):
            rcovjur = rcovdata.rcovdata
            index11 = int(types[jat] - 1)
            index1 = int(znucl[index11])
            rcovj = rcovjur[index1][2]
            for ix in range(-ixyz, ixyz+1):
                for iy in range(-ixyz, ixyz+1):
                    for iz in range(-ixyz, ixyz+1):
                        xj = rxyz[jat][0] + ix*lat[0][0] + iy*lat[1][0] + iz*lat[2][0]
                        yj = rxyz[jat][1] + ix*lat[0][1] + iy*lat[1][1] + iz*lat[2][1]
                        zj = rxyz[jat][2] + ix*lat[0][2] + iy*lat[1][2] + iz*lat[2][2]
                        d2 = (xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2
                        if d2 <= cutoff2:
                            n_sphere += 1
                            if n_sphere > nx:
                                raise Exception("FP WARNING: Cutoff radius is too large, \
                                                increase nx or decrease cutoff.")
                            # amp.append((1.0-d2*fc)**NC)
                            # nd2 = d2/cutoff2
                            ampt = (1.0-d2*fc)**(NC-1)
                            amp.append(ampt * (1.0-d2*fc))
                            damp.append(-2.0 * fc * NC * ampt)
                            indori.append(jat)
                            # amp.append(1.0)
                            # print (1.0-d2*fc)**NC
                            rxyz_sphere.append([xj, yj, zj])
                            rcov_sphere.append(rcovj)
                            alpha.append(0.5 / rcovj**2)
                            if jat == iat and ix == 0 and iy == 0 and iz == 0:
                                ityp_sphere = 0
                                icenter = n_sphere-1
                            else:
                                ityp_sphere = types[jat]
                            for il in range(lseg):
                                if il == 0:
                                    # print len(ind)
                                    # print ind
                                    # print il+lseg*(n_sphere-1)
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l
                                else:
                                    ind[il+lseg*(n_sphere-1)] = ityp_sphere * l + 1
        n_sphere_list.append(n_sphere)
        rxyz_sphere = np.array(rxyz_sphere)
        # full overlap matrix
        nid = lseg * n_sphere
        (gom, mamp) = get_gom(lseg, rxyz_sphere, alpha, amp)
        gomamp = gom * mamp
        val, vec = np.linalg.eigh(gomamp)
        # val = np.real(val)
        fp0 = np.zeros(nx*lseg)
        for i in range(len(val)):
            # print (val[i])
            fp0[i] = val[len(val)-1-i]
        fp0 = fp0/np.linalg.norm(fp0)
        np.append(lfp,fp0)
        lfp[iat] = fp0
        # pvec = np.real(np.transpose(vec)[0])

        vectmp = np.transpose(vec)
        vecs = []
        for i in range(len(vectmp)):
            vecs.append(vectmp[len(vectmp)-1-i])

        pvec = vecs[0]
        # derivative
        if ldfp:
            dgom = get_dgom(gom, amp, damp, rxyz_sphere, alpha, icenter)
            # print (dgom[0][0][0])
            dvdr = np.zeros((n_sphere, n_sphere, 3))
            for iats in range(n_sphere):
                for iorb in range(n_sphere):
                    vvec = vecs[iorb]
                    for ik in range(3):
                        matt = dgom[iats][ik]
                        vv1 = np.dot(vvec, matt)
                        vv2 = np.dot(vv1, np.transpose(vvec))
                        dvdr[iats][iorb][ik] = vv2
            for iats in range(n_sphere):
                iiat = indori[iats]
                for iorb in range(n_sphere):
                    for ik in range(3):
                        dfp[iat][iiat][ik][iorb] += dvdr[iats][iorb][ik]

        # contracted overlap matrix
        if contract:
            nids = l * (ntyp + 1)
            omx = np.zeros((nids, nids))
            for i in range(nid):
                for j in range(nid):
                    # print ind[i], ind[j]
                    omx[ind[i]][ind[j]] = omx[ind[i]][ind[j]] + pvec[i] * gom[i][j] * pvec[j]
            # for i in range(nids):
            #     for j in range(nids):
            #         if abs(omx[i][j] - omx[j][i]) > 1e-6:
            #             print ("ERROR", i, j, omx[i][j], omx[j][i])
            # print omx
            sfp0 = np.linalg.eigvals(omx)
            sfp.append(sorted(sfp0))


    # print ("n_sphere_min", min(n_sphere_list))
    # print ("n_shpere_max", max(n_sphere_list))

    if contract:
        sfp = np.array(sfp, float)
        return sfp, dfp

    else:
        lfp = np.array(lfp, float)
        return lfp, dfp

# @numba.jit()
def get_ixyz(lat, cutoff):
    lat2 = np.matmul(lat, np.transpose(lat))
    # print lat2
    vec = np.linalg.eigvals(lat2)
    # print (vec)
    ixyz = int(np.sqrt(1.0/max(vec))*cutoff) + 1
    return ixyz

# @numba.jit()
def get_fpdist(ntyp, types, fp1, fp2, mx=False):
    nat, lenfp = np.shape(fp1)
    fpd = 0.0
    for ityp in range(ntyp):
        itype = ityp + 1
        MX = np.zeros((nat, nat))
        for iat in range(nat):
            if types[iat] == itype:
                for jat in range(nat):
                    if types[jat] == itype:
                        tfpd = fp1[iat] - fp2[jat]
                        MX[iat][jat] = np.sqrt(np.vdot(tfpd, tfpd))

        row_ind, col_ind = linear_sum_assignment(MX)
        # print(row_ind, col_ind)
        total = MX[row_ind, col_ind].sum()
        fpd += total

    fpd = fpd / nat
    if mx:
        return fpd, col_ind
    else:
        return fpd

# @numba.jit()
def get_ef(fp, dfp, ntyp, types):
    nat = len(fp)
    e = 0.
    for ityp in range(ntyp):
        itype = ityp + 1
        e0 = 0.
        for i in range(nat):
            for j in range(nat):
                if types[i] == itype and types[j] == itype:
                    vij = fp[i] - fp[j]
                    t = np.dot(vij, vij)
                    e0 += t
        # print ("e0", e0)
        e += e0
    # print ("e", e)

    force = np.zeros((nat, 3))

    for k in range(nat):
        for ityp in range(ntyp):
            itype = ityp + 1
            for i in range(nat):
                for j in range(nat):
                    if  types[i] == itype and types[j] == itype:
                        vij = fp[i] - fp[j]
                        dvij = dfp[i][k] - dfp[j][k]
                        for l in range(3):
                            t = -2 * np.dot(vij, dvij[l])
                            force[k][l] += t
    return e, force


# @numba.jit()
def get_fpe(fp, ntyp, types):
    nat = len(fp)
    e = 0.
    for ityp in range(ntyp):
        itype = ityp + 1
        e0 = 0.
        for i in range(nat):
            for j in range(nat):
                if types[i] == itype and types[j] == itype:
                    vij = fp[i] - fp[j]
                    t = np.dot(vij, vij)
                    e0 += t
        e += e0
    return e

# @numba.jit()
def get_stress(lat, rxyz, types, znucl,
               contract,
               ntyp,
               nx,
               lmax,
               cutoff):
    pos = np.dot(rxyz, np.linalg.inv(lat))
    rxyz_delta = np.zeros_like(rxyz)
    cell_vol = np.linalg.det(lat)
    # stress = np.zeros(6)
    stress = np.zeros((3,3))
    step_size = 1.e-4
    strain_delta_tmp = step_size*(np.random.randint(1, 9999, (3, 3))/9999)
    # print (strain_delta_tmp)
    # Make strain tensor symmetric
    strain_delta = 0.5*(strain_delta_tmp + strain_delta_tmp.T - \
                        np.diag(np.diag(strain_delta_tmp))) 
    # print (strain_delta)
    rxyz_ratio = np.diag(np.ones(3))
    rxyz_ratio_new = rxyz_ratio.copy()
    for m in range(3):
        for n in range(3):
            h = strain_delta[m][n]
            rxyz_ratio_left = np.diag(np.ones(3))
            rxyz_ratio_right = np.diag(np.ones(3))
            rxyz_ratio_left[m][n] = rxyz_ratio[m][n] - h
            rxyz_ratio_right[m][n] = rxyz_ratio[m][n] + h
            lat_left = np.dot(lat, rxyz_ratio_left)
            lat_right = np.dot(lat, rxyz_ratio_right)
            rxyz_left = np.dot(pos, lat_left)
            rxyz_right = np.dot(pos, lat_right)
            ldfp = False
            fp_left, dfptmp = get_fp(lat_left, rxyz_left, types, znucl, \
                                     contract, ldfp, ntyp, nx, lmax, cutoff)
            fp_right, dfptmp = get_fp(lat_right, rxyz_right, types, znucl, \
                                      contract, ldfp, ntyp, nx, lmax, cutoff)
            fp_energy_left = get_fpe(fp_left, ntyp, types)
            fp_energy_right = get_fpe(fp_right, ntyp, types)
            # fp_energy_left = get_fp_energy(lat_left, rxyz_left, types, contract, \
            #                                ntyp, nx, lmax, znucl, cutoff)
            # fp_energy_right = get_fp_energy(lat_right, rxyz_right, types, contract, \
            #                                 ntyp, nx, lmax, znucl, cutoff)
            # print(fp_energy_left, fp_energy_right, h, cell_vol)
            stress[m][n] = - (fp_energy_right - fp_energy_left)/(2.0*h*cell_vol)
        #################
        
    #################
    # print (stress)
    stress = stress.flat[[0, 4, 8, 5, 2, 1]]
    return stress