#!/usr/bin/env python
# Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Analytic PP integrals.  See also pyscf/pbc/gto/pesudo/pp.py

For GTH/HGH PPs, see:
    Goedecker, Teter, Hutter, PRB 54, 1703 (1996)
    Hartwigsen, Goedecker, and Hutter, PRB 58, 3641 (1998)
'''

import ctypes
import copy
import numpy
import scipy.special
from pyscf import lib
from pyscf import gto

libpbc = lib.load_library('libpbc')

def get_pp_loc_part1(cell, kpts=None):
    '''PRB, 58, 3641 Eq (1), integrals associated to erf
    '''
    raise NotImplementedError

def get_gth_vlocG_part1(cell, Gv):
    '''PRB, 58, 3641 Eq (5) first term
    '''
    from pyscf.pbc import tools
    coulG = tools.get_coulG(cell, Gv=Gv)
    G2 = numpy.einsum('ix,ix->i', Gv, Gv)
    G0idx = numpy.where(G2==0)[0]

    if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
        vlocG = numpy.zeros((cell.natm, len(G2)))
        for ia in range(cell.natm):
            Zia = cell.atom_charge(ia)
            symb = cell.atom_symbol(ia)
            # Note the signs -- potential here is positive
            vlocG[ia] = Zia * coulG
            if symb in cell._pseudo:
                pp = cell._pseudo[symb]
                rloc, nexp, cexp = pp[1:3+1]
                vlocG[ia] *= numpy.exp(-0.5*rloc**2 * G2)
                # alpha parameters from the non-divergent Hartree+Vloc G=0 term.
                vlocG[ia,G0idx] = -2*numpy.pi*Zia*rloc**2

    elif cell.dimension == 2:
        # The following 2D ewald summation is taken from:
        # Minary, Tuckerman, Pihakari, Martyna J. Chem. Phys. 116, 5351 (2002)
        vlocG = numpy.zeros((cell.natm,len(G2)))
        b = cell.reciprocal_vectors()
        inv_area = numpy.linalg.norm(numpy.cross(b[0], b[1]))/(2*numpy.pi)**2
        lzd2 = cell.vol * inv_area / 2
        lz = lzd2*2.

        G2[G0idx] = 1e200
        Gxy = numpy.linalg.norm(Gv[:,:2],axis=1)
        Gz = abs(Gv[:,2])

        for ia in range(cell.natm):
            Zia = cell.atom_charge(ia)
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                vlocG[ia] = Zia * coulG
                continue

            pp = cell._pseudo[symb]
            rloc, nexp, cexp = pp[1:3+1]

            ew_eta = 1./numpy.sqrt(2)/rloc
            JexpG2 = 4*numpy.pi / G2 * numpy.exp(-G2/(4*ew_eta**2))
            fac = 4*numpy.pi / G2 * numpy.cos(Gz*lzd2)
            JexpG2 -= fac * numpy.exp(-Gxy*lzd2)
            eta_z1 = (ew_eta**2 * lz + Gxy) / (2.*ew_eta)
            eta_z2 = (ew_eta**2 * lz - Gxy) / (2.*ew_eta)
            JexpG2 += fac * 0.5*(numpy.exp(-eta_z1**2)*scipy.special.erfcx(eta_z2) +
                                 numpy.exp(-eta_z2**2)*scipy.special.erfcx(eta_z1))
            vlocG[ia,:] = Zia * JexpG2

            JexpG0 = ( - numpy.pi * lz**2 / 2. * scipy.special.erf( ew_eta * lzd2 )
                       + numpy.pi/ew_eta**2 * scipy.special.erfc(ew_eta*lzd2)
                       - numpy.sqrt(numpy.pi)*lz/ew_eta * numpy.exp( - (ew_eta*lzd2)**2 ) )
            vlocG[ia,G0idx] = -2*numpy.pi*Zia*rloc**2 + Zia*JexpG0
    else:
        raise NotImplementedError('Low dimension ft_type %s'
                                  ' not implemented for dimension %d' %
                                  (cell.low_dim_ft_type, cell.dimension))
    return vlocG

# part2 Vnuc - Vloc
def get_pp_loc_part2(cell, kpts=None):
    '''PRB, 58, 3641 Eq (1), integrals associated to C1, C2, C3, C4
    '''
    from pyscf.pbc.df import incore
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
              'int3c1e_r4_origk', 'int3c1e_r6_origk')
    kptij_lst = numpy.hstack((kpts_lst,kpts_lst)).reshape(-1,2,3)
    buf = 0
    for cn in range(1, 5):
        fakecell = fake_cell_vloc(cell, cn)
        if fakecell.nbas > 0:
            v = incore.aux_e2(cell, fakecell, intors[cn], aosym='s2', comp=1,
                              kptij_lst=kptij_lst)
            buf += numpy.einsum('...i->...', v)

    if isinstance(buf, int):
        if any(cell.atom_symbol(ia) in cell._pseudo for ia in range(cell.natm)):
            pass
        else:
            lib.logger.warn(cell, 'cell.pseudo was specified but its elements %s '
                             'were not found in the system.', cell._pseudo.keys())
        vpploc = [0] * nkpts
    else:
        buf = buf.reshape(nkpts,-1)
        vpploc = []
        for k, kpt in enumerate(kpts_lst):
            v = lib.unpack_tril(buf[k])
            if abs(kpt).sum() < 1e-9:  # gamma_point:
                v = v.real
            vpploc.append(v)
    if kpts is None or numpy.shape(kpts) == (3,):
        vpploc = vpploc[0]
        print('!!!!!! gamma point in original func part2', numpy.shape(kpts) )
    print('!!!!!! gamma point unnoticed in original func part2', numpy.shape(kpts) )
    return vpploc


def get_pp_nl(cell, kpts=None):
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    fakecell, hl_blocks = fake_cell_vnl(cell)
    ppnl_half = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
    nao = cell.nao_nr()
    buf = numpy.empty((3*9*nao), dtype=numpy.complex128)

    # We set this equal to zeros in case hl_blocks loop is skipped
    # and ppnl is returned
    ppnl = numpy.zeros((nkpts,nao,nao), dtype=numpy.complex128)
    for k, kpt in enumerate(kpts_lst):
        offset = [0] * 3
        for ib, hl in enumerate(hl_blocks):
            l = fakecell.bas_angular(ib)
            nd = 2 * l + 1
            hl_dim = hl.shape[0]
            ilp = numpy.ndarray((hl_dim,nd,nao), dtype=numpy.complex128, buffer=buf)
            for i in range(hl_dim):
                p0 = offset[i]
                ilp[i] = ppnl_half[i][k][p0:p0+nd]
                offset[i] = p0 + nd
            ppnl[k] += numpy.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)

    if abs(kpts_lst).sum() < 1e-9:  # gamma_point:
        ppnl = ppnl.real

    if kpts is None or numpy.shape(kpts) == (3,):
        ppnl = ppnl[0]
    return ppnl

##### TODO rm my funcs
def get_pp_loc_part2_atomic(cell, kpts=None):
    '''PRB, 58, 3641 Eq (1), integrals associated to C1, C2, C3, C4
    '''
    '''
        Fake cell created to "house" each coeff.*gaussian (on each atom that has it) 
        for V_loc of pseudopotential (1 fakecell has max 1 gaussian per atom). 
        Ergo different nr of coeff. ->diff. nr of ints to loop and sum over for diff. atoms
        See: "Each term of V_{loc} (erf, C_1, C_2, C_3, C_4) is a gaussian type
        function. The integral over V_{loc} can be transfered to the 3-center
        integrals, in which the auxiliary basis is given by the fake cell."
        Later the cell and fakecells are concatenated to compute 3c overlaps between 
        basis funcs on the real cell & coeff*gaussians on fake cell?
        TODO check if this is correct
        <X_P(r)| sum_A^Nat [ -Z_Acore/r erf(r/sqrt(2)r_loc) + sum_i C_iA (r/r_loc)^(2i-2) ] |X_Q(r)>
        -> 
        int X_P(r - R_P)     :X_P actual basis func. that sits on atom P  ??
        * Ci              :coeff for atom A, coeff nr i
    '''
    from pyscf.pbc.df import incore
    #print('kpts in get_pp_loc_part2_atomic', kpts, type(kpts) )
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    natm = cell.natm
    #print('part2_atomic: atoms in cell', natm)

    intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
              'int3c1e_r4_origk', 'int3c1e_r6_origk')
    kptij_lst = numpy.hstack((kpts_lst,kpts_lst)).reshape(-1,2,3)
    # TODO check why 2 lists of kpts needed for incore.aux_e2
    #print('in pp part2 atomic, kptij_lst', kptij_lst)
    buf = 0
    buf2 = []

    # Loop over coefficients to generate: erf, C1, C2, C3, C4
    # each coeff.-gaussian put in its own fakecell
    # If cn = 0, the erf term is generated.  C_1,..,C_4 are generated with cn = 1..4
    for cn in range(1, 5):
        fakecell = fake_cell_vloc(cell, cn)
        #print('in part2_atomic: cn, natm and nbas in fakecell', cn , fakecell.natm, fakecell.nbas)
        # If the atoms in fake cell have pp (and how many) TODO why nbas=nr shells?
        if fakecell.nbas > 0:
            #print('fakecell for cn=', cn, ' has nbas=', fakecell.nbas)
            # Make a list on which atoms the gaussians sit (for the current Ci coeff.)
            fakebas_atom_lst = []
            for i in range(fakecell.nbas):
                #print('atom symbol', i, fakecell.atom_symbol(i))
                #print('atom symbol real cell', i, cell.atom_symbol(i))
                #print('atom id where i=',i,' gaussian sits on', fakecell.bas_atom(i))
                #print('atom coord where i=',i,' gaussian sits on', fakecell.bas_coord(i))
                #print('real cell atom coord i=',i,' gaussian sits on', cell.atom_coord(fakecell.bas_atom(i)))
                fakebas_atom_lst.append(fakecell.bas_atom(i))
            fakebas_atom_ids = numpy.array(fakebas_atom_lst)
            #print('')
            #print('fakebas_atom_lst ', fakebas_atom_ids)
            # 
            #print('')
            # The int over V_{loc} can be transfered to the 3-center
            # integrals, in which the aux. basis is given by the fake cell.
            v = incore.aux_e2(cell, fakecell, intors[cn], aosym='s2', comp=1,
                              kptij_lst=kptij_lst)
            # v is (naopairs, naux).TODO  where does nkptij dim go/sums over?
            # TODO can naopair be assigned before the loop?
            buf_cn = numpy.zeros( (natm, numpy.shape(v)[0]) )
            #print('buf_cn', numpy.shape(buf_cn))
            #print('v', numpy.shape(v))
            # Put the ints in the right places in the buffer (i.e. assign to the right atom)
            for i_aux, id_atm in enumerate(fakebas_atom_ids):
                buf_cn[id_atm, :] = v[:,i_aux]
            #print('buf_cn', numpy.shape(buf_cn))
            #print('!!!!!')
            #print('      intors       in part2', intors[cn])
            #print('      cn, v        in part2', cn, numpy.shape(v))
            #print('!!!!!')
            # TODO is buf always the same shape, i.e. does not fail when in place addition is performed
            buf += numpy.einsum('...i->...', v)
            # FIXME this is only correct if all the atoms have pp and exact same nr C coeff. that are nonzero
            # i.e. same nr coeff.
            # preallocate size and shape of the array wanted and transfer things correctly
            # for each i transfers into the correct position of the output array
            #buf2 += numpy.einsum('...i->i...', v)
            buf2.append(buf_cn) 
            #print('      buf,buf2        in part2', numpy.shape(buf), len(buf2) )
        # need to check if nbas=0 for all cellselse:
        #    buf2 = 0
    # Add up all ints for each atom. The buf2 is then (natm, naopairs)
    buf2 = numpy.sum(buf2, axis=0)
    #print('shape of buf2 after addition', numpy.shape(buf2))
    print('allclose buf & buf2 ', numpy.allclose(buf, numpy.einsum('ix->x', buf2)) )
    
    # TODO make this valid for my calc, too
    # if fakecell.nbas are all < 0, buf is 0 and we check for elements in the system 
    if isinstance(buf, int):
    #    print('  buf is int   ')
        if any(cell.atom_symbol(ia) in cell._pseudo for ia in range(cell.natm)):
    #        print( '     passed')
            pass
        else:
            lib.logger.warn(cell, 'cell.pseudo was specified but its elements %s '
                             'were not found in the system.', cell._pseudo.keys())
        # list of zeros, length nkpts
        # TODO update here
        vpploc = [0] * nkpts
    #    print('       vploc created as ', numpy.shape(vpploc) )
    else:
        buf = buf.reshape(nkpts,-1)
        buf2 = buf2.reshape(natm, nkpts,-1)
        # indices: k-kpoint, i-atom, x-aopair
        buf2 = numpy.einsum('ikx->kix', buf2)
    #    print('       buf, buf2 reshaped as ', numpy.shape(buf), numpy.shape(buf2) )
        vpploc = []
        vpploc2 = []
        # now have the triangular matrix for each k (triangular of nao x nao is n_aopairs)
        # unpack here to nao x nao for each atom
        for k, kpt in enumerate(kpts_lst):
    #        print('   before unpacking buf[k]  ', numpy.shape(buf[k]) )
            vpploc2_1at_kpts = []
            v = lib.unpack_tril(buf[k])
            for i in range(natm):
    #            print('   before unpacking buf2[k]  ', numpy.shape(buf2[k,i]) )
                v2 = lib.unpack_tril(buf2[k,i,:])
                if abs(kpt).sum() < 1e-9:  # gamma_point:
                    v2 = v2.real
                vpploc2_1at_kpts.append(v2)
    #            print('   after unpacking buf2[k,i] to v[i]  ', numpy.shape(vpploc2_1at_kpts[i]))
    #        print('   after unpacking buf[k] to v  ', numpy.shape(v))
            if abs(kpt).sum() < 1e-9:  # gamma_point:
                v = v.real
            vpploc.append(v)
            vpploc2.append(vpploc2_1at_kpts)
    #        print('gamma point: vpploc appended v ', numpy.shape(vpploc) )
    #        print('gamma point: vpploc2 appended v2 ', numpy.shape(vpploc2) )
    #print('kpts', kpts, numpy.shape(kpts))
    # when only gamma point, the n_k x nao x nao tensor -> nao x nao matrix 
    if kpts is None or numpy.shape(kpts) == (3,):
        print('went here')
        vpploc = vpploc[0]
        vpploc2 = vpploc2[0]
#        print('chosen vpploc[0]', numpy.shape(vpploc))
#        print('chosen vpploc2[0]', numpy.shape(vpploc2))
#        for k, kpt in enumerate(kpts_lst):
#            print('   before unpacking buf[k]  ', numpy.shape(buf[k]))
#            v = lib.unpack_tril(buf[k])
#            print('   after unpacking buf[k] to v  ', numpy.shape(v))
#            if abs(kpt).sum() < 1e-9:  # gamma_point:
#                v = v.real
#            vpploc.append(v)
#            print('gamma point: vpploc appended v ', numpy.shape(vpploc) )
#    if kpts is None or numpy.shape(kpts) == (3,):
#        vpploc = vpploc[0]
#        print('chosen vpploc[0]', numpy.shape(vpploc))
#    print('is vpploc same as vpploc2?')
#    print(numpy.shape(vpploc), numpy.shape(vpploc2))
#    print(numpy.allclose(vpploc, numpy.einsum('kiab->kab', vpploc2) ) )
    return vpploc2



def get_pp_nl_atomic(cell, kpts=None):
    # non local contribution
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    #Generate fake cell for V_{nl}.gaussian function p_i^l Y_{lm}. 
    # Function p_i^l (PRB, 58, 3641 Eq 3) 
    # TODO need to import fake_cell_vnl(cell), _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
    # Y_lm: spherical harmonic, l ang.mom. qnr
    # p_i^l: Gaussian projectors; rela&recipr.space: projectors have a form of Gaussian x polyn.
    # hl_blocks: coeff. for nonlocal projectors.
    # i &j run up to 3 ..never larger atom cores than l=3 (d-orbs)
    ## fake cell for V_nl. Has the atoms but instead of basis funcs, they have projectors 
    ## sitting omn them (confirm). Later the cells are concatenated to compute overlaps between 
    ## basis funcs on the real cell & proj. on fake cell (splitting the int into two ints to multiply)
    ## <X_P(r)| sum_A^Nat sum_i^3 sum_j^3 sum_m^(2l+1) Y_lm(r_A) p_lmi(r_A) h^l_i,j p_lmj(r'_A) Y*_lm(r'_A) |X_Q(r')>
    ## -> (Y_lm implici in p^lm)
    ## int X_P(r - R_P) p^lm_i(r - R_A) dr     :X_P actual basis func. that sits on atom P  
    ## \times h^A,lm_i,j                       :coeff for atom A, lm,ij
    ## int p^lm_j(r' - R_A) X(r' - R_Q) dr     :X_Q actual basis func. that sits on atom Q  
    ## A sums over all atoms since each might have a pp that needs projecting out core sph. harm.
    fakecell, hl_blocks = fake_cell_vnl(cell)
    #'''Vnuc - Vloc'''
    ppnl_half = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
    ppnl_half1 = _int_vnl_atomic(cell, fakecell, hl_blocks, kpts_lst)
    #print('ppnl_half in get_pp_nl_atomic', numpy.shape(ppnl_half) )
    #print('ppnl_half1 in get_pp_nl_atomic', numpy.shape(ppnl_half1) )
    nao = cell.nao_nr()
    natm = cell.natm
    buf = numpy.empty((3*9*nao), dtype=numpy.complex128)

    # We set this equal to zeros in case hl_blocks loop is skipped
    # and ppnl is returned
    ppnl = numpy.zeros((nkpts,nao,nao), dtype=numpy.complex128)
    ppnl1 = numpy.zeros((nkpts,natm,nao,nao), dtype=numpy.complex128)
    for k, kpt in enumerate(kpts_lst):
        offset = [0] * 3
      #  print('hlblocks in get_pp_nl_atomic', numpy.shape(hl_blocks), hl_blocks)
        # hlblocks: for each atom&ang.mom. i have a matrix of coeff. 
        # e.g. 2ang.mom. on two atoms A and B would give A1 1x1 matrix, A2 1x1 matrix, 
        # B1 1x1 matrix, B2 1x1 matrix. if only one kind of a projector for this ang.mom. for this atom
        for ib, hl in enumerate(hl_blocks):
            # NEW this loop is over hlij for all atoms and ang.mom.(i)
            # i think this is shell, hl coeff pair, but could be shell, atom-hl coefficients pair (how likely?)..
            # either way ib is bas_id and called with bas_atom gives diff. atom ids..
          #  print('ib, hl in hl_blocks', ib, hl)
            l = fakecell.bas_angular(ib)
            #print('atom in fakecell that bas sits on', fakecell.bas_atom(ib))
            atm_id_hl = fakecell.bas_atom(ib)
            # TODO use atom id to put into the right ppnl1[nkpts, NATM, nao, nao]
            # l is the angular mom. qnr associated with given basis (ib used here as bas_id)
          #  print('l in get_pp_nl_atomic', type(l), l)
            # orb magn nr 2L+1
            nd = 2 * l + 1
            # dim of the hl (non-local) coeff. array
            hl_dim = hl.shape[0]
          #  print('nd, hldim[0] in get_pp_nl_atomic', nd, hl_dim)
            ilp = numpy.ndarray((hl_dim,nd,nao), dtype=numpy.complex128, buffer=buf)
            for i in range(hl_dim):
          #      print('loop over hldim, i   in get_pp_nl_atomic', i)
          #      print('inside loop over i: hl block nr, value   in get_pp_nl_atomic', ib, hl)
                p0 = offset[i]
          #      print('first p0  in get_pp_nl_atomic', p0, nd)
                # so maybe this is p_il projectors with specific i,j associated with 
                # hl coeff. nrs (r(i) probably)
                # that gets contracted later with  h_lij, p_jl 
                # p0 takes care that the right m,l spherical harm are taken in projectors?
                ilp[i] = ppnl_half[i][k][p0:p0+nd]
          #      print('in get_pp_nl_atomic, ilp[i] = ppnl_half[i][k][p0:p0+nd] i k', i, k)
          #      print('in get_pp_nl_atomic, ppnl_half is (nkpts, ni, nj)', ppnl_half)
          #      print('ilp[i] in loop   in get_pp_nl_atomic', ilp[i])
                offset[i] = p0 + nd
                #print('second offset', offset)
          #   print('shape ilp in get_pp_nl_atomic', numpy.shape(ilp))
            # indices: i,j - hlblock (3 total. TODO why?), l - ang.momentum qnumber,
            # p,q - sph.harm. projectors TODO?
         #   print('hl before einsum in get_pp_nl_atomic', hl)
            # to be able to contract without summing over atoms, ppnl_half need to be xilp or similar (x:atom dim)
            ppnl[k] += numpy.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)
            ppnl1[k,atm_id_hl] += numpy.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)
         #   print('ppnl', numpy.shape(ppnl) )
         #   print('ppnl1', numpy.shape(ppnl1) )
    
    if abs(kpts_lst).sum() < 1e-9:  # gamma_point:
        ppnl = ppnl.real
        ppnl1 = ppnl1.real

    if kpts is None or numpy.shape(kpts) == (3,):
        ppnl = ppnl[0]
        ppnl1 = ppnl1[0]
    return ppnl, ppnl1

def _int_vnl_atomic(cell, fakecell, hl_blocks, kpts):
    '''Vnuc - Vloc'''
    rcut = max(cell.rcut, fakecell.rcut)
    Ls = cell.get_lattice_Ls(rcut=rcut)
    nimgs = len(Ls)
    expkL = numpy.asarray(numpy.exp(1j*numpy.dot(kpts, Ls.T)), order='C')
    nkpts = len(kpts)

    fill = getattr(libpbc, 'PBCnr2c_fill_ks1')
    intopt = lib.c_null_ptr()
    # intopt some class in pycsf
    #print('intopt in _int_vnl_atomic', type(intopt))

    def int_ket(_bas, intor):
        if len(_bas) == 0:
            return []
        # str for which int to get
        intor = cell._add_suffix(intor)
        # i think evt one needs:
        #1-electron integrals from two cells like
        #\langle \mu | intor | \nu \rangle, \mu \in cell1, \nu \in cell2
        # so between real & fakecell (orbitals in cell & sph harm in fakecell to make the orbs ~orth to core shells 
        # represented by pp/cancel out the parts of orbs that are not there due to pp)?
        # 
        #print('int_ket intor in _int_vnl_atomic/int_ket', type(intor) )
        #print('intor in _int_vnl_atomic/int_ket', intor )
        atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                     fakecell._atm, _bas, fakecell._env)
        atm = numpy.asarray(atm, dtype=numpy.int32)
        bas = numpy.asarray(bas, dtype=numpy.int32)
        env = numpy.asarray(env, dtype=numpy.double)
        natm = len(atm)
        # 2*natm in cell, fakecell 
    #    print('natm in _int_vnl_atomic/int_ket', natm)
        nbas = len(bas)
        #bas : int32 ndarray, libcint integral function argument
        # nbas: nr of shells. So the slice is nr of shells in cell, nr of shells in concatenated
        # cell+fakecell, 0, nr of shells in cell 
        # e.g. diam.prim.: [4, 6, 0, 4] 
        shls_slice = (cell.nbas, nbas, 0, cell.nbas)
        #print('shls_slice _int_vnl_atomic/int_ket', shls_slice)
        #print('shls_slice[1] _int_vnl_atomic/int_ket', shls_slice[1])
        #print('shls_slice[2] _int_vnl_atomic/int_ket', shls_slice[2])
        #print('shls_slice[3] _int_vnl_atomic/int_ket', shls_slice[3])
        #print('shls_slice[0] _int_vnl_atomic/int_ket', shls_slice[0])
        # TODO a bit lost here.. but i guess the ints picked her give info which integrals
        # to compute in this concatenated system..?
        ao_loc = gto.moleintor.make_loc(bas, intor)
        #print('ao_loc in _int_vnl_atomic/int_ket ', type(ao_loc), numpy.shape(ao_loc) )
        #print('ao_loc in _int_vnl_atomic/int_ket ', ao_loc)
        ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
        nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
        #print('ni nj in _int_vnl_atomic/int_ket ', numpy.shape(ni), numpy.shape(nj) )
        #print('ni nj in _int_vnl_atomic/int_ket ', ni, nj )
        # 
        # since for diam.prim. ni=2=n_atm=n_ang.moms, nj=8=n_aos
        # i probably need to make sure i get it back like this from drv, too
        out = numpy.empty((nkpts,ni,nj), dtype=numpy.complex128)
        comp = 1

        fintor = getattr(gto.moleintor.libcgto, intor)
        # fintor is function for these ints? <class 'ctypes.CDLL.__init__.<locals>._FuncPtr'>
        #print('in _int_vnl_atomic/int_ket fintor:', type(fintor))

        drv = libpbc.PBCnr2c_drv
        drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p),
            expkL.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*4)(*(shls_slice[:4])),
            ao_loc.ctypes.data_as(ctypes.c_void_p), intopt, lib.c_null_ptr(),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))
        print('out returned by _int_vnl_atomic/int_ket ', numpy.shape(out) )
        return out
    
    # extract how many nl proj. coeff. are there for each atom in fakecell
    hl_dims = numpy.asarray([len(hl) for hl in hl_blocks])
    #print('hl_blocks, hl_dims in _int_vnl_atomic')
    #print(hl_blocks)
    #print(hl_dims)
    #print('hl_dims in _int_vnl_atomic >0, >1, >2')
    # _bas: [atom-id,angular-momentum,num-primitive-GTO,num-contracted-GTO,0,ptr-of-exps,
    # each element reperesents one shell
    # e.g. diam. prim.fakecell: two lists,  [at_id=0 or 1, ang.mom.=0, nr.primGTOs=1, num.contr.GTOs=1,
    # 0, ptr-of-exp=6 or 8, ptr.contract.coeff=7 or 9, ..=0 ] 
    #print('in _int_vnl_atomic, fakecell._bas',fakecell._bas )
    #print(' in _int_vnl_atomic fakecell._bas[hl_dims>0]', fakecell._bas[hl_dims>0])
    #print(' in _int_vnl_atomic fakecell._bas[hl_dims>1]', fakecell._bas[hl_dims>1])
    #print(' in _int_vnl_atomic fakecell._bas[hl_dims>2]', fakecell._bas[hl_dims>2])
    # each element in tuple out is ... computed for one shell, l qnr
    out = (int_ket(fakecell._bas[hl_dims>0], 'int1e_ovlp'),
           int_ket(fakecell._bas[hl_dims>1], 'int1e_r2_origi'),
           int_ket(fakecell._bas[hl_dims>2], 'int1e_r4_origi'))
  #  print('out returned by int_vnl_atomic ', out)
    #print('print out[0] in by int_vnl_atomic', numpy.shape(out[0]), out[0])
    #print('print out[1] in by int_vnl_atomic', out[1])
    #print('print out[2] in by int_vnl_atomic', out[2])
    return out

##### TODO rm my funcs

def fake_cell_vloc(cell, cn=0):
    '''Generate fake cell for V_{loc}.

    Each term of V_{loc} (erf, C_1, C_2, C_3, C_4) is a gaussian type
    function.  The integral over V_{loc} can be transfered to the 3-center
    integrals, in which the auxiliary basis is given by the fake cell.

    The kwarg cn indiciates which term to generate for the fake cell.
    If cn = 0, the erf term is generated.  C_1,..,C_4 are generated with cn = 1..4
    '''
    fake_env = [cell.atom_coords().ravel()]
    fake_atm = cell._atm.copy()
    fake_atm[:,gto.PTR_COORD] = numpy.arange(0, cell.natm*3, 3)
    ptr = cell.natm * 3
    fake_bas = []
    half_sph_norm = .5/numpy.sqrt(numpy.pi)
    for ia in range(cell.natm):
        if cell.atom_charge(ia) == 0:  # pass ghost atoms
            continue

        symb = cell.atom_symbol(ia)
        if cn == 0:
            if symb in cell._pseudo:
                pp = cell._pseudo[symb]
                rloc, nexp, cexp = pp[1:3+1]
                alpha = .5 / rloc**2
            else:
                alpha = 1e16
            norm = half_sph_norm / gto.gaussian_int(2, alpha)
            fake_env.append([alpha, norm])
            fake_bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
            ptr += 2
        elif symb in cell._pseudo:
            pp = cell._pseudo[symb]
            rloc, nexp, cexp = pp[1:3+1]
            if cn <= nexp:
                alpha = .5 / rloc**2
                norm = cexp[cn-1]/rloc**(cn*2-2) / half_sph_norm
                fake_env.append([alpha, norm])
                fake_bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
                ptr += 2

    fakecell = copy.copy(cell)
    fakecell._atm = numpy.asarray(fake_atm, dtype=numpy.int32)
    fakecell._bas = numpy.asarray(fake_bas, dtype=numpy.int32)
    fakecell._env = numpy.asarray(numpy.hstack(fake_env), dtype=numpy.double)
    return fakecell

# sqrt(Gamma(l+1.5)/Gamma(l+2i+1.5))
_PLI_FAC = 1/numpy.sqrt(numpy.array((
    (1, 3.75 , 59.0625  ),  # l = 0,
    (1, 8.75 , 216.5625 ),  # l = 1,
    (1, 15.75, 563.0625 ),  # l = 2,
    (1, 24.75, 1206.5625),  # l = 3,
    (1, 35.75, 2279.0625),  # l = 4,
    (1, 48.75, 3936.5625),  # l = 5,
    (1, 63.75, 6359.0625),  # l = 6,
    (1, 80.75, 9750.5625))))# l = 7,

def fake_cell_vnl(cell):
    '''Generate fake cell for V_{nl}.

    gaussian function p_i^l Y_{lm}
    '''
    # TODO in (nl, rl, hl) hl is the proj part hproj
    '''{ atom: ( (nelec_s, nele_p, nelec_d, ...),
                rloc, nexp, (cexp_1, cexp_2, ..., cexp_nexp),
                nproj_types,
                (r1, nproj1, ( (hproj1[1,1], hproj1[1,2], ..., hproj1[1,nproj1]),
                               (hproj1[2,1], hproj1[2,2], ..., hproj1[2,nproj1]),
                               ...
                               (hproj1[nproj1,1], hproj1[nproj1,2], ...        ) )),
                (r2, nproj2, ( (hproj2[1,1], hproj2[1,2], ..., hproj2[1,nproj1]),
                ... ) )
                )
        ... }]i
    '''
    fake_env = [cell.atom_coords().ravel()]
    fake_atm = cell._atm.copy()
    fake_atm[:,gto.PTR_COORD] = numpy.arange(0, cell.natm*3, 3)
    ptr = cell.natm * 3
    fake_bas = []
    hl_blocks = []
    for ia in range(cell.natm):
        if cell.atom_charge(ia) == 0:  # pass ghost atoms
            #print('in fake_cell_vnl: ghost atomed passed..')
            continue

        symb = cell.atom_symbol(ia)
        #print('in fake_cell_vnl: ia(#atm), atm symbol', ia, symb)
        if symb in cell._pseudo:
            #print('in fake_cell_vnl: symb was in cell._psuedo')
            pp = cell._pseudo[symb]
            #print('in fake_cell_vnl: pp from cell._pseudo', type(pp), numpy.shape(pp), pp)
            # nproj_types = pp[4]
            for l, (rl, nl, hl) in enumerate(pp[5:]):
                #print('in fake_cell_vnl: l, (rl, nl, hl) ', l, (rl, nl, hl) )
                if nl > 0:
                    alpha = .5 / rl**2
                    norm = gto.gto_norm(l, alpha)
                    fake_env.append([alpha, norm])
                    fake_bas.append([ia, l, 1, 1, 0, ptr, ptr+1, 0])

#
# Function p_i^l (PRB, 58, 3641 Eq 3) is (r^{2(i-1)})^2 square normalized to 1.
# But here the fake basis is square normalized to 1.  A factor ~ p_i^l / p_1^l
# is attached to h^l_ij (for i>1,j>1) so that (factor * fake-basis * r^{2(i-1)})
# is normalized to 1.  The factor is
#       r_l^{l+(4-1)/2} sqrt(Gamma(l+(4-1)/2))      sqrt(Gamma(l+3/2))
#     ------------------------------------------ = ----------------------------------
#      r_l^{l+(4i-1)/2} sqrt(Gamma(l+(4i-1)/2))     sqrt(Gamma(l+2i-1/2)) r_l^{2i-2}
#
                    fac = numpy.array([_PLI_FAC[l,i]/rl**(i*2) for i in range(nl)])
                    #print('in fake_cell_vnl: fac', fac)
                    hl = numpy.einsum('i,ij,j->ij', fac, numpy.asarray(hl), fac)
                    #print('in fake_cell_vnl: hl contracted with fac', hl)
                    hl_blocks.append(hl)
                    ptr += 2

    fakecell = copy.copy(cell)
    fakecell._atm = numpy.asarray(fake_atm, dtype=numpy.int32)
    fakecell._bas = numpy.asarray(fake_bas, dtype=numpy.int32)
    fakecell._env = numpy.asarray(numpy.hstack(fake_env), dtype=numpy.double)
    return fakecell, hl_blocks

def _int_vnl(cell, fakecell, hl_blocks, kpts):
    '''Vnuc - Vloc'''
    rcut = max(cell.rcut, fakecell.rcut)
    Ls = cell.get_lattice_Ls(rcut=rcut)
    nimgs = len(Ls)
    expkL = numpy.asarray(numpy.exp(1j*numpy.dot(kpts, Ls.T)), order='C')
    nkpts = len(kpts)

    fill = getattr(libpbc, 'PBCnr2c_fill_ks1')
    intopt = lib.c_null_ptr()

    def int_ket(_bas, intor):
        if len(_bas) == 0:
            return []
        intor = cell._add_suffix(intor)
        atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                     fakecell._atm, _bas, fakecell._env)
        atm = numpy.asarray(atm, dtype=numpy.int32)
        bas = numpy.asarray(bas, dtype=numpy.int32)
        env = numpy.asarray(env, dtype=numpy.double)
        natm = len(atm)
        nbas = len(bas)
        shls_slice = (cell.nbas, nbas, 0, cell.nbas)
        ao_loc = gto.moleintor.make_loc(bas, intor)
        ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
        nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
        out = numpy.empty((nkpts,ni,nj), dtype=numpy.complex128)
        comp = 1

        fintor = getattr(gto.moleintor.libcgto, intor)

        drv = libpbc.PBCnr2c_drv
        drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p),
            expkL.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*4)(*(shls_slice[:4])),
            ao_loc.ctypes.data_as(ctypes.c_void_p), intopt, lib.c_null_ptr(),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))
        return out

    hl_dims = numpy.asarray([len(hl) for hl in hl_blocks])
    out = (int_ket(fakecell._bas[hl_dims>0], 'int1e_ovlp'),
           int_ket(fakecell._bas[hl_dims>1], 'int1e_r2_origi'),
           int_ket(fakecell._bas[hl_dims>2], 'int1e_r4_origi'))
    return out

