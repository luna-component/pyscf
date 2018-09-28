#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

'''
Non-relativistic magnetizability tensor for DFT
(In testing)

Refs:
[1] R. Cammi, J. Chem. Phys., 109, 3185 (1998)
[2] Todd A. Keith, Chem. Phys., 213, 123 (1996)
'''


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import jk
from pyscf.dft import numint
from pyscf.prop.nmr import rhf as rhf_nmr
from pyscf.prop.nmr import rks as rks_nmr
from pyscf.prop.magnetizability import rhf as rhf_mag


def dia(magobj, gauge_orig=None):
    mol = magobj.mol
    mf = magobj._scf
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    orbo = mo_coeff[:,mo_occ > 0]
    dm0 = numpy.dot(orbo, orbo.T) * 2
    dm0 = lib.tag_array(dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)
    dme0 = numpy.dot(orbo * mo_energy[mo_occ > 0], orbo.T) * 2

    if gauge_orig is None:
        h2 = mol.intor('int1e_rr_origj', comp=9)
    else:
        mol.set_common_origin(gauge_orig)
        h2 = mol.intor('int1e_rr', comp=9)

    if getattr(magobj._scf, 'with_x2c', None):
        raise NotImplementedError('X2C for magnetizability')

    if getattr(magobj._scf, 'with_qmmm', None):
        raise NotImplementedError('Magnetizability with QM/MM')

    if getattr(magobj._scf, 'with_solvent', None):
        raise NotImplementedError('Magnetizability with Solvent')

    e2 = numpy.einsum('xpq,qp->x', h2, dm0)
    diag = [0, 4, 8]  # XX, YY, ZZ
    e2[diag] -= e2[diag].sum()
    e2 *= -.25

    if gauge_orig is not None:
        return e2

    # If gauge_orig is None, computing the GIAO contributions
    # 1-electron operator integrals
    e2 += numpy.einsum('qp,xpq->x', dm0, mol.intor('int1e_grjxp', comp=9))
    e2 += numpy.einsum('qp,xpq->x', dm0, mol.intor('int1e_ggkin', comp=9))
    e2 += numpy.einsum('qp,xpq->x', dm0, mol.intor('int1e_ggnuc', comp=9))
    if mol.has_ecp():
        raise NotImplementedError
        e2+= numpy.einsum('qp,xpq->x', dm0, mol.intor('ECPscalar_ggnuc', comp=9))
    e2 -= numpy.einsum('qp,xpq->x', dme0, mol.intor('int1e_ggovlp', comp=9))

    # 2nd order Vxc integrals
    grids = mf.grids
    ni = mf._numint
    xc_code = mf.xc
    xctype = ni._xc_type(xc_code)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc_code, mol.spin)

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm0, hermi=1)
    ngrids = len(grids.weights)
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    BLKSIZE = numint.BLKSIZE
    blksize = min(int(max_memory/12*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)

    vmat = numpy.zeros((3,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory,
                                 blksize=blksize):
            rho = make_rho(0, ao, mask, 'LDA')
            vxc = ni.eval_xc(xc_code, rho, 0, deriv=1)[1]
            vrho = vxc[0]
            r_ao = numpy.einsum('pi,px->pxi', ao, coords)
            aow = numpy.einsum('pxi,p,p->pxi', r_ao, weight, vrho)
            vmat += lib.einsum('pxi,pyj->xyij', r_ao, aow)
            rho = vxc = vrho = aow = None

    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory,
                                 blksize=blksize):
            rho = make_rho(0, ao, mask, 'GGA')
            vxc = ni.eval_xc(xc_code, rho, 0, deriv=1)[1]
            wv = numint._rks_gga_wv0(rho, vxc, weight)

            # Computing \nabla (r * AO) = r * \nabla AO + [\nabla,r]_- * AO
            r_ao = numpy.einsum('npi,px->npxi', ao, coords)
            r_ao[1,:,0] += ao[0]
            r_ao[2,:,1] += ao[0]
            r_ao[3,:,2] += ao[0]

            aow = numpy.einsum('npxi,np->pxi', r_ao, wv)
            vmat += lib.einsum('pxi,pyj->xyij', r_ao[0], aow)
            rho = vxc = vrho = vsigma = wv = aow = None

        vmat = vmat + vmat.transpose(0,1,3,2)

    else:
        raise NotImplementedError('meta-GGA')

    # Add the factor i/2 (Ri-Rj) of the GIAO phase e^{i/2 (Ri-Rj) times r}
    ao_coords = _get_ao_coords(mol)
    Rx = .5 * (ao_coords[:,0:1] - ao_coords[:,0])
    Ry = .5 * (ao_coords[:,1:2] - ao_coords[:,1])
    Rz = .5 * (ao_coords[:,2:3] - ao_coords[:,2])
    vxc20 = numpy.empty_like(vmat)
    vxc20[0]  = Ry * vmat[2] - Rz * vmat[1]
    vxc20[1]  = Rz * vmat[0] - Rx * vmat[2]
    vxc20[2]  = Rx * vmat[1] - Ry * vmat[0]
    vxc20, vmat = vmat, vxc20
    vxc20[:,0]  = Ry * vmat[:,2] - Rz * vmat[:,1]
    vxc20[:,1]  = Rz * vmat[:,0] - Rx * vmat[:,2]
    vxc20[:,2]  = Rx * vmat[:,1] - Ry * vmat[:,0]
    vxc20 *= -1

    e2 += numpy.einsum('qp,xypq->xy', dm0, vxc20).ravel()
    vxc20 = vmat = None

    # Handle the hybrid functional and the range-separated functional
    if abs(hyb) > 1e-10:
        vs = jk.get_jk(mol, [dm0]*3, ['ijkl,ji->s2kl',
                                      'ijkl,jk->s1il',
                                      'ijkl,li->s1kj'],
                       'int2e_gg1', 's4', 9, hermi=1)
        e2 += numpy.einsum('xpq,qp->x', vs[0], dm0)
        e2 -= numpy.einsum('xpq,qp->x', vs[1], dm0) * .25 * hyb
        e2 -= numpy.einsum('xpq,qp->x', vs[2], dm0) * .25 * hyb
        vk = jk.get_jk(mol, dm0, 'ijkl,jk->s1il',
                       'int2e_g1g2', 'aa4', 9, hermi=0)
        e2 -= numpy.einsum('xpq,qp->x', vk, dm0) * .5 * hyb

        if abs(omega) > 1e-10:
            with mol.with_range_coulomb(omega):
                vs = jk.get_jk(mol, [dm0]*2, ['ijkl,jk->s1il',
                                              'ijkl,li->s1kj'],
                               'int2e_gg1', 's4', 9, hermi=1)
                e2 -= numpy.einsum('xpq,qp->x', vs[0], dm0) * .25 * (alpha-hyb)
                e2 -= numpy.einsum('xpq,qp->x', vs[1], dm0) * .25 * (alpha-hyb)
                vk = jk.get_jk(mol, dm0, 'ijkl,jk->s1il',
                               'int2e_g1g2', 'aa4', 9, hermi=0)
                e2 -= numpy.einsum('xpq,qp->x', vk, dm0) * .5 * (alpha-hyb)

    else:
        vj = jk.get_jk(mol, dm0, 'ijkl,ji->s2kl',
                       'int2e_gg1', 's4', 9, hermi=1)
        e2 += numpy.einsum('xpq,qp->x', vj, dm0)

    return e2.reshape(3, 3)


def _get_ao_coords(mol):
    atom_coords = mol.atom_coords()
    nao = mol.nao_nr()
    ao_coords = numpy.empty((nao, 3))
    aoslices = mol.aoslice_by_atom()
    for atm_id, (ish0, ish1, i0, i1) in enumerate(aoslices):
        ao_coords[i0:i1] = atom_coords[atm_id]
    return ao_coords


class Magnetizability(rhf_mag.Magnetizability):

    get_fock = rhf_nmr.get_fock

    dia = dia

    def solve_mo1(self, mo_energy=None, mo_coeff=None, mo_occ=None,
                  h1=None, s1=None, with_cphf=None):
        if with_cphf is None:
            with_cphf = self.cphf
        libxc = self._scf._numint.libxc
        with_cphf = with_cphf and libxc.is_hybrid_xc(self._scf.xc)
        return rhf_nmr.solve_mo1(self, mo_energy, mo_coeff, mo_occ,
                                 h1, s1, with_cphf)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['Ne' , (0. , 0. , 0.)], ]
    mol.basis='631g'
    mol.build()
#
#    mf = dft.RKS(mol).run()
#    mag = Magnetizability(mf).kernel()
#    print(mag)
#
#    mf.set(xc = 'b3lyp').run()
#    mag = Magnetizability(mf).kernel()
#    print(mag)

    mol.atom = [
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.  )], ]
    mol.basis = '6-31g'
    mol.build()

    mf = dft.RKS(mol).set(xc='lda,vwn_rpa').run()
#    mf = dft.RKS(mol).set(xc='b3lypg').run()
    mag = Magnetizability(mf).kernel()
    print(mag)

#    mol.basis = 'ccpvdz'
#    mol.build(0, 0)
#    mf = dft.RKS(mol)
#    mf.xc = 'b3lyp'
#    mf.kernel()
#    mag = Magnetizability(mf).kernel()
#    print(mag)
#
#
#    mol = gto.M(atom='''O      0.   0.       0.
#                        H      0.  -0.757    0.587
#                        H      0.   0.757    0.587''',
#                basis='ccpvdz')
#    mf = scf.RHF(mol)
#    mf.xc = 'b3lyp'
#    mf.run()
#    mag = Magnetizability(mf).kernel()
#    print(lib.finger(m) - 0.43596639996758657)
