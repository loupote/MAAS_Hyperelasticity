# -*- coding: utf-8 -*-
#
# Data structures for finite element model
#

"""
HYPER Homework
Module 3: kinematic tensor field on a mesh
complete code below where indicated by
# ============
# TODO:
# ...
# ============

REMARK: do not worry for the functions which remain "NOT IMPLEMENTED": you will
complete them in further assignments (M5 and M6). Only complete the "TODO:" parts.
"""

import numpy as np
from scipy.linalg import polar
from scipy.linalg import logm

import tensor
from geometry import SFT3
from geometry import IPTri
from geometry import SFQ4
from geometry import IPGauss2D



class FiniteElement:
    """
    Data structure for finite element
    
    Attributes:
        - type: 2=Triangle, 3=Quadrangle
        - shape: shape functions in reference coordinates N(Ksi)
        - dShape: derivatives of shape functions in global coordinates
                    dN/dX = dN/dKsi*(dX/dKsi)^-1
        - weight: weight of the integration points w*det(dX/dKsi)
        - F: deformation gradient F = Grad(u) + I
        - C: right Cauchy-Green C = F^T*F
        - hencky: Hencky strain ln(V) with F = V.R
        - E_GL: lagrangian strain E = 1/2 (C - I)
        - E_EA: Euler-Almansi strain e = 1/2 (I - b^-1)
        - PK1: piola kirchoff I stress P = F*S
        - PK2: piola kirchoff II stress S
        - sigma: cauchy stress
        - K: lagrangian tangent operator dP/dF
    """
    def __init__(self, t, xNod):
        self.type   = t
        self.shape  = []
        self.dShape = []
        self.weight = []
        #
        #--- select element type and relevant shape functions
        #
        if (t == 2): # triangle T3 (1 int. pt.)
            
            for i in range(1): # loop on integration points
                self.shape.append(SFT3.shape(IPTri.X[i]))
                dShape0 = np.array(SFT3.dShape(IPTri.X[i])) # dN/dKsi
                J = tensor.tensor(2) # dX/dKsi
                for n in range(3): # loop on nodes
                    # J can also be interpreted as the sum of many matrices obtained by outer products
                    J += tensor.outerProd(dShape0[n], xNod[n][0:2])
                self.dShape.append(np.dot(dShape0, tensor.inv(J))) # [dN/dX] = [J]^-1 [dN/dKsi]
                self.weight.append(IPTri.W[i] * tensor.det(J))


        elif (t == 3): # quadrangle Q4 (4 int. pt.)
            for i in range(4): # loop on integration points
                self.shape.append(SFQ4.shape(IPGauss2D.X[i]))
                dShape0 = np.array(SFQ4.dShape(IPGauss2D.X[i])) # dN/dKsi
                J = tensor.tensor(2) # dX/dKsi
                for n in range(4): # loop on nodes
                    J += tensor.outerProd(xNod[n][0:2], dShape0[n])
                self.dShape.append(np.dot(dShape0, tensor.inv(J))) # [dN/dX] = [J]^-1 [dN/dKsi]
                self.weight.append(IPGauss2D.W[i] * tensor.det(J))
        
        #
        #--- initialise mechanical tensors at each integration point
        #
        d = self.getDim() # space dimension
        nip = self.getnIntPts() # number of integration points
        # deformation gradient F = Grad(u) + I
        self.F = [tensor.tensor(d) for n in range(nip)]
        # hencky strain ln(V) with F = V.R
        self.hencky = [tensor.tensor(d) for n in range(nip)]
        # right Cauchy-Green C = F^T*F
        self.C = [tensor.tensor(d) for n in range(nip)]
        # lagrangian strain E = 1/2 (C - I)
        self.E_GL = [tensor.tensor(d) for n in range(nip)]
        # Euler-Almansi strain e = 1/2 (I - b^-1)
        self.E_EA = [tensor.tensor(d) for n in range(nip)]
        # piola kirchoff I : P = F*S
        self.PK1 = [tensor.tensor(d) for n in range(nip)]
        # piola kirchoff II : S
        self.PK2 = [tensor.tensor(d) for n in range(nip)]
        # cauchy stress : sigma
        self.sigma = [tensor.tensor(d) for n in range(nip)]
        # lagrangian tangent operator dP/dF
        self.K = [tensor.tensor4(d) for n in range(nip)]

    def getDim(self,):
        """
        Get mesh dimension
        """
        return np.shape(self.dShape)[2]

    def getnIntPts(self,):
        """
        Get number of integration points per element
        """
        return len(self.shape)

    def getnNodes(self,):
        """
        Get number of nodes per element
        """
        return np.shape(self.dShape)[1]

    def gradient(self, dShp, uNod):
        
        """
        Compute gradient of the displacement field

        Parameters
        ----------
        dShp : 2-entry array
            derivatives of shape functions. (n_nodes, u_dim)
            [ [ dN0/dx  dN0/dy ]
              [ dN1/dx  dN1/dy ]
              [ dN2/dx  dN2/dy ] ]

        uNod : 2-entry array
            local displacement field. (n_nodes, u_dim)
            [ [ u0_x  u0_y ]
              [ u1_x  u1_y ]
              [ u2_x  u2_y ] ]

        Returns
        -------
        G : 2-entry array
            gradient of the displacement field G_{ij} = du_i/dX_j.
        """
        
        G = np.dot(np.transpose(uNod), dShp)
        return G
        # =====================================================================

    def update(self, uNod, mater):
        """
        update strain and stress tensors of the element from the displacement
        field 'uNod' with the material model 'mater'
        """
        for i in range(self.getnIntPts()): # loop on integration points
            ### kinematic values
            # compute CG stretch tensor
            G = self.gradient(self.dShape[i], uNod)
            # compute deformation gradient
            F = G + tensor.I(len(G))
            self.F[i] = F #store deformation gradient at integration point i
            self.C[i] = tensor.rightCauchyGreen(F)
            # compute GL strain tensor
            self.E_GL[i] = 0.5*(self.C[i]-tensor.I(len(self.C[i])))
            #compute spatial description
            _, V = polar(F, side='left') #from scipy.linalg.polar() method with u=R and p=V,
            V[V<1e-10] = 1e-15 # replace pure zeros by very low values to prevent "nan" in np.log(V)
            # compute hencky = ln(V) with F = V.R, "true" strain
            self.hencky[i] = logm(V)
            b = tensor.leftCauchyGreen(F)
            # compute EA strain tensor
            self.E_EA[i] = 0.5*(tensor.I(len(b))-tensor.inv(b))
            

            ### stress values
            if (mater == 0): #skip next lines: do not compute stress
                continue
            # compute PK2 stress tensor and material tangent operator M = 2*dS/dC
            (self.PK2[i], M) = mater.stress_stiffness(self.C[i])
            # compute PK1 stress
            self.PK1[i] = tensor.PK2toPK1(F, self.PK2[i])
            # compute lagrangian tangent operator K = dP/dF
            self.K[i] = tensor.MaterialToLagrangian(F, self.PK2[i], M)
            # compute cauchy stress (spatial description)
            self.sigma[i] = tensor.PK1toCauchy(F, self.PK1[i])

    def computeForces(self):
        """
        compute internal forces of the element

        Returns
        -------
        fNod : 2-entry tensor
            internal nodal forces of the element.

        """
        fNod = np.zeros((self.getnNodes(), self.getDim()))

        for i in range(self.getnIntPts()):
            
            w   = self.weight[i]
            Pk1 = np.transpose(self.PK1[i])
            dN  = self.dShape[i]
            
            fNod += w * np.matmul(dN, Pk1)
            
        return fNod

    def computeStiffness(self):
        """
        compute internal stiffness of the element

        Returns
        -------
        KNod : 4-entry tensor
            internal stiffness of the element.

        """
        KNod = np.zeros((self.getnNodes(),self.getDim(),
                         self.getnNodes(),self.getDim()))

        for i in range (self.getnIntPts()):
            
            w  = self.weight[i]
            dN = self.dShape[i]
            K  = self.K[i]
            
            KNod += w * np.einsum('aj,ijkL,bL -> aibk', dN, K, dN)
        
        return KNod