# -*- coding: utf-8 -*-
#
# Data structures for hyperelastic constitutive models
#
"""
HYPER Homework
Module 4: Hyperelasticity
complete code below where indicated by
# ============
# TODO:
# ...
# ============

TODO:
complete stress(), stiffness() and stress_stiffness() methods for 
StVenantKirchhoffElasticity and NeoHookeanElasticity classes.

stress:
    return 2ND Piola Kirchhoff tensor (PK2) as a function of right Cauchy-
    Green tensor (C)
sitffness:
    return Material tangent operator (M) as a function of right Cauchy-Green 
    tensor (C)
stress_stiffness:
    return 2ND Piola Kirchhoff tensor (PK2) and Material tangent operator (M) 
    as a function of right Cauchy-Green tensor (C). It can either use both 
    previous methods or rewrite PK2 and M (can save CPU time if same quantities 
    are used in both computations)

WARNING:
in FiniteElement.update(), (S,M) = material.stress_stiffness(C)
--> stress_stiffness must be a function of right Cauchy Green tensor
===============================================================================
"""

import tensor
import numpy as np

class StVenantKirchhoffElasticity:
    """
    Data structure for (isotropic) St-Venant-Kirchhoff hyperelaticity models
    """
    
    def __init__(self, E, nu):
        self.prop = dict()
        self.prop["YOUNG_MODULUS"] = E
        self.prop["POISSON_COEFFICIENT"] = nu
        self.prop["SHEAR_MODULUS"] = 0.5*E/(1.+nu)
        self.prop["BULK_MODULUS"] = E/3./(1.-2*nu)
        self.prop["1ST_LAME_CONSTANT"] = self.prop["BULK_MODULUS"]-2./3.*self.prop["SHEAR_MODULUS"]
        self.prop["2ND_LAME_CONSTANT"] = self.prop["SHEAR_MODULUS"]
    
    def getLame1(self,):
        return self.prop["1ST_LAME_CONSTANT"]
    
    def getLame2(self,):
        return self.prop["2ND_LAME_CONSTANT"]
    
    def potential(self, C):
        """
        Compute hyperelastic potential: phi = 1/2*lambda*tr(E)^2 - mu*(E:E)
        """
        lam = self.getLame1()
        mu = self.getLame2()
        EL = 0.5*(C - tensor.I(len(C))) # Lagrangian strain E
        phi = lam/2.*(tensor.trace(EL))**2 + mu*np.tensordot(EL,EL,2)
        return phi

    def stress(self, C):
        """
        Compute 2nd Piola-Kirchhoff stress
        """
        PK2 = tensor.tensor(len(C)) 
        
                    
        E = 0.5*(C-tensor.I())
        
        PK2 = self.prop["1ST_LAME_CONSTANT"]*tensor.trace(E)*tensor.I() + 2*self.prop["SHEAR_MODULUS"]*E
        return PK2

    def stiffness(self, C):
        """
        Compute material tangent M = 2*dS/dC
        C_{ijkl} =λδ_{ij}δ_{kl} + 2μδ_{ik}δ_{jl}
        """
        M = tensor.tensor4(len(C))
        M = self.prop["1ST_LAME_CONSTANT"]*tensor.outerProd4(tensor.I(), tensor.I()) + 2*self.prop["SHEAR_MODULUS"]*tensor.IISym()
        return M

    def stress_stiffness(self, C):
        """
        Compute 2nd Piola-Kirchhoff stress and material tangent at the same time
        """
        PK2 = tensor.tensor(len(C))
        M = tensor.tensor4(len(C))
        
        PK2 = self.stress(C)
        M = self.stiffness(C)
        return (PK2, M)

class NeoHookeanElasticity:
    """
    Data structure for (isotropic) NeoHookean hyperelaticity models
    """
    
    def __init__(self, E, nu):
        self.prop = dict()
        self.prop["YOUNG_MODULUS"] = E
        self.prop["POISSON_COEFFICIENT"] = nu
        self.prop["SHEAR_MODULUS"] = 0.5*E/(1.+nu)
        self.prop["BULK_MODULUS"] = E/3./(1.-2*nu)
        self.prop["1ST_LAME_CONSTANT"] = self.prop["BULK_MODULUS"]-2./3.*self.prop["SHEAR_MODULUS"]
        self.prop["2ND_LAME_CONSTANT"] = self.prop["SHEAR_MODULUS"]
    
    def getLame1(self,):
        return self.prop["1ST_LAME_CONSTANT"]
    
    def getLame2(self,):
        return self.prop["2ND_LAME_CONSTANT"]
    
    def potential(self, C):
        """
        Compute hyperelastic potential: phi = mu/2 * (tr(C)-3) - mu*ln(J) + lam/2 *ln(J)^2
        """
        lam = self.getLame1()
        mu = self.getLame2()
        J = np.sqrt(tensor.det(C)) # J = det(F) and det(C) = J^2
        phi = mu/2.*(tensor.trace(C)-3.) - mu*np.log(J) + lam/2.*(np.log(J))**2.
        return phi

    def stress(self, C):
        """
        Compute 2nd Piola-Kirchhoff stress
        """
        PK2 = tensor.tensor(len(C))
        J = np.sqrt(tensor.det(C))
        PK2 = self.prop["2ND_LAME_CONSTANT"]*(tensor.I(len(C))-tensor.inv(C))+self.prop["1ST_LAME_CONSTANT"]*np.log(J)*tensor.inv(C)
        return PK2

    def stiffness(self, C):
        """
        Compute material tangent M = 2*dS/dC
        """
        M = tensor.tensor4(len(C))
        lam = self.prop["1ST_LAME_CONSTANT"]
        mu = self.prop["2ND_LAME_CONSTANT"]
        invC = tensor.inv(C)
        J = np.sqrt(tensor.det(C))

        deriv = tensor.tensor4(len(C))
        for i in range(len(C)):
            for j in range(len(C)):
                for k in range(len(C)):
                    for l in range(len(C)):
                        deriv[i, j, k, l] = -1/2*(np.dot(invC[i,k], invC[j,l]) + np.dot(invC[i,l], invC[j,k]))
        
        M = self.prop["1ST_LAME_CONSTANT"]*tensor.outerProd4(invC,invC)+2*(lam*np.log(J)-mu)*deriv
        return M

    def stress_stiffness(self, C):
        """
        Compute 2nd Piola-Kirchhoff stress and material tangent at the same time
        """
        PK2 = tensor.tensor(len(C))
        M = tensor.tensor4(len(C))
        PK2 = self.stress(C)
        M = self.stiffness(C)
        return (PK2, M)
