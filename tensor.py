# -*- coding: utf-8 -*-
#
# Set of methods for manipulating 2nd-order and 4th-order tensors
#
"""
HYPER Homework
Module 1: operations on tensors
complete code below where indicated by
# ============
# TODO:
# ...
# ============
and rename file as tensor.py
WARNING:
    - do NOT write matrices explicitely, use python methods instead
    - complete every functions but PK2toPK1, PK1toCauchy and MaterialToLagragian
    that will be needed later
REMARK:
    use numpy package, most of the required functions are already implemented in it
    cf. documentation at https://docs.scipy.org/doc/numpy/genindex.html
"""
#
# --- Namespace
#
import numpy as np #access to a method from the "numpy" library using "np.method()"
import numpy.linalg as la #access to a method from the "linalg" module of the
                          #"numpy" library using "la.method()"

#
# --- Vectors
#

def vector(d=2):
    """
    Constructor of a vector object (dimension d)
    """
    return np.zeros(d)

#
# --- 2nd order tensors
#

def tensor(d=2):
    """
    Constructor of 2nd-order tensor (dimension d)
    """
    return np.zeros((d, d))

def I(d=2):
    """
    Identity second-order tensor
    """
        
    return np.eye(d)
    # =========================================================================

def det(A):
    '''
    Déterminant d'une matrice
    '''
    return la.det(A)
    # =========================================================================

def inv(A):
   '''
   L'inverse d'une matrice
   '''
   
   return la.inv(A)
    # =========================================================================

def trace(A):
    '''
    Trace d'une matrice
    '''
    
    return np.trace(A)
    # =========================================================================

def outerProd(a,b):
    '''
    Produit de 2 vecteurs
    '''
    
    return np.outer(a, b)
    # =========================================================================

def rightCauchyGreen(F):
    # F^T F
    return np.dot(np.transpose(F), F)
    # =========================================================================

def leftCauchyGreen(F):
    # F F^T
    return  np.dot(F, np.transpose(F))
    # =========================================================================

def PK2toPK1(F, S):
    """
    Compute Piola stress tensor from second Piola-Kirchhoff stress
    """
    
    P = np.dot(F, S) # First Piola-Kirchhoff stress
    return P

def PK1toCauchy(F, P):
    """
    Compute Cauchy stress tensor from first Piola-Kirchhoff stress
    """
    
    J = det(F)
    sigma = 1/J*np.dot(P, np.transpose(F))
    return sigma
  

#
# --- 4th order tensors
#

def tensor4(d=2):
    """
    Constructor of 4th-order tensor (dimension d)
    """
    
    return np.zeros((d, d, d, d))

def II(d=2):
    """
    Identity fourth-order tensor
    """
    
    Id4 = tensor4(d)
    
    for i in range(d):
        for j in range(d):
            Id4[i, j, i, j] = 1.
    return Id4
   
    # =========================================================================

def IISym(d=2):
    """
    Symmetrical identity fourth-order tensor:
    IISym_ijkl = 1/2 * (delta_ik delta_jl + delta_il delta_jk)
    """
    
    IISym = tensor4(d)
    for i in range(d):
        for j in range (d):
            for k in range (d):
                for l in range (d):
                    if i==k and j==l:
                        IISym[i, j, k, l] += 1.0
                    if i==l and j==k:
                        IISym[i, j, k, l] += 1.0
    IISym = 0.5 * IISym
    
    return IISym
    # =========================================================================

def KK(d=2):
    """
    Spherical operator:
    returns the spherical part of a given 2nd-order tensor
    KK_ijkl = delta_ij * delta_kl
    """
    KK = tensor4(d)
    for i in range(d):
        for j in range (d):
            for k in range (d):
                for l in range (d):
                    if i==j and k==l:
                        KK[i, j, k, l] = 1.0
    return KK
    # =========================================================================

def outerProd4(a, b):
    '''
    Produit de 2 matrices
    '''
    
    return np.tensordot(a, b, axes=0)
    # =========================================================================

def MaterialToLagrangian(F, S, M):
    """
    Compute Lagrangian tangent operator from material tensor and stress
    """
    #dP/dF_{iJkL} = delta_{ik}*S_{JL} + F_{iI}*M_{IJKL}*F_{kK}
    (a, b) = np.shape(S)
    delta = np.eye(a)
    ML = np.einsum('ik,JL->iJkL', delta, S) + np.einsum('iI,IJKL,kK->iJkL', F, M, F)
    return ML
