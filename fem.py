# -*- coding: utf-8 -*-
#
# Finite element model algorithm
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
complete them in further assignments (M5 and M6). Only complete the "TODO:" 
parts.
"""

import numpy as np

import tensor
from element import FiniteElement


class FEModel:
    """
    Data structure for FE model
    
    Attributes:
        - dim: mesh dimension
        - connect: table of connectivity (list of lists)
            example: connect=[[0,1,2],[1,2,3]]
            -> element 0 contains nodes 0,1,2
            -> element 1 contains nodes 1,2,3
        - elems: list of FiniteElement instances
        - material: constitutive model instance from elasticity module
    """

    def __init__(self, theMesh, m=0):
        self.dim = theMesh.getDimension()
        self.connect = []
        self.elems = []
        for n in range(theMesh.nElements()): # loop on elements
            if (theMesh.elems[n].type < 1) or (theMesh.elems[n].type > 3):
                continue # skip following lines if unknown type
            self.connect.append(theMesh.elems[n].nodes) # add element's nodes to table of connectivity
            xNod = []
            for i in range(theMesh.elems[n].nNodes()): # loop on element's nodes
                xNod.append(theMesh.nodes[theMesh.elems[n].nodes[i]].X) # save nodes coordinates
            newElem = FiniteElement(theMesh.elems[n].type, xNod) # create instance of FiniteElement
            self.elems.append(newElem) # add FiniteElement to self.elems
        self.material = m
    
    def getDim(self):
        """
        Return dimension of the FE model
        """
        return self.dim
    
    def nElements(self):
        """
        Get number of elements
        """
        
        return len(self.elems)
        # =====================================================================
    
    def extract(self, U, e):
        """
        extract nodal values of field U for element e

        Parameters
        ----------
        U : 2-entry array
            nodal displacement field.
        e : int
            element.

        Returns
        -------
        uNod : 2-entry array
            nodal values of displacement field for element e.
        """

        uNod = np.zeros([len(self.connect[e]), 2])
        
        for i in range(len(self.connect[e])):
            uNod[i,:] = U[(self.connect[e])[i]]
        
        return uNod


    def computeStrain(self, U):
        """
        Compute strains on all elements
        """

        # loop on elements
        for n in range(len(self.elems)):
            # get nodal displacements for element
            uNod = self.extract(U, n)
            # compute gradients (and strains)
            self.elems[n].update(uNod, mater=0)


    def computeStress(self, U):
        """
        Compute stresses on all elements
        """

        # loop on elements
        for n in range(len(self.elems)):
            # get nodal displacements for element
            uNod = self.extract(U, n)
            # compute stresses
            self.elems[n].update(uNod, self.material)
    

    def assemble2(self, e, vNod, V):
        """
        Assemble nodal values for 2-entry array:
            add nodal values for element e to global array V

        Parameters
        ----------
        e : int
            element.
        vNod : 2-entry array
            nodal values for element e.
        V : 2-entry array
            all nodal values of the mesh.

        example: nodes = [2, 6, 8]
        i = 1 -> j = 6

        Returns
        -------
        None.
        """
        
        nodes = self.connect[e]
        for i in range(len(nodes)):
            j = nodes[i]
            V[j,:] += vNod[i,:]
        

    def assemble4(self, e, vNod, V):
        """
        Assemble nodal values for 4-entry array:
            add nodal values for element e to global array V

        Parameters
        ----------
        e : int
            element.
        vNod : 4-entry array
            nodal values for element e.
        V : 4-entry array
            all nodal values of the mesh.

        Returns
        -------
        None.
        """

        nodes=self.connect[e]
        for i in range(len(nodes)):
            j = nodes[i]
            for k in range(len(nodes)):
                l = nodes[k]
                V[j,:,l,:] += vNod[i,:,k,:]


    def computeInternalForces(self, U):
        """
        Compute generalized internal forces Tint

        Parameters
        ----------
        U : 2-entry array
            displacement field.

        Returns
        -------
        Tint : 2-entry array
            internal forces.
        """
        meshnNodes, meshDim = U.shape
        Tint = np.zeros((meshnNodes, meshDim))
        
        self.computeStress(U)
        
        nElems = self.nElements()
        
        for i in range(nElems):
            
            elem = self.elems[i]
            fint = elem.computeForces()
            self.assemble2(i, fint, Tint)            
        
        return Tint


    def computeResidual(self, U):
        """
        Compute residual R = Tint - Text
        """
        R = self.computeInternalForces(U) # R = Tint
        return R


    def computeInternalStiffness(self, U):
        """
        Compute generalized internal stiffness tangent Kint

        Parameters
        ----------
        U : 2-entry array
            displacement field.

        Returns
        -------
        Kint : 4-entry array
            internal stiffness tangent.

        """

        meshnNodes, meshDim = U.shape
        Kint = np.zeros((meshnNodes, meshDim, meshnNodes, meshDim))
        
        self.computeStress(U)

        nElems = self.nElements()
        
        for i in range(nElems):
            
            elem = self.elems[i]
            Ks = elem.computeStiffness()
            self.assemble4(i, Ks, Kint)
            
        return Kint
    
    def computeTangent(self, U):
        """
        Compute tangent K = Kint - Kext = Kint: in the absence of displacement-depend external forces Kext = 0.
        """
        K = self.computeInternalStiffness(U) # K = Kint
        return K

    def getDeformationGradient(self, n):
        """
        Return average of deformation gradient at all integration points of element n
        """
        avg = tensor.tensor(self.getDim())
        for tens in self.elems[n].F:
            avg += tens
        avg /= self.elems[n].getnIntPts()
        return avg

    def getRightCauchyGreen(self, n):
        """
        Return average of right CG at all integration points of element n
        """
        avg = tensor.tensor(self.getDim())
        for tens in self.elems[n].C:
            avg += tens
        avg /= self.elems[n].getnIntPts()
        return avg
    
    def getStrainHencky(self, n):
        """
        Return average of hencky strain at all integration points of element n
        """
        avg = tensor.tensor(self.getDim())
        for tens in self.elems[n].hencky:
            avg += tens
        avg /= self.elems[n].getnIntPts()
        return avg
        
    def getStrainGreenLagrange(self, n):
        """
        Return average of GL strain at all integration points of element n
        """
        avg = tensor.tensor(self.getDim())
        for tens in self.elems[n].E_GL:
            avg += tens
        avg /= self.elems[n].getnIntPts()
        return avg
    
    def getStrainEulerAlmansi(self,n):
        """
        Return average of EA strain at all integration points of element n
        """
        avg = tensor.tensor(self.getDim())
        for tens in self.elems[n].E_EA:
            avg += tens
        avg /= self.elems[n].getnIntPts()
        return avg
    
    def getStressPK1(self, n):
        """
        Return average of PK1 stress at all integration points of element n
        """
        avg = tensor.tensor(self.getDim())
        for tens in self.elems[n].PK1:
            avg += tens
        avg /= self.elems[n].getnIntPts()
        return avg

    def getStressPK2(self, n):
        """
        Return average of PK2 stress at all integration points of element n
        """
        avg = tensor.tensor(self.getDim())
        for tens in self.elems[n].PK2:
            avg += tens
        avg /= self.elems[n].getnIntPts()
        return avg
    
    def getStressCauchy(self, n):
        """
        Return average of cauchy stress at all integration points of element n
        """
        avg = tensor.tensor(self.getDim())
        for tens in self.elems[n].sigma:
            avg += tens
        avg /= self.elems[n].getnIntPts()
        return avg