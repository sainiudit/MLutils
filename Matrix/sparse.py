import scipy.sparse as sp
from scipy.sparse import *
from scipy import *
import numpy as np

def createcsr_SparseMatrix(D,Attributes):
    rowindex = range(0, D.shape[0])
    col = np.zeros(len(D.people_char_10))
    colmatrix = csr_matrix((D.people_char_10, (rowindex, col)), shape=(len(rowindex), 1))
    for features in Attributes:
        csm = csr_matrix((D[features], (rowindex, col)), shape=(len(rowindex), 1))
        colmatrix = sp.hstack([colmatrix, csm])
    return colmatrix


def createDummycsr_SparseMatrix(D,catfeatures):
    data = np.ones(D.shape[0])
    rowindex = range(0, D.shape[0])
    sparse_matrix = csr_matrix((data, (rowindex, D[catfeatures[0]])),
                               shape=(len(rowindex), D[catfeatures[0]].nunique()))
    for col in catfeatures[1:]:
        spm = csr_matrix((data, (rowindex, D[col])), shape=(len(rowindex), D[col].nunique()))
        sparse_matrix = sp.hstack([sparse_matrix, spm])
    return sparse_matrix







