# -*- coding: utf-8 -*-
"""
2018/20/28
@author: ALA
"""
# Import

import os
import csv
import numpy as np
from scipy.io import loadmat
from classes import Div

"""
read2div(filename, filetype=None)
saisir2div(filename)
fileread(fileName,delimiter=";")
gtfileread(filename)
"""


def read2div(filename, filetype=None):
    """
    read2div import a file and create a div structure
    Parameters
    ----------
    filename: str (mandatory)
        name of the file
    filetype: str (optional, default=None)
        type of file
    Returns
    -------
    d,i,v: d data np array, i list of strings names of the row 'individuals' ,
    v list of strings names of the columns ('variables')
    """
    # Find the type of file
    _, file_extension = os.path.splitext(filename)

    # If .csv file, use the fileread function
    if file_extension.lower() == '.csv':
        x_div = fileread(filename)
    # If .mat file, assume this is a saisir structure
    elif file_extension.lower() == '.mat':
        x_div = saisir2div(filename)
            
    return x_div

def saisir2div(filename):
    """
    saisir2div reads saisir matlab structure
    Parameters
    ----------
    filename: str (mandatory)
        name of the file
    Returns
    -------
    d,i,v: d data np array, i list of strings names of the row 'individuals' ,
    v list of strings names of the columns ('variables')
    """
    x = loadmat(filename)
    # x is a dict, can acceed the data by finding the key without '__'
    for k in x.keys():
        if k[:2] != '__':
            key_access = k
    
    d = x[key_access][0][0][0]
    i = x[key_access][0][0][1]
    v = x[key_access][0][0][2]

    # Div structure creation
    div = Div(d=d, i=i, v=v, id=filename)

    return div

def fileread(fileName,delimiter=";"):    
    """
    reads the d,i,v outputs from a .csv file
    (Mainly used from class Div)    
    Parameters
    ----------
    fileName: string, name of the csv file 
    delimiter= delimiter string, default :";"
    Returns
    -------
    d,i,v: d data np array, i list of strings names of the row 'individuals' ,
    v list of strings names of the columns ('variables')
    """
    if(not(os.path.isfile(fileName))):
        # Ici on peut utiliser les erreurs Python. J'utilise la ValueError qui n'est pas adaptée ici car il s'agit d'une erreur d'importation. On peut éventuellement créer une classe erreur adaptée pour ce type d'erreur pour être plus clair avec l'utilisateur.
        # Du coup on ne retourne rien, l'erreur stop le programme.
        raise ValueError(fileName + ' is not here.')
        #print(fileName + ' not here')        
        #return(0,0,0)

    # Ici on peut utiliser la syntaxe 
    # with open(fileName,'r') as of:

    # Add encoding to ensure that french character are not an issue
    of=open(fileName,'r', encoding='cp1252')
    txt=of.readline().strip() # first line =variables name"
    aux1=txt.split(delimiter)
    del aux1[0] # first element is not a variable due to excel organization    
    aux2=[]
    aux4=[]    
    
    while 1:    
        txt=of.readline().strip()
        if txt=='':
            break
        #print(txt)
        place=txt.find(delimiter)
        aux2.append(txt[0:(place)].strip())
        aux3=np.fromstring(txt[(place+1) :],dtype=float,sep=delimiter)
        aux4.append(aux3)
    of.close()    
    aux4=np.asarray(aux4)

    div = Div(d=aux4, i=aux2, v=aux1, id=fileName)

    return div

def gtfileread(filename):
    """
    Reads gtcsv

    Parameters
    ----------
    filename: str
        name of the file

    Returns
    -------
    """

    def check_column_name(columnnames):
        """
        check_column_name returns the columns name and two list containing
        numeric column names and string column names

        Parameters
        ----------
        columnnames: list (mandatory)
            list of columns name (str)

        Returns
        -------
        columns_dict: dict
            a dict with columns name as keys and column indexes as values

        numericcol: list
            a list of float of numeric column names

        stringcol: list
            a list of string of string column names

        """
        numericcol = []
        stringcol = []
        colmuns_dict = {}
        for ind, col in enumerate(columnnames):
            colmuns_dict[col] = ind
            # Test if float
            try:
                float(col)
                numericcol.append(float(col))
            except ValueError:
                stringcol.append(col)
        return (colmuns_dict, numericcol, stringcol)

    oldversion = True
    if oldversion:
        header_row = 3
    else:
        header_row = 6
    with open(filename, 'r') as f:
        data = csv.reader(f, delimiter=',')

        datamat = []
        for ind, row in enumerate(data):
            # Extract the domain unit
            if ind == 4:
                if not (oldversion):
                    domain_unit = row[0][13:]
                    # Extract the column name row
            elif ind == header_row:
                columns, numcol, strcol = check_column_name(row)
            elif ind > header_row:
                datamat.append(row)
        datamat = np.asarray(datamat)

    # Get the domain unit
    if oldversion:
        if max(numcol) > 3000:
            domain_unit = 'cm-1'
            numcol_ordered = sorted(numcol, reverse=True)
        else:
            domain_unit = 'nm'
            numcol_ordered = sorted(numcol)
    else:
        if domain_unit == 'nm':
            numcol_ordered = sorted(numcol)
        elif domain_unit == 'cm-1':
            numcol_ordered = sorted(numcol, reverse=True)

    # Reorder the datamat by column according to the domain
    spectralmat = []
    for ind, domainval in enumerate(numcol_ordered):
        # Find the domainval in columns dict to know the colmun index to choose
        # in datamat
        index_datamat = columns[str(domainval)]
        # Put in spectralmat
        spectralmat.append(datamat[:, index_datamat])
    # Convert to numpy array and transpose to have sample by lines and
    # variables by columns
    spectralmat = np.asarray(spectralmat).astype(float).T
    
    metadata = {}
    y_val_array = []
    y_val_name = []
    for metaname in strcol:
        colindex = columns[metaname]
        metadata[metaname] = np.asarray(datamat[:, colindex])
        y_val_array.append(np.asarray(datamat[:, colindex]))
        y_val_name.append(metaname)

    identifiers = metadata['File nomenclature'].astype(str)
    domain = np.array(numcol_ordered).astype(str)

    y_val_array = np.array(y_val_array).T
    y_val_name = np.array(y_val_name).astype(str)

    x_div = Div(d=spectralmat, i=identifiers, v=domain, id=filename)
    y_div = Div(d=y_val_array, i=identifiers, v=y_val_name, id=filename)

    return x_div, y_div