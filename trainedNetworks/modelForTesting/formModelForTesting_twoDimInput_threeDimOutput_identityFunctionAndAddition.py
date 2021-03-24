# Fanoos: Multi-Resolution, Multi-Strength, Interactive Explanations for Learned Systems ; David Bayani and Stefan Mitsch ; paper at https://arxiv.org/abs/2006.12453
# Copyright (C) 2021  David Bayani
# 
# This file is part of Fanoos.
# 
# Fanoos is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License only.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 
# Contact Information:
# 
# Electronic Mail:
#   dcbayani@alumni.cmu.edu
# 
# Paper Mail:
#   David Bayani
#   Computer Science Department
#   Carnegie Mellon University
#   5000 Forbes Ave.
#   Pittsburgh, PA 15213
#   USA
# 
# 


import numpy as np;
from sklearn.linear_model import LinearRegression;

from scipy.special import comb;

from sklearn.datasets import fetch_openml;

def requires(booleanStatement):
    assert(booleanStatement);
    return;

def getDefaultTargetAndNonTargetData(valueNames, targetNames):
    requires(isinstance(targetNames, list));
    requires(all([isinstance(x, str) for x in targetNames]));
    requires(set(valueNames).issuperset(targetNames));
    requires(len(targetNames) == len(set(targetNames)));

    setOfTargetNames = set(targetNames);

    indicesOfNewTargets = np.zeros(len(valueNames), dtype=bool); 
    for thisIndex in range(0, len(valueNames)):
        if(valueNames[thisIndex] in setOfTargetNames):
            indicesOfNewTargets[thisIndex] = 1;

    indicesOfNonTargets =  ~indicesOfNewTargets;
    return {"indicesOfNewTargets" : indicesOfNewTargets, \
            "indicesOfNonTargets": indicesOfNonTargets};



def normalizeThisData(thisData, indicesOfTheTrainSet):
    B = thisData  - np.min(thisData[indicesOfTheTrainSet, :], axis=0);
    B = B / np.max(B, axis=0);
    return B;


def tryDatasetWithPolynomialLinear(indicesOfNewTargets, indicesOfNonTargets):
    requires(isinstance(indicesOfNewTargets, np.ndarray));
    requires(isinstance(indicesOfNonTargets, np.ndarray));
    requires(not np.any(indicesOfNewTargets & indicesOfNonTargets));
    requires(indicesOfNewTargets.dtype == np.dtype('bool'));
    requires(indicesOfNonTargets.dtype == np.dtype('bool'));
   
    newFeatureNames = ["in_x", "in_y"] ;
    newTargetData_featureNames = ["out_u", "out_v", "out_w"];
    newBaseData_featureNames = newFeatureNames;    


    """
note: the layout of coefficients is:
    np.array( [
        [<coefficients for first output variable>],
        [<coefficients for second output variable>],
        .
        . 
        .
        [<coefficients for nth output variable>],
        ]);

example: 

     input box: [[0. 1.]
                 [0. 1.]]
 
     output: [[   3.5        6.105  ]
              [-100.2      -98.99999]]
    """



    resultsToReturn = \
        {"newFeatureNames" : newFeatureNames ,\
            "coefficients" : np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) , \
            "intercept" : np.array([0.0, 0.0, -1.0]) , \
            "namesOfTargetValues" : newTargetData_featureNames, \
            "orderOfNonFeaturizedObservationNames" : newBaseData_featureNames \
        };
    return resultsToReturn;



import pickle;
valueNames = ["in_x", "in_y", "out_u", "out_v", "out_w"];
targetNames =["out_u", "out_v", "out_w"]

A = getDefaultTargetAndNonTargetData(valueNames, targetNames);


trainedModelInfoDicts = tryDatasetWithPolynomialLinear(\
    A["indicesOfNewTargets"], \
    A["indicesOfNonTargets"] );

# TODO: see the todos at the end of the code for preparing the one 
#     dimensional identity function...
fh = open("./modelForTesting_twoDimInput_threeDimOutput_identityFunctionAndAddition.pickle", "wb");
pickle.dump(trainedModelInfoDicts, fh);
fh.close();



