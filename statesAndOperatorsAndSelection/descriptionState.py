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


import pickle;
import numpy as np;
import sys;
from utils.contracts import *;

import config;

import uuid;

from databaseInterface.databaseValueTracker import ObjDatabaseValueTracker;
from databaseInterface.databaseIOManager import objDatabaseInterface, executeDatabaseCommandList;


from domainsAndConditions.baseClassConditionsToSpecifyPredictsWith import \
    MetaCondition_Conjunction, \
    Condition_TheBoxItself ;

from boxesAndBoxOperations.getBox import boxSize, getDimensionOfBox;

from utils.distributionStatics import distributionStatics;

class DescriptionState():

    @staticmethod
    def convertDescriptionStateIDToStringUniformly(thisID):
        # TODO: write requires to check it is a proper ID...
        valueToReturn = "";
        if(isinstance(thisID, str)):
            assert("frozen" not in str(thisID));
            valueToReturn = thisID;
        else:
            assert(isinstance(thisID, frozenset));
            valueToReturn = str(sorted(list(thisID))).replace("'", "\"").replace("[", "frozenset({").replace("]", "})");
            assert(eval(valueToReturn) == thisID);
        ensures(isinstance(valueToReturn, str));
        ensures(len(valueToReturn) == len(str(thisID)) );
        ensures( (set(valueToReturn).symmetric_difference(set(str(thisID)))).issubset({"'", "\""}) );
        return valueToReturn;

    def setID(self):
        self.uuid = str(uuid.uuid4());
        ObjDatabaseValueTracker.set_QAStateUUID_mostRecentBeingComputed(self.uuid);# In a lot of ways, this is
            # not ideal to set like this, since multiple states could and will exist in use at any particular time
            # (for example of a pair of such states: prior state and state being computed)
        return

    def getID(self):
        return self.uuid;

    def __init__(self):
        self.internalDictionary = dict();
        self.internalDictionary["description"] = list();
        self.internalDictionary["boxes"]= list();
        self.internalDictionary["mostRecentOperatorParameters"] = dict();
        self.internalDictionary["mostRecentOperatorParameters"]["removedPredicates"] = set();
        self.internalDictionary["rawInputDomainBoxes"] = dict(); # I dislike storing this due to memory consumption....
        self.setID();


        self.sideInformationDict = dict();
        self.sideInformationDict["dictMappingConditionIDToVolumeCoveredAndUniqueVolumeCovered"] = dict();
        return;

    def setDescription(self, thisDescription):
        requires(isinstance(thisDescription, list));
        self.internalDictionary["description"] = thisDescription;
        return;

    def getDescription(self):
        return self.internalDictionary["description"];

    def setRawInputDomainBoxes(self, theseBoxes):
        requires(isinstance(theseBoxes, list));
        self.internalDictionary["rawInputDomainBoxes"] = theseBoxes;
        return;

    def getRawInputDomainBoxes(self):
        return self.internalDictionary["rawInputDomainBoxes"];

    
    def setBoxes(self, theseBoxes):
        requires(isinstance(theseBoxes, list));
        self.internalDictionary["boxes"] = theseBoxes;
        return;

    def getBoxes(self):
        return self.internalDictionary["boxes"];


    def getCopyOfParameters(self):
        return self.internalDictionary["mostRecentOperatorParameters"].copy();


    def setSideInformation(self, key, value):
        requires(isinstance(key, str));
        requires(len(key) > 0);
        self.sideInformationDict[key] = value;
        return;

    def getSideInformation(self, key):
        requires(isinstance(key, str));
        requires(len(key) > 0);
        if(key not in self.sideInformationDict):
            raise Exception("Side information of the sort requested does not exist." + \
                            " HINT: check that you spelled the key correctly.");
        return self.sideInformationDict[key];


    def convertBoxAndLabelToList(self, label, d2Array, listOfAxisForBox):
        requires(d2Array.shape[0] == len(listOfAxisForBox));
        for thisIndex in range(0, len(listOfAxisForBox)):
            for upperOrLowerSelector in [0, 1]:
                specificLabel = (\
                    label + ":" + str(listOfAxisForBox[thisIndex]) + ":" + \
                    ("u" if (upperOrLowerSelector == 1) else "l") \
                );
                commandToExecute = \
                    "INSERT INTO QAStateValues ( QAStateUUID , fieldName, fieldValue) VALUES ('" + \
                    self.getID() + "', '" + specificLabel + "', ? );";
                objDatabaseInterface.interfaceBleed_insertValuesForBlob(\
                    commandToExecute, [d2Array[thisIndex, upperOrLowerSelector]]  );                
        objDatabaseInterface.commit();
        return
                
    @staticmethod
    def _getAnalysisResult(thisData):
        tempDict = distributionStatics(thisData);
        tempDict.pop("numberOfDataPoints"); # We want the number of data points only
            # to be recorded once for the boxes, namely under the general, non-domain
            # specific category, not for each item that we might find a distribution for.
        return tempDict;

    def _convertBoxesToSQLCommandsToRecordSummaryStatistics(self, variablesBoxesProducedMayBeOver, universeForBoxes):
        labelBeginning = "bstats"; # short for "box statistics";

 
        #V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
        # domain specific statistics - i.e., those that are most clearly applicale to this specific domain....
        #=============================================================
        labelForDomainSpecificStats = labelBeginning + ":d";
        variablesBoxesProducedMayBeOver = [str(x) for x in variablesBoxesProducedMayBeOver]; 
        resultValue = self._getAnalysisResult(self.internalDictionary["boxes"]);
        for thisKey in resultValue:
            self.convertBoxAndLabelToList(\
                labelForDomainSpecificStats + ":" + thisKey, \
                resultValue[thisKey], \
                variablesBoxesProducedMayBeOver );

        generalSummaryFunctionsAndLabelsForThem = [\
            ( (lambda A : np.prod(np.diff(A, axis=1)) ), "bvolume"), \
            ( (lambda A : np.min(np.diff(A, axis=1)) ), "bminSideLength"), \
            ( (lambda A : np.max(np.diff(A, axis=1)) ), "bmaxSideLength"), \
            ( (lambda A : np.sum(np.diff(A, axis=1)) ), "bsumSideLengths"), \
        ];

        for thisFunctAndLabel in generalSummaryFunctionsAndLabelsForThem:
            theseValues = [ thisFunctAndLabel[0](x) for x in self.getBoxes()];
            resultValue = self._getAnalysisResult(theseValues);
            specificLabel = labelForDomainSpecificStats + ":" + thisFunctAndLabel[1];
            for thisKey in resultValue:
                commandToExecute = \
                    "INSERT INTO QAStateValues ( QAStateUUID , fieldName, fieldValue) VALUES ('" + \
                    self.getID() + "', '" + (specificLabel + ":" + thisKey)  + "', ? );";
                objDatabaseInterface.interfaceBleed_insertValuesForBlob(\
                    commandToExecute, [resultValue[thisKey]]  );                
            objDatabaseInterface.commit();
        #^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^


        #V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
        # general statistics - i.e., those that may be easily useable across domains...
        #=============================================================
        labelForGeneralStats =  labelBeginning + ":g";
        numberOfBoxes = len(self.internalDictionary["boxes"]);
        commandToExecute = \
            "INSERT INTO QAStateValues ( QAStateUUID , fieldName, fieldValue) VALUES ('" + \
             self.getID() + "', '" + (labelForGeneralStats + ":numberOfDataPoints") + "', ? );";
        objDatabaseInterface.interfaceBleed_insertValuesForBlob(\
            commandToExecute, [numberOfBoxes]  );

        scalingFactors = np.diff(universeForBoxes, axis=1); 
        # NOTE: below, we find the NORMALIZED side-lengths. Yes, this is normalized
        #     even though the constant offset term (subtracting the minimal value) is not shown - those terms would 
        #     simply cancel when we compute the interval length....
        getScaledSideLengths = \
            (lambda A: (np.diff(A, axis=1) / scalingFactors));

 
        labelForGeneralStats =  labelForGeneralStats + ":normalizedInputBoxes";

        generalSummaryFunctionsAndLabelsForThem = [\
            ( (lambda A : np.prod(getScaledSideLengths(A)) ), "bvolume"), \
            ( (lambda A : np.min(getScaledSideLengths(A)) ), "bminSideLength"), \
            ( (lambda A : np.max(getScaledSideLengths(A)) ), "bmaxSideLength"), \
            ( (lambda A : np.sum(getScaledSideLengths(A)) ), "bsumSideLengths"), \
        ];
        
        for thisFunctAndLabel in generalSummaryFunctionsAndLabelsForThem:
            theseValues = [ thisFunctAndLabel[0](x) for x in self.getRawInputDomainBoxes()];
            resultValue = self._getAnalysisResult(theseValues);
            specificLabel = labelForGeneralStats + ":" + thisFunctAndLabel[1];
            for thisKey in resultValue:
                commandToExecute = \
                    "INSERT INTO QAStateValues ( QAStateUUID , fieldName, fieldValue) VALUES ('" + \
                    self.getID() + "', '" + (specificLabel + ":" + thisKey)  + "', ? );";
                objDatabaseInterface.interfaceBleed_insertValuesForBlob(\
                    commandToExecute, [resultValue[thisKey]]  );                
            objDatabaseInterface.commit();
        #^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^
        return;

    """
ssh-rsa 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 abc@defg
"""

    def _helper_convertDescriptionToSQLCommands(self, thisObj):
            thisObjUUIDAsString = DescriptionState.convertDescriptionStateIDToStringUniformly(thisObj.getID()); # need for the case of a MetaCondition_Conjunction, but does the proper thing in other cases as well
            if(isinstance(thisObj, MetaCondition_Conjunction)):
                for childObj in thisObj.listOfConditionsToConjunct:
                    objDatabaseInterface.exec(\
                        "INSERT INTO QAStateValues ( QAStateUUID , fieldName, parentUUID , childUUID) VALUES ('" + \
                        self.getID() + "', 'd:parent_child', '" + thisObjUUIDAsString + "', '" + childObj.getID() + "' );" \
                    );
                    self._helper_convertDescriptionToSQLCommands(childObj);
            elif(isinstance(thisObj, Condition_TheBoxItself)):
                commandToExecute = \
                    "INSERT INTO QAStateValues ( QAStateUUID , fieldName, childUUID, fieldValue) VALUES ('" + \
                    self.getID() + "', 'd:box:value', '" + thisObjUUIDAsString + "', ? );";
                objDatabaseInterface.interfaceBleed_insertValuesForBlob(\
                    commandToExecute, [str(thisObj)]  );  
            return;

    def _convertDescriptionToSQLCommands(self):
        for thisObj in self.getDescription():
            thisObjUUIDAsString = DescriptionState.convertDescriptionStateIDToStringUniformly(thisObj.getID()); # need for the case of a MetaCondition_Conjunction, but does the proper thing in other cases as well
            objDatabaseInterface.exec(\
                "INSERT INTO QAStateValues ( QAStateUUID , fieldName, childUUID) VALUES ('" + \
                self.getID() + "', 'd:root', '" + thisObjUUIDAsString + "' );"  \
            );
            self._helper_convertDescriptionToSQLCommands(thisObj);       
        objDatabaseInterface.commit();
        return;


    def _convertDictionaryToSQLCommands(self,leadingTag, dictToRecord):
        requires(isinstance(leadingTag, str));
        requires(len(leadingTag) > 0);
        requires(isinstance(dictToRecord, dict));
        requires(all([isinstance(x, str) for x in dictToRecord.keys()]));
        requires(all([(len(x) > 0) for x in dictToRecord.keys()]));
        for keyName in dictToRecord:
            assert(isinstance(keyName, str));
            fieldName = leadingTag + ":" + keyName;
            objDatabaseInterface.interfaceBleed_insertValuesForBlob(\
                "INSERT INTO QAStateValues (QAStateUUID, fieldName, fieldValue) VALUES ( '" + self.getID() + "', '" + fieldName + "' , ?);", \
                [dictToRecord[keyName]] \
            );
        objDatabaseInterface.commit();
        return;

    def recordInDatabase(self, variablesBoxesProducedMayBeOver, universeForBoxes):
        self._convertDictionaryToSQLCommands("p", self.internalDictionary["mostRecentOperatorParameters"])
        self._convertDictionaryToSQLCommands("s", self.sideInformationDict);
        self._convertDescriptionToSQLCommands();
        self._convertBoxesToSQLCommandsToRecordSummaryStatistics(variablesBoxesProducedMayBeOver, universeForBoxes);
        return;


class FirstState_DescriptionState(DescriptionState):

    def getFloatValueForBoxDivisionCutoff(self):
        print("Enter a fraction of the universe box length to limit refinement to at the beginning. " + \
              "Value must be a positive real number less than or equal to one.", flush=True);
        thisLine = sys.stdin.readline();
        thisLineMissingNewLine = thisLine.replace("\n", "");
        floatValueForBoxDivisionCutoff = None;
        try:
            floatValueForBoxDivisionCutoff = float(thisLineMissingNewLine);
        except:
            raise Exception("Unable to convert input into a floating point value: " + \
                            str(thisLineMissingNewLine));
        assert(floatValueForBoxDivisionCutoff != None);
        if(floatValueForBoxDivisionCutoff <= 0 or floatValueForBoxDivisionCutoff > 1):
            raise Exception("Provided value is outside of the range (0.0, 1.0]. Value provided: " + \
                            str(thisLineMissingNewLine));
        return floatValueForBoxDivisionCutoff;

    def __init__(self):
        DescriptionState.__init__(self);
        parameterDict = self.internalDictionary["mostRecentOperatorParameters"];
        self.internalDictionary["mostRecentOperatorParameters"]["removedPredicates"] = set();
        parameterDict["floatValueForBoxDivisionCutoff"] = self.getFloatValueForBoxDivisionCutoff();
        parameterDict["limitOnNumberOfTimesToMerge"] = config.defaultValues.limitOnNumberOfTimesToMerge;
        parameterDict["splitOnlyOnRelaventVariables"]=True;
        parameterDict["precisionForMerging"]= config.defaultValues.precisionForMerging;
        parameterDict["numberOfSamplesToTry"]=int(config.defaultValues.numberOfSamplesToTry);
        parameterDict["produceGreaterAbstraction"]=False;
        parameterDict["exponentialComponent"]= 0.0; 
        return;



