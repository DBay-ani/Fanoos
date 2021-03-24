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



from utils.contracts import *;

from boxesAndBoxOperations.getBox import isProperBox, getBox, getDimensionOfBox, getJointBox, getContainingBox, getRandomBox, boxSize, getRandomVectorInBox;

import z3;

from boxesAndBoxOperations.codeForGettingSamplesBetweenBoxes import getSampleVectorsToCheckAgainst, getBoxCenter;

from domainsAndConditions.baseClassConditionsToSpecifyPredictsWith import CharacterizationConditionsBaseClass,\
         Condition_TheBoxItself, MetaCondition_Conjunction;

import config;


def removePredicatesImpliedByOthers(coveringDescriptionsFiltered, \
    dictMappingConditionToBoxesItIsConsistentWith, listOfBoxes, \
    listMappingAxisIndexToVariableInQuestion, dictMappingConditionIDToVolumeCoveredAndUniqueVolumeCovered_initial):

    z3Solver = coveringDescriptionsFiltered[0].z3Solver;

    boxItselfConditionIDs = frozenset([x.getID() for x in coveringDescriptionsFiltered if \
        isinstance(x, Condition_TheBoxItself)]);
    conjunctionConditionIDs = frozenset([x.getID() for x in coveringDescriptionsFiltered if \
        isinstance(x, MetaCondition_Conjunction)]);
    assert(boxItselfConditionIDs.isdisjoint(conjunctionConditionIDs));
    otherPredicateIDs = frozenset([x.getID() for x in coveringDescriptionsFiltered if \
            not ((x.getID() in boxItselfConditionIDs) or (x.getID() in conjunctionConditionIDs))]); 
    assert(otherPredicateIDs.isdisjoint(boxItselfConditionIDs));
    assert(conjunctionConditionIDs.isdisjoint(otherPredicateIDs));

    orderToConsiderElements = [\
        sorted( list(x), \
            key=(lambda x: dictMappingConditionIDToVolumeCoveredAndUniqueVolumeCovered_initial[x][
                "uniqueVolumeCovered"] ) \
        )
        for x in [\
            boxItselfConditionIDs, conjunctionConditionIDs, otherPredicateIDs \
        ] \
    ];

    newDescription = [x.getID() for x in coveringDescriptionsFiltered];
    perminentSetOfBoxesToCheckOver = set([]);
    for thisListToConsider in orderToConsiderElements:
        for thisPredID in thisListToConsider:
            restOfPreds = [x for x in coveringDescriptionsFiltered if \
                ((x.getID() in newDescription) and (x.getID() != thisPredID))];
            assert(len(restOfPreds) == len(newDescription) -1 );
            boxesCoveredByThisPred = dictMappingConditionToBoxesItIsConsistentWith[thisPredID];
            setOfBoxesToCheckOver = perminentSetOfBoxesToCheckOver.union(boxesCoveredByThisPred);
            removeThisPred = True;
            for thisBoxIndex in setOfBoxesToCheckOver :
                verdict = checkIfPredicateRepetativeForThisBox(listOfBoxes[thisBoxIndex], restOfPreds, z3Solver, \
                    listMappingAxisIndexToVariableInQuestion);
                if( not verdict ): 
                    removeThisPred = False;
                    break;
            if(removeThisPred):
                assert(thisPredID in newDescription);
                newDescription.remove(thisPredID);
                assert(thisPredID not in newDescription);
                perminentSetOfBoxesToCheckOver = setOfBoxesToCheckOver;
                assert(boxesCoveredByThisPred.issubset(perminentSetOfBoxesToCheckOver));
  
    coveringDescriptionsFiltered = [x for x in coveringDescriptionsFiltered if (x.getID() in newDescription)];
    return coveringDescriptionsFiltered;


def _helper_getFunctionToCheckWhetherNoPointsInTheBoxStatisfyCondition_convertBoxToFormulaConstraints(listMappingAxisIndexToVariableInQuestion, thisBox):
    requires(isProperBox(thisBox));
    requires(getDimensionOfBox(thisBox) == len(listMappingAxisIndexToVariableInQuestion));
    F = z3.And([ \
        z3.And( float(thisBox[index, 0]) <= listMappingAxisIndexToVariableInQuestion[index], \
                listMappingAxisIndexToVariableInQuestion[index] <= float(thisBox[index, 1]) \
              ) \
        for index in range(0, len(listMappingAxisIndexToVariableInQuestion))     ]);
    return F;


def checkIfPredicateRepetativeForThisBox(thisBox, restOfConditions, z3Solver, listMappingAxisIndexToVariableInQuestion):

    # TODO: split the two sections below into two functions...

    #V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
    # for efficiency, some probabilistic checks to see if some quick random sampling
    # shows that the disjunction of the conditions fail to cover the whole box...
    #===========================================================================

    numberOfSamples=\
        config.defaultValues.numberOfStatisticalSamplesToTakeIn_numberOfStatisticalSamplesToTakeIn_getFunctionToCheckWhetherNoPointsInTheBoxStatisfyCondition

    for thisSampleIndex in range(0, numberOfSamples):
        randomVector = getRandomVectorInBox(thisBox).reshape(getDimensionOfBox(thisBox), 1);
        noConditionsHold = True;
        for thisCondition in restOfConditions:
            if(thisCondition.pythonFormatEvaluation(randomVector)):
                noConditionsHold = False;
                break;
        if(noConditionsHold):
            return False;
    #^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^

    #V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
    # Formal check in the case that probabilistic sampling wasn't able to disprove
    # that the disjunction of the conditions to prove the statement...
    #===========================================================================

    z3Solver.reset(); # this might be the expensive.... TODO: check
    # disjunctive normal form - each element in the list is a clause which we or-together....
    formulaToCheck = \
        (\
            z3.ForAll( listMappingAxisIndexToVariableInQuestion , \
                z3.Implies(\
                    _helper_getFunctionToCheckWhetherNoPointsInTheBoxStatisfyCondition_convertBoxToFormulaConstraints(\
                        listMappingAxisIndexToVariableInQuestion, thisBox), \
                    z3.Or([x.z3FormattedCondition for x in restOfConditions]) \
                ) \
            ) \
        );
    z3Solver.add(formulaToCheck);
    verdict = (z3Solver.check() == z3.z3.sat);
    z3Solver.reset();
    return verdict;
    #^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^




