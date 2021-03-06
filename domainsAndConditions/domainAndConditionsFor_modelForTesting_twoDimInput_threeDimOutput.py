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


import config;
_LOCALDEBUGFLAG = config.debugFlags.get_v_print_ForThisFile(__file__);
    

import numpy as np;
from utils.contracts import *;

from boxesAndBoxOperations.getBox import getDimensionOfBox;

import z3;

from domainsAndConditions.baseClassConditionsToSpecifyPredictsWith import CharacterizationConditionsBaseClass, CharacterizationCondition_FromPythonFunction;
from domainsAndConditions.baseClassDomainInformation import BaseClassDomainInformation ;

from domainsAndConditions.utilsForDefiningPredicates import *;

class DomainFor_modelForTesting_twoDimInput_threeDimOutput(BaseClassDomainInformation):

    def __init__(self, z3SolverInstance):
        requires(isinstance(z3SolverInstance, z3.z3.Solver));
        self.initializedConditions = None;
        self.initialize_baseConditions(z3SolverInstance);
        assert(self.initializedConditions != None);
        return;

    @staticmethod
    def getUUID():
        return "862fb8c4-fee4-4181-a79f-9127cd8b2b64";

    @staticmethod
    def getInputSpaceUniverseBox():
        orderOfVariables = __class__.inputSpaceVariables();
        dictMappingVariableToBound = {\
            "in_x" : [-15.0, 15.0], \
            "in_y" : [-10.0, 10.0] \
        };
        thisUniverseBox =  __class__._helper_getInputSpaceUniverseBox(\
                               orderOfVariables, dictMappingVariableToBound);
        ensures(getDimensionOfBox(thisUniverseBox) == len(DomainFor_modelForTesting_twoDimInput_threeDimOutput.inputSpaceVariables()));
        return thisUniverseBox;

    @staticmethod
    def inputSpaceVariables():
        return [\
            z3.Real(x) for x in ["in_x", "in_y"] ];

    @staticmethod
    def outputSpaceVariables():
        return [z3.Real(x) for x in ["out_u", "out_v", "out_w"]];

    @staticmethod
    def getName():
        return "Domain For modelForTesting_twoDimInput_threeDimOutput";

    def initialize_baseConditions(self, z3SolverInstance):
        dictMappingPredicateStringNameToUUID = \
        {
        "IN_X_NEG20DOT0TONEG19DOT0" : "a4433189-76b4-40ee-b9ca-3cc9a37287e0" , \
        "IN_Y_NEG20DOT0TONEG19DOT0" : "af2466bf-d1f8-4038-9925-9bbfe361d2bf" , \
        "IN_X_NEG20DOT0TONEG19DOT5" : "00fa6c5b-297d-4ac3-98cd-77f3e1adb219" , \
        "IN_Y_NEG20DOT0TONEG19DOT5" : "ada870ed-9146-4c1c-871f-79e7b3cc874a" , \
        "IN_X_NEG19DOT5TONEG19DOT0" : "27db3e7c-2325-4d57-bd14-195801daa63e" , \
        "IN_Y_NEG19DOT5TONEG19DOT0" : "8fb0f9e5-a8e1-4d9a-b8a8-d137f6b71752" , \
        "IN_X_NEG19DOT0TONEG18DOT0" : "cc1adca9-7447-4533-8a67-23c48bfe308a" , \
        "IN_Y_NEG19DOT0TONEG18DOT0" : "4baa96b3-d607-4349-a0d2-a3c3049f8e58" , \
        "IN_X_NEG19DOT0TONEG18DOT5" : "439f46f3-1069-4113-8ffe-189290726769" , \
        "IN_Y_NEG19DOT0TONEG18DOT5" : "6b654a18-e769-4e33-907a-8d82ee17ffa7" , \
        "IN_X_NEG18DOT5TONEG18DOT0" : "968532b2-229b-4bbe-a081-229869a84b88" , \
        "IN_Y_NEG18DOT5TONEG18DOT0" : "fadef770-7076-4caa-87cd-8c98a678f8c2" , \
        "IN_X_NEG18DOT0TONEG17DOT0" : "587249c7-08e4-4c96-bfcc-48814c846835" , \
        "IN_Y_NEG18DOT0TONEG17DOT0" : "853a5e21-7938-427d-b7de-6b57829a2b77" , \
        "IN_X_NEG18DOT0TONEG17DOT5" : "29980b88-66f3-4a02-92b7-ac31cd4996c6" , \
        "IN_Y_NEG18DOT0TONEG17DOT5" : "c76acb90-aca2-49b6-a6ec-52b9c7ba160e" , \
        "IN_X_NEG17DOT5TONEG17DOT0" : "5618eeed-2ac3-4493-b41d-b9e7a5f7514a" , \
        "IN_Y_NEG17DOT5TONEG17DOT0" : "a2882b0c-3e14-4d05-bb66-01a5a21a87c0" , \
        "IN_X_NEG17DOT0TONEG16DOT0" : "64d06e42-7b30-420d-ae68-56ee9752e1a1" , \
        "IN_Y_NEG17DOT0TONEG16DOT0" : "0ff594d9-7b0f-47df-bd51-2b33dbe962ec" , \
        "IN_X_NEG17DOT0TONEG16DOT5" : "ace4d588-dd75-4ea5-af22-c1ba593e7f61" , \
        "IN_Y_NEG17DOT0TONEG16DOT5" : "e77d0f66-7d91-4ff7-baa6-799678ffe5c8" , \
        "IN_X_NEG16DOT5TONEG16DOT0" : "8540b40f-7c06-4d0d-bccb-01f8ab531e3b" , \
        "IN_Y_NEG16DOT5TONEG16DOT0" : "f1c9493c-9909-4d79-94e3-ad836a2402c3" , \
        "IN_X_NEG16DOT0TONEG15DOT0" : "25940dfb-139a-4818-9506-e5b6ccbefbf8" , \
        "IN_Y_NEG16DOT0TONEG15DOT0" : "f52f964c-6806-48dc-8120-b0dbe6dd5e1b" , \
        "IN_X_NEG16DOT0TONEG15DOT5" : "69104c3c-d41d-4a3a-b2d0-007835efc458" , \
        "IN_Y_NEG16DOT0TONEG15DOT5" : "43edc681-1ce2-4a66-99bc-f30645215daa" , \
        "IN_X_NEG15DOT5TONEG15DOT0" : "45f2dafc-1aa2-463a-88bc-ee966121e715" , \
        "IN_Y_NEG15DOT5TONEG15DOT0" : "71c14806-e483-4cb1-99a3-75d626d15300" , \
        "IN_X_NEG15DOT0TONEG14DOT0" : "626a60a3-0861-4d2c-9927-841ba0ac532b" , \
        "IN_Y_NEG15DOT0TONEG14DOT0" : "db645a1d-5432-42fe-8767-f542cde05416" , \
        "IN_X_NEG15DOT0TONEG14DOT5" : "c1abdd83-3ff1-4e50-a56b-d58184e5cd8a" , \
        "IN_Y_NEG15DOT0TONEG14DOT5" : "0565882b-39c1-427d-9767-8499c7b515a0" , \
        "IN_X_NEG14DOT5TONEG14DOT0" : "c601caa0-319d-4f7f-b1e1-13bccbd24f41" , \
        "IN_Y_NEG14DOT5TONEG14DOT0" : "cbd840b3-e2d0-4c2b-876a-cc64ba1cc427" , \
        "IN_X_NEG14DOT0TONEG13DOT0" : "51300a19-6787-4a41-bc79-7691d961cf70" , \
        "IN_Y_NEG14DOT0TONEG13DOT0" : "b047ac7a-b6bd-4b09-b81b-d1f8a9dcfab9" , \
        "IN_X_NEG14DOT0TONEG13DOT5" : "b4d2ec8c-710f-4592-a869-80dab2d1414f" , \
        "IN_Y_NEG14DOT0TONEG13DOT5" : "1ee85eae-0993-4e97-b752-e463f376efe9" , \
        "IN_X_NEG13DOT5TONEG13DOT0" : "78a666f7-804a-4ce3-923b-a14045920887" , \
        "IN_Y_NEG13DOT5TONEG13DOT0" : "40731390-2894-48fd-b0d1-cd7f80f10d18" , \
        "IN_X_NEG13DOT0TONEG12DOT0" : "2c3b3591-85d9-45c0-baf5-1909cefacd5a" , \
        "IN_Y_NEG13DOT0TONEG12DOT0" : "d36c9989-9814-4f8b-980a-50bd4c8f238d" , \
        "IN_X_NEG13DOT0TONEG12DOT5" : "69c53f8c-0d10-4778-bed0-cec7bcfb2063" , \
        "IN_Y_NEG13DOT0TONEG12DOT5" : "03dad021-26ed-4179-96b7-523f9dc912e8" , \
        "IN_X_NEG12DOT5TONEG12DOT0" : "2ccfd41f-2a05-448c-a6a2-142da6637b39" , \
        "IN_Y_NEG12DOT5TONEG12DOT0" : "d317fcc9-ae63-4161-b880-a7bd67375e73" , \
        "IN_X_NEG12DOT0TONEG11DOT0" : "51479b7d-f175-40d7-be9e-f16137771e31" , \
        "IN_Y_NEG12DOT0TONEG11DOT0" : "f1eee679-5a5f-4a9c-bf63-16d91caec1aa" , \
        "IN_X_NEG12DOT0TONEG11DOT5" : "3b91a11e-c4a4-40d6-b140-a5b501d070a6" , \
        "IN_Y_NEG12DOT0TONEG11DOT5" : "6974f47e-3252-4a2b-9f63-d7ae9b8c8217" , \
        "IN_X_NEG11DOT5TONEG11DOT0" : "be66388d-96a7-4716-9def-e3e0af1f1752" , \
        "IN_Y_NEG11DOT5TONEG11DOT0" : "16e137f2-4ba1-49a6-bffb-19dc4dcc6d81" , \
        "IN_X_NEG11DOT0TONEG10DOT0" : "39a9f3e7-4231-4df6-93e1-64eb7b7f6341" , \
        "IN_Y_NEG11DOT0TONEG10DOT0" : "67af9c29-ab5e-442b-9a64-d797feda8468" , \
        "IN_X_NEG11DOT0TONEG10DOT5" : "625c48f3-9f57-4a86-bca4-59ec473b3f8e" , \
        "IN_Y_NEG11DOT0TONEG10DOT5" : "b0f237bc-58f3-459b-8366-2656f47d3cc4" , \
        "IN_X_NEG10DOT5TONEG10DOT0" : "4b8522a3-c832-468e-a916-8b11fc624794" , \
        "IN_Y_NEG10DOT5TONEG10DOT0" : "2b7dcb52-9072-4dab-b418-c0232ec392f1" , \
        "IN_X_NEG10DOT0TONEG9DOT0" : "4d62bd4d-a6d9-4f82-9a6c-2ba109458ca6" , \
        "IN_Y_NEG10DOT0TONEG9DOT0" : "8ff7ed78-4fdf-45b6-a3f2-cabd94b96f94" , \
        "IN_X_NEG10DOT0TONEG9DOT5" : "4b1e2b67-6eff-42cf-ab9a-d01bd5423b5e" , \
        "IN_Y_NEG10DOT0TONEG9DOT5" : "48134cc8-fe5a-4606-a09b-a927ef70ebe3" , \
        "IN_X_NEG9DOT5TONEG9DOT0" : "770095c5-cb48-4f56-93ba-724eb4c0a7a0" , \
        "IN_Y_NEG9DOT5TONEG9DOT0" : "047e9fdb-1264-46e5-81d5-c1e7091056f1" , \
        "IN_X_NEG9DOT0TONEG8DOT0" : "8d7c5b32-8d27-4648-b6b5-e4f2efd27f7f" , \
        "IN_Y_NEG9DOT0TONEG8DOT0" : "da5d8383-804c-42f4-96a0-eeea550ee60b" , \
        "IN_X_NEG9DOT0TONEG8DOT5" : "40638a98-e293-4e75-a6b3-a246623edc72" , \
        "IN_Y_NEG9DOT0TONEG8DOT5" : "8e3181f8-1f2d-4641-96b6-182617994179" , \
        "IN_X_NEG8DOT5TONEG8DOT0" : "395a242b-22ff-4d59-95dc-0ad22709b9fb" , \
        "IN_Y_NEG8DOT5TONEG8DOT0" : "9c2be67e-0795-48c3-924d-e8d926301ffb" , \
        "IN_X_NEG8DOT0TONEG7DOT0" : "84a1bd60-113f-49ea-a754-ef57b0f3ce43" , \
        "IN_Y_NEG8DOT0TONEG7DOT0" : "4836315b-1edb-469b-94ab-3e8a03b5c730" , \
        "IN_X_NEG8DOT0TONEG7DOT5" : "383a743b-7298-4896-87fc-8ceafc4d9a40" , \
        "IN_Y_NEG8DOT0TONEG7DOT5" : "c25f7a16-b487-4469-a3bd-3e718dcc13a3" , \
        "IN_X_NEG7DOT5TONEG7DOT0" : "ceb05436-82d1-4cf8-97f5-79d3c1540733" , \
        "IN_Y_NEG7DOT5TONEG7DOT0" : "a0b257a9-0780-4417-837b-8f1203191bf2" , \
        "IN_X_NEG7DOT0TONEG6DOT0" : "ce82425e-5534-4234-a0ba-756d84bdb139" , \
        "IN_Y_NEG7DOT0TONEG6DOT0" : "226a157c-a16e-47b9-b7d6-6dc84d58b00f" , \
        "IN_X_NEG7DOT0TONEG6DOT5" : "3adfa009-6f21-4e6f-b5cc-585226624db6" , \
        "IN_Y_NEG7DOT0TONEG6DOT5" : "9cc45874-6aa3-4b5f-a4b6-b9847c0199d6" , \
        "IN_X_NEG6DOT5TONEG6DOT0" : "532b4414-5489-4826-9274-40c2af334581" , \
        "IN_Y_NEG6DOT5TONEG6DOT0" : "4b6a8963-ee0c-476c-bf49-fd44037dc249" , \
        "IN_X_NEG6DOT0TONEG5DOT0" : "9ff58550-049f-4868-9f57-7fc0fcd5e52e" , \
        "IN_Y_NEG6DOT0TONEG5DOT0" : "618a76c3-8bae-4e3b-9059-337840da93b5" , \
        "IN_X_NEG6DOT0TONEG5DOT5" : "6aebb372-f16c-4397-8f9d-e5d9100c53ee" , \
        "IN_Y_NEG6DOT0TONEG5DOT5" : "f61ccd75-76c4-4a68-8240-78936a100940" , \
        "IN_X_NEG5DOT5TONEG5DOT0" : "b7b4cbdd-d288-4ae9-82bb-b85acd8e01bd" , \
        "IN_Y_NEG5DOT5TONEG5DOT0" : "92a4f6ca-85cc-48ea-b34f-2860e87a8aa3" , \
        "IN_X_NEG5DOT0TONEG4DOT0" : "7ba75dc7-aa8a-4077-a876-f28d82346d8a" , \
        "IN_Y_NEG5DOT0TONEG4DOT0" : "4ab1118e-e2f2-4fd3-8d0b-a668596d550e" , \
        "IN_X_NEG5DOT0TONEG4DOT5" : "92c3bbf0-fac3-4d14-9e7c-0345329fb813" , \
        "IN_Y_NEG5DOT0TONEG4DOT5" : "7aa4bd98-db81-4701-8a33-086fe6cc912d" , \
        "IN_X_NEG4DOT5TONEG4DOT0" : "d8a547d6-01ab-404e-bb63-343e0a7fd9b9" , \
        "IN_Y_NEG4DOT5TONEG4DOT0" : "2b539615-fe2b-4b11-8ea4-233cb4ba14b7" , \
        "IN_X_NEG4DOT0TONEG3DOT0" : "5faf30a6-2399-46d6-bc2f-c12743a37442" , \
        "IN_Y_NEG4DOT0TONEG3DOT0" : "b81341ff-a584-46d8-9df8-783c7c2768f3" , \
        "IN_X_NEG4DOT0TONEG3DOT5" : "a79603c6-ea18-4e39-984f-6d49ecd39557" , \
        "IN_Y_NEG4DOT0TONEG3DOT5" : "5b3183d5-7301-4e86-a0c5-1ed46e24e22d" , \
        "IN_X_NEG3DOT5TONEG3DOT0" : "bee6978c-d3d8-4e9a-9736-3f1ecf43e782" , \
        "IN_Y_NEG3DOT5TONEG3DOT0" : "06583dc8-c512-4c78-be9b-271d784886b3" , \
        "IN_X_NEG3DOT0TONEG2DOT0" : "1a11629b-cacf-4482-b531-7938991dc1ca" , \
        "IN_Y_NEG3DOT0TONEG2DOT0" : "3ff6efc1-2040-4f03-bef7-de4ac0172a22" , \
        "IN_X_NEG3DOT0TONEG2DOT5" : "b50b4ad7-da01-4984-a5dc-6188d44a439a" , \
        "IN_Y_NEG3DOT0TONEG2DOT5" : "e4d05686-0da9-40b3-a1dd-4b8b899c20a1" , \
        "IN_X_NEG2DOT5TONEG2DOT0" : "e42efd59-cb86-4006-8727-f9387216de01" , \
        "IN_Y_NEG2DOT5TONEG2DOT0" : "01753372-fcb9-44d5-b43e-fffd5a95e863" , \
        "IN_X_NEG2DOT0TONEG1DOT0" : "0fcee40c-128d-4f26-a411-63044702a2a6" , \
        "IN_Y_NEG2DOT0TONEG1DOT0" : "a81363db-2b5f-4e26-9129-5cec5d7ac394" , \
        "IN_X_NEG2DOT0TONEG1DOT5" : "4f3949fd-5c94-4b55-aeac-09d04a4f9e01" , \
        "IN_Y_NEG2DOT0TONEG1DOT5" : "b02bd089-f2c0-44f7-bba2-33eff483e523" , \
        "IN_X_NEG1DOT5TONEG1DOT0" : "3e8d1c9d-4db4-4a5b-b802-c45c336ad36e" , \
        "IN_Y_NEG1DOT5TONEG1DOT0" : "4c3f8d47-c56d-467f-9404-03da78b8179c" , \
        "IN_X_NEG1DOT0TO0DOT0" : "1bb3e692-595c-4dfa-8f81-f8f2b08737fe" , \
        "IN_Y_NEG1DOT0TO0DOT0" : "a52a4f32-82b2-4159-9c13-e0e315b82afb" , \
        "IN_X_NEG1DOT0TONEG0DOT5" : "528b9ed9-0653-41fc-bdf8-005f8ada4612" , \
        "IN_Y_NEG1DOT0TONEG0DOT5" : "ce11302c-a9d7-47ae-a08e-e79606ee9a3e" , \
        "IN_X_NEG0DOT5TO0DOT0" : "fc631ce8-50ff-4144-bcdc-503da3087b83" , \
        "IN_Y_NEG0DOT5TO0DOT0" : "85b4d189-2cf9-41a7-b5a9-44cc63e238d0" , \
        "IN_X_0DOT0TO1DOT0" : "5e7c692d-ed60-4bf2-939a-8ee88a8e11f0" , \
        "IN_Y_0DOT0TO1DOT0" : "d82fd6e7-00ea-4c35-8ca5-65dcc3fb1323" , \
        "IN_X_0DOT0TO0DOT5" : "80e2ac72-f9a3-40fa-9233-34b6b9d5129f" , \
        "IN_Y_0DOT0TO0DOT5" : "8ebeb4ad-a3e0-4d51-8e28-ec0d9b39f4a1" , \
        "IN_X_0DOT5TO1DOT0" : "cd2bae95-fa25-4585-bd8b-5c4e447895e3" , \
        "IN_Y_0DOT5TO1DOT0" : "cb4bf7bc-de85-41cd-8cf6-0aaaa7d8089d" , \
        "IN_X_1DOT0TO2DOT0" : "101ba361-d53b-4d97-b551-d4b34c1b157a" , \
        "IN_Y_1DOT0TO2DOT0" : "752f195f-55a8-4217-b312-156942f8991c" , \
        "IN_X_1DOT0TO1DOT5" : "84b8bd46-4458-45cc-aa36-b0b1d8ca957e" , \
        "IN_Y_1DOT0TO1DOT5" : "82ba291a-b3fe-44f3-a351-aca4701b3284" , \
        "IN_X_1DOT5TO2DOT0" : "329307b5-60ef-4220-ae17-d58d0a634746" , \
        "IN_Y_1DOT5TO2DOT0" : "342cc253-b267-414e-8f05-2ff955cfd809" , \
        "IN_X_2DOT0TO3DOT0" : "bae8d113-3126-4d62-a7a5-91e63a4bcc9c" , \
        "IN_Y_2DOT0TO3DOT0" : "4b02ecef-8e81-4531-8fc2-e50b2128e222" , \
        "IN_X_2DOT0TO2DOT5" : "1b33c149-7a56-48b2-b135-af394bb81360" , \
        "IN_Y_2DOT0TO2DOT5" : "92ddb9ac-93bc-4e6d-b680-96f42134c110" , \
        "IN_X_2DOT5TO3DOT0" : "178084f0-3a6b-49d2-8946-531d84cba704" , \
        "IN_Y_2DOT5TO3DOT0" : "7011e195-2d5e-4872-9973-514b7f1a5bcf" , \
        "IN_X_3DOT0TO4DOT0" : "68f24b75-751b-4d1b-8c24-eef3105f1326" , \
        "IN_Y_3DOT0TO4DOT0" : "289705c4-1bf5-4140-9ab1-cb6e2b67e44f" , \
        "IN_X_3DOT0TO3DOT5" : "f8ec99b4-1682-4521-b797-442633940930" , \
        "IN_Y_3DOT0TO3DOT5" : "6cfbc0c6-cad1-484a-9173-1e3275a2cc0b" , \
        "IN_X_3DOT5TO4DOT0" : "7aa418e6-bc01-4af8-9309-01393148e694" , \
        "IN_Y_3DOT5TO4DOT0" : "564ab9f0-3b40-4285-ab99-23a4a98a9310" , \
        "IN_X_4DOT0TO5DOT0" : "33b6d03b-a9e2-4fbc-9252-4197df1bd46b" , \
        "IN_Y_4DOT0TO5DOT0" : "788bee6a-4599-43c9-93ab-b20bfe592a92" , \
        "IN_X_4DOT0TO4DOT5" : "d71f7753-c5ba-4b38-b3d8-4668d5dc23e0" , \
        "IN_Y_4DOT0TO4DOT5" : "3c6b0c26-635a-44f8-8b0b-c4cda931104e" , \
        "IN_X_4DOT5TO5DOT0" : "6fe5df4a-0c10-466e-80d4-431deff596a4" , \
        "IN_Y_4DOT5TO5DOT0" : "d072c736-5c6e-48e2-bebb-d95009621f94" , \
        "IN_X_5DOT0TO6DOT0" : "4163d4af-1555-4508-b1e4-0dc471550090" , \
        "IN_Y_5DOT0TO6DOT0" : "f845a53b-dd7e-4dbf-a902-787f47f8510a" , \
        "IN_X_5DOT0TO5DOT5" : "dee6c8cd-3a34-4ebd-b36e-4d16cbd30b7d" , \
        "IN_Y_5DOT0TO5DOT5" : "4981a768-4a3d-490d-93e8-b688663ebf4c" , \
        "IN_X_5DOT5TO6DOT0" : "c8950432-f64e-48c7-a3b9-09f076d8cd4a" , \
        "IN_Y_5DOT5TO6DOT0" : "00c69e7a-07d6-4780-baba-58a40d7e0617" , \
        "IN_X_6DOT0TO7DOT0" : "3eeaabb3-0187-4c94-88d2-e08df5667046" , \
        "IN_Y_6DOT0TO7DOT0" : "7513a1bd-dd23-4674-8fbc-e0fdfebb7c9f" , \
        "IN_X_6DOT0TO6DOT5" : "0cf7d9ba-8ce7-4bd9-9d9b-319cc3752c48" , \
        "IN_Y_6DOT0TO6DOT5" : "dc9851cf-f5d6-4bb6-9b9b-0d42078f34a1" , \
        "IN_X_6DOT5TO7DOT0" : "176d7020-63da-4639-a322-644f2dd4bcce" , \
        "IN_Y_6DOT5TO7DOT0" : "2f930d1c-5f50-4830-aa5e-7393dd8bfe1b" , \
        "IN_X_7DOT0TO8DOT0" : "98af9e31-4e34-4e34-acb5-600c8e30af51" , \
        "IN_Y_7DOT0TO8DOT0" : "dbec776a-4404-40e1-8725-10e840aaabe1" , \
        "IN_X_7DOT0TO7DOT5" : "3573bed3-51d9-489d-ba17-a1615ea49d86" , \
        "IN_Y_7DOT0TO7DOT5" : "9462d943-92ea-49d2-9143-45f43fb8c8e5" , \
        "IN_X_7DOT5TO8DOT0" : "cc077f35-3585-462f-8e55-b6f9f702c1a4" , \
        "IN_Y_7DOT5TO8DOT0" : "accab2ce-14e7-495f-b89e-a8b504b27b24" , \
        "IN_X_8DOT0TO9DOT0" : "be42244e-a669-40c6-9b2a-bb5670793a47" , \
        "IN_Y_8DOT0TO9DOT0" : "040af208-5557-4179-83c8-0d95e177806d" , \
        "IN_X_8DOT0TO8DOT5" : "b4b5f321-5947-4a21-9384-f124619521d0" , \
        "IN_Y_8DOT0TO8DOT5" : "21273ece-77ae-4090-bd9f-5d4b4ab9fbff" , \
        "IN_X_8DOT5TO9DOT0" : "108710ca-1e87-45ff-878f-0aa0445936ac" , \
        "IN_Y_8DOT5TO9DOT0" : "15c66172-c735-47e3-9e75-4012c6ad62ca" , \
        "IN_X_9DOT0TO10DOT0" : "8db2dcbb-36bb-4362-a833-0e0b8ffe8e1c" , \
        "IN_Y_9DOT0TO10DOT0" : "acec202b-f003-4fa3-9e25-574ea85d1727" , \
        "IN_X_9DOT0TO9DOT5" : "be84f09a-0043-41b0-b5cd-0ff6a26ec3f8" , \
        "IN_Y_9DOT0TO9DOT5" : "0477ca4f-ec8d-4dc4-9b6e-7ded67c48f33" , \
        "IN_X_9DOT5TO10DOT0" : "da99305c-9fae-46c8-adc6-c8da9a968521" , \
        "IN_Y_9DOT5TO10DOT0" : "ec8af7ea-9203-4039-b239-ab102c986d0c" , \
        "IN_X_10DOT0TO11DOT0" : "b7a71df0-5802-4a76-892c-b18df73ab8a7" , \
        "IN_Y_10DOT0TO11DOT0" : "013cc284-9815-42d6-8f9f-b7d9ad1aeffa" , \
        "IN_X_10DOT0TO10DOT5" : "137ff452-1f00-4d1d-90ac-6a6c95da329a" , \
        "IN_Y_10DOT0TO10DOT5" : "6a7e3c62-e10f-40f8-b0b8-bda5ec3ab977" , \
        "IN_X_10DOT5TO11DOT0" : "d0ac2499-f368-4552-af2f-c95efc0a2a8c" , \
        "IN_Y_10DOT5TO11DOT0" : "0933204f-1b6d-4222-b10e-1035cd1168af" , \
        "IN_X_11DOT0TO12DOT0" : "9d9fb80e-6e68-40b4-941e-07655eff7793" , \
        "IN_Y_11DOT0TO12DOT0" : "5383099e-f2fd-4cae-9e74-9877e2cff74b" , \
        "IN_X_11DOT0TO11DOT5" : "c0c94728-c407-4c1d-bda9-e529781157c6" , \
        "IN_Y_11DOT0TO11DOT5" : "7c0b0531-62b3-4e42-bf21-9ffbee857c19" , \
        "IN_X_11DOT5TO12DOT0" : "4e731d8c-4af1-41c0-b646-35274a27e0a7" , \
        "IN_Y_11DOT5TO12DOT0" : "4b670337-5fb6-4cc7-8cf1-9b13c874107b" , \
        "IN_X_12DOT0TO13DOT0" : "4354b8bd-175d-47c2-bcb1-6430099915ad" , \
        "IN_Y_12DOT0TO13DOT0" : "e2c2ef43-e96c-48b0-ba05-9a7c80c478e3" , \
        "IN_X_12DOT0TO12DOT5" : "0477c0b5-d26e-4ae3-9e1f-d5ea5a687a59" , \
        "IN_Y_12DOT0TO12DOT5" : "5fbeee8e-66f3-487d-843c-570f66fd8ce7" , \
        "IN_X_12DOT5TO13DOT0" : "8d65ed79-824d-4f4a-9509-d6af07894159" , \
        "IN_Y_12DOT5TO13DOT0" : "6af9fec5-346d-41c7-9d23-bb79b43addc6" , \
        "IN_X_13DOT0TO14DOT0" : "7b8bf3c2-3795-4353-96eb-e8a4dcc9240d" , \
        "IN_Y_13DOT0TO14DOT0" : "587c0470-2f6d-412b-b74c-b06acd291e98" , \
        "IN_X_13DOT0TO13DOT5" : "1811a603-a153-4187-8c7c-1cb8157dae64" , \
        "IN_Y_13DOT0TO13DOT5" : "374d040f-dce0-4710-99a1-5f2f3ebd0fbe" , \
        "IN_X_13DOT5TO14DOT0" : "2dbdc1e8-7097-4a7a-99cd-357beb04fb94" , \
        "IN_Y_13DOT5TO14DOT0" : "5d094076-1438-4ffa-b37d-e70b924a1a89" , \
        "IN_X_14DOT0TO15DOT0" : "41c99974-a930-439e-9af9-1c4a874ecd0b" , \
        "IN_Y_14DOT0TO15DOT0" : "56df92f8-9a5e-4938-b321-4340909c7857" , \
        "IN_X_14DOT0TO14DOT5" : "bb7122e1-fd3d-4de3-bbd1-e8c22727dc4d" , \
        "IN_Y_14DOT0TO14DOT5" : "cb669d00-ed49-4392-82b1-eb7415537f62" , \
        "IN_X_14DOT5TO15DOT0" : "7929c7eb-96ad-4a3b-a9c0-94307f6010c2" , \
        "IN_Y_14DOT5TO15DOT0" : "9eae5ee9-8521-48bf-9e84-bbe35e9de6b4" , \
        "IN_X_15DOT0TO16DOT0" : "d5aed402-3c2f-483e-840a-e387224e66af" , \
        "IN_Y_15DOT0TO16DOT0" : "2ef013d6-af60-4a3d-aab6-ef2762717167" , \
        "IN_X_15DOT0TO15DOT5" : "9148b03d-0580-49cf-916f-4b53f7d0080a" , \
        "IN_Y_15DOT0TO15DOT5" : "c4b8ff1e-96f8-4e86-bd1f-a3ccaceb203d" , \
        "IN_X_15DOT5TO16DOT0" : "49c70262-2575-406e-97b1-281a8f89c7d4" , \
        "IN_Y_15DOT5TO16DOT0" : "54febe2b-ea0d-4bd5-a1d1-0683fac97221" , \
        "IN_X_16DOT0TO17DOT0" : "a56c472d-16b6-49ec-aabd-aabce909248c" , \
        "IN_Y_16DOT0TO17DOT0" : "ccb746ba-2def-41c3-951d-28850a6a00ad" , \
        "IN_X_16DOT0TO16DOT5" : "c4e59fc3-0d75-4d73-8933-a29b505a5822" , \
        "IN_Y_16DOT0TO16DOT5" : "2adc6b8e-af50-4816-aee1-7e18318fe509" , \
        "IN_X_16DOT5TO17DOT0" : "a80cb4b0-2fbb-45be-a5a7-44e2fa50c18f" , \
        "IN_Y_16DOT5TO17DOT0" : "a0f6c3c3-2aae-4661-b94f-25549fee7bde" , \
        "IN_X_17DOT0TO18DOT0" : "2fdd6618-0b7a-417d-a3e8-51a23d654e84" , \
        "IN_Y_17DOT0TO18DOT0" : "4f256ce6-89fe-4d9a-8fbd-def4eca91a9b" , \
        "IN_X_17DOT0TO17DOT5" : "b5234917-1341-40bf-b435-01f5f2bc4d49" , \
        "IN_Y_17DOT0TO17DOT5" : "ad7c8e5a-3f28-4fe5-8912-7f5d99a43ef1" , \
        "IN_X_17DOT5TO18DOT0" : "1356f838-2af7-4a71-97c9-2faf64a27d3b" , \
        "IN_Y_17DOT5TO18DOT0" : "7481ea0c-8a38-4f63-a57a-d7ab78582ac7" , \
        "IN_X_18DOT0TO19DOT0" : "e4a691bf-3dba-43b1-876b-02f320b76174" , \
        "IN_Y_18DOT0TO19DOT0" : "0e4fa640-ec9a-49c8-8406-9d62c054adc2" , \
        "IN_X_18DOT0TO18DOT5" : "d3362e51-8b6b-4821-81e3-d79f5358bd0c" , \
        "IN_Y_18DOT0TO18DOT5" : "a0288b89-deae-4985-a222-d245347e4425" , \
        "IN_X_18DOT5TO19DOT0" : "cc3110ce-4e51-4d3c-a727-7890492a591f" , \
        "IN_Y_18DOT5TO19DOT0" : "d81cbdb0-b895-47fc-b00c-ed1804ecb3d5" , \
        "IN_X_19DOT0TO20DOT0" : "13f564ca-de0a-47ae-bf9a-eef973cbea35" , \
        "IN_Y_19DOT0TO20DOT0" : "10bd1380-bb72-4e72-9879-9947150cedee" , \
        "IN_X_19DOT0TO19DOT5" : "a936a781-dca0-497a-8c1a-8f941de73e68" , \
        "IN_Y_19DOT0TO19DOT5" : "64837bca-80bf-4032-a24f-1c8f73752a47" , \
        "IN_X_19DOT5TO20DOT0" : "52b574b4-b3f6-4bfa-9511-655c0598aa85" , \
        "IN_Y_19DOT5TO20DOT0" : "9a14e009-09ec-408b-ae67-4d1cba135d28" , \
        "OUT_U_NEG25DOT0TONEG24DOT0" : "36aec72b-242d-42e7-9a95-b4be4214ad32" , \
        "OUT_V_NEG25DOT0TONEG24DOT0" : "d047b7cc-7d72-4637-a808-41b35cdeda48" , \
        "OUT_W_NEG25DOT0TONEG24DOT0" : "f2ba7e85-7089-4364-a5a6-55412aae1684" , \
        "OUT_U_NEG25DOT0TONEG24DOT5" : "7103ed36-4bb2-4d02-85b6-7d2a20ecba11" , \
        "OUT_V_NEG25DOT0TONEG24DOT5" : "ba81511a-bbf8-44fa-b00c-2a418f83a709" , \
        "OUT_W_NEG25DOT0TONEG24DOT5" : "6a27fc7e-f3a9-4402-a54d-5dd5fe912c6f" , \
        "OUT_U_NEG24DOT5TONEG24DOT0" : "00de3d55-1268-4496-946c-43423d3733bd" , \
        "OUT_V_NEG24DOT5TONEG24DOT0" : "30b9472c-5913-4754-b863-908a14bce311" , \
        "OUT_W_NEG24DOT5TONEG24DOT0" : "3ddbda5e-6476-4024-af5a-a31a66fdc9b0" , \
        "OUT_U_NEG24DOT0TONEG23DOT0" : "07963d8c-45e9-4be1-ade7-90ddfc9731cd" , \
        "OUT_V_NEG24DOT0TONEG23DOT0" : "e3b96ac9-b454-430a-a28d-409c90c491a7" , \
        "OUT_W_NEG24DOT0TONEG23DOT0" : "c258b08c-382b-46f2-92a6-964893ba19ef" , \
        "OUT_U_NEG24DOT0TONEG23DOT5" : "1f9424e8-f35b-41e2-a412-a51b32b30f81" , \
        "OUT_V_NEG24DOT0TONEG23DOT5" : "0613b0c1-b459-4d28-b72f-c30429d6867d" , \
        "OUT_W_NEG24DOT0TONEG23DOT5" : "982af104-b96e-437b-b580-2a8d2b2ce2d2" , \
        "OUT_U_NEG23DOT5TONEG23DOT0" : "b0c09ddb-e399-45f5-be5d-e235a2f5cee7" , \
        "OUT_V_NEG23DOT5TONEG23DOT0" : "df7c7744-f491-485f-82a1-8c0ba1a01457" , \
        "OUT_W_NEG23DOT5TONEG23DOT0" : "ad1f2272-a6d3-477f-bbf4-d7a16b6b21b9" , \
        "OUT_U_NEG23DOT0TONEG22DOT0" : "33d6e154-d40a-4c88-8949-83b93afc21d5" , \
        "OUT_V_NEG23DOT0TONEG22DOT0" : "e5da1cf3-e977-4261-af63-c54d2d855e9c" , \
        "OUT_W_NEG23DOT0TONEG22DOT0" : "5371a203-13c6-4ba3-bea9-3a4c99a057f5" , \
        "OUT_U_NEG23DOT0TONEG22DOT5" : "1205b423-669f-4878-be73-886d09f7351f" , \
        "OUT_V_NEG23DOT0TONEG22DOT5" : "4513144d-4d5a-4e35-aeaa-a0616d96a58a" , \
        "OUT_W_NEG23DOT0TONEG22DOT5" : "488b91c1-b5f6-4f61-9691-87a62be6f838" , \
        "OUT_U_NEG22DOT5TONEG22DOT0" : "00ecfac4-7989-436d-8cbd-b0f425699dfe" , \
        "OUT_V_NEG22DOT5TONEG22DOT0" : "2fa22855-e1c6-412d-94c5-45d135d192f0" , \
        "OUT_W_NEG22DOT5TONEG22DOT0" : "bfdf149d-4705-4783-84db-ded323dc73d0" , \
        "OUT_U_NEG22DOT0TONEG21DOT0" : "298aab04-bbb3-4f14-8016-88251a60c48b" , \
        "OUT_V_NEG22DOT0TONEG21DOT0" : "a8e3d378-aaa1-4d98-aefb-21bb8c16388b" , \
        "OUT_W_NEG22DOT0TONEG21DOT0" : "cdec1eba-fd32-4c14-8785-5c442ec8d3dc" , \
        "OUT_U_NEG22DOT0TONEG21DOT5" : "c91fffef-bbe9-4ae2-a122-4be44557d78e" , \
        "OUT_V_NEG22DOT0TONEG21DOT5" : "ec3faeb6-3a5c-4dd8-9a5f-4bd832066685" , \
        "OUT_W_NEG22DOT0TONEG21DOT5" : "9ae377de-95f0-4f57-8030-e2d7e7cc8ca8" , \
        "OUT_U_NEG21DOT5TONEG21DOT0" : "d11647b0-40e3-45f5-b38b-1fca9f5ff860" , \
        "OUT_V_NEG21DOT5TONEG21DOT0" : "63f7c161-a6f6-4144-a4c8-85ccad0c6b98" , \
        "OUT_W_NEG21DOT5TONEG21DOT0" : "7b6f6a05-7c11-494d-9e13-90743995bc04" , \
        "OUT_U_NEG21DOT0TONEG20DOT0" : "c92b796f-2b4d-4bb3-b707-d04dc1764c33" , \
        "OUT_V_NEG21DOT0TONEG20DOT0" : "d1a56b79-028d-4400-9c94-8ea07c2d6b5d" , \
        "OUT_W_NEG21DOT0TONEG20DOT0" : "c1612ed7-77ac-4366-94ed-2551d9e4cb81" , \
        "OUT_U_NEG21DOT0TONEG20DOT5" : "89253478-92d6-4425-adbd-538eb7986354" , \
        "OUT_V_NEG21DOT0TONEG20DOT5" : "1ae06644-5b79-4b52-8ffc-78ba545f8bb2" , \
        "OUT_W_NEG21DOT0TONEG20DOT5" : "72ac71f5-ae86-4a2e-b6aa-ee6b47f5a36d" , \
        "OUT_U_NEG20DOT5TONEG20DOT0" : "88f23d6d-39df-4d1a-afaf-6a4150b96f76" , \
        "OUT_V_NEG20DOT5TONEG20DOT0" : "3d9a4a34-0ab0-4da4-86fc-51766c2010f2" , \
        "OUT_W_NEG20DOT5TONEG20DOT0" : "8efd2da1-1c5f-412b-a403-879d82007166" , \
        "OUT_U_NEG20DOT0TONEG19DOT0" : "246a9bca-1872-4e70-9ca6-85ed4667c1fa" , \
        "OUT_V_NEG20DOT0TONEG19DOT0" : "3830270e-b33a-4b84-aeb4-6f974d769dac" , \
        "OUT_W_NEG20DOT0TONEG19DOT0" : "67e7da95-fb40-4bc9-af3b-fbfcf52feaeb" , \
        "OUT_U_NEG20DOT0TONEG19DOT5" : "c1c0ed2c-20f3-41bc-94d8-7bd79a5dde05" , \
        "OUT_V_NEG20DOT0TONEG19DOT5" : "34d9d356-88f9-4731-9980-63160ebfcd28" , \
        "OUT_W_NEG20DOT0TONEG19DOT5" : "fc5c6a08-3b4c-4505-8c8e-7d5860de6b88" , \
        "OUT_U_NEG19DOT5TONEG19DOT0" : "96cc7c7d-27b7-49e6-9b0a-efac9eb425f5" , \
        "OUT_V_NEG19DOT5TONEG19DOT0" : "45a29a44-4f28-4f4f-8a5b-54928364cd91" , \
        "OUT_W_NEG19DOT5TONEG19DOT0" : "42ae4b85-e78c-4d49-b47d-7fab475a10b9" , \
        "OUT_U_NEG19DOT0TONEG18DOT0" : "db61d5a0-92c4-46d9-9ea2-588571cfa3f8" , \
        "OUT_V_NEG19DOT0TONEG18DOT0" : "68f6da40-a393-42f6-9d23-d2fd512eaf75" , \
        "OUT_W_NEG19DOT0TONEG18DOT0" : "d8a87da0-2009-44a8-951e-e941e81e5952" , \
        "OUT_U_NEG19DOT0TONEG18DOT5" : "868d6ffa-196d-4b95-9c6b-9915037c2a79" , \
        "OUT_V_NEG19DOT0TONEG18DOT5" : "f6f31575-d3d4-481c-a3e8-0d5654f7951e" , \
        "OUT_W_NEG19DOT0TONEG18DOT5" : "9518a350-a115-49c8-a956-cb4b88efcdaf" , \
        "OUT_U_NEG18DOT5TONEG18DOT0" : "fee9939a-f502-4d6e-9f25-a4e8b86f2b6f" , \
        "OUT_V_NEG18DOT5TONEG18DOT0" : "90b8356b-e1a9-4624-8521-de5922918191" , \
        "OUT_W_NEG18DOT5TONEG18DOT0" : "a6c7fc39-6556-4a24-af83-08c489747485" , \
        "OUT_U_NEG18DOT0TONEG17DOT0" : "1f08d63d-b2dc-46ee-a1e6-759ea78635cc" , \
        "OUT_V_NEG18DOT0TONEG17DOT0" : "c9cdc6c4-9176-45ea-9eda-dca020c6eac1" , \
        "OUT_W_NEG18DOT0TONEG17DOT0" : "06fd9f20-8562-49ea-90b9-1840eae3cf4b" , \
        "OUT_U_NEG18DOT0TONEG17DOT5" : "b83ee8e3-65ad-4c8e-9804-1a2b414fd2e9" , \
        "OUT_V_NEG18DOT0TONEG17DOT5" : "a65addd3-4070-4d31-a918-d45b5833e94f" , \
        "OUT_W_NEG18DOT0TONEG17DOT5" : "5ba4aa72-da6c-4667-ae8d-5e4f8a009efd" , \
        "OUT_U_NEG17DOT5TONEG17DOT0" : "5eed0347-24ec-4c94-abd7-d622dbe56a1f" , \
        "OUT_V_NEG17DOT5TONEG17DOT0" : "72f7d856-ce6e-4c3a-928c-2e86abd7b0c8" , \
        "OUT_W_NEG17DOT5TONEG17DOT0" : "c57b8910-a49f-4b15-9630-8dd9ead0bcf9" , \
        "OUT_U_NEG17DOT0TONEG16DOT0" : "e80e4e1f-9053-43e1-974a-c942b65b3e31" , \
        "OUT_V_NEG17DOT0TONEG16DOT0" : "42e98f7c-e6df-4dd8-b514-0eeab46e277d" , \
        "OUT_W_NEG17DOT0TONEG16DOT0" : "548fd44f-109e-4437-a844-fca66cb6eb76" , \
        "OUT_U_NEG17DOT0TONEG16DOT5" : "e9d236b1-0146-429d-b74d-108a358a9c40" , \
        "OUT_V_NEG17DOT0TONEG16DOT5" : "5f825279-4f9c-4a83-9a5c-9396bedd64e3" , \
        "OUT_W_NEG17DOT0TONEG16DOT5" : "b7092d48-93b2-4400-b14e-1ccf62a6a806" , \
        "OUT_U_NEG16DOT5TONEG16DOT0" : "32efa15a-55ec-4487-8cc6-8afcba669fe8" , \
        "OUT_V_NEG16DOT5TONEG16DOT0" : "89daa459-7965-4868-9b20-2134d3b40b13" , \
        "OUT_W_NEG16DOT5TONEG16DOT0" : "d6710fa9-dbcd-4f98-9aa9-55b4fe73d9fd" , \
        "OUT_U_NEG16DOT0TONEG15DOT0" : "bc3409d9-1247-4020-bf3c-a2e063a704ce" , \
        "OUT_V_NEG16DOT0TONEG15DOT0" : "8247a71d-1bd7-4175-93e9-c572ab976008" , \
        "OUT_W_NEG16DOT0TONEG15DOT0" : "20b086a9-4051-4869-8f4e-b48b066a6748" , \
        "OUT_U_NEG16DOT0TONEG15DOT5" : "d1973de0-a4c8-4940-b302-0a04b519bb3f" , \
        "OUT_V_NEG16DOT0TONEG15DOT5" : "0e2a96b9-9250-486d-91cf-d6d67d6adbd3" , \
        "OUT_W_NEG16DOT0TONEG15DOT5" : "2c7dda13-e3f4-4479-825b-e5bdbd9cc798" , \
        "OUT_U_NEG15DOT5TONEG15DOT0" : "d8ca6ea0-24e8-4c1a-a53a-083c9d014665" , \
        "OUT_V_NEG15DOT5TONEG15DOT0" : "bdace2fc-1ee0-4469-a822-8fa1fd1a5bdf" , \
        "OUT_W_NEG15DOT5TONEG15DOT0" : "a47e804d-da40-45b9-9d11-8daa20d0a015" , \
        "OUT_U_NEG15DOT0TONEG14DOT0" : "2c8c772c-6bee-4c03-b67c-d06d6425a57e" , \
        "OUT_V_NEG15DOT0TONEG14DOT0" : "0287b0f1-ef1f-47bb-ad75-6aa89cfc8e6c" , \
        "OUT_W_NEG15DOT0TONEG14DOT0" : "3c44e588-2ac9-4c7e-a4e5-3be02961f6cb" , \
        "OUT_U_NEG15DOT0TONEG14DOT5" : "6c49f8ee-988a-4dbd-be1a-b6f3dadeb092" , \
        "OUT_V_NEG15DOT0TONEG14DOT5" : "69352542-5fe1-40ac-8397-2374d6ee44f1" , \
        "OUT_W_NEG15DOT0TONEG14DOT5" : "91771f82-dbe6-4912-9a66-ff2995635563" , \
        "OUT_U_NEG14DOT5TONEG14DOT0" : "fd9e2b84-0cc1-473e-b6b3-f8e75275fffc" , \
        "OUT_V_NEG14DOT5TONEG14DOT0" : "6f83d4a3-be89-4aa3-be3d-8bf74a5ec4e6" , \
        "OUT_W_NEG14DOT5TONEG14DOT0" : "e6955c0e-0c10-4ad6-bbf1-53fa18244d11" , \
        "OUT_U_NEG14DOT0TONEG13DOT0" : "0855c1f1-9f6e-44b8-b9dd-8d900232dd0b" , \
        "OUT_V_NEG14DOT0TONEG13DOT0" : "2a279cb0-78f6-4a66-89d8-c108b2846464" , \
        "OUT_W_NEG14DOT0TONEG13DOT0" : "a694e8e9-fbed-4ed3-b847-396c60dbc8cd" , \
        "OUT_U_NEG14DOT0TONEG13DOT5" : "246e8937-29d2-4b58-8b81-1fd9884c19c0" , \
        "OUT_V_NEG14DOT0TONEG13DOT5" : "6ebb225e-6097-472c-8903-0d5a0c8c9346" , \
        "OUT_W_NEG14DOT0TONEG13DOT5" : "4ff7b31b-1f53-4cb5-a0c0-816634d62de1" , \
        "OUT_U_NEG13DOT5TONEG13DOT0" : "6cfc5a71-3bea-4c31-9211-b14eeff0b982" , \
        "OUT_V_NEG13DOT5TONEG13DOT0" : "9dd2434c-6bf2-45da-8073-79af3ff5976a" , \
        "OUT_W_NEG13DOT5TONEG13DOT0" : "f0524153-e1ed-4d06-831e-ab3d5536ea95" , \
        "OUT_U_NEG13DOT0TONEG12DOT0" : "75d9ef62-83e1-40a3-8359-2f6ed241698f" , \
        "OUT_V_NEG13DOT0TONEG12DOT0" : "45b82517-e48f-494e-9684-c21a7743407d" , \
        "OUT_W_NEG13DOT0TONEG12DOT0" : "9493e477-c7b9-44a7-8e4c-980f3a9ec7b3" , \
        "OUT_U_NEG13DOT0TONEG12DOT5" : "369dd09f-76c6-4f68-bea7-067120e8c2b8" , \
        "OUT_V_NEG13DOT0TONEG12DOT5" : "3523d601-21f4-4613-bebc-9a41e52141a5" , \
        "OUT_W_NEG13DOT0TONEG12DOT5" : "8fadf7a1-9b99-479d-ab3a-cc86cc31c904" , \
        "OUT_U_NEG12DOT5TONEG12DOT0" : "945224d6-5d67-459d-a107-c4bc8a23a583" , \
        "OUT_V_NEG12DOT5TONEG12DOT0" : "58be66be-7b9f-4c82-af45-1bd8c99974ab" , \
        "OUT_W_NEG12DOT5TONEG12DOT0" : "c142d7e7-2c8b-4819-bfd2-bd951590cb28" , \
        "OUT_U_NEG12DOT0TONEG11DOT0" : "bcf21f23-7ec4-443d-90e0-969e5c706750" , \
        "OUT_V_NEG12DOT0TONEG11DOT0" : "7fc00b03-b9b7-47cd-994b-a8e8c0a463d1" , \
        "OUT_W_NEG12DOT0TONEG11DOT0" : "f1caa5fa-b921-468f-acc4-7349dbd71f28" , \
        "OUT_U_NEG12DOT0TONEG11DOT5" : "5dc20454-0d32-42e0-9351-da7279d2e4d2" , \
        "OUT_V_NEG12DOT0TONEG11DOT5" : "7e46d2b7-fa5a-4864-9ac5-263bf9b65345" , \
        "OUT_W_NEG12DOT0TONEG11DOT5" : "f0581da0-18f9-4d7b-8a08-4af8a33728a0" , \
        "OUT_U_NEG11DOT5TONEG11DOT0" : "4f4cfb53-a13e-44bd-bca5-d74c72d31b16" , \
        "OUT_V_NEG11DOT5TONEG11DOT0" : "0daba05e-fc39-41b9-897c-af8154baa97d" , \
        "OUT_W_NEG11DOT5TONEG11DOT0" : "7ae6f85f-a5b6-4e28-a506-fa0d4929981a" , \
        "OUT_U_NEG11DOT0TONEG10DOT0" : "c338cb1e-964b-4c57-9003-a7b38091f9e2" , \
        "OUT_V_NEG11DOT0TONEG10DOT0" : "b2cd950c-dfd6-47c7-9a7f-6112d8eecb25" , \
        "OUT_W_NEG11DOT0TONEG10DOT0" : "65f244d9-6191-4e94-a6d8-fc1e8879f89c" , \
        "OUT_U_NEG11DOT0TONEG10DOT5" : "e4605504-e90a-4539-b776-b1027416e898" , \
        "OUT_V_NEG11DOT0TONEG10DOT5" : "36a14614-0750-4327-a7ee-419c924377f2" , \
        "OUT_W_NEG11DOT0TONEG10DOT5" : "01405bb1-a94d-4a96-8163-1f61cf524e83" , \
        "OUT_U_NEG10DOT5TONEG10DOT0" : "549c1908-afe2-4cda-8e52-056e187cf173" , \
        "OUT_V_NEG10DOT5TONEG10DOT0" : "4541ede1-252b-409a-9b88-091e687bf0e1" , \
        "OUT_W_NEG10DOT5TONEG10DOT0" : "615e2125-40d2-4ace-a796-1db2587b6bfa" , \
        "OUT_U_NEG10DOT0TONEG9DOT0" : "39404a03-5383-4674-aa5f-a5014e7880d2" , \
        "OUT_V_NEG10DOT0TONEG9DOT0" : "d4d513e4-aa7f-447b-abae-ba8a67c07d8e" , \
        "OUT_W_NEG10DOT0TONEG9DOT0" : "6fd8031f-2d4c-4e07-8dee-d3ce56dc3159" , \
        "OUT_U_NEG10DOT0TONEG9DOT5" : "af5b2a10-0177-4a0a-860d-cc645a1f84aa" , \
        "OUT_V_NEG10DOT0TONEG9DOT5" : "04d047d5-3814-4f4f-ada7-1cd3c7cf23a0" , \
        "OUT_W_NEG10DOT0TONEG9DOT5" : "a9294466-8320-4f9d-a7e3-6463db94285e" , \
        "OUT_U_NEG9DOT5TONEG9DOT0" : "92c3db1e-b7a2-4e99-9f1d-547955ab7170" , \
        "OUT_V_NEG9DOT5TONEG9DOT0" : "9d08564f-2d8e-4732-afba-f9fb3f3e4939" , \
        "OUT_W_NEG9DOT5TONEG9DOT0" : "93309ecb-e604-4a4a-93a5-7f3ae4469d5f" , \
        "OUT_U_NEG9DOT0TONEG8DOT0" : "92dccc3a-f450-448a-94ea-811820789233" , \
        "OUT_V_NEG9DOT0TONEG8DOT0" : "31bee0d4-a0b7-4a13-86ef-bfd2c4cb091b" , \
        "OUT_W_NEG9DOT0TONEG8DOT0" : "680ddb11-987d-4cbe-84e1-10468dcb0296" , \
        "OUT_U_NEG9DOT0TONEG8DOT5" : "1c40af42-b072-4c82-8a46-7a302338306a" , \
        "OUT_V_NEG9DOT0TONEG8DOT5" : "f0bab64b-4fd9-4ee5-8987-0dfdb4e822f4" , \
        "OUT_W_NEG9DOT0TONEG8DOT5" : "5b0aaa3d-ce85-4ac2-a17d-ba08b6a3ed3a" , \
        "OUT_U_NEG8DOT5TONEG8DOT0" : "eb1d189a-9910-49da-9afc-f2a1a36fd0eb" , \
        "OUT_V_NEG8DOT5TONEG8DOT0" : "1080d198-7936-4f69-83bf-0090f2ce7e4c" , \
        "OUT_W_NEG8DOT5TONEG8DOT0" : "af8ac348-9db5-4127-8093-dde37965acb0" , \
        "OUT_U_NEG8DOT0TONEG7DOT0" : "229c1743-d4a6-4f49-8ef9-4363025528c2" , \
        "OUT_V_NEG8DOT0TONEG7DOT0" : "8d90724d-14ec-4c2e-a4e5-1b678262db99" , \
        "OUT_W_NEG8DOT0TONEG7DOT0" : "c6417f5d-2151-4300-82e8-f79f73688a84" , \
        "OUT_U_NEG8DOT0TONEG7DOT5" : "d1b71139-08e9-4576-97b7-d20c069033a3" , \
        "OUT_V_NEG8DOT0TONEG7DOT5" : "db791c3f-9514-4b62-962c-98d4466c89fb" , \
        "OUT_W_NEG8DOT0TONEG7DOT5" : "c9a4b827-76a5-41e9-8bf4-527b4c06a7ab" , \
        "OUT_U_NEG7DOT5TONEG7DOT0" : "5d933298-bcbb-4944-8d39-c47a33c23303" , \
        "OUT_V_NEG7DOT5TONEG7DOT0" : "027bcd31-cded-4312-ad38-a77ca29b08d0" , \
        "OUT_W_NEG7DOT5TONEG7DOT0" : "1f042e6c-b4b0-441c-99c6-88840c1c00f8" , \
        "OUT_U_NEG7DOT0TONEG6DOT0" : "278f7ba6-ad7d-47b2-ba68-49d544919c28" , \
        "OUT_V_NEG7DOT0TONEG6DOT0" : "417ee740-d816-474f-8fc2-092e27d11d24" , \
        "OUT_W_NEG7DOT0TONEG6DOT0" : "a963b075-e64e-4aac-addb-87e77ab8b918" , \
        "OUT_U_NEG7DOT0TONEG6DOT5" : "c17c8894-7172-4bcb-98b9-8be0bc8f91b4" , \
        "OUT_V_NEG7DOT0TONEG6DOT5" : "ef93b375-bc8b-47ae-9f48-9ed598346f57" , \
        "OUT_W_NEG7DOT0TONEG6DOT5" : "9195ae09-f4a6-4749-9a42-63b64e275999" , \
        "OUT_U_NEG6DOT5TONEG6DOT0" : "0d2607f2-1e4f-420d-aac0-74159adf910e" , \
        "OUT_V_NEG6DOT5TONEG6DOT0" : "0c69be10-edd4-4a75-87fa-89ec5a202501" , \
        "OUT_W_NEG6DOT5TONEG6DOT0" : "f9f84c2b-b1c7-4947-ab83-5d125df3216d" , \
        "OUT_U_NEG6DOT0TONEG5DOT0" : "ec0e93bc-2888-46a0-8873-75490060a0e8" , \
        "OUT_V_NEG6DOT0TONEG5DOT0" : "dd041b28-5cd3-4ab9-9068-e6a8364edbe2" , \
        "OUT_W_NEG6DOT0TONEG5DOT0" : "89a7db08-6e02-4d65-85fb-68620422b8cc" , \
        "OUT_U_NEG6DOT0TONEG5DOT5" : "cd915631-53a2-4899-968e-e45a1fdd1050" , \
        "OUT_V_NEG6DOT0TONEG5DOT5" : "46966672-d27e-48d6-9f4a-7338c7e4bf92" , \
        "OUT_W_NEG6DOT0TONEG5DOT5" : "fadc8105-d82b-4a1c-9546-b5d04ce5d1c6" , \
        "OUT_U_NEG5DOT5TONEG5DOT0" : "327db73c-9da7-4dc8-b0b3-b63cf4e3f5d7" , \
        "OUT_V_NEG5DOT5TONEG5DOT0" : "8348a655-75fa-4b24-b4d3-2c2a7c9e49e4" , \
        "OUT_W_NEG5DOT5TONEG5DOT0" : "8560bece-51bd-4d42-a582-92445b33bbbc" , \
        "OUT_U_NEG5DOT0TONEG4DOT0" : "1879854a-ac85-474b-882b-dfda13997ce0" , \
        "OUT_V_NEG5DOT0TONEG4DOT0" : "e4cc3b91-80bd-405b-a66a-07b68295753c" , \
        "OUT_W_NEG5DOT0TONEG4DOT0" : "cb3971f1-5001-4e5e-96fe-617b0522fe15" , \
        "OUT_U_NEG5DOT0TONEG4DOT5" : "33eb8237-5c87-44a3-ab51-ee5ffece1a26" , \
        "OUT_V_NEG5DOT0TONEG4DOT5" : "d2d7cf89-7e83-4bf9-af4c-4721a1fee66d" , \
        "OUT_W_NEG5DOT0TONEG4DOT5" : "fe85a5a2-5ed9-413f-b0ef-e48a06082d63" , \
        "OUT_U_NEG4DOT5TONEG4DOT0" : "92680dc0-951c-4f52-b04d-3cff9efff722" , \
        "OUT_V_NEG4DOT5TONEG4DOT0" : "bb8ad89e-2ddc-4859-b659-dcba06909cc3" , \
        "OUT_W_NEG4DOT5TONEG4DOT0" : "90ee2eb4-6076-4f28-94b4-05f5430b4669" , \
        "OUT_U_NEG4DOT0TONEG3DOT0" : "5ee43133-56cf-49e7-8a86-a95b139de14c" , \
        "OUT_V_NEG4DOT0TONEG3DOT0" : "d6c196af-2cb2-4f24-b8f5-2b60277e8c7a" , \
        "OUT_W_NEG4DOT0TONEG3DOT0" : "844b86e9-031c-48a0-a666-482a9b633623" , \
        "OUT_U_NEG4DOT0TONEG3DOT5" : "85a4a061-f21a-465b-9c32-75d1717e0ff9" , \
        "OUT_V_NEG4DOT0TONEG3DOT5" : "ddee67e4-35bb-4b6d-8818-75222ecdb82f" , \
        "OUT_W_NEG4DOT0TONEG3DOT5" : "a70e0ee6-1050-49e4-a69a-0ec31b970d79" , \
        "OUT_U_NEG3DOT5TONEG3DOT0" : "9976ede5-fa72-48be-a649-7fc10b9f322e" , \
        "OUT_V_NEG3DOT5TONEG3DOT0" : "e4f363c5-3e41-43f8-867a-82a40ee54210" , \
        "OUT_W_NEG3DOT5TONEG3DOT0" : "8bbe7276-8c6d-4517-ae43-65d80bb5dd56" , \
        "OUT_U_NEG3DOT0TONEG2DOT0" : "72e628a1-2fde-4659-a04a-c53c193d0256" , \
        "OUT_V_NEG3DOT0TONEG2DOT0" : "87d2bb01-4b6f-43d5-b66d-e3aa8e00ecec" , \
        "OUT_W_NEG3DOT0TONEG2DOT0" : "41146ed5-761a-43a5-8001-d524ea670ad6" , \
        "OUT_U_NEG3DOT0TONEG2DOT5" : "4836a4fa-0b7e-41b6-991d-76671c763f6f" , \
        "OUT_V_NEG3DOT0TONEG2DOT5" : "26a0ead7-d776-4830-8251-e49ff2b8842e" , \
        "OUT_W_NEG3DOT0TONEG2DOT5" : "d29746c7-8714-4cc6-97cc-f3abcae4fc15" , \
        "OUT_U_NEG2DOT5TONEG2DOT0" : "059e1d64-e3d8-42a8-9b94-e1c2ed643232" , \
        "OUT_V_NEG2DOT5TONEG2DOT0" : "29253612-3504-4e66-969c-515408002817" , \
        "OUT_W_NEG2DOT5TONEG2DOT0" : "47da7a90-c97a-47de-81e3-99b0d7f5ba4a" , \
        "OUT_U_NEG2DOT0TONEG1DOT0" : "9d08b9d0-7a23-41de-9bc8-6017eb4a57ea" , \
        "OUT_V_NEG2DOT0TONEG1DOT0" : "d69290b1-e796-4c96-8037-0f0f4798fd1e" , \
        "OUT_W_NEG2DOT0TONEG1DOT0" : "8ed66cb6-ba53-427f-80b3-0d580882f126" , \
        "OUT_U_NEG2DOT0TONEG1DOT5" : "c55b54c7-ed16-4a0f-ba89-7546f27f8d82" , \
        "OUT_V_NEG2DOT0TONEG1DOT5" : "2a8783a5-cb88-4137-b907-5f28592124e3" , \
        "OUT_W_NEG2DOT0TONEG1DOT5" : "d47ad741-f565-42fb-8731-001d6f6d00dd" , \
        "OUT_U_NEG1DOT5TONEG1DOT0" : "6676d2af-a125-4bca-9016-1879b3bb6341" , \
        "OUT_V_NEG1DOT5TONEG1DOT0" : "b77cda92-302d-409c-97b1-6b5c36feeaee" , \
        "OUT_W_NEG1DOT5TONEG1DOT0" : "969c7c15-ad16-423e-b126-6703a36e041a" , \
        "OUT_U_NEG1DOT0TO0DOT0" : "cfd992e5-738b-4573-badc-5c5510552b76" , \
        "OUT_V_NEG1DOT0TO0DOT0" : "dc56ef6a-e191-4cac-a353-bd9dbb38b6b1" , \
        "OUT_W_NEG1DOT0TO0DOT0" : "1ae840b1-c651-4583-a7aa-6028fbd6da57" , \
        "OUT_U_NEG1DOT0TONEG0DOT5" : "a2f5e475-94ff-4436-a7bf-12da379ddda5" , \
        "OUT_V_NEG1DOT0TONEG0DOT5" : "de69845b-3b49-42bc-acf9-d6f408418f24" , \
        "OUT_W_NEG1DOT0TONEG0DOT5" : "0123b914-8ad3-46b1-abb8-26dc7b8a973c" , \
        "OUT_U_NEG0DOT5TO0DOT0" : "fa6945b7-27ca-4a04-a0ba-44c610b9f5e4" , \
        "OUT_V_NEG0DOT5TO0DOT0" : "ffa8350b-cb9a-434e-aaaf-911709b9d29b" , \
        "OUT_W_NEG0DOT5TO0DOT0" : "d4163df7-d235-4756-b09d-18e8c69703b1" , \
        "OUT_U_0DOT0TO1DOT0" : "057947e0-44ac-47d5-b688-3987f3ec3bb4" , \
        "OUT_V_0DOT0TO1DOT0" : "6c391306-fc3a-4a6f-bac6-745c7bf8f598" , \
        "OUT_W_0DOT0TO1DOT0" : "037f8d78-7ca0-4423-a288-cee3f7c0275a" , \
        "OUT_U_0DOT0TO0DOT5" : "c1e80afd-949d-4f7c-a5f3-7fc4215419fb" , \
        "OUT_V_0DOT0TO0DOT5" : "71b1cf6f-f229-4ec1-b2e4-f96d138f5c11" , \
        "OUT_W_0DOT0TO0DOT5" : "82e2f5e9-f792-411d-abb5-b7e491cfe96f" , \
        "OUT_U_0DOT5TO1DOT0" : "8a76ef57-eead-4e3e-b0e1-21bc7fb1f2bc" , \
        "OUT_V_0DOT5TO1DOT0" : "c11e059b-a6a4-46ad-b68b-1c0bf3404a9c" , \
        "OUT_W_0DOT5TO1DOT0" : "1ece566e-0024-480a-9f84-68d497367fe3" , \
        "OUT_U_1DOT0TO2DOT0" : "1174e942-7a64-4a79-8daf-16306508f54b" , \
        "OUT_V_1DOT0TO2DOT0" : "d08a11e6-73e3-4f24-818d-b42fe6f0dcb8" , \
        "OUT_W_1DOT0TO2DOT0" : "2fc3f2f1-8036-4be6-9f02-e84dfcade8a7" , \
        "OUT_U_1DOT0TO1DOT5" : "421d9ebf-d16f-4fa2-9460-efa7a0cc793c" , \
        "OUT_V_1DOT0TO1DOT5" : "d403c072-7f7f-490c-9486-4b9aed08d8df" , \
        "OUT_W_1DOT0TO1DOT5" : "0f906e40-6c24-45ec-a89c-e42cd6adccaf" , \
        "OUT_U_1DOT5TO2DOT0" : "58735ae7-6f87-465a-8411-5aaecd02f1bb" , \
        "OUT_V_1DOT5TO2DOT0" : "09803248-757a-4e3c-b1bf-70d5206b5243" , \
        "OUT_W_1DOT5TO2DOT0" : "946acd88-2d81-4cbd-9305-4cc186a8d039" , \
        "OUT_U_2DOT0TO3DOT0" : "efa21c93-d9ba-43e7-a622-f30fb8a217b5" , \
        "OUT_V_2DOT0TO3DOT0" : "7fba2231-7d37-4b7b-9d01-52540eb02bf6" , \
        "OUT_W_2DOT0TO3DOT0" : "f2f21f24-d52a-48a3-8c4b-197f55d96c08" , \
        "OUT_U_2DOT0TO2DOT5" : "751c1a29-a137-4b7d-bfcf-f00a3610753e" , \
        "OUT_V_2DOT0TO2DOT5" : "752bb1bc-2bdf-4f4d-9aa3-f9c6a572ee4f" , \
        "OUT_W_2DOT0TO2DOT5" : "0c0ecfc9-674e-4341-a532-0950866e1935" , \
        "OUT_U_2DOT5TO3DOT0" : "42f1e26c-44d9-49bf-b1f5-bdc76a34e837" , \
        "OUT_V_2DOT5TO3DOT0" : "3c02c8be-6b92-4901-b074-bbbb18ff7c67" , \
        "OUT_W_2DOT5TO3DOT0" : "73839f74-6ef9-4219-8766-5eca24da179e" , \
        "OUT_U_3DOT0TO4DOT0" : "bce78e90-88f8-44e4-aea6-a45907ebb64c" , \
        "OUT_V_3DOT0TO4DOT0" : "0e28f77e-5f80-406d-8193-acffc8e1885d" , \
        "OUT_W_3DOT0TO4DOT0" : "658c2eb1-9f51-4e11-9134-a7a92ea032c4" , \
        "OUT_U_3DOT0TO3DOT5" : "8f061f8a-4851-4619-bb35-ab87b336b85a" , \
        "OUT_V_3DOT0TO3DOT5" : "c145194a-4217-4e91-94a1-646997db8f12" , \
        "OUT_W_3DOT0TO3DOT5" : "966fdb0e-23b7-4964-939f-08c1e0a2f183" , \
        "OUT_U_3DOT5TO4DOT0" : "51ace9de-13d8-43d6-a12c-558f6101061c" , \
        "OUT_V_3DOT5TO4DOT0" : "6a8fd123-496d-4004-9b4c-c7d061046d8a" , \
        "OUT_W_3DOT5TO4DOT0" : "62375996-b03e-49a1-b53f-3e838489f426" , \
        "OUT_U_4DOT0TO5DOT0" : "b353eb3a-baa3-484a-b8f1-63565d7574d5" , \
        "OUT_V_4DOT0TO5DOT0" : "9a454f1d-5325-4c00-98b2-c429874ee677" , \
        "OUT_W_4DOT0TO5DOT0" : "5405eeb2-db49-4b0f-8ca2-7b337d2f1978" , \
        "OUT_U_4DOT0TO4DOT5" : "b6159529-38ca-4d50-b9a7-8f92e19d8b5b" , \
        "OUT_V_4DOT0TO4DOT5" : "c37b483d-35ef-42fe-bf6d-0d50eadb7d86" , \
        "OUT_W_4DOT0TO4DOT5" : "f0bad062-b3a9-4251-952e-9393ff0f1cae" , \
        "OUT_U_4DOT5TO5DOT0" : "c7cf7620-9fa3-47d5-aae7-f0dcbad3769f" , \
        "OUT_V_4DOT5TO5DOT0" : "4eab4e10-bc48-44db-9f03-69f07aca2e84" , \
        "OUT_W_4DOT5TO5DOT0" : "7352a600-f8e2-4838-a374-3c589616ab13" , \
        "OUT_U_5DOT0TO6DOT0" : "cc72c315-1a30-4267-866b-5403ef827608" , \
        "OUT_V_5DOT0TO6DOT0" : "44049260-4aa8-4846-b285-82413558bc9d" , \
        "OUT_W_5DOT0TO6DOT0" : "7b9e7df0-d9fb-42c8-852f-270d73b87fb4" , \
        "OUT_U_5DOT0TO5DOT5" : "088d5399-f868-4074-af30-25a4e2a9c782" , \
        "OUT_V_5DOT0TO5DOT5" : "4abffdd5-2666-4a42-ae87-6b1ea91f9300" , \
        "OUT_W_5DOT0TO5DOT5" : "fee4311a-9fa6-45d1-84da-a722b6669774" , \
        "OUT_U_5DOT5TO6DOT0" : "97dbe86e-e522-4c8d-96b7-dcaf51e014f9" , \
        "OUT_V_5DOT5TO6DOT0" : "f1bde450-2b0f-4035-bbb5-48c3f20fa806" , \
        "OUT_W_5DOT5TO6DOT0" : "180aeb32-4ad8-4d12-a821-1c7d7e50810f" , \
        "OUT_U_6DOT0TO7DOT0" : "6886b98c-b7bd-45e3-8236-d9e4dc984d0b" , \
        "OUT_V_6DOT0TO7DOT0" : "6a4973b0-fda1-4a5a-908a-d4b9c6617c5d" , \
        "OUT_W_6DOT0TO7DOT0" : "00578db1-01cb-458b-bbe6-ce5cb4e87eee" , \
        "OUT_U_6DOT0TO6DOT5" : "d911a510-1763-4f42-a2b2-047a54229f24" , \
        "OUT_V_6DOT0TO6DOT5" : "77220a62-ea00-4bdb-873d-3cb941ab0df2" , \
        "OUT_W_6DOT0TO6DOT5" : "21eadd91-69d4-4ef1-85c1-f28d17ffe98e" , \
        "OUT_U_6DOT5TO7DOT0" : "484a2ad4-a7cd-4681-9d6f-65d774b05137" , \
        "OUT_V_6DOT5TO7DOT0" : "1d019214-4047-4130-8258-a92c3fa1b936" , \
        "OUT_W_6DOT5TO7DOT0" : "a17a4f91-ff9c-4e28-830c-383f238281dc" , \
        "OUT_U_7DOT0TO8DOT0" : "a17336ed-b098-4d6e-bc3d-78b938e43aa8" , \
        "OUT_V_7DOT0TO8DOT0" : "1c4f9f45-6b7e-4a14-a81d-a22e2b3bcb30" , \
        "OUT_W_7DOT0TO8DOT0" : "9e148160-60e4-47e0-9f34-eae9d02520af" , \
        "OUT_U_7DOT0TO7DOT5" : "5ce17a02-4872-4203-8689-452889cab106" , \
        "OUT_V_7DOT0TO7DOT5" : "f9e583ca-45f9-4891-82f2-897af56df166" , \
        "OUT_W_7DOT0TO7DOT5" : "ee97787c-38c0-4c20-909d-1cf6cec3ff41" , \
        "OUT_U_7DOT5TO8DOT0" : "e6023ff4-1086-461b-9323-d567fbedd8ad" , \
        "OUT_V_7DOT5TO8DOT0" : "84d180cc-6ebc-4b74-97fc-547fdc4bcd3e" , \
        "OUT_W_7DOT5TO8DOT0" : "92688b6c-1a3c-4a1b-a582-07abe0eb4983" , \
        "OUT_U_8DOT0TO9DOT0" : "8bf2d272-5816-4d74-8ef5-14dbb816c22f" , \
        "OUT_V_8DOT0TO9DOT0" : "782a139c-bb97-4155-b613-b919b2980503" , \
        "OUT_W_8DOT0TO9DOT0" : "ea3bf5c2-dba1-4757-8cf1-5e3b67825dcb" , \
        "OUT_U_8DOT0TO8DOT5" : "d8a413f5-c833-4260-9098-e3569738703a" , \
        "OUT_V_8DOT0TO8DOT5" : "79e4551b-1bef-436b-abf6-0e06440c0868" , \
        "OUT_W_8DOT0TO8DOT5" : "a6644e56-ca00-47ea-9110-17620cdadb09" , \
        "OUT_U_8DOT5TO9DOT0" : "cffe29df-9a8a-403e-b67a-2c3c2897e18f" , \
        "OUT_V_8DOT5TO9DOT0" : "a0ffe61a-beef-4f7c-b9e8-13655c88a3dd" , \
        "OUT_W_8DOT5TO9DOT0" : "980cfa10-53fb-4626-9f13-6d6bf58cad69" , \
        "OUT_U_9DOT0TO10DOT0" : "66589cb0-5ab4-4d66-bd04-e81415f12611" , \
        "OUT_V_9DOT0TO10DOT0" : "2a279ef7-23b2-46c2-a820-3e21905103ed" , \
        "OUT_W_9DOT0TO10DOT0" : "e0420b69-1b86-46d1-8be3-0b000db7607b" , \
        "OUT_U_9DOT0TO9DOT5" : "634bd87d-e743-4f08-89cf-a582f732950d" , \
        "OUT_V_9DOT0TO9DOT5" : "4304731b-9b09-4da0-bdb6-88a85e91549b" , \
        "OUT_W_9DOT0TO9DOT5" : "a4b40ddd-048a-4960-9b44-d66ba2d9dc92" , \
        "OUT_U_9DOT5TO10DOT0" : "c8d647a9-aeb9-4714-92f2-a34880842f76" , \
        "OUT_V_9DOT5TO10DOT0" : "73505e1d-2c86-4c11-89ed-3e9e101257c8" , \
        "OUT_W_9DOT5TO10DOT0" : "5e076135-920d-49c2-ba82-195e04192359" , \
        "OUT_U_10DOT0TO11DOT0" : "9882ecc5-5110-4ca6-bbb3-b6047e0bd398" , \
        "OUT_V_10DOT0TO11DOT0" : "dae4c3e4-d398-4a84-8fc1-bcf602d0c604" , \
        "OUT_W_10DOT0TO11DOT0" : "a56fd570-a1ec-4dcc-b87c-f7f6af474479" , \
        "OUT_U_10DOT0TO10DOT5" : "0897d1f0-49c1-4ee9-a536-61f3d8a18f98" , \
        "OUT_V_10DOT0TO10DOT5" : "ce6aba6b-c0d4-4701-855a-52cc4cd65795" , \
        "OUT_W_10DOT0TO10DOT5" : "9a2845c0-5a1f-4ce1-89e6-0e3849d2a4bc" , \
        "OUT_U_10DOT5TO11DOT0" : "5f79be58-676b-48dd-ba51-074b3069bf0a" , \
        "OUT_V_10DOT5TO11DOT0" : "378a4868-08b2-4271-b862-4092e2bab025" , \
        "OUT_W_10DOT5TO11DOT0" : "5304f8f5-c079-4781-8ae6-6ac4aa6968b0" , \
        "OUT_U_11DOT0TO12DOT0" : "39627381-01af-4146-94bd-ed938f2c67fa" , \
        "OUT_V_11DOT0TO12DOT0" : "24011bcf-fbe0-4e19-8d75-315f7aa91ec7" , \
        "OUT_W_11DOT0TO12DOT0" : "078f6aa7-f939-4a21-b23c-0b5066108af2" , \
        "OUT_U_11DOT0TO11DOT5" : "49203555-3b30-4516-ac98-df5252ed8cbd" , \
        "OUT_V_11DOT0TO11DOT5" : "c3781d3f-82e7-45ed-990b-1a79463a1b3f" , \
        "OUT_W_11DOT0TO11DOT5" : "4766d48e-652f-497f-9efb-e34d92effe14" , \
        "OUT_U_11DOT5TO12DOT0" : "ff3d45eb-207f-4871-8d66-6a3daedc0ba9" , \
        "OUT_V_11DOT5TO12DOT0" : "8141e8ba-ebee-44ca-b998-089d6952cf61" , \
        "OUT_W_11DOT5TO12DOT0" : "79da6cb3-022c-4d9e-ac14-faeb8e4668e7" , \
        "OUT_U_12DOT0TO13DOT0" : "a8aa1199-6b8e-43a6-accd-a5b16b4ef194" , \
        "OUT_V_12DOT0TO13DOT0" : "43c28d6b-83fd-40ab-a5f4-560bed339c19" , \
        "OUT_W_12DOT0TO13DOT0" : "93e81b2a-d808-401d-b54a-48045d1361ed" , \
        "OUT_U_12DOT0TO12DOT5" : "00b802cf-ccd1-40d5-8c4e-80b38da0877f" , \
        "OUT_V_12DOT0TO12DOT5" : "eee42375-aff6-47b6-a1a8-7cf39f6becdf" , \
        "OUT_W_12DOT0TO12DOT5" : "30299f2d-5bca-4f58-a9b3-8b32786891dc" , \
        "OUT_U_12DOT5TO13DOT0" : "7a299503-c221-4ed9-8b29-c93d8d6dc869" , \
        "OUT_V_12DOT5TO13DOT0" : "b63b4388-f923-4236-bf2a-6faa81317808" , \
        "OUT_W_12DOT5TO13DOT0" : "b6c0aa12-ac81-4930-9ee8-5d205e557526" , \
        "OUT_U_13DOT0TO14DOT0" : "84a47068-27fc-48ca-bf9b-db473c460905" , \
        "OUT_V_13DOT0TO14DOT0" : "55d0b3e0-a631-4ed0-8a17-cccf29a79639" , \
        "OUT_W_13DOT0TO14DOT0" : "c8e2f739-8efb-40dd-b7b2-d9a9b3ebb0aa" , \
        "OUT_U_13DOT0TO13DOT5" : "1c6039df-7c7c-474d-96f1-c0eb2c737636" , \
        "OUT_V_13DOT0TO13DOT5" : "a3ba5e88-a451-49e1-87b1-e7b0dc0e9d33" , \
        "OUT_W_13DOT0TO13DOT5" : "68cf9a53-1342-496d-91b0-423d69641262" , \
        "OUT_U_13DOT5TO14DOT0" : "2ad1ba55-49ca-4604-b37c-d1bf7e7a48ca" , \
        "OUT_V_13DOT5TO14DOT0" : "84991731-cf30-4f96-9504-0ad7b1182c43" , \
        "OUT_W_13DOT5TO14DOT0" : "dd05a5cc-295c-4563-b802-c8e3c9e81d3c" , \
        "OUT_U_14DOT0TO15DOT0" : "7e880d4d-ba1c-4c9a-834d-21287ba94b92" , \
        "OUT_V_14DOT0TO15DOT0" : "a44e2770-1066-4f5d-9291-c222e403003c" , \
        "OUT_W_14DOT0TO15DOT0" : "d072b579-432c-4ace-bdca-4f74ae4e2d66" , \
        "OUT_U_14DOT0TO14DOT5" : "ddeb88c3-5c1e-4dd9-b526-e16275f89cb9" , \
        "OUT_V_14DOT0TO14DOT5" : "8812aea6-42bf-4535-9512-bfb6a6e17145" , \
        "OUT_W_14DOT0TO14DOT5" : "9ad8333b-ca38-48f0-8192-ffa9ae8549b1" , \
        "OUT_U_14DOT5TO15DOT0" : "71011be7-6eff-4e80-97bd-aa6c3941f2c6" , \
        "OUT_V_14DOT5TO15DOT0" : "ee1624d5-6857-43c6-89f3-9be893f46e49" , \
        "OUT_W_14DOT5TO15DOT0" : "1982c969-0c11-4df0-b521-32a9c2c5ebf7" , \
        "OUT_U_15DOT0TO16DOT0" : "d5ffd4de-7bf3-401d-9577-0869d47e3210" , \
        "OUT_V_15DOT0TO16DOT0" : "ffb3bb43-d01a-491f-9fb1-2bc5560b3555" , \
        "OUT_W_15DOT0TO16DOT0" : "02241f3a-682d-4040-ba51-9448ba469485" , \
        "OUT_U_15DOT0TO15DOT5" : "f50b2f86-e88b-4bca-83ba-7f15c86161d8" , \
        "OUT_V_15DOT0TO15DOT5" : "1c278483-564b-4f98-8bfb-c1d8f6fc6597" , \
        "OUT_W_15DOT0TO15DOT5" : "3570518b-33b9-4b98-b4f1-2fed49964cd6" , \
        "OUT_U_15DOT5TO16DOT0" : "c3cce42c-405f-4049-9e06-6fb0e5b65660" , \
        "OUT_V_15DOT5TO16DOT0" : "b30d11dd-8def-442e-81a0-391cabd6abcd" , \
        "OUT_W_15DOT5TO16DOT0" : "3c1f0afe-4507-48ba-a371-e0947243981b" , \
        "OUT_U_16DOT0TO17DOT0" : "ce677953-9125-4e3d-9db7-ac917a18e0ed" , \
        "OUT_V_16DOT0TO17DOT0" : "672fd031-148c-45bd-b0af-b6a37a74b9eb" , \
        "OUT_W_16DOT0TO17DOT0" : "8adcba33-aca1-4ba3-a06e-34801149d381" , \
        "OUT_U_16DOT0TO16DOT5" : "781b3dd5-f6ad-44aa-acd0-f2a6b3e00721" , \
        "OUT_V_16DOT0TO16DOT5" : "ee168de9-d0ba-4c3e-8cdb-ca5395506d88" , \
        "OUT_W_16DOT0TO16DOT5" : "01d0da09-24d3-4541-ab5a-a4f3ac631c1c" , \
        "OUT_U_16DOT5TO17DOT0" : "836e0587-6195-4aaf-ae1f-79d905f81cf8" , \
        "OUT_V_16DOT5TO17DOT0" : "9ae82186-8e35-4796-bb94-445bc67504bb" , \
        "OUT_W_16DOT5TO17DOT0" : "ec207ba0-7b9a-4e43-ae2d-a1ea54246123" , \
        "OUT_U_17DOT0TO18DOT0" : "c76484d5-61f8-404a-b6ab-bc96211603af" , \
        "OUT_V_17DOT0TO18DOT0" : "e45184fb-c17f-4bf0-8f80-3b5613982a8b" , \
        "OUT_W_17DOT0TO18DOT0" : "9b7ef67a-d1a6-4c8c-911e-1c77a8e6311c" , \
        "OUT_U_17DOT0TO17DOT5" : "8db982e0-cb05-40d1-a0a2-7704c0e58538" , \
        "OUT_V_17DOT0TO17DOT5" : "8acefb84-e950-4b64-9c23-b58a255beb53" , \
        "OUT_W_17DOT0TO17DOT5" : "d1a4ce6b-dca2-4fde-a725-006cae0f992e" , \
        "OUT_U_17DOT5TO18DOT0" : "52c66c2a-073c-47c8-af7d-851cb3ace0b6" , \
        "OUT_V_17DOT5TO18DOT0" : "0d16bfd5-b274-4a02-8c81-c6d072b6299b" , \
        "OUT_W_17DOT5TO18DOT0" : "1835c505-4639-4eb9-b520-2433881fa937" , \
        "OUT_U_18DOT0TO19DOT0" : "d65a1f13-bc75-4763-b07d-4e99378df8c7" , \
        "OUT_V_18DOT0TO19DOT0" : "7ceeb929-d25a-4012-92cb-f298a8939f29" , \
        "OUT_W_18DOT0TO19DOT0" : "8c899d90-8079-439b-869b-3586221a9010" , \
        "OUT_U_18DOT0TO18DOT5" : "6f051e00-72cc-472a-a653-c74871cced90" , \
        "OUT_V_18DOT0TO18DOT5" : "b2f0d489-d923-458c-8f34-23bd84a33939" , \
        "OUT_W_18DOT0TO18DOT5" : "0ac838cb-2067-4e24-9065-46f31364bd60" , \
        "OUT_U_18DOT5TO19DOT0" : "0ae5326b-545a-43a8-aaf7-5fa5486237ad" , \
        "OUT_V_18DOT5TO19DOT0" : "a5627227-79a7-4a35-b6b5-03ba4aef7e6a" , \
        "OUT_W_18DOT5TO19DOT0" : "eb3eda85-10fd-4497-89a5-61be0ddead7d" , \
        "OUT_U_19DOT0TO20DOT0" : "1616e30d-60bd-4c12-8a26-02d39e0d69ae" , \
        "OUT_V_19DOT0TO20DOT0" : "7b15f03c-8297-47df-9299-0e0df22d95f7" , \
        "OUT_W_19DOT0TO20DOT0" : "019ec06e-cf04-4c99-9b90-40d499ffaa7e" , \
        "OUT_U_19DOT0TO19DOT5" : "4328dcfe-6745-4486-aa87-79979cb38fd2" , \
        "OUT_V_19DOT0TO19DOT5" : "ec54356b-5408-499c-a13a-a0d5b5106e7a" , \
        "OUT_W_19DOT0TO19DOT5" : "51fa31e0-7c6c-47ad-8efc-25315d9003b1" , \
        "OUT_U_19DOT5TO20DOT0" : "25d19d6d-ed22-43d1-8b0b-58861d385d43" , \
        "OUT_V_19DOT5TO20DOT0" : "4ed8bdf0-6017-4d04-b8e8-97eead5b1754" , \
        "OUT_W_19DOT5TO20DOT0" : "ec8065ce-7beb-400c-bd2e-49aa0a6765f6" , \
        "OUT_U_20DOT0TO21DOT0" : "35ac9f5c-4768-4743-99cb-b2dfad45fa77" , \
        "OUT_V_20DOT0TO21DOT0" : "c7a11023-4a6a-464a-8d18-f8c6ac526eb4" , \
        "OUT_W_20DOT0TO21DOT0" : "762c147c-fb47-4088-929f-dccb374177e7" , \
        "OUT_U_20DOT0TO20DOT5" : "ffe132b5-377c-455c-96f3-68c7ecfadf21" , \
        "OUT_V_20DOT0TO20DOT5" : "a44e5669-f1cc-45a9-b6ca-80abb5cd7a68" , \
        "OUT_W_20DOT0TO20DOT5" : "9e3ceca8-1467-4b06-9b65-2435536905bf" , \
        "OUT_U_20DOT5TO21DOT0" : "eb30b936-fc65-4d16-a0c5-d44de7525102" , \
        "OUT_V_20DOT5TO21DOT0" : "bbe267bc-71b8-420a-8884-d62abcce038c" , \
        "OUT_W_20DOT5TO21DOT0" : "62369b32-e7a9-45b6-9938-00660c61519b" , \
        "OUT_U_21DOT0TO22DOT0" : "a3d293b6-7017-4d1b-889b-ff7398eeb4c7" , \
        "OUT_V_21DOT0TO22DOT0" : "7399a022-3266-43d7-9ee1-4655530081d4" , \
        "OUT_W_21DOT0TO22DOT0" : "45135a50-1400-4ef7-907b-b080738f4534" , \
        "OUT_U_21DOT0TO21DOT5" : "f72d4436-cd80-43c5-ad99-c9ace2862e38" , \
        "OUT_V_21DOT0TO21DOT5" : "40152a78-db1c-4602-8418-2bf37f74e8ce" , \
        "OUT_W_21DOT0TO21DOT5" : "2d544ab6-d3e9-4583-99fd-26eab0564ad4" , \
        "OUT_U_21DOT5TO22DOT0" : "284aaac6-4d91-45f7-99b1-44dd63f44d5f" , \
        "OUT_V_21DOT5TO22DOT0" : "645a822e-839f-4480-a07e-06d5b92f3654" , \
        "OUT_W_21DOT5TO22DOT0" : "ad080a4e-dc86-4c18-90ff-0b7627d0914e" , \
        "OUT_U_22DOT0TO23DOT0" : "0dd674fa-2943-4e4d-ae77-86866612d285" , \
        "OUT_V_22DOT0TO23DOT0" : "18aca929-8f9c-4e1d-ad44-b5cb8db8a2b6" , \
        "OUT_W_22DOT0TO23DOT0" : "21632f84-1aa7-45d6-a5e5-30d73531430b" , \
        "OUT_U_22DOT0TO22DOT5" : "4e8c14da-85d9-4054-b721-6cd861ada0a1" , \
        "OUT_V_22DOT0TO22DOT5" : "11846e41-2d90-4d12-bf89-04dc6cc53235" , \
        "OUT_W_22DOT0TO22DOT5" : "aa726cde-d38e-44d9-9c8d-6ffade84e81c" , \
        "OUT_U_22DOT5TO23DOT0" : "9cd0fd71-263b-48e8-a25b-e6e1e4b9fdb8" , \
        "OUT_V_22DOT5TO23DOT0" : "e7948ed9-4a58-48e1-8fa1-cd28ac8b4138" , \
        "OUT_W_22DOT5TO23DOT0" : "3e0dec16-decf-4b30-b9d5-abe1b10e0d78" , \
        "OUT_U_23DOT0TO24DOT0" : "34b8b11f-0d5f-4a04-be7d-749500ee92e2" , \
        "OUT_V_23DOT0TO24DOT0" : "97346dc3-258b-4751-bcba-5261720abb42" , \
        "OUT_W_23DOT0TO24DOT0" : "c1f3434e-6f5d-4fae-8f50-e520a9710cd4" , \
        "OUT_U_23DOT0TO23DOT5" : "5b10c5db-8c22-42a2-b5b2-3c85bdfd878d" , \
        "OUT_V_23DOT0TO23DOT5" : "edd3c391-6415-40f6-a2b9-7bdccc3ad7c9" , \
        "OUT_W_23DOT0TO23DOT5" : "95cfb82d-19c8-4f34-a3fb-e6a34fcc6115" , \
        "OUT_U_23DOT5TO24DOT0" : "ffb8e085-3e79-427a-890c-9db499765672" , \
        "OUT_V_23DOT5TO24DOT0" : "b33ebc35-9db7-441e-93b2-585e293237a7" , \
        "OUT_W_23DOT5TO24DOT0" : "c0ec9b23-c038-48c6-a812-8ddaaf115880" , \
        "OUT_U_24DOT0TO25DOT0" : "eb4bce3c-93e7-4a56-90ce-9d0eb55e3cc3" , \
        "OUT_V_24DOT0TO25DOT0" : "f2763f31-031e-4ada-a5d1-75d956ef3b4e" , \
        "OUT_W_24DOT0TO25DOT0" : "f5db6cd0-66d0-4232-860f-bb83e32291bb" , \
        "OUT_U_24DOT0TO24DOT5" : "43af33ae-e914-435b-9e72-4d4c71e9c7dc" , \
        "OUT_V_24DOT0TO24DOT5" : "ba9bcbc9-2ead-49b0-9dea-f6ba6d179a58" , \
        "OUT_W_24DOT0TO24DOT5" : "e4c62288-6db1-4fa6-8a1a-a1fe97ba8775" , \
        "OUT_U_24DOT5TO25DOT0" : "03cda718-804c-4383-9f66-79b27c97a342" , \
        "OUT_V_24DOT5TO25DOT0" : "91208cec-4ba5-41ce-8faa-e4fb98ed237c" , \
        "OUT_W_24DOT5TO25DOT0" : "a09776d7-1240-43ac-8fe0-e82af0f1f81a" \
        };
        # Note: manualAbstractionInformation, generally speaking, is a
        #     structured used purely in analysis scripts (as developed for
        #     the paper describing Fanoos); placing this information
        #     in the class defining the domain proved to be a convieniant place to store the
        #     information during the time of development and testing. Fanoos does not access 
        #     the information in manualAbstractionInformation when determining how to make 
        #     adjustments to respond to users. Again, it is only used in analysis scripts
        #     used to prepare results for the paper. While this sanity-checking
        #     code does not have results discussed in the paper at the time of
        #     writting this comment, we needed to fill the information for 
        #     this structure; while Fanoos itself does not examin content in
        #     manualAbstractionInformation, some code (such as checking code, e.g., contracts)
        #     expect the structure to be present and obey basic properties such as
        #     number of entries.
        #
        #     While it was convieniant for development, clearly it is not
        #     ideal have this data stored here or this structure required
        #     to be present. TODO: resolve the issue just described.
        self.manualAbstractionInformation = {\
             "predicatesAndLabels" : [\
            ("IN_X_NEG20DOT0TONEG19DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG20DOT0TONEG19DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG20DOT0TONEG19DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG20DOT0TONEG19DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG19DOT5TONEG19DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG19DOT5TONEG19DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG19DOT0TONEG18DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG19DOT0TONEG18DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG19DOT0TONEG18DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG19DOT0TONEG18DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG18DOT5TONEG18DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG18DOT5TONEG18DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG18DOT0TONEG17DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG18DOT0TONEG17DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG18DOT0TONEG17DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG18DOT0TONEG17DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG17DOT5TONEG17DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG17DOT5TONEG17DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG17DOT0TONEG16DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG17DOT0TONEG16DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG17DOT0TONEG16DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG17DOT0TONEG16DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG16DOT5TONEG16DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG16DOT5TONEG16DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG16DOT0TONEG15DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG16DOT0TONEG15DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG16DOT0TONEG15DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG16DOT0TONEG15DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG15DOT5TONEG15DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG15DOT5TONEG15DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG15DOT0TONEG14DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG15DOT0TONEG14DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG15DOT0TONEG14DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG15DOT0TONEG14DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG14DOT5TONEG14DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG14DOT5TONEG14DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG14DOT0TONEG13DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG14DOT0TONEG13DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG14DOT0TONEG13DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG14DOT0TONEG13DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG13DOT5TONEG13DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG13DOT5TONEG13DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG13DOT0TONEG12DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG13DOT0TONEG12DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG13DOT0TONEG12DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG13DOT0TONEG12DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG12DOT5TONEG12DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG12DOT5TONEG12DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG12DOT0TONEG11DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG12DOT0TONEG11DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG12DOT0TONEG11DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG12DOT0TONEG11DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG11DOT5TONEG11DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG11DOT5TONEG11DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG11DOT0TONEG10DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG11DOT0TONEG10DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG11DOT0TONEG10DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG11DOT0TONEG10DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG10DOT5TONEG10DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG10DOT5TONEG10DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG10DOT0TONEG9DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG10DOT0TONEG9DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG10DOT0TONEG9DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG10DOT0TONEG9DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG9DOT5TONEG9DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG9DOT5TONEG9DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG9DOT0TONEG8DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG9DOT0TONEG8DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG9DOT0TONEG8DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG9DOT0TONEG8DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG8DOT5TONEG8DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG8DOT5TONEG8DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG8DOT0TONEG7DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG8DOT0TONEG7DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG8DOT0TONEG7DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG8DOT0TONEG7DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG7DOT5TONEG7DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG7DOT5TONEG7DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG7DOT0TONEG6DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG7DOT0TONEG6DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG7DOT0TONEG6DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG7DOT0TONEG6DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG6DOT5TONEG6DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG6DOT5TONEG6DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG6DOT0TONEG5DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG6DOT0TONEG5DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG6DOT0TONEG5DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG6DOT0TONEG5DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG5DOT5TONEG5DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG5DOT5TONEG5DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG5DOT0TONEG4DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG5DOT0TONEG4DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG5DOT0TONEG4DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG5DOT0TONEG4DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG4DOT5TONEG4DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG4DOT5TONEG4DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG4DOT0TONEG3DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG4DOT0TONEG3DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG4DOT0TONEG3DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG4DOT0TONEG3DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG3DOT5TONEG3DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG3DOT5TONEG3DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG3DOT0TONEG2DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG3DOT0TONEG2DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG3DOT0TONEG2DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG3DOT0TONEG2DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG2DOT5TONEG2DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG2DOT5TONEG2DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG2DOT0TONEG1DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG2DOT0TONEG1DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG2DOT0TONEG1DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG2DOT0TONEG1DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG1DOT5TONEG1DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG1DOT5TONEG1DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG1DOT0TO0DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_NEG1DOT0TO0DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_NEG1DOT0TONEG0DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG1DOT0TONEG0DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_NEG0DOT5TO0DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_NEG0DOT5TO0DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_0DOT0TO1DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_0DOT0TO1DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_0DOT0TO0DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_0DOT0TO0DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_0DOT5TO1DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_0DOT5TO1DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_1DOT0TO2DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_1DOT0TO2DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_1DOT0TO1DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_1DOT0TO1DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_1DOT5TO2DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_1DOT5TO2DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_2DOT0TO3DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_2DOT0TO3DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_2DOT0TO2DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_2DOT0TO2DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_2DOT5TO3DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_2DOT5TO3DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_3DOT0TO4DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_3DOT0TO4DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_3DOT0TO3DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_3DOT0TO3DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_3DOT5TO4DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_3DOT5TO4DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_4DOT0TO5DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_4DOT0TO5DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_4DOT0TO4DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_4DOT0TO4DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_4DOT5TO5DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_4DOT5TO5DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_5DOT0TO6DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_5DOT0TO6DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_5DOT0TO5DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_5DOT0TO5DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_5DOT5TO6DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_5DOT5TO6DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_6DOT0TO7DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_6DOT0TO7DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_6DOT0TO6DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_6DOT0TO6DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_6DOT5TO7DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_6DOT5TO7DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_7DOT0TO8DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_7DOT0TO8DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_7DOT0TO7DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_7DOT0TO7DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_7DOT5TO8DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_7DOT5TO8DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_8DOT0TO9DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_8DOT0TO9DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_8DOT0TO8DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_8DOT0TO8DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_8DOT5TO9DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_8DOT5TO9DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_9DOT0TO10DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_9DOT0TO10DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_9DOT0TO9DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_9DOT0TO9DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_9DOT5TO10DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_9DOT5TO10DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_10DOT0TO11DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_10DOT0TO11DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_10DOT0TO10DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_10DOT0TO10DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_10DOT5TO11DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_10DOT5TO11DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_11DOT0TO12DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_11DOT0TO12DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_11DOT0TO11DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_11DOT0TO11DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_11DOT5TO12DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_11DOT5TO12DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_12DOT0TO13DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_12DOT0TO13DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_12DOT0TO12DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_12DOT0TO12DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_12DOT5TO13DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_12DOT5TO13DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_13DOT0TO14DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_13DOT0TO14DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_13DOT0TO13DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_13DOT0TO13DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_13DOT5TO14DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_13DOT5TO14DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_14DOT0TO15DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_14DOT0TO15DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_14DOT0TO14DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_14DOT0TO14DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_14DOT5TO15DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_14DOT5TO15DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_15DOT0TO16DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_15DOT0TO16DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_15DOT0TO15DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_15DOT0TO15DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_15DOT5TO16DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_15DOT5TO16DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_16DOT0TO17DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_16DOT0TO17DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_16DOT0TO16DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_16DOT0TO16DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_16DOT5TO17DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_16DOT5TO17DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_17DOT0TO18DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_17DOT0TO18DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_17DOT0TO17DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_17DOT0TO17DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_17DOT5TO18DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_17DOT5TO18DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_18DOT0TO19DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_18DOT0TO19DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_18DOT0TO18DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_18DOT0TO18DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_18DOT5TO19DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_18DOT5TO19DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_19DOT0TO20DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_Y_19DOT0TO20DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("IN_X_19DOT0TO19DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_19DOT0TO19DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_X_19DOT5TO20DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("IN_Y_19DOT5TO20DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG25DOT0TONEG24DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG25DOT0TONEG24DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG25DOT0TONEG24DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG25DOT0TONEG24DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG25DOT0TONEG24DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG25DOT0TONEG24DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG24DOT5TONEG24DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG24DOT5TONEG24DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG24DOT5TONEG24DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG24DOT0TONEG23DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG24DOT0TONEG23DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG24DOT0TONEG23DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG24DOT0TONEG23DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG24DOT0TONEG23DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG24DOT0TONEG23DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG23DOT5TONEG23DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG23DOT5TONEG23DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG23DOT5TONEG23DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG23DOT0TONEG22DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG23DOT0TONEG22DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG23DOT0TONEG22DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG23DOT0TONEG22DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG23DOT0TONEG22DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG23DOT0TONEG22DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG22DOT5TONEG22DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG22DOT5TONEG22DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG22DOT5TONEG22DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG22DOT0TONEG21DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG22DOT0TONEG21DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG22DOT0TONEG21DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG22DOT0TONEG21DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG22DOT0TONEG21DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG22DOT0TONEG21DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG21DOT5TONEG21DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG21DOT5TONEG21DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG21DOT5TONEG21DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG21DOT0TONEG20DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG21DOT0TONEG20DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG21DOT0TONEG20DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG21DOT0TONEG20DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG21DOT0TONEG20DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG21DOT0TONEG20DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG20DOT5TONEG20DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG20DOT5TONEG20DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG20DOT5TONEG20DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG20DOT0TONEG19DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG20DOT0TONEG19DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG20DOT0TONEG19DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG20DOT0TONEG19DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG20DOT0TONEG19DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG20DOT0TONEG19DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG19DOT5TONEG19DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG19DOT5TONEG19DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG19DOT5TONEG19DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG19DOT0TONEG18DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG19DOT0TONEG18DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG19DOT0TONEG18DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG19DOT0TONEG18DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG19DOT0TONEG18DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG19DOT0TONEG18DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG18DOT5TONEG18DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG18DOT5TONEG18DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG18DOT5TONEG18DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG18DOT0TONEG17DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG18DOT0TONEG17DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG18DOT0TONEG17DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG18DOT0TONEG17DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG18DOT0TONEG17DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG18DOT0TONEG17DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG17DOT5TONEG17DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG17DOT5TONEG17DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG17DOT5TONEG17DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG17DOT0TONEG16DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG17DOT0TONEG16DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG17DOT0TONEG16DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG17DOT0TONEG16DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG17DOT0TONEG16DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG17DOT0TONEG16DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG16DOT5TONEG16DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG16DOT5TONEG16DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG16DOT5TONEG16DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG16DOT0TONEG15DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG16DOT0TONEG15DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG16DOT0TONEG15DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG16DOT0TONEG15DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG16DOT0TONEG15DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG16DOT0TONEG15DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG15DOT5TONEG15DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG15DOT5TONEG15DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG15DOT5TONEG15DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG15DOT0TONEG14DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG15DOT0TONEG14DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG15DOT0TONEG14DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG15DOT0TONEG14DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG15DOT0TONEG14DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG15DOT0TONEG14DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG14DOT5TONEG14DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG14DOT5TONEG14DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG14DOT5TONEG14DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG14DOT0TONEG13DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG14DOT0TONEG13DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG14DOT0TONEG13DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG14DOT0TONEG13DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG14DOT0TONEG13DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG14DOT0TONEG13DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG13DOT5TONEG13DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG13DOT5TONEG13DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG13DOT5TONEG13DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG13DOT0TONEG12DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG13DOT0TONEG12DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG13DOT0TONEG12DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG13DOT0TONEG12DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG13DOT0TONEG12DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG13DOT0TONEG12DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG12DOT5TONEG12DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG12DOT5TONEG12DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG12DOT5TONEG12DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG12DOT0TONEG11DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG12DOT0TONEG11DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG12DOT0TONEG11DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG12DOT0TONEG11DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG12DOT0TONEG11DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG12DOT0TONEG11DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG11DOT5TONEG11DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG11DOT5TONEG11DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG11DOT5TONEG11DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG11DOT0TONEG10DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG11DOT0TONEG10DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG11DOT0TONEG10DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG11DOT0TONEG10DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG11DOT0TONEG10DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG11DOT0TONEG10DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG10DOT5TONEG10DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG10DOT5TONEG10DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG10DOT5TONEG10DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG10DOT0TONEG9DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG10DOT0TONEG9DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG10DOT0TONEG9DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG10DOT0TONEG9DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG10DOT0TONEG9DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG10DOT0TONEG9DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG9DOT5TONEG9DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG9DOT5TONEG9DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG9DOT5TONEG9DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG9DOT0TONEG8DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG9DOT0TONEG8DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG9DOT0TONEG8DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG9DOT0TONEG8DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG9DOT0TONEG8DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG9DOT0TONEG8DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG8DOT5TONEG8DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG8DOT5TONEG8DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG8DOT5TONEG8DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG8DOT0TONEG7DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG8DOT0TONEG7DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG8DOT0TONEG7DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG8DOT0TONEG7DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG8DOT0TONEG7DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG8DOT0TONEG7DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG7DOT5TONEG7DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG7DOT5TONEG7DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG7DOT5TONEG7DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG7DOT0TONEG6DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG7DOT0TONEG6DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG7DOT0TONEG6DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG7DOT0TONEG6DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG7DOT0TONEG6DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG7DOT0TONEG6DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG6DOT5TONEG6DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG6DOT5TONEG6DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG6DOT5TONEG6DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG6DOT0TONEG5DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG6DOT0TONEG5DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG6DOT0TONEG5DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG6DOT0TONEG5DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG6DOT0TONEG5DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG6DOT0TONEG5DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG5DOT5TONEG5DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG5DOT5TONEG5DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG5DOT5TONEG5DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG5DOT0TONEG4DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG5DOT0TONEG4DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG5DOT0TONEG4DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG5DOT0TONEG4DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG5DOT0TONEG4DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG5DOT0TONEG4DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG4DOT5TONEG4DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG4DOT5TONEG4DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG4DOT5TONEG4DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG4DOT0TONEG3DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG4DOT0TONEG3DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG4DOT0TONEG3DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG4DOT0TONEG3DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG4DOT0TONEG3DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG4DOT0TONEG3DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG3DOT5TONEG3DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG3DOT5TONEG3DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG3DOT5TONEG3DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG3DOT0TONEG2DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG3DOT0TONEG2DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG3DOT0TONEG2DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG3DOT0TONEG2DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG3DOT0TONEG2DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG3DOT0TONEG2DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG2DOT5TONEG2DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG2DOT5TONEG2DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG2DOT5TONEG2DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG2DOT0TONEG1DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG2DOT0TONEG1DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG2DOT0TONEG1DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG2DOT0TONEG1DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG2DOT0TONEG1DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG2DOT0TONEG1DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG1DOT5TONEG1DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG1DOT5TONEG1DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG1DOT5TONEG1DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG1DOT0TO0DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_NEG1DOT0TO0DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_NEG1DOT0TO0DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_NEG1DOT0TONEG0DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG1DOT0TONEG0DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG1DOT0TONEG0DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_NEG0DOT5TO0DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_NEG0DOT5TO0DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_NEG0DOT5TO0DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_0DOT0TO1DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_0DOT0TO1DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_0DOT0TO1DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_0DOT0TO0DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_0DOT0TO0DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_0DOT0TO0DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_0DOT5TO1DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_0DOT5TO1DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_0DOT5TO1DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_1DOT0TO2DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_1DOT0TO2DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_1DOT0TO2DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_1DOT0TO1DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_1DOT0TO1DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_1DOT0TO1DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_1DOT5TO2DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_1DOT5TO2DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_1DOT5TO2DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_2DOT0TO3DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_2DOT0TO3DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_2DOT0TO3DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_2DOT0TO2DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_2DOT0TO2DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_2DOT0TO2DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_2DOT5TO3DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_2DOT5TO3DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_2DOT5TO3DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_3DOT0TO4DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_3DOT0TO4DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_3DOT0TO4DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_3DOT0TO3DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_3DOT0TO3DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_3DOT0TO3DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_3DOT5TO4DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_3DOT5TO4DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_3DOT5TO4DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_4DOT0TO5DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_4DOT0TO5DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_4DOT0TO5DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_4DOT0TO4DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_4DOT0TO4DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_4DOT0TO4DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_4DOT5TO5DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_4DOT5TO5DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_4DOT5TO5DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_5DOT0TO6DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_5DOT0TO6DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_5DOT0TO6DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_5DOT0TO5DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_5DOT0TO5DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_5DOT0TO5DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_5DOT5TO6DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_5DOT5TO6DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_5DOT5TO6DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_6DOT0TO7DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_6DOT0TO7DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_6DOT0TO7DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_6DOT0TO6DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_6DOT0TO6DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_6DOT0TO6DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_6DOT5TO7DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_6DOT5TO7DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_6DOT5TO7DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_7DOT0TO8DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_7DOT0TO8DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_7DOT0TO8DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_7DOT0TO7DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_7DOT0TO7DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_7DOT0TO7DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_7DOT5TO8DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_7DOT5TO8DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_7DOT5TO8DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_8DOT0TO9DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_8DOT0TO9DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_8DOT0TO9DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_8DOT0TO8DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_8DOT0TO8DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_8DOT0TO8DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_8DOT5TO9DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_8DOT5TO9DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_8DOT5TO9DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_9DOT0TO10DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_9DOT0TO10DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_9DOT0TO10DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_9DOT0TO9DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_9DOT0TO9DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_9DOT0TO9DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_9DOT5TO10DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_9DOT5TO10DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_9DOT5TO10DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_10DOT0TO11DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_10DOT0TO11DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_10DOT0TO11DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_10DOT0TO10DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_10DOT0TO10DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_10DOT0TO10DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_10DOT5TO11DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_10DOT5TO11DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_10DOT5TO11DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_11DOT0TO12DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_11DOT0TO12DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_11DOT0TO12DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_11DOT0TO11DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_11DOT0TO11DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_11DOT0TO11DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_11DOT5TO12DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_11DOT5TO12DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_11DOT5TO12DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_12DOT0TO13DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_12DOT0TO13DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_12DOT0TO13DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_12DOT0TO12DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_12DOT0TO12DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_12DOT0TO12DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_12DOT5TO13DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_12DOT5TO13DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_12DOT5TO13DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_13DOT0TO14DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_13DOT0TO14DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_13DOT0TO14DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_13DOT0TO13DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_13DOT0TO13DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_13DOT0TO13DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_13DOT5TO14DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_13DOT5TO14DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_13DOT5TO14DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_14DOT0TO15DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_14DOT0TO15DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_14DOT0TO15DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_14DOT0TO14DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_14DOT0TO14DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_14DOT0TO14DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_14DOT5TO15DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_14DOT5TO15DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_14DOT5TO15DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_15DOT0TO16DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_15DOT0TO16DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_15DOT0TO16DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_15DOT0TO15DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_15DOT0TO15DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_15DOT0TO15DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_15DOT5TO16DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_15DOT5TO16DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_15DOT5TO16DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_16DOT0TO17DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_16DOT0TO17DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_16DOT0TO17DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_16DOT0TO16DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_16DOT0TO16DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_16DOT0TO16DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_16DOT5TO17DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_16DOT5TO17DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_16DOT5TO17DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_17DOT0TO18DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_17DOT0TO18DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_17DOT0TO18DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_17DOT0TO17DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_17DOT0TO17DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_17DOT0TO17DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_17DOT5TO18DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_17DOT5TO18DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_17DOT5TO18DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_18DOT0TO19DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_18DOT0TO19DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_18DOT0TO19DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_18DOT0TO18DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_18DOT0TO18DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_18DOT0TO18DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_18DOT5TO19DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_18DOT5TO19DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_18DOT5TO19DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_19DOT0TO20DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_19DOT0TO20DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_19DOT0TO20DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_19DOT0TO19DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_19DOT0TO19DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_19DOT0TO19DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_19DOT5TO20DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_19DOT5TO20DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_19DOT5TO20DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_20DOT0TO21DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_20DOT0TO21DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_20DOT0TO21DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_20DOT0TO20DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_20DOT0TO20DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_20DOT0TO20DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_20DOT5TO21DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_20DOT5TO21DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_20DOT5TO21DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_21DOT0TO22DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_21DOT0TO22DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_21DOT0TO22DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_21DOT0TO21DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_21DOT0TO21DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_21DOT0TO21DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_21DOT5TO22DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_21DOT5TO22DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_21DOT5TO22DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_22DOT0TO23DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_22DOT0TO23DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_22DOT0TO23DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_22DOT0TO22DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_22DOT0TO22DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_22DOT0TO22DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_22DOT5TO23DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_22DOT5TO23DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_22DOT5TO23DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_23DOT0TO24DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_23DOT0TO24DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_23DOT0TO24DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_23DOT0TO23DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_23DOT0TO23DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_23DOT0TO23DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_23DOT5TO24DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_23DOT5TO24DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_23DOT5TO24DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_24DOT0TO25DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_V_24DOT0TO25DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_W_24DOT0TO25DOT0" , "4609e380-322b-4a66-b5b6-b0e5cf0dd820"), \
            ("OUT_U_24DOT0TO24DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_24DOT0TO24DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_24DOT0TO24DOT5" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_U_24DOT5TO25DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_V_24DOT5TO25DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0"), \
            ("OUT_W_24DOT5TO25DOT0" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0") \
             ], \
             "labelDag_firstParent_secondChild" : [ \
             ("4609e380-322b-4a66-b5b6-b0e5cf0dd820" , "859a21b5-d37d-4a64-9dd3-779b1c14f9c0") \
             ] \
        };

        self.manualAbstractionInformation["predicatesAndLabels"] = \
             [ (dictMappingPredicateStringNameToUUID[x[0]] , x[1]) for x in self.manualAbstractionInformation["predicatesAndLabels"]];

        functToGetUuidProvided = (lambda predicateObjectBeingInitialized : 
            dictMappingPredicateStringNameToUUID[str(predicateObjectBeingInitialized)] );
    
        self.initializedConditions = \
            [CharacterizationCondition_FromPythonFunction(z3SolverInstance, DomainFor_modelForTesting_twoDimInput_threeDimOutput, x, functToGetUuidProvided=functToGetUuidProvided) \
             for x in getListFunctionsToBaseCondtionsOn_forInputOfDomainThisUse() + \
                      getListFunctionsToBaseCondtionsOn_forOutputOfDomainThisUse() + \
                      getListFunctionsToBaseCondtionsOn_forJointInputAndOutputDomainsInThisUse() ];
        assert(all([ (x.getID() == functToGetUuidProvided(x)) for x in self.initializedConditions]));
        self._writeInfoToDatabase();
        return;

    def getBaseConditions(self):
        return self.initializedConditions;




#V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
# class-specific utilities for defining domains
#===========================================================================

def getFiniteInterval(variableType, nameOfPredicate, lowerBound, upperBound):
    requires(isinstance(nameOfPredicate, str));
    requires(variableType in {"input", "output"});
    requires(isinstance(nameOfPredicate, str));
    requires(len(nameOfPredicate) > 0);
    requires(len(set(nameOfPredicate).intersection([" ", "\n", "\r", "\t"])) == 0);
    requires(isinstance(lowerBound, float));
    requires(isinstance(upperBound, float));
    requires(np.isfinite(lowerBound));
    requires(np.isfinite(upperBound));
    requires(lowerBound <= upperBound);

    templateString = """
def funct_{0}({1}):
    \"\"\"{0}\"\"\"
    if(isinstance({1}, z3.z3.ArithRef)):
        return z3.And( {1} <= {3}, {1} >= {2} );
    else:
        return ({1} <= {3}) and ({1} >= {2} );
    raise Exception("Control should not reach here");
    return;
    """;
 
    listOfVariableNames = ["in_x", "in_y"];
    if(variableType == "output"):
        listOfVariableNames = ["out_u", "out_v", "out_w"];
    assert(listOfVariableNames in [["in_x", "in_y"], ["out_u", "out_v", "out_w"]]);

    return [ \
        templateString.format(\
            (variableNameString.upper() + "_" + nameOfPredicate), \
             variableNameString, str(lowerBound), str(upperBound) ) \
        for variableNameString in listOfVariableNames ];


# The below function in principle could be done with getFiniteInterval if z3 supported infinite values, but its standard theory does not seem to support
# them, which, honestly, is reasonable.
def getInfiniteInterval(variableType, nameOfPredicate, boundary, aboveOrBelow):
    requires(isinstance(nameOfPredicate, str));
    requires(variableType in {"input", "output"});
    requires(isinstance(nameOfPredicate, str));
    requires(len(nameOfPredicate) > 0);
    requires(len(set(nameOfPredicate).intersection([" ", "\n", "\r", "\t"])) == 0);
    requires(isinstance(boundary, float));
    requires(np.isfinite(boundary));
    requires(aboveOrBelow in {"lowerBound", "upperBound"});

    templateString = """
def funct_{0}({1}):
    \"\"\"{0}\"\"\"
    if(isinstance({1}, z3.z3.ArithRef)):
        return {2} <= {3};
    else:
        return {2} <= {3};
    raise Exception("Control should not reach here");
    return;
    """;

    variableNameString = "in_x";
    if(variableType == "output"):
        variableNameString = "out_y";
    assert(variableNameString in {"in_x", "out_y"});

    raise Exception("TODO: update this");

    stringToReturn = "";
    if(aboveOrBelow == "upperBound"):
        stringToReturn = templateString.format(nameOfPredicate, variableNameString, variableNameString, str(boundry) )
    else:
        assert(aboveOrBelow == "lowerBound");
        stringToReturn = templateString.format(nameOfPredicate, variableNameString, str(boundry), variableNameString)

    return stringToReturn ;




#^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^



#V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
# Conditions over the input domain
#===========================================================================

otherInputSpaceFunctionsToUse = """

"""



def getListFunctionsToBaseCondtionsOn_forInputOfDomainThisUse():
    listOfFunctionCodes =[];

    def boundsToName(lower, upper):
        A = str(lower).replace(".", "DOT").replace("-", "NEG");
        B = str(upper).replace(".", "DOT").replace("-", "NEG");
        return A + "TO" + B;

    def formPredicateHere(lower, upper):
        return getFiniteInterval("input", boundsToName(lower, upper), lower, upper);

    for thisStartIndex in range(-20, 20):
        thisStartIndex = float(thisStartIndex);
        upperIndex = thisStartIndex + 1.0;
        middleIndex = thisStartIndex + 0.5;
        #  nameOfPredicate, lowerBound, upperBound)
        listOfFunctionCodes = listOfFunctionCodes + \
            formPredicateHere(thisStartIndex, upperIndex) + \
            formPredicateHere(thisStartIndex, middleIndex) + \
            formPredicateHere(middleIndex, upperIndex);

    return convertCodeListToListOfFunctions(listOfFunctionCodes);


#^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^


#V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
# Conditions over the output domain
#===========================================================================


otherOutputSpaceFunctionsToUse = """

""";


def getListFunctionsToBaseCondtionsOn_forOutputOfDomainThisUse():
    listOfFunctionCodes =[];

    def boundsToName(lower, upper):
        A = str(lower).replace(".", "DOT").replace("-", "NEG");
        B = str(upper).replace(".", "DOT").replace("-", "NEG");
        return A + "TO" + B;

    def formPredicateHere(lower, upper):
        return getFiniteInterval("output", boundsToName(lower, upper), lower, upper);

    # We have 25 below as oppossed to 20 so that it is distinct from the 
    # [-20,20] range used for the input space, allowing for further testing of
    # expected behaviour, etc.
    for thisStartIndex in range(-25, 25):
        thisStartIndex = float(thisStartIndex);
        upperIndex = thisStartIndex + 1.0;
        middleIndex = thisStartIndex + 0.5;
        #  nameOfPredicate, lowerBound, upperBound)
        listOfFunctionCodes = listOfFunctionCodes + \
            formPredicateHere(thisStartIndex, upperIndex) + \
            formPredicateHere(thisStartIndex, middleIndex) + \
            formPredicateHere(middleIndex, upperIndex);



    return convertCodeListToListOfFunctions(listOfFunctionCodes);

#^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^



#V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
# Conditions over the joint domain
#===========================================================================


def getBox(nameOfPredicate, lowerBoundInput, upperBoundInput, lowerBoundOutput, upperBoundOutput):
    requires(isinstance(lowerBoundInput, float));
    requires(isinstance(upperBoundInput, float));
    requires(isinstance(lowerBoundOutput, float));
    requires(isinstance(upperBoundOutput, float));
    requires(np.isfinite(lowerBoundInput));
    requires(np.isfinite(upperBoundInput));
    requires(np.isfinite(lowerBoundOutput));
    requires(np.isfinite(upperBoundOutput));
    requires(lowerBoundInput <= upperBoundInput);
    requires(lowerBoundOutput <= upperBoundOutput);


    templateString = """
def funct_{0}(in_x, out_y):
    \"\"\"{0}\"\"\"
    if(isinstance(in_x, z3.z3.ArithRef)):
        return z3.And( in_x <= {3}, in_x >= {2}, out_y  <= {5}, out_y >= {4} );
    else:
        return (in_x <= {3}) and (in_x >= {2} ) and (out_y  <= {5}) and (out_y >= {4});
    raise Exception("Control should not reach here");
    return;
    """;

    return templateString.format(nameOfPredicate, str(lowerBoundInput), str(upperBoundInput), str(lowerBoundOutput), str(upperBoundOutput) );


# circle, 
# halfplane
# negation (or maybe just allow the user to pass in the inequality.. but actually negation would be useful for things later on... )


def getHalfPlane(nameOfPredicate, slope, intercept, inequality):
    requires(isinstance(slope, float));
    requires(isinstance(intercept, float));
    requires(np.isfinite(intercept));
    requires(np.isfinite(slope));
    requires(isinstance(inequality, str));
    requires(inequality in {"=<", "=>", "<", ">"});


    templateString = """
def funct_{0}(in_x, out_y):
    \"\"\"{0}\"\"\"
    return  in_x * {1} + {2} {3} out_y ;
    raise Exception("Control should not reach here");
    return;
    """;

    return templateString.format(nameOfPredicate, str(slope), str(intercept), str(inequality));


def getCicle(nameOfPredicate, in_x_center, out_y_center, radius, inequality):
    requires(isinstance(in_x_center, float));
    requires(np.isfinite(in_x_center));
    requires(isinstance(out_y_center, float));
    requires(np.isfinite(out_y_center));
    requires(isinstance(radius, float));
    requires(np.isfinite(radius));
    requires(isinstance(inequality, str));
    requires(inequality in {"=<", "=>", "<", ">"});


    templateString = """
def funct_{0}(in_x, out_y):
    \"\"\"{0}\"\"\"
    return  (in_x - {1}) ** 2  + (out_y - {2}) {3} {4} ;
    raise Exception("Control should not reach here");
    return;
    """;

    return templateString.format(nameOfPredicate, str(in_x_center), str(out_y_center), str(inequality), str(radius ** 2) );







def getListFunctionsToBaseCondtionsOn_forJointInputAndOutputDomainsInThisUse():
    listOfFunctionCodes =[];
    return convertCodeListToListOfFunctions(listOfFunctionCodes);

#^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^












