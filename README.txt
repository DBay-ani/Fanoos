** Fanoos: Multi-Resolution, Multi-Strength, Interactive Explanations for Learned Systems
David Bayani and Stefan Mitsch ; paper at https://arxiv.org/abs/2006.12453

For information regarding this code's license, see the "LICENSE" section below.
For information on how to cite this code and/or the corresponding write-ups, 
see the section "Information on Citing this Code and Corresponding Papers" 
below.

One can begin playing with the system by running:
python3 fanoos.py
See DEPENDENCIES.txt for a list of software dependencies this code has.

Configuration default values, which we highly encourage every user to visit,
can be found in config/defaultValues.py .

We also suggest that users take advantage of the efficiency improvements
available in the development branch (note that this README is for the main
branch). While code in the development branch has undergone a variety of tests,
we hold-off from merging it in here in order to keep the main branch's core
relatively stable, as well as to more extensively scrutinize the additions prior
via further hardening tests. Further discussion of the branches in the repo can
be found later in this README file.

Note, day23 month3 year2021: The code to run the experiments is found in this
repository. We will push the collection and summarization scripts at a later point.


V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
LICENSE
===============================================================================

V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
License for this Repository's Material, as of Day22 Month3 Year2021
-------------------------------------------------------------------------------
See the LICENSE.txt file found in this repository for a copy of the GNU General
Public License, Version  3, referenced below.
===============================================================================

Fanoos: Multi-Resolution, Multi-Strength, Interactive Explanations for Learned Systems ; David Bayani and Stefan Mitsch ; paper at https://arxiv.org/abs/2006.12453
Copyright (C) 2021  David Bayani

This file is part of Fanoos.

Fanoos is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License only.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact Information

Electronic Mail: 
  dcbayani@alumni.cmu.edu

Paper Mail:
  David Bayani
  Computer Science Department
  Carnegie Mellon University
  5000 Forbes Ave.
  Pittsburgh, PA 15213
  USA

^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^


V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
Further Comments Regarding the License for this Code, as of Day22 Month3 Year2021
===============================================================================

While we may change the license for this repository at a later date to be more 
permissive, at this time we use GPL3 - specifically, GPL3-only not 
GPL3-or-later. For those who have a compelling case where a more permissive 
open source license (MIT, BSD, etc.) would be exceedingly better for your aims,
please email the repository maintainer so that we may consider your situation 
and determine, when balanced with other evidence and goals, if the community 
would be better served by a license change. As it stands, we would like to 
ensure extensions or improvements to Fanoos, when exposed outside of private 
use, are as shareable, inspectable, and extensible as the original
("the original" meaning the implementation(s) that have been, are, or will be 
housed in this repository and its git history).

^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^


^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^


V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
Information on Citing this Code and Corresponding Papers
===============================================================================

To cite the paper corresponding to this material, please use:
    @inproceedings{DBLP:conf/vmcai/BayaniM22,
      author    = {David Bayani and
                   Stefan Mitsch},
      editor    = {Bernd Finkbeiner and
                   Thomas Wies},
      title     = {Fanoos: Multi-resolution, Multi-strength, Interactive Explanations
                   for Learned Systems},
      booktitle = {Verification, Model Checking, and Abstract Interpretation - 23rd International
                   Conference, {VMCAI} 2022, Philadelphia, PA, USA, January 16-18, 2022,
                   Proceedings},
      series    = {Lecture Notes in Computer Science},
      volume    = {13182},
      pages     = {43--68},
      publisher = {Springer},
      year      = {2022},
      url       = {https://doi.org/10.1007/978-3-030-94583-1\_3},
      doi       = {10.1007/978-3-030-94583-1\_3},
      timestamp = {Fri, 21 Jan 2022 22:02:46 +0100},
      biburl    = {https://dblp.org/rec/conf/vmcai/BayaniM22.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
or (perhaps less preferably) cite the extended write-up at:
    @article{DBLP:journals/corr/abs-2006-12453,
      author    = {David Bayani and
                   Stefan Mitsch},
      title     = {Fanoos: Multi-Resolution, Multi-Strength, Interactive Explanations
                   for Learned Systems},
      journal   = {CoRR},
      volume    = {abs/2006.12453},
      year      = {2020},
      url       = {https://arxiv.org/abs/2006.12453},
      eprinttype = {arXiv},
      eprint    = {2006.12453},
      timestamp = {Tue, 23 Jun 2020 17:57:22 +0200},
      biburl    = {https://dblp.org/rec/journals/corr/abs-2006-12453.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }

For a citable version of any of the material found exclusively / specifically
in this repo, the copy of this code found on Zenodo may be of use to you, since
it provides Bibtex citation export and a DOI:
    @software{david_bayani_2021_5513079,
      author       = {David Bayani},
      title        = {{Code for the Fanoos Multi-Resolution, Multi- 
                       Strength, Interactive XAI System}},
      month        = mar,
      year         = 2021,
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.5513079},
      url          = {https://doi.org/10.5281/zenodo.5513079}
    }
The content on Zenodo is immutable, so I cannot alter or takedown any material
that appears there after it is uploaded (barring extreme cases that fit narrow 
guidelines established by that organization, and which require permission and 
oversight from those governing Zenodo - even then, upload timestamps most 
likely would remain uneffected). To confirm the state of this code prior to the
earliest Zenodo upload, please see commit 
2ee67db4250339e3308cd611e260528d9c129639 and earlier of the repo at
https://github.com/DBay-ani/FanoosFurtherMaterials ; the file manifest.xml found
there explains how to confirm the integrity and state of the copies of the code
there-included.

^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^


V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
Learned Systems And Domains Available in this Release
===============================================================================

Below files contain trained policies discussed in the paper along with code
used to generate/learn/tweak them:
    -------------------
    trainedNetworks/cpuPolynomialRegressionModel:
        ------------------
        polynomialRegressionTrial.py
            -----------------
            NOTE: data used to learn the model here-produced is from 
            https://www.openml.org/api/v1/json/data/562 and
            shared under CC-BY-4 (see https://creativecommons.org/licenses/by/4.0/
            and https://www.openml.org/cite). The data itself is not stored here 
            (one must run the script above to download it), but the model generated 
            (provided in the pickle file below) potentially may be considered a
            derivative work since it is trained using said data.
        -------------------
        trainedPolynomialModelInfo.pickle
        -------------------
    trainedNetworks/invertedDoublePendulumBulletEnv_v0:
        ---------------------
        convertInvertedPendulumNetworkToAlternateFormat.py
        ---------------------
        networkLayers_putIntoProperFormat.pickle
            ------------------
            This learned model has the potential to be considered a derived
            work from rl-baselines-zoo , which is shared under an MIT license.
            Please see https://github.com/araffin/rl-baselines-zoo/blob/master/LICENSE 
            for the license on the original learned network and 
            https://github.com/araffin/rl-baselines-zoo/tree/master/trained_agents/ppo2/InvertedDoublePendulumBulletEnv-v0
            for the original networks.
                ---------------------
                See the domain specification code for the inverted double
                pendulum (./domainsAndConditions/domainAndConditionsForInvertedDoublePendulum.py)
                to see further points to where the training code and environment specification
                for the network can be found. 

In addition to the above files which were discussed in the paper, two additional
models for testing and sanity-checking purposes can be found in:
    ------------------------------------
    trainedNetworks/modelForTesting
        ------------------------------------
        formModelForTesting_oneDimInput_oneDimOutput_identityFunction.py
            -------------------------------
            Produces the system stored in
            modelForTesting_oneDimInput_oneDimOutput_identityFunction.pickle .
            While the code in this file easily extends to more interesting functions,
            the function implemented and saved is simply a one-dimensional identity
            (i.e., f(x) = x)--- BEAR IN MIND the input-space bounding box when
            interpreting results (see the function getInputSpaceUniverseBox in 
            domainsAndConditions/domainAndConditionsFor_modelForTesting_oneDimInput_oneDimOutput.py )  
        ------------------------------------
        formModelForTesting_twoDimInput_threeDimOutput_identityFunctionAndAddition.py
            -------------------------------
            Produces the system stored in
            modelForTesting_twoDimInput_threeDimOutput_identityFunctionAndAddition.pickle .
            While the code in this file easily extends to more interesting functions,
            the function implemented and saved is a simple linear function from two inputs
            to three: f(x, y) = (x, y, x + y - 1) ---
            BEAR IN MIND the input-space bounding box when interpreting results
            (see the function getInputSpaceUniverseBox in 
            domainsAndConditions/domainAndConditionsFor_modelForTesting_twoDimInput_threeDimOutput.py )
        ------------------------------------
        modelForTesting_oneDimInput_oneDimOutput_identityFunction.pickle
        ------------------------------------
        modelForTesting_twoDimInput_threeDimOutput_identityFunctionAndAddition.pickle

We had intended to include in this release another series of learned system we 
analyzed which controlled a Dubins's-Car-Like. The learned systems were implemented
using the same architecture as policy network in the inverted double pendulum 
above. One network learned how to drive and steer effectively
while skidding/sliding around a circle, and another learned to drive straight under
similar environmental conditions. Unfortunately, due to lack of clarity in the
license for the training code, we are currently uncomfortable including the 
controllers learned in that software in this code release. An example of an early
domain used for these Dubins's-Car-Like system can be found at: 
domainsAndConditions/domainAndConditionsForCircleFollowing.py

^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^



V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
Installations Instructions
-------------------------------------------------------------------------------
Both methods of installation listed below should result in Fanoos functioning
for typical users. We --highly-- recommend using AWS (or other computer 
infrastructure which allows a native install) for the sake of performance. For
those who simply wish to casually interact with the system and get a high-level
idea of what interactions are like, the Docker install may be sufficient and
would be more convenient to setup.
===============================================================================

V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
Instructions for Installing on an AWS EC2 machine
===============================================================================

EC2 Instance:
    Ubuntu 18.04 instance (t2.micro EC2 instance - a "free-tier" server instance)
        specifically: Ubuntu Server 18.04 LTS (HVM), SSD Volume Type - ami-0a63f96e85105c6d3 (64-bit
        x86)
Security group settings:
    inbound: SSH, only from "my ip address"
    outbound:
        allow HTTPS from 0.0.0.0/0
        allow HTTP from 0.0.0.0/0
        same information ,but in table format just copied from AWS:
            Type Protocol Port range Destination Description - optional
            HTTP TCP 80 0.0.0.0/0 -
            HTTPS TCP 443 0.0.0.0/0
Commands to run once EC2 instance is running, and you access it by SSH:
    sudo apt update
    sudo apt upgrade -y python3
    sudo apt install -y python3-pip
    pip3 install scipy
    pip3 install sklearn
    pip3 install matplotlib
    git clone https://github.com/Z3Prover/z3
    cd z3
    python3 scripts/mk_make.py --python
    time ( cd build
      make
      sudo make install);
      # NOTE: For the author, these were the timing statistics:
      # real 22m58.712s
      # user 21m49.574s
      # sys 1m7.419s
    cd ..
    git clone https://github.com/DBay-ani/Fanoos
    cd Fanoos
    # NOTE: from here you should be able to interact with the system using 
    # "python3 fanoos.py" without issue

^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^


V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
Instructions for Installing via Docker Container
===============================================================================

To setup the latest version of Fanoos on docker, run the following commands in the
top-level of the code repo:
    docker build -t image_for_fanoos . ;
    docker create --name container_for_fanoos -ti --mount src=$(pwd),dst="/home/user8d0a0629178c48bdab52171a1f3b981d/fanoosCode",type=bind image_for_fanoos ;
    docker start container_for_fanoos ;
    docker exec -it -d -u root container_for_fanoos bash -c "cd /home/user8d0a0629178c48bdab52171a1f3b981d/fanoosCode; bash restOfInstall.sh; "

Using the above commands, first the basic environment is setup in a Docker 
container, then the rest of the install - particularly the time-consuming 
process of installing Z3 - is done in the background. Obvious caution should
be used when querying Docker, etc., prior to the install process finishing. 
Notice that, unlike the commands for AWS, these commands for Docker use the 
local version of the code (i.e., the code in this repository when you download 
it , whether that be today or years ago) as opposed to pulling the latest 
version of Fanoos's code available from GitHub.

^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^


^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^

V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V~V
Regarding Other Branches
===============================================================================

We suggest that users take advantage of the efficiency improvements available 
in the development branch (note that this README is for the main branch). While 
code in the development branch has undergone a variety of tests, we hold-off
from merging it in here in order to keep the main branch's core relatively
stable, as well as to more extensively scrutinize the additions prior via
further hardening tests.

Additional extensions and improvements can be found in the experimental branch,
which extends off of the development branch. Changes in the experimental branch,
however, are more likely to alter the results of Fanoos (either intentionally
or unintentionally), and are provided with a lower degree of assurance than the
commits in the development branch.

We recognize this versioning may not be typical practice for release-holding
git repos, at least in regard to branch naming. In the future, we may change
the branch names from "master", "development",  and "experimental" to "stable",
"master", and "development", respectively - but at this time we believe the
current names more accurately reflect our intents. For now, while perhaps not
entirely standard practice, we believe this branching structure will support
the release pattern and release types we plan to make in this repo in the near
future.

^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^


ssh-rsa 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 abc@defg
