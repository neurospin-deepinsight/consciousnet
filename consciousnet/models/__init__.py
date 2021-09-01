# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common architectures.
"""

# Imports
from .tdvae import TDVAE
from .gmvae import GMVAE, VAEGMP
from .barlowtwins import BarlowTwins
from .deepcluster import DeepCluster
from .brainnetcnn import BrainNetCNN
