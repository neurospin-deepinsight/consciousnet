# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Module current version
version_major = 0
version_minor = 0
version_micro = 0

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = """
Module providing Consciousness Exploration Tools for PyTorch.
"""
SUMMARY = """
.. container:: summary-carousel

    `consciousnet` is a PyTorch toolbox that provides Consciousness
    Exploration Tools.
"""
long_description = (
    "Module providing Consciousness Exploration Tools for PyTorch.\n")

# Main setup parameters
NAME = "consciousnet"
ORGANISATION = "CEA"
MAINTAINER = "Antoine Grigis"
MAINTAINER_EMAIL = "antoine.grigis@cea.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
EXTRANAME = "NeuroSpin webPage"
EXTRAURL = (
    "https://joliot.cea.fr/drf/joliot/Pages/Entites_de_recherche/"
    "NeuroSpin.aspx")
LINKS = {"deepinsight": "https://github.com/neurospin-deepinsight/deepinsight"}
URL = "https://github.com/neurospin-deepinsight/consciousnet"
DOWNLOAD_URL = "https://github.com/neurospin-deepinsight/consciousnet"
LICENSE = "CeCILL-B"
AUTHOR = """
consciousnet developers
"""
AUTHOR_EMAIL = "antoine.grigis@cea.fr"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["consciousnet"]
REQUIRES = [
    "numpy>=1.17.1",
    "scipy>=0.19.1",
    "Pillow>=6.2.1",
    "matplotlib>=3.0.3",
    "scikit-image>=0.16.2",
    "torch>=1.8.1",
    "torchvision>=0.9.1"
]
