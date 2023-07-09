#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="adamantium",
    version="1.0.0",
    description="EMLOv3 Base Setup",
    author="Salil Gautam",
    author_email="salil.gtm@gmail.com",
    url="https://github.com/salil-gtm/emlov3_assignment_6",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "adamantium_train = adamantium.train:main",
            "adamantium_eval = adamantium.eval:main",
            "adamantium_infer = adamantium.infer:main",
        ]
    },
)
