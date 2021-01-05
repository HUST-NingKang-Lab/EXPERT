from setuptools import setup
from setuptools import find_packages

# change this.
NAME = "expert"
AUTHOR = "Hui Chong"
EMAIL = "huichong.me@gmail.com"
URL = "https://github.com/HUST-NingKang-Lab/EXPERT"
LICENSE = "MIT"
DESCRIPTION = "Exact and pervasive expert model for source tracking based on transfer learning"


if __name__ == "__main__":
    setup(
        name=NAME,
        version="0.3-cpu",
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        license=LICENSE,
        description=DESCRIPTION,
        packages=find_packages(),
        package_dir={'expert': 'expert'},
        include_package_data=True,
        install_requires=open("./requirements_cpu.txt", "r").read().splitlines(),
        long_description=open("./README.md", "r").read(),
        long_description_content_type='text/markdown',
        # change package_name to your package name.
        entry_points={
            "console_scripts": [
                "expert=expert.CLI:main"
            ]
        },
        package_data={
            # change package_name to your package name.
            "config": ["./resources/config.ini"],
            "general_model": ["./resources/general_model"],
			"phylo":["./resources/phylogeny.csv"],
			"tmp":["./resources/tmp"]
        },
        zip_safe=True,
        classifiers=[
            "Programming Language :: Python :: 3",
            # change $license to your license.
            "License :: OSI Approved :: $license",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.6"
    )
