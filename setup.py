from setuptools import setup
from setuptools import find_packages

# change this.
NAME = "your package name"
AUTHOR = "your name"
EMAIL = "your mail"
URL = "your project url"
LICENSE = "your license"
DESCRIPTION = "your project description"

if __name__ == "__main__":
    setup(
        name=NAME,
        version="0.0.1",
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        license=LICENSE,
        description=DESCRIPTION,
        packages=find_packages(),
        include_package_data=True,
        install_requires=open("./requirements.txt", "r").read().splitlines(),
        long_description=open("./README.md", "r").read(),
        long_description_content_type='text/markdown',
        # change package_name to your package name.
        entry_points={
            "console_scripts": [
                "package_name=package_name.shell:run"
            ]
        },
        package_data={
            # change package_name to your package name.
            "package_name": ["src/*.txt"]
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
