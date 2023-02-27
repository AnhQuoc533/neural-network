# https://packaging.python.org/en/latest/tutorials/packaging-projects
# https://docs.python.org/3/tutorial/modules.html#packages

from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Determine version number from module
VERSION = None
with (HERE / "neural_network" / "__init__.py").open() as f:
    for line in f:
        if line.startswith("__version__ ="):
            VERSION = line.strip().split()[2][1:-1]
            break

setup(name='neural-network',
      version=VERSION,
      description='A Neural Network framework for building Multi-layer Perceptron model.',
      long_description=README,
      long_description_content_type="text/markdown",
      url="https://github.com/AnhQuoc533/neural-network",
      packages=['neural_network'],  # packages=find_packages(exclude=("test",))
      author='Anh Quoc',
      author_email='lhoanganhquoc@gmail.com',
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10'
      ],
      keywords='neural-network, deep-learning, machine-learning, neural-networks, machine-learning-algorithms',
      install_requires=["numpy>=1.22.1", "matplotlib>=3.5.1"],
      python_requires='>=3.8',
      zip_safe=False,
      project_urls={  # Optional
          "Bug Reports": "https://github.com/AnhQuoc533/neural-network/issues",
          "Funding": "https://donate.pypi.org",
          "Source": "https://github.com/AnhQuoc533/neural-network",
      },
      )

# py setup.py sdist bdist_wheel
# twine upload dist/*
