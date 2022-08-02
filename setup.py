# https://packaging.python.org/en/latest/tutorials/packaging-projects/
from setuptools import setup
import pathlib
import neural_networks

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(name='neural-networks',
      version=neural_networks.__version__,
      description='Basic neural networks package.',
      long_description=README,
      long_description_content_type="text/markdown",
      url="https://github.com/AnhQuoc533/neural-networks",
      packages=['neural_networks'],
      author='Anh Quoc',
      author_email='lhoanganhquoc@gmail.com',
      license='MIT',
      classifiers=[
            'Development Status :: 1 - Planning',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10'
      ],
      keywords='neural-network AI deep-learning ai',
      install_requires=["numpy", "matplotlib"],
      python_requires='>=3.7',
      zip_safe=False,
      project_urls={  # Optional
          "Bug Reports": "https://github.com/AnhQuoc533/neural-networks/issues",
          "Funding": "https://donate.pypi.org",
          "Source": "https://github.com/AnhQuoc533/neural-networks",
      },
      )
