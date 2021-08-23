import os
from glob import glob
from setuptools import setup

exec(open('nmrgnn/version.py').read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='nmrgnn',
      version=__version__,
      description='Chemical shift predictor',
      author='Ziyue Yang, Andrew White',
      author_email='andrew.white@rochester.edu',
      url='https://github.com/ur-whitelab/nmrgnn',
      license='MIT',
      packages=['nmrgnn'],
      install_requires=[
          'tensorflow >= 2.3',
          'MDAnalysis < 2',
          'click',
          'numpy',
          'pandas', 'tqdm',
          'nmrgnn-data >= 0.4',
          'keras-tuner==1.0.2'],
      test_suite='tests',
      zip_safe=True,
      entry_points='''
        [console_scripts]
        nmrgnn=nmrgnn.main:main
            ''',
      include_package_data=True,
      package_data={'nmrgnn': [
          'models/baseline/saved_model.pb', 'models/baseline/variables/variables*']},
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering :: Bio-Informatics",
          "Topic :: Scientific/Engineering :: Artificial Intelligence"
      ]
      )
