import os
from glob import glob
from setuptools import setup

exec(open('nmrgnn/version.py').read())

setup(name='nmrgnn',
      version=__version__,
      scripts=glob(os.path.join('scripts', '*')),
      description='Chemical shift predictor',
      author='Ziyue Yang, Andrew White',
      author_email='andrew.white@rochester.edu',
      url='http://thewhitelab.org/Software',
      license='MIT',
      packages=['nmrgnn'],
      install_requires=[
          'tensorflow >= 2.3',
          'MDAnalysis',
          'click',
          'numpy',
          'pandas',
          'nmrdata@https://api.github.com/repos/ur-whitelab/nmrdata/tarball/master',
          'keras-tuner==1.0.2'],
      test_suite='tests',
      zip_safe=True,
      entry_points='''
        [console_scripts]
        nmrgnn=nmrgnn.main:main
            ''',
      include_package_data=True,
      package_data={'nmrgnn': [
          'models/baseline/saved_model.pb', 'models/baseline/variables/variables*']}
      )
