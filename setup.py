from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
  name='scmpy',
  packages=['scmpy'],
  version='0.1.2',
  license='MIT',
  description='Set Covering Machine is a binary classification algorithm',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author='Omar Younis',
  author_email='omar.younis98@gmail.com',
  url='https://github.com/younik',
  download_url='https://github.com/younik/SetCoveringMachine/archive/refs/tags/v0.1.2.tar.gz',
  keywords=['SET', 'COVER', 'BINARY', 'CLASSIFICATION', 'MACHINE', 'LEARNING'],
  install_requires=[
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)
