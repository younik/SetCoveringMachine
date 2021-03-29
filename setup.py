from distutils.core import setup
setup(
  name='scmpy',
  packages=['scmpy'],
  version='0.1',
  license='MIT',
  description='The Set Covering Machine is a binary classifier',
  author='Omar Younis',
  author_email='omar.younis98@gmail.com',
  url='https://github.com/younik',
  download_url='https://github.com/younik/SetCoveringMachine/archive/refs/tags/v0.1.tar.gz',
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
