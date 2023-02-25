from distutils.core import setup
setup(
  name = 'treeg',
  packages = ['treeg'],
  version = '0.1'
  license='MIT',
  description = 'TREE-G is a library of decision trees specialized for graph data. The library is scikit compatible, and the implementation follows the paper "Decision Trees with Dynamic Graph Features".',ibrary
  author = 'Maya Bechler-Speicher',
  author_email = 'mayab4@mail.tau.ac.il',
  url = 'https://github.com/mayabechlerspeicher',
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',
  keywords = ['Decision Trees', 'Graph', 'Graph Learning', 'Graph Machine Learning', 'Decision Trees for graphs'],
  install_requires=['numpy>=1.19.2', 'scikit-learn>= 0.23.1', 'scipy>=pi1.5.2' ],
                   classifiers = ["Programming Language :: Python :: 3",
                                  "License :: OSI Approved :: MIT License",
                                  "Operating System :: OS Independent"],
)