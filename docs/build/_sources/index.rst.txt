.. livingTree documentation master file, created by
   sphinx-quickstart on Thu Mar 26 19:16:43 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to livingTree's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Introduction
======

Taxonomic data is generally considered as a tree-like 
structure. In more and more fields of bioinformatics, 
metagenomics, microbiology, etc., analysis is needed 
with the help of species classification trees (such as 
reading counting, abundance calculation, etc.)

Installation
======

The easy way to install livingTree is using `pip` package manager
   
   .. code-block:: sh

      $ pip install livingTree

or just run the setup script.

   .. code-block:: sh

      $ python setup.py install

Example
======

>>> import livingTree as lt

Constructing tree
------

Node by node
++++++

>>> tree = lt.SuperTree()
>>> tree.create_node(identifier = 'root')
>>> tree.create_node(identifier = 'taxa_1', parent = 'root')
>>> tree.create_node(identifier = 'taxa_2', parent = 'taxa_1')
>>> tree.create_node(identifier = 'taxa_3', parent = 'taxa_2')
>>> tree.create_node(identifier = 'taxa_4', parent = 'taxa_1')
>>> tree.create_node(identifier = 'taxa_5', parent = 'root')
>>> tree.create_node(identifier = 'taxa_6', parent = 'taxa_5')
>>> tree.create_node(identifier = 'taxa_7', parent = 'taxa_6')
>>> tree.show()

Using node id paths
++++++

>>> tree = lt.SuperTree()
>>> tree.create_node(identifier = 'root')
>>> paths = [['taxa_1', 'taxa_2', 'taxa_3'], 
             ['taxa_1', 'taxa_4'],
             ['taxa_5', 'taxa_6', 'taxa_7']]
>>> tree.from_paths(paths)
>>> tree.show()

Restore from pkl file
++++++

>>> tree = lt.SuperTree()
>>> mm_tree = tree.from_pickle('trees/mammalia_living_tree.pkl')
>>> mm_tree.depth()
>>> mm_tree.size()

Using taxonomic data from NCBI taxonomy database
++++++

>>> builder = lt.TreeBuilder('Mammalia')
>>> builder.build()
>>> tree = builder.tree
>>> tree.show()

Deeling with abundance data
------

Initiate all nodes' data at once
++++++

>>> tree.init_nodes_data(data = 0)

Fill node data in batch
++++++

>>> abundances = {'taxa_3': 10,
                  'taxa_4': 5,
                  'taxa_7': 3,
                  'taxa_6': 4}
>>> tree.fill_with(abundances)
>>> tree.get_node('root').data
>>> tree.update_value()
>>> tree.get_node('root').data

Get a numpy n-dimensional array from all taxonomy abundances
++++++

>>> import numpy as np
>>> tree.get_matrix(dtype = np.float32) 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
