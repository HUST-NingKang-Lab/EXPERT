import sys
import os


def init(cfg, args):
    import ete3
    ncbi = ete3.NCBITaxa()
    ncbi.update_taxonomy_database()
    print('NCBI Taxomomy database is installed in {}.'.format(ncbi.dbfile))

