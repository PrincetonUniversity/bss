import os.path
from unittest import TestCase

from bss.utils.data import load_xy_file, load_cor_file, load_data

DATA_DIR = os.path.join(os.path.dirname(__file__), 'sample_data')


class DataTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_load_xy(self):
        genos, phenos, eqtls = load_xy_file(os.path.join(DATA_DIR, 'real0_yx_10000.out'))
        self.assertEqual(genos.shape, (480, 1509))
        self.assertEqual(phenos.shape, (480,))
        self.assertEqual(eqtls.shape, (1509,))

    def test_load_cor(self):
        cor = load_cor_file('sample_data/real0_cor1_10000.out')
        self.assertEqual(cor.shape, (1509, 1509))

    def test_load_data(self):
        genos, phenos, eqtls, cor = load_data(os.path.join(DATA_DIR, 'real0_*_10000.out'))
        self.assertEqual(genos.shape, (480, 1509))
        self.assertEqual(phenos.shape, (480,))
        self.assertEqual(eqtls.shape, (1509,))
        self.assertEqual(cor.shape, (1509, 1509))
