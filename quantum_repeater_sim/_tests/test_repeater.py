import unittest
import numpy as np
import random
import random

from quantum_repeater_sim.repeater import (
    Repeater, SwapPolicy, NO_PARTNER, QUBIT_FREE, QUBIT_OCCUPIED,
    fidelity_to_werner, werner_to_fidelity,
    bbpssw_success_prob, bbpssw_new_werner,
)

def random_params_generator():
    return  {'n_ch' : np.random.randint(2, 20),
            'p_gen' : np.random.rand(),
            'p_swap' : np.random.rand(),
            'cutoff' : np.random.randint(1, 100)}

class TestRepeaterNetwork_CoreTests(unittest.TestCase):

    def setUP(self):
        ...
    
    def test_init(self):
        for _ in range(100):
            params = random_params_generator()
            r = Repeater(rid=1, **params)
            self.assertFalse(r.num_occupied())
            self.assertFalse(r.available_indices())
            self.assertTrue(r.has_free_qubit())
            self.assertFalse(r.can_swap())
            self.assertFalse(r.num_locked())

    def test_1_entangle(self):
        n_ch = random.randint(2, 10)
        r1 = Repeater(rid=1, p_gen=1)
        r2 = Repeater(rid=2, p_gen=1)

        for k in range(n_ch):
            self.assertEqual(r1.num_occupied(), k)
            self.assertEqual(r2.num_occupied(), k)
            
            q1 = r1.allocate_qubit()
            q2 = r2.allocate_qubit()
            r1.set_link(q1, 2, q2, 1, 1, None)
            r2.set_link(q2, 1, q1, 1, 1, None)

            self.assertTrue(r1.num_occupied()==1)
            self.assertTrue(r2.num_occupied()==1)
            self.assertTrue(r1.partner_qubit == q2)
            self.assertTrue(r2.partner_qubit == q1)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRepeaterNetwork_CoreTests))
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)  
    success = result.wasSuccessful()
    exit(0 if success else 1)