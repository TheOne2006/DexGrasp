import itertools
import unittest
from night_schedule import validate_schedule, tour_numbers, planned_visits, reached

class Schedule(unittest.TestCase):
    def config(self, continuous=False):
        return dict(continuous=continuous,tours=None if continuous else 2,
                    max_group_visits=None if continuous else 4,groups=[{},{}],
                    hard_deadline_epoch=None if continuous else 100,
                    stop_new_init_epoch=None if continuous else 90)
    def test_finite_and_resume(self):
        c=self.config()
        self.assertEqual(list(tour_numbers(c)),[0,1])
        self.assertEqual(list(tour_numbers(c,1)),[1])
        self.assertEqual(planned_visits(c),4)
    def test_continuous_passes_old_two_tours(self):
        c=self.config(True)
        self.assertEqual(list(itertools.islice(tour_numbers(c,3),4)),[3,4,5,6])
        self.assertIsNone(planned_visits(c))
        self.assertFalse(reached(None,10**20))
    def test_explicit_drain_boundary(self):
        self.assertFalse(reached(90,89));self.assertTrue(reached(90,90))
    def test_refuses_inherited_limits(self):
        for key,value in [('tours',2),('max_group_visits',4),('hard_deadline_epoch',100),('stop_new_init_epoch',90)]:
            c=self.config(True);c[key]=value
            with self.assertRaises(AssertionError):validate_schedule(c)

if __name__=='__main__':unittest.main()
