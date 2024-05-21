import unittest

import laminer
import laminer.util


class BasicTest(unittest.TestCase):

    def test_date_not_midnight(self):
        s = laminer.util.build_file_prefix()
        # print(f"{s}")
        self.assertFalse("00" in s)
