import unittest

import laminer
import laminer.util


class BasicTest(unittest.TestCase):
    """
    Basic brain-dead simple low level tests
    """
    def test_date_not_midnight(self):
        """ 
         sometimes buid_file_prefix included a nasty 00 instead of the correct hour.
        """
        s = laminer.util.build_file_prefix()
        # print(f"{s}")
        self.assertFalse("00" in s)

    def test_load(self):
        df=laminer.util.load_rag_quesitons("data-cobol",True)