import unittest
import os

# don't forget imports 
# to run all tests at once: python3 -m unittest unit_tests_classes.py

# https://docs.python.org/3/library/unittest.html#
# new: added docstrings, naming convention (function definitions have to start with "test_"), working tests

# note: actual valid arguments have to be specified for the tests to run ( None by default )

class unit_test_label_pred(unittest.TestCase):
    def test_prediction(self):
        """
        first checks if the model predicted anything, then if it predicted the correct label; strings only.
        """
        self.assertIsNotNone(pred=None)
        self.assertMultiLineEqual(actual=None, pred=None) # have to be strings, if not string, cast into string first 

class unit_test_check_img(unittest.TestCase):
    def test_check_img(self): 
        """
        checks if a specified image is in a directory.
        don't specify a dir, but rather a list--> i.e target_folder (str img, list dir_).
        """
        self.assertIn(img=None, dir_=None) 

class unit_test_exists(unittest.TestCase):
    def test_exists(self):
        """
        checks if a specified image exists.
        """
        self.assertIsNotNone(img=None)

class unit_test_check_dir_len(unittest.TestCase):
    def test_check_dir_len(self):
        """
        checks the size of the directory against a pre-specified size.
        """
        self.assertEqual(actual_dir_len = len(os.listdir(dir)), specified_dir_len=None)


# This is actually an integration test as it tests the predictions of two diferent models, not just a unit::

class integration_test_check_audio_vs_img(unittest.TestCase):
    def test_check_audio_vs_img(self): 
        """
        checks the predictions of the audio model against the image model's.
        """
        self.assertMultiLineEqual(audio_pred=None, img_pred=None)
        
#r = unit_test.birdUnitTest("a", "a", None) # the third argument is for the error message

if __name__ == '__main__':
    unittest.main()