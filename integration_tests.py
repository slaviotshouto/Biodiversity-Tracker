import unit_tests_classes_new
import unittest
import os

# don't forget potential other imports

"""
the "big-bang approach" to integration testing ( or at least an attempt )

Idea for the workflow - we don't have as many modules of course::

Modules A and D are all tested individually for the system.
Module C is combined with module B and module F is combined with module E to be tested as a group.
Module A is tested first, followed by module B, then D, and finally F.
Each module is tested for functionality and compatibility with the other modules.

CALL THIS ELSEWHERE IN THE CODEBASE
**After all, modules are tested, they are integrated into the system and the entire system is tested for functionality**

"""

# test module A - DON'T FORGET WE HAVE TO SPECIFY EVERYTHING

class integration_test_cnn(unittest.TestCase):
    def test_pipeline_cnn(self):
        """
        pipeline to test everything regarding the cnn model
        """
        self.assertIsNotNone(prediction=None) #checks if prediction exists
        self.assertMultiLineEqual(actual=None, pred=None) # have to be strings, if not string, cast into string first - checks if we predicted the correct label
        self.assertIsNotNone(img=None) #checks if img exists
        self.assertIn(img=None, dir_=None) # checks if img in dir
        self.assertEqual(actual_dir_len = len(os.listdir(dir)), specified_dir_len=None) # checks if our dir is as expected


# test module B

class integration_test_audio(unittest.TestCase):
    def test_pipeline_audio(self):
        self.assertIsNotNone(prediction=None) #checks if prediction exists
        self.assertMultiLineEqual(actual=None, pred=None) # have to be strings, if not string, cast into string first - checks if we predicted the correct label
        self.assertIsNotNone(img=None) #checks if img exists
        self.assertIn(img=None, dir_=None) # checks if img in dir
        self.assertEqual(actual_dir_len = len(os.listdir(dir)), specified_dir_len=None) # checks if our dir is as expected


# test if both predict the same ( essentially both modules combined and tested as a group? )

class integration_test_check_audio_vs_img(unittest.TestCase):
    def test_check_audio_vs_img(self):
        """
        checks the predictions of the audio model against the image model's.
        """
        self.assertMultiLineEqual(pred1=None, pred2=None)


class integration_test_model1_vs_model2(unittest.TestCase):
    def test_check_model1_vs_model2(self):
            """
            checks the predictions of one model against another ( since we have multiple )
            """
            self.assertMultiLineEqual(pred1=None, pred2=None)


if __name__ == '__main__':
    unittest.main()