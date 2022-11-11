import unittest
import os

# https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertMultiLineEqual

class unit_test_label_pred(unittest.TestCase):
        def prediction(self, actual, pred):
            self.assertIsNotNone(pred)
            self.assertMultiLineEqual(actual, pred)  #strings only --> checks if a model predicted the correct label

class unit_test_check_img(unittest.TestCase):
    def check_img(self, img, dir_): #checks if a specified image is in a directory
        self.assertIn(img, dir_) #don't specify a dir, but rather a list--> i.e target_folder (str img, list dir_)

class unit_test_exists(unittest.TestCase):
    def exists(self, entry):
        self.assertIsNotNone(entry)

class unit_test_check_dir_len(unittest.TestCase):
    def check_dir_len(self, dir, specified_len):
        self.assertEqual(len(os.listdir(dir)), specified_len)

class unit_test_check_audio_vs_img(unittest.TestCase):
    def check_audio_vs_img(self, audio_pred, img_pred): #couldn't come up with anything else
        self.assertMultiLineEqual(audio_pred, img_pred)
        
#r = unit_test.birdUnitTest("a", "a", None) #third argument is for the error message

if __name__ == '__main__':
    unittest.main()

