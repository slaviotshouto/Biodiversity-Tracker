import unittest
import os


"""

we could also do it another way:

    instead of doing this:

    def pred_unit_test(self, actual, pred):
        self.assertIsNotNone(pred)
        self.assertMultiLineEqual(actual, pred)
    
    we could integrate the specific tests into the functions, idk which is correct:

    def pred_unit_test(self):
        self.assertIsNotNone("example_img")
        self.assertMultiLineEqual("actual_label_for_image", "predicted_label_for_image)


"""
# https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertMultiLineEqual

class unit_test(unittest.TestCase):

    def pred_unit_test(self, actual, pred):
        self.assertIsNotNone(pred)
        self.assertMultiLineEqual(actual, pred)  #strings only --> checks if model predicted the correct label

    def check_img(self, img, dir_): #checks if a specified image is in a directory
        self.assertIn(img, dir_) #don't specify a dir, but rather a list--> i.e target_folder (str img, list dir_)

    def exists(self, entry):
        self.assertIsNotNone(entry)
        self.assertTrue(entry)

    def check_dir_len(self, dir, specified_length):
        self.assertEqual(len(os.listdir(dir)), specified_length)

    def check_audio_vs_img(self, audio_pred, img_pred): #couldn't come up with anything else
        self.assertMultiLineEqual(audio_pred, img_pred)


r = unit_test.birdUnitTest("a", "a", None) #third argument is for the potential error message
print(r)

if __name__ == '__main__':
    unittest.main()

