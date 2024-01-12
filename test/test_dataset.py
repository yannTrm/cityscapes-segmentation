# -*- coding: utf-8 -*-
# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import unittest
import matplotlib.pyplot as plt



from src.dataset import CustomCityscapes
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class TestCityscapes(unittest.TestCase):
    def setUp(self):
        self.dataset = CustomCityscapes('../data/', split='train', mode='fine', target_type='semantic', transform=None)
        self.skip_plot = "No need to plot"

    def test_plot_image(self):
        # Test the plot_image method
        self.skipTest(self.skip_plot)
        plt.figure()
        self.dataset.plot_image(index=10)   

    def test_plot_segmentation(self):
        # Test the plot_segmentation method
        self.skipTest(self.skip_plot)
        plt.figure()
        self.dataset.plot_segmentation(index=10)  

    def test_plot_image_and_segmentation(self):
        # Test the plot_image_and_segmentation method
        self.skipTest(self.skip_plot)
        plt.figure()
        self.dataset.plot_image_and_segmentation()  
        
    def test_number_class(self):
        # Test the __init_mapping_class__ method
        assert self.dataset.n_classes == 20
    


        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------