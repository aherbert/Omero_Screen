from omero.gateway import BlitzGateway
from omero_screen.data_structure import ExperimentData
from omero_screen.aggregator import ImageAggregator
from omero_screen.general_functions import save_fig, scale_img, generate_image, generate_random_image, get_well_pos
from skimage import io
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import json
from tqdm import tqdm
import matplotlib

matplotlib.use('MacOSX')  # avoid matplotlib warning about interactive backend


class FlatFieldCorr:
    """
    Takes images from a well and applies the aggregation and convolution functions from the Aggregator module
    for flatfield correlations for each channel. Class methethiods are private and results in generation of
    of a list of masks in order of channel number that is stores a the .mask_list attribute. For each mask a sample
    image is generated that shows the original and corrected image with diagonal line plots and also the mask. These
    images are saved in the flatfield_correction_representative_images folder. The mask are saved as tif files with row,
    col and channel number in the flatfield_correction_images folder.
    Public attributes:
    FlatFieldCorr.mask_list, a list of numpy nd arrays. These images are used for flatfiel correction for each channel.
    The order in the list correslpinds to the channel number.
    """

    def __init__(self, current_well: 'Omero Well Object', experiment_data: ExperimentData):
        """ Initiates the class with well parameter and path and channel information from the ExperimentData class"""

        self.mask_dict = None
        self._well = current_well
        self._exp_data = experiment_data
        self._well_pos = get_well_pos(self._exp_data.plate_layout, self._well.getId())
        self.generate_channel_masks()


    def generate_channel_masks(self):
        """
        Method that is instantiated with the class object and assembles the object.mask_dict attributes
        Saves each flat field mask file with well position and channel name
        :return: a dictionary with channel_name : flatfield correction masks
        """
        print("\nAssembling Flatfield Correction Masks for each channel\n"
              "------------------------------------------------------------------------------------------------------\n")
        mask_dict = {}
        # iteration extracts tuple (channel_name, channel_number) as channel
        for channel in tqdm(list(self._exp_data.channels.items())):
            norm_mask = self.aggregate_imgs(channel)
            io.imsave(self._exp_data.flatfield_path / f"{self._well_pos}_{channel[0]}_flatfield_masks.tif", norm_mask)
            mask_dict[channel[0]] = norm_mask  # associates channel name with flatfield mask
        self.mask_dict = mask_dict

    def aggregate_imgs(self, channel: dict) -> np.ndarray:
        """
        Aggregate images in well for specified channel and generate correction mask using the Aggregator Module
        :param channel: dictionary from self.exp_data.channels
        :return: flatfield correction mask for given channel
        """
        index = self._well.countWellSample()
        agg = ImageAggregator(60)
        for i in range(index):
            image = self._well.getImage(i)
            img = generate_image(image, channel[1])
            agg.add_image(img)
        blurred_agg_img = agg.get_gaussian_image(30)
        mask = blurred_agg_img / blurred_agg_img.mean()
        example_img = generate_random_image(self._well, channel)
        # random_img_num = randint(0, index - 1)  # to select random image for flatfield test
        # random_image = self._well.getImage(random_img_num)
        # example_img = generate_image(random_image, channel[1])

        self.check_flatfield_corr(example_img, mask, channel)
        return mask

    def check_flatfield_corr(self, img: np.ndarray, mask: np.ndarray, channel: dict):
        """
        Generate Figure to check flatfield correction for channels
        :param img: ndarray retrieved from generate image function
        :param mask: flatfield mask to test
        :param channel: channel dictionary
        saves figure as pdf files
        :return: None
        """
        # assemble images for matplotlib
        scaled = scale_img(img)
        corr_img = img / mask
        corr_scaled = scale_img(corr_img)
        # order all images for plotting
        data_list = [(scaled, 'original image'), (np.diagonal(img), 'diag. intensities'),
                     (corr_scaled, 'corrected image'), (np.diagonal(corr_img), 'diag. intensities'),
                     (mask, 'flatfield correction mask')]
        # generate figure
        fig, ax = plt.subplots(1, 5, figsize=(20, 5))
        for i, data_tuple in enumerate(data_list):
            plt.sca(ax[i])
            if i in [0, 2, 4]:
                plt.imshow(data_tuple[0], cmap='gray')
            else:
                plt.plot(data_tuple[0])
                plt.ylim(img.min(), 5 * img.min())
            plt.title(data_tuple[1])
        # save and close figure
        fig_id = f"{self._well_pos}_{channel[0]}_flatfield_check"  # using channel name
        save_fig(self._exp_data.flatfield_imgs_path, fig_id)
        plt.close(fig)


if __name__ == "__main__":
    with open('../secrets/config.json') as file:
        data = json.load(file)
    username = data['username']
    password = data['password']
    excel_path = '/Users/hh65/Desktop/221102_cellcycle_exp5.xlsx'

    conn = BlitzGateway(username, password, host="ome2.hpc.susx.ac.uk")

    conn.connect()
    well = conn.getObject("Well", 10707)
    exp_data = ExperimentData(excel_path, conn=conn)
    flatfield_corr = FlatFieldCorr(well, exp_data)
    conn.close()

    print(flatfield_corr.mask_dict['DAPI'].shape)
