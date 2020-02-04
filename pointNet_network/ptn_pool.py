import random
import torch

class PtnPool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size, batch_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        self.batch_size = batch_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.pts_pool = []
            self.ptsv_pool = []

    def query(self, pts, ptsv):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return pts, ptsv

        return_pts = []
        return_ptsv = []

        for i in range(self.batch_size):
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.pts_pool.append(pts[i])
                self.ptsv_pool.append(ptsv[i])
                return_pts.append(pts[i])
                return_ptsv.append(ptsv[i])
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    return_pts.append(self.pts_pool[random_id].clone())
                    return_ptsv.append(self.ptsv_pool[random_id].clone())
                    self.pts_pool[random_id] = pts[i]
                    self.ptsv_pool[random_id] = ptsv[i]
                else:       # by another 50% chance, the buffer will return the current image
                    return_pts.append(pts[i])
                    return_ptsv.append(ptsv[i])
        return_pts = torch.stack(return_pts, 0)   # collect all the images and return
        return_ptsv = torch.stack(return_ptsv, 0)
        return return_pts, return_ptsv
