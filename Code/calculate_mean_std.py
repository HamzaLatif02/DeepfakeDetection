import torch

# calcualte the average mean and std values that will be used to normalize the dataset
def calculate_mean_std(loader):
    # initialise variables
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images_count = 0

    # loop through images in folder
    for images, _ in loader:
        # update total count of images
        batch_samples = images.size(0)
        total_images_count += batch_samples

        # reshape the images to (batch_size, channels, -1)
        # -1 flattens the remaining two dimensions (width and height)
        images = images.view(batch_samples, images.size(1), -1)
        
        # sum up the mean and std values for each channel
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    # average the mean and std by the number of images
    mean /= total_images_count
    std /= total_images_count

    return mean, std