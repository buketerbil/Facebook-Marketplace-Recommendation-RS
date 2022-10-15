from PIL import Image
import os
from glob import glob


def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.Resampling.LANCZOS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

if __name__ == '__main__':
    path = "/Users/macbook/Downloads/images/*.jpg"
    dest_path = "/Users/macbook/Downloads/resized_images/"
    dirs = glob(path)
    final_size = 20
    for n, item in enumerate(dirs[:500], 1):
        im = Image.open(item)
        new_im = resize_image(final_size, im)
        new_im.save(dest_path + item.rsplit('/')[-1], 'JPEG', quality=100)