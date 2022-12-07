import random
import torchvision.transforms.functional as TF
from torchvision import transforms, utils


class DataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot90=False,
            with_random_rot180=False,
            with_random_rot270=False,
            with_random_crop=False,
            with_color_jittering=False,
            crop_scale = (0.5, 1.0),
            crop_ratio=(0.9, 1.1)
    ):
        self.img_size = img_size
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot90 = with_random_rot90
        self.with_random_rot180 = with_random_rot180
        self.with_random_rot270 = with_random_rot270
        self.with_random_crop = with_random_crop
        self.with_color_jittering = with_color_jittering
        self.crop_ratio = crop_ratio
        self.crop_scale = crop_scale

    def transform(self, img):

        # resize image and covert to tensor
        img = TF.to_pil_image(img)
        img = TF.resize(img, [self.img_size, self.img_size])

        if self.with_random_hflip and random.random() > 0.5:
            img = TF.hflip(img)

        if self.with_random_vflip and random.random() > 0.5:
            img = TF.vflip(img)

        if self.with_random_rot90 and random.random() > 0.5:
            img = TF.rotate(img, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img = TF.rotate(img, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img = TF.rotate(img, 270)

        if self.with_color_jittering and random.random() > 0.5:
            img = TF.adjust_hue(img, hue_factor=random.random()*0.5-0.25)  # -0.25 ~ +0.25
            img = TF.adjust_saturation(img, saturation_factor=random.random()*0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img, scale=self.crop_scale, ratio=self.crop_ratio)
            img = TF.resized_crop(
                img, i, j, h, w, size=(self.img_size, self.img_size))

        # to tensor
        img = TF.to_tensor(img)
        return img


    def transform_twins(self, img1, img2):

        # resize image and covert to tensor
        img1 = TF.to_pil_image(img1)
        img1 = TF.resize(img1, [self.img_size, self.img_size])

        img2 = TF.to_pil_image(img2)
        img2 = TF.resize(img2, [self.img_size, self.img_size])

        if self.with_random_hflip and random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        if self.with_random_vflip and random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)

        if self.with_random_rot90 and random.random() > 0.5:
            img1 = TF.rotate(img1, 90)
            img2 = TF.rotate(img2, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img1 = TF.rotate(img1, 180)
            img2 = TF.rotate(img2, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img1 = TF.rotate(img1, 270)
            img2 = TF.rotate(img2, 270)

        if self.with_color_jittering and random.random() > 0.5:
            img1 = TF.adjust_hue(img1, hue_factor=random.random()*0.5-0.25)  # -0.25 ~ +0.25
            img1 = TF.adjust_saturation(img1, saturation_factor=random.random()*0.8 + 0.8)  # 0.8 ~ +1.6
            img2 = TF.adjust_hue(img2, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            img2 = TF.adjust_saturation(img2, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img1, scale=self.crop_scale, ratio=self.crop_ratio)
            img1 = TF.resized_crop(
                img1, i, j, h, w, size=(self.img_size, self.img_size))
            img2 = TF.resized_crop(
                img2, i, j, h, w, size=(self.img_size, self.img_size))

        # to tensor
        # img1 = TF.to_tensor(img1)
        # img2 = TF.to_tensor(img2)

        return img1, img2

    def transform_triplets(self, img1, img2, img3):

        # resize image and covert to tensor
        img1 = TF.to_pil_image(img1)
        img1 = TF.resize(img1, [self.img_size, self.img_size])

        img2 = TF.to_pil_image(img2)
        img2 = TF.resize(img2, [self.img_size, self.img_size])

        img3 = TF.to_pil_image(img3)
        img3 = TF.resize(img3, [self.img_size, self.img_size])

        if self.with_random_hflip and random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            img3 = TF.hflip(img3)

        if self.with_random_vflip and random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            img3 = TF.vflip(img3)

        if self.with_random_rot90 and random.random() > 0.5:
            img1 = TF.rotate(img1, 90)
            img2 = TF.rotate(img2, 90)
            img3 = TF.rotate(img3, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img1 = TF.rotate(img1, 180)
            img2 = TF.rotate(img2, 180)
            img3 = TF.rotate(img3, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img1 = TF.rotate(img1, 270)
            img2 = TF.rotate(img2, 270)
            img3 = TF.rotate(img3, 270)

        if self.with_color_jittering and random.random() > 0.5:
            img1 = TF.adjust_hue(img1, hue_factor=random.random()*0.5-0.25)  # -0.25 ~ +0.25
            img1 = TF.adjust_saturation(img1, saturation_factor=random.random()*0.8 + 0.8)  # 0.8 ~ +1.6
            img2 = TF.adjust_hue(img2, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            img2 = TF.adjust_saturation(img2, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6
            img3 = TF.adjust_hue(img3, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            img3 = TF.adjust_saturation(img3, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img1, scale=self.crop_scale, ratio=self.crop_ratio)
            img1 = TF.resized_crop(
                img1, i, j, h, w, size=(self.img_size, self.img_size))
            img2 = TF.resized_crop(
                img2, i, j, h, w, size=(self.img_size, self.img_size))
            img3 = TF.resized_crop(
                img3, i, j, h, w, size=(self.img_size, self.img_size))

        # to tensor
        # img1 = TF.to_tensor(img1)
        # img2 = TF.to_tensor(img2)

        return img1, img2, img3

    def transform_quadruplets(self, img1, img2, img3, img4):
        
        # resize image and covert to tensor
        img1 = TF.to_pil_image(img1)
        img1 = TF.resize(img1, [self.img_size, self.img_size])

        img2 = TF.to_pil_image(img2)
        img2 = TF.resize(img2, [self.img_size, self.img_size])

        img3 = TF.to_pil_image(img3)
        img3 = TF.resize(img3, [self.img_size, self.img_size])

        img4 = TF.to_pil_image(img4)
        img4 = TF.resize(img4, [self.img_size, self.img_size])

        if self.with_random_hflip and random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            img3 = TF.hflip(img3)
            img4 = TF.hflip(img4)

        if self.with_random_vflip and random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            img3 = TF.vflip(img3)
            img4 = TF.vflip(img4)

        if self.with_random_rot90 and random.random() > 0.5:
            img1 = TF.rotate(img1, 90)
            img2 = TF.rotate(img2, 90)
            img3 = TF.rotate(img3, 90)
            img4 = TF.rotate(img4, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img1 = TF.rotate(img1, 180)
            img2 = TF.rotate(img2, 180)
            img3 = TF.rotate(img3, 180)
            img4 = TF.rotate(img4, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img1 = TF.rotate(img1, 270)
            img2 = TF.rotate(img2, 270)
            img3 = TF.rotate(img3, 270)
            img4 = TF.rotate(img4, 270)

        if self.with_color_jittering and random.random() > 0.5:
            img1 = TF.adjust_hue(img1, hue_factor=random.random()*0.5-0.25)  # -0.25 ~ +0.25
            img1 = TF.adjust_saturation(img1, saturation_factor=random.random()*0.8 + 0.8)  # 0.8 ~ +1.6
            img2 = TF.adjust_hue(img2, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            img2 = TF.adjust_saturation(img2, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6
            img3 = TF.adjust_hue(img3, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            img3 = TF.adjust_saturation(img3, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6
            img4 = TF.adjust_hue(img4, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            img4 = TF.adjust_saturation(img4, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img1, scale=self.crop_scale, ratio=self.crop_ratio)
            img1 = TF.resized_crop(
                img1, i, j, h, w, size=(self.img_size, self.img_size))
            img2 = TF.resized_crop(
                img2, i, j, h, w, size=(self.img_size, self.img_size))
            img3 = TF.resized_crop(
                img3, i, j, h, w, size=(self.img_size, self.img_size))
            img4 = TF.resized_crop(
                img4, i, j, h, w, size=(self.img_size, self.img_size))

        return img1, img2, img3, img4

    def transform_quintuplets(self, img1, img2, img3, img4, img5):

        # resize image and covert to tensor
        img1 = TF.to_pil_image(img1)
        img1 = TF.resize(img1, [self.img_size, self.img_size])

        img2 = TF.to_pil_image(img2)
        img2 = TF.resize(img2, [self.img_size, self.img_size])

        img3 = TF.to_pil_image(img3)
        img3 = TF.resize(img3, [self.img_size, self.img_size])

        img4 = TF.to_pil_image(img4)
        img4 = TF.resize(img4, [self.img_size, self.img_size])

        img5 = TF.to_pil_image(img5)
        img5 = TF.resize(img5, [self.img_size, self.img_size])

        if self.with_random_hflip and random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            img3 = TF.hflip(img3)
            img4 = TF.hflip(img4)
            img5 = TF.hflip(img5)

        if self.with_random_vflip and random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            img3 = TF.vflip(img3)
            img4 = TF.vflip(img4)
            img5 = TF.vflip(img5)

        if self.with_random_rot90 and random.random() > 0.5:
            img1 = TF.rotate(img1, 90)
            img2 = TF.rotate(img2, 90)
            img3 = TF.rotate(img3, 90)
            img4 = TF.rotate(img4, 90)
            img5 = TF.rotate(img5, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img1 = TF.rotate(img1, 180)
            img2 = TF.rotate(img2, 180)
            img3 = TF.rotate(img3, 180)
            img4 = TF.rotate(img4, 180)
            img5 = TF.rotate(img5, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img1 = TF.rotate(img1, 270)
            img2 = TF.rotate(img2, 270)
            img3 = TF.rotate(img3, 270)
            img4 = TF.rotate(img4, 270)
            img5 = TF.rotate(img5, 270)

        if self.with_color_jittering and random.random() > 0.5:
            img1 = TF.adjust_hue(img1, hue_factor=random.random()*0.5-0.25)  # -0.25 ~ +0.25
            img1 = TF.adjust_saturation(img1, saturation_factor=random.random()*0.8 + 0.8)  # 0.8 ~ +1.6
            img2 = TF.adjust_hue(img2, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            img2 = TF.adjust_saturation(img2, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6
            img3 = TF.adjust_hue(img3, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            img3 = TF.adjust_saturation(img3, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6
            img4 = TF.adjust_hue(img4, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            img4 = TF.adjust_saturation(img4, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6
            img5 = TF.adjust_hue(img5, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            img5 = TF.adjust_saturation(img5, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img1, scale=self.crop_scale, ratio=self.crop_ratio)
            img1 = TF.resized_crop(
                img1, i, j, h, w, size=(self.img_size, self.img_size))
            img2 = TF.resized_crop(
                img2, i, j, h, w, size=(self.img_size, self.img_size))
            img3 = TF.resized_crop(
                img3, i, j, h, w, size=(self.img_size, self.img_size))
            img4 = TF.resized_crop(
                img4, i, j, h, w, size=(self.img_size, self.img_size))
            img5 = TF.resized_crop(
                img5, i, j, h, w, size=(self.img_size, self.img_size))

        # to tensor
        # img1 = TF.to_tensor(img1)
        # img2 = TF.to_tensor(img2)

        return img1, img2, img3, img4, img5