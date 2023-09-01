import albumentations as A



Transform = A.Compose(

    [
                    A.OneOf([
                        A.MotionBlur(p=.2),
                        A.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.2),
                    A.PiecewiseAffine(p=0.2),
                    A.OneOf([
                        A.Sharpen(),
                        A.Emboss(),
                    ], p=0.3),
                    A.OneOf([
                        A.HueSaturationValue(p=0.3),
                        A.RandomBrightnessContrast(p=0.3),
                    ], p=0.4),
                    A.OneOf([
                        A.ImageCompression(p=0.3),
                        A.ISONoise(p=0.4),
                    ], p=0.6)
                    #A.RandomFog(p=0.3),
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)
