from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        # rewrite devalue values
        parser.set_defaults(model='test')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        # Additional arguments for GradCAM feedback in pix2pix
        parser.add_argument('--resnet18_path', type=str, default='', help='Path to pretrained ResNet18 for GradCAM guidance')
        parser.add_argument('--lambda_gradcam', type=float, default=10.0, help='Weight for GradCAM loss')

        self.isTrain = False
        return parser
