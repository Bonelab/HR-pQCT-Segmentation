'''
Written by Nathan Neeteson
A module containing code for a modified UNet.
Layer is a class containing a repeating unit used in the UNet.
UNet is a class containing the full structure of the UNet, with a
configurable number of layers and number of filters in each layer.
'''
import torch
import torch.nn as nn


# the repeating structure of the UNet, which occurs in each layer on the
# way down and up the encoder and decoder
class Layer(nn.Module):

    def __init__(self, inputs, outputs, kernel_size, padding, stride, groups, dropout):
        super(Layer, self).__init__()

        # as per https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
        # We order the layers as conv->ReLU->normalization-> (repeat)
        # group normalization is used instead of batch normlization
        # because we lacked the VRAM to be putting through 30+ images at a time
        # and if your batch size is too small, batch norm is useless
        self.layer = nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.GroupNorm(groups, outputs),
            nn.Dropout2d(dropout),
            nn.Conv2d(outputs, outputs, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.GroupNorm(groups, outputs),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.layer(x)


class UNet(nn.Module):

    def __init__(self,
                 input_channels,
                 output_classes,
                 num_filters,
                 channels_per_group,
                 dropout
                 ):
        # input_channels (int): number of channels in expected input to the UNet,
        # - channels is simply an extra dimension to the data, it could be that
        # you have multiple color channels, it could be that you want to
        # input numerous slices of the same field, etc.
        # output_classes (int): the number of output predictions you want at each pixel
        # num_filters (list of ints): an entry for each layer you want, and in
        # each entry, the number of filters in that layer

        super(UNet, self).__init__()

        # define the internal parameters of the UNet
        # for now I don't think we need to surface these parameters, I don't
        # see the merit in fiddling with larger kernel sizes or the
        # down/up-sampling factors
        self.layer_kernel_size = 3
        self.layer_padding = (self.layer_kernel_size - 1) // 2  # for same conv
        self.layer_stride = 1

        self.down_pool = 2

        self.up_kernel_size = 2
        self.up_stride = 2

        self.channels_per_group = channels_per_group

        self.dropout = dropout

        # initialize some module lists for the 4 types of operations
        self.layer_down = nn.ModuleList()
        self.down = nn.ModuleList()
        self.layer_up = nn.ModuleList()
        self.up = nn.ModuleList()

        # there is one more down layer than there are up/down sampling ops or
        # up layers, since we class the layer at the bottom of the U-Net as
        # a down layer, so create the first one before the loop
        self.layer_down.append(
            Layer(
                input_channels, num_filters[0],
                self.layer_kernel_size, self.layer_padding, self.layer_stride,
                num_filters[0] // self.channels_per_group, self.dropout
            )
        )

        # then we can iterate over the channel number list, excluding the first
        # entry because all of these operations go between layers
        for fi in range(1, len(num_filters)):
            # downsample is easy, always the same operation
            self.down.append(nn.MaxPool2d(self.down_pool))

            # the next convolutional layer on the downward trajectory slightly
            # reduces image size while moving to the next number of filters
            self.layer_down.append(
                Layer(
                    num_filters[fi - 1], num_filters[fi],
                    self.layer_kernel_size, self.layer_padding, self.layer_stride,
                    num_filters[fi] // self.channels_per_group, self.dropout
                )
            )

            # upsampling always takes our current image and (almost) doubles the
            # spatial resolution by "deconvolution" (*technically not real
            # deconvolution), while going back to the previous number of filters
            self.up.append(
                nn.ConvTranspose2d(
                    num_filters[fi], num_filters[fi - 1],
                    kernel_size=self.up_kernel_size, stride=self.up_stride
                )
            )

            # the next convolutional layer on the upward trajectory takes an
            # input that is the previous upward layer upsampled and then
            # concatenated (along the channel axis) with the (cropped) output
            # from the downward layer across the U at the same height, this is
            # why the number of input channels is twice the current # filters
            self.layer_up.append(
                Layer(
                    2 * num_filters[fi - 1], num_filters[fi - 1],
                    self.layer_kernel_size, self.layer_padding, self.layer_stride,
                    num_filters[fi - 1] // self.channels_per_group, self.dropout
                )
            )

        # the last operation in the U-Net is to use a single voxel kernel to
        # map the top-level filter outputs to the output classes we actually
        # want to predict. CrossEntropyLoss wants pre-softmax outputs, so
        # that's what we output
        self.map_to_output = nn.Conv2d(
            num_filters[0], output_classes,
            kernel_size=1, stride=1
        )

    # this is the method that gets called when the UNet is called on an input
    # argument as if it is a function
    def forward(self, x):

        # initialize a list of the outputs of the downward layers
        x_down = []

        # put the input tensor through the first downward layer and append the
        # output to our list
        x_down.append(self.layer_down[0](x))

        # then iterate through pairs of layers and downsampling operations
        for layer_down, down in zip(self.layer_down[1:], self.down):
            # each time, we take the current last output value, run it through
            # a downsample followed by a layer, and append the new output
            x_down.append(layer_down(down(x_down[-1])))

        # now to prepare for traversing the upward trajectory, we pop off the
        # last downward layer's output and assign it as the "active tensor"
        # (we no longer need to keep any incremental outputs in memory)
        x = x_down.pop()

        # we constructed the upsampling and upward layers in order from top to
        # bottom, so now we need to iterate through these operations in reverse
        for layer_up, up in zip(reversed(self.layer_up), reversed(self.up)):
            # first, pass the active tensor through the upsampling op
            x = up(x)
            # next, concatenate the current tensor with the output from the
            # downwards layer of the same level (using pop to get rid of these
            # tensors as they are processed) and pass that through the current
            # upwards layer
            x = layer_up(torch.cat([x_down.pop(), x], dim=1))

        # then pass the active tensor through the output/"head" layer and out
        return self.map_to_output(x)
