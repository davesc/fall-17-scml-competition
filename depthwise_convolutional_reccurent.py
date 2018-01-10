# -*- coding: utf-8 -*-
"""Convolutional-recurrent layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
# from keras import activations
# from keras import initializers
# from keras import regularizers
# from keras import constraints
from keras.layers.convolutional_recurrent import ConvLSTM2D

import numpy as np
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces





class DepthwiseConvLSTM2D(ConvLSTM2D):
    """Depthwise Convolutional LSTM.
    
    This is a modified version of 
    keras.layers.ConvSLTM2D
    github.com/keras-team/keras/blob/master/keras/layers/convolutional_recurrent.py

    This is similar to ConvLSTM2D, but it computes the depthwise part of a
    separable convolution. 

    # Arguments
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        depth_multiplier: An integer value specifying the multiplier for 
        	the number of output filters. Default = 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, time, ..., channels)`
            while `channels_first` corresponds to
            inputs with shape `(batch, time, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Use in combination with `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, rocess the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.

    # Input shape
        - if data_format='channels_first'
            5D tensor with shape:
            `(samples,time, channels, rows, cols)`
        - if data_format='channels_last'
            5D tensor with shape:
            `(samples,time, rows, cols, channels)`

     # Output shape
        - if `return_sequences`
             - if data_format='channels_first'
                5D tensor with shape:
                `(samples, time, filters, output_row, output_col)`
             - if data_format='channels_last'
                5D tensor with shape:
                `(samples, time, output_row, output_col, filters)`
        - else
            - if data_format ='channels_first'
                4D tensor with shape:
                `(samples, filters, output_row, output_col)`
            - if data_format='channels_last'
                4D tensor with shape:
                `(samples, output_row, output_col, filters)`
            where o_row and o_col depend on the shape of the filter and
            the padding

    # Raises
        ValueError: in case of invalid constructor arguments.

    # References
        - [Convolutional LSTM Network: A Machine Learning Approach for
        Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
        The current implementation does not include the feedback loop on the
        cells output
    """

    @interfaces.legacy_convlstm2d_support
    def __init__(self, kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(DepthwiseConvLSTM2D, self).__init__(filters=None,
                                                  kernel_size=kernel_size,
                                                  strides=strides,
                                                  padding=padding,
                                                  data_format=data_format,
                                                  dilation_rate=dilation_rate,
                                                  activation=activation,
                                                  recurrent_activation=recurrent_activation,
                                                  use_bias=use_bias,
                                                  kernel_initializer=kernel_initializer,
                                                  recurrent_initializer=recurrent_initializer,
                                                  bias_initializer=bias_initializer,
                                                  unit_forget_bias=unit_forget_bias,
                                                  kernel_regularizer=kernel_regularizer,
                                                  recurrent_regularizer=recurrent_regularizer,
                                                  bias_regularizer=bias_regularizer,
                                                  activity_regularizer=activity_regularizer,
                                                  kernel_constraint=kernel_constraint,
                                                  recurrent_constraint=recurrent_constraint,
                                                  bias_constraint=bias_constraint,
                                                  return_sequences=return_sequences,
                                                  go_backwards=go_backwards,
                                                  stateful=stateful,
                                                  dropout=dropout,
                                                  recurrent_dropout=recurrent_dropout,
                                                  **kwargs)
        self.depth_multiplier = depth_multiplier

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if self.data_format == 'channels_first':
            rows = input_shape[3]
            cols = input_shape[4]
            self.output_filters = input_shape[2] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[2]
            cols = input_shape[3]
            self.output_filters = input_shape[4] * self.depth_multiplier
        rows = conv_utils.conv_output_length(rows,
                                             self.kernel_size[0],
                                             padding=self.padding,
                                             stride=self.strides[0],
                                             dilation=self.dilation_rate[0])
        cols = conv_utils.conv_output_length(cols,
                                             self.kernel_size[1],
                                             padding=self.padding,
                                             stride=self.strides[1],
                                             dilation=self.dilation_rate[1])
        if self.return_sequences:
            if self.data_format == 'channels_first':
                output_shape = (input_shape[0], input_shape[1],
                                self.output_filters, rows, cols)
            elif self.data_format == 'channels_last':
                output_shape = (input_shape[0], input_shape[1],
                                rows, cols, self.output_filters)
        else:
            if self.data_format == 'channels_first':
                output_shape = (input_shape[0], self.output_filters, rows, cols)
            elif self.data_format == 'channels_last':
                output_shape = (input_shape[0], rows, cols, self.output_filters)

        if self.return_state:
            if self.data_format == 'channels_first':
                output_shape = [output_shape] + [(input_shape[0], self.output_filters, rows, cols) for _ in range(2)]
            elif self.data_format == 'channels_last':
                output_shape = [output_shape] + [(input_shape[0], rows, cols, self.output_filters) for _ in range(2)]
        print('output shape is: ',output_shape)
        return output_shape


    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        batch_size = input_shape[0] if self.stateful else None
        self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:])
        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (filters)
            self.states = [None, None]

        if self.data_format == 'channels_first':
            channel_axis = 2
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        self.output_filters = input_dim * self.depth_multiplier
        state_shape = [None] * 4
        state_shape[channel_axis] = input_dim
        state_shape = tuple(state_shape)
        self.state_spec = [InputSpec(shape=state_shape), InputSpec(shape=state_shape)]
        kernel_shape = self.kernel_size + (input_dim, self.depth_multiplier * 4)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (input_dim, self.depth_multiplier * 4)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.depth_multiplier * 4,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if self.unit_forget_bias:
                bias_value = np.zeros((self.depth_multiplier * 4,))
                bias_value[self.depth_multiplier: self.depth_multiplier * 2] = 1.
                K.set_value(self.bias, bias_value)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :, :, :self.depth_multiplier]
        self.recurrent_kernel_i = self.recurrent_kernel[:, :, :, :self.depth_multiplier]
        self.kernel_f = self.kernel[:, :, :, self.depth_multiplier: self.depth_multiplier * 2]
        self.recurrent_kernel_f = self.recurrent_kernel[:, :, :, self.depth_multiplier: self.depth_multiplier * 2]
        self.kernel_c = self.kernel[:, :, :, self.depth_multiplier * 2: self.depth_multiplier * 3]
        self.recurrent_kernel_c = self.recurrent_kernel[:, :, :, self.depth_multiplier * 2: self.depth_multiplier * 3]
        self.kernel_o = self.kernel[:, :, :, self.depth_multiplier * 3:]
        self.recurrent_kernel_o = self.recurrent_kernel[:, :, :, self.depth_multiplier * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.depth_multiplier]
            self.bias_f = self.bias[self.depth_multiplier: self.depth_multiplier * 2]
            self.bias_c = self.bias[self.depth_multiplier * 2: self.depth_multiplier * 3]
            self.bias_o = self.bias[self.depth_multiplier * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    def get_initial_state(self, inputs):
        # (samples, timesteps, rows, cols, filters)
        initial_state = K.zeros_like(inputs)
        # (samples, rows, cols, filters)
        initial_state = K.sum(initial_state, axis=1)
        shape = list(self.kernel_shape)
        shape[-1] = self.depth_multiplier
        initial_state = self.input_conv(initial_state,
                                        K.zeros(tuple(shape)),
                                        padding=self.padding)

        initial_states = [initial_state for _ in range(2)]
        return initial_states

    def reset_states(self):
        if not self.stateful:
            raise RuntimeError('Layer must be stateful.')
        input_shape = self.input_spec[0].shape
        output_shape = self.compute_output_shape(input_shape)
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete '
                             'input_shape must be provided '
                             '(including batch size). '
                             'Got input shape: ' + str(input_shape))
        if self.return_sequences:
            if self.return_state:
                output_shape = output_shape[1]
            else:
                output_shape = (input_shape[0],) + output_shape[2:]
        else:
            if self.return_state:
                output_shape = output_shape[1]
            else:
                output_shape = (input_shape[0],) + output_shape[1:]

        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros(output_shape))
            K.set_value(self.states[1],
                        np.zeros(output_shape))
        else:
            self.states = [K.zeros(output_shape),
                           K.zeros(output_shape)]

    def get_constants(self, inputs, training=None):
        constants = []
        if self.implementation == 0 and 0 < self.dropout < 1:
            ones = K.zeros_like(inputs)
            ones = K.sum(ones, axis=1)
            ones += 1

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.recurrent_dropout < 1:
            shape = list(self.kernel_shape)
            shape[-1] = self.depth_multiplier
            ones = K.zeros_like(inputs)
            ones = K.sum(ones, axis=1)
            ones = self.input_conv(ones, K.zeros(shape),
                                   padding=self.padding)
            ones += 1.

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)
            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.depthwise_conv2d(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = K.depthwise_conv2d(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out
