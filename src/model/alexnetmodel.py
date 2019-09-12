import tensorflow as tf


class AlexNetModel:

    """
    All the values used for initialization are as according to the paper mentioned.
    More about cross entropy used below in the code, refer

    https://towardsdatascience.com/cross-entropy-from-an-information-theory-point-of-view-456b34fd939d
    """

    def __init__(self, width=227, height=227, channels=3, no_classes=1000, learning_rate=0.01,
                 momentum=0.9, act_prob=0.5):

        self.width = width
        self.height = height
        self.channels = channels
        self.no_classes = no_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.act_prob = act_prob

        self.random_mean = 0
        self.random_stddev = 0.01

        """
        According to paper, the neuron biases in the second, fourth, and fifth convolutional layers, as well
        as in the fully-connected hidden layers are initialized with the constant 1.
        The neuron biases in the remaining layers are initialized with the constant 0.
        """

        with tf.name_scope('input'):
            self.X = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.input_height, self.input_width, self.input_channels], name='X')

        with tf.name_scope('labels'):
            self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='Y')

        with tf.name_scope('dropout'):
            self.dropout_act_prob = tf.placeholder(dtype=tf.float32, shape=(), name='dropout_act_prob')

        # Layer 1
        with tf.name_scope('layer1'):
            layer1_acts = self.get_activations(input=self.X, filter_width=11, filter_height=11, filters_count=96,
                                             stride_x=4, stride_y=4, padding='VALID',
                                             init_biases_with_the_constant_1=False)
            layer1_normalized = self.normalized_value(input=layer1_acts)
            layer1 = self.get_max_pool(input=layer1_normalized, filter_width=3, filter_height=3, stride_x=2, stride_y=2,
                                          padding='VALID')

        # Layer 2
        with tf.name_scope('layer2'):
            layer2_acts = self.get_activations(input=layer1, filter_width=5, filter_height=5, filters_count=256,
                                             stride_x=1, stride_y=1, padding='SAME',
                                             init_biases_with_the_constant_1=True)
            layer2_normalized = self.normalized_value(input=layer2_acts)
            layer2 = self.get_max_pool(input=layer2_normalized, filter_width=3, filter_height=3, stride_x=2, stride_y=2,
                                          padding='VALID')

        # Layer 3.
        with tf.name_scope('layer3'):
            layer3 = self.get_activations(input=layer2, filter_width=3, filter_height=3, filters_count=384,
                                             stride_x=1, stride_y=1, padding='SAME',
                                             init_biases_with_the_constant_1=False)

        # Layer 4.
        with tf.name_scope('layer4'):
            layer4 = self.get_activations(input=layer3, filter_width=3, filter_height=3,
                                             filters_count=384, stride_x=1, stride_y=1, padding='SAME',
                                             init_biases_with_the_constant_1=True)

        # Layer 5.
        with tf.name_scope('layer5'):
            layer5_acts = self.get_activations(input=layer4, filter_width=3, filter_height=3,
                                             filters_count=256, stride_x=1, stride_y=1, padding='SAME',
                                             init_biases_with_the_constant_1=True)
            layer5 = self.get_max_pool(input=layer5_acts, filter_width=3, filter_height=3, stride_x=2,
                                          stride_y=2, padding='VALID')

        # Layer 6.
        with tf.name_scope('layer6'):
            layer5_shape = layer5.get_shape().as_list()
            flattened_input_size = layer5_shape[1] * layer5_shape[2] * layer5_shape[3]
            layer6 = self.fully_connected(input=tf.reshape(layer5, shape=[-1, flattened_input_size]),
                                               inputs_count=flattened_input_size, outputs_count=4096, relu=True,
                                               init_biases_with_the_constant_1=True)
            layer6_with_dropout = self.dropout(input=layer6)

        # Layer 7.
        with tf.name_scope('layer7'):
            layer7 = self.fully_connected(input=layer6_with_dropout, inputs_count=4096, outputs_count=4096, relu=True,
                                               init_biases_with_the_constant_1=True)
            layer7_with_dropout = self.dropout(input=layer7)

        # Layer 8.
        with tf.name_scope('layer8'):
            layer8 = self.fully_connected(input=layer7_with_dropout, inputs_count=4096,
                                                   outputs_count=self.num_classes, relu=False, name='logits')

        # Cross Entropy.
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer8, labels=self.Y,
                                                                       name='cross_entropy')
            self.summary(cross_entropy)

        with tf.name_scope('training'):
            loss_operation = tf.reduce_mean(cross_entropy, name='loss_operation')
            tf.summary.scalar(name='loss', tensor=loss_operation)

            momentum_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)

            gradients = momentum_optimizer.compute_gradients(loss_operation)
            self.training_operation = momentum_optimizer.apply_gradients(gradients, name='training_operation')

            for gradient, variable in gradients:
                if gradient is not None:
                    with tf.name_scope(variable.op.name + '/gradients'):
                        self.summary(gradient)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(layer8_logits, 1), tf.argmax(self.Y, 1), name='correct_prediction')
            self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')
            tf.summary.scalar(name='accuracy', tensor=self.accuracy_operation)

    # Stochastic gradient descent with a batch size of 128.
    def sgd(self, tf_session, X_data, Y_data, batch_size=128, file_writer=None, summary_operation=None,
                    epoch_number=None):
        num_examples = len(X_data)
        step = 0
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_data[offset:end], Y_data[offset:end]
            if file_writer is not None and summary_operation is not None:
                _, summary = tf_session.run([self.training_operation, summary_operation],
                                      feed_dict={self.X: batch_x, self.Y: batch_y,
                                                 self.dropout_keep_prob: self.keep_prob})
                file_writer.add_summary(summary, epoch_number * (num_examples // batch_size + 1) + step)
                step += 1
            else:
                tf_session.run(self.training_operation, feed_dict={self.X: batch_x, self.Y: batch_y,
                                                             self.dropout_keep_prob: self.keep_prob})

    def evaluate(self, tf_session, X_data, Y_data, batch_size=128):
        num_examples = len(X_data)
        total_accuracy = 0
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_data[offset:end], Y_data[offset:end]
            batch_accuracy = tf_session.run(self.accuracy_operation, feed_dict={self.X: batch_x, self.Y: batch_y,
                                                                          self.dropout_keep_prob: 1.0})
            total_accuracy += (batch_accuracy * len(batch_x))
        return total_accuracy / num_examples

    def save(self, tf_session, file_name):
        saver = tf.train.Saver()
        saver.save(tf_session, file_name)

    def restore(self, tf_session, checkpoint_dir):
        saver = tf.train.Saver()
        saver.restore(tf_session, tf.train.latest_checkpoint(checkpoint_dir))

    def random_values(self, shape):
        return tf.random_normal(shape=shape, mean=self.random_mean, stddev=self.random_stddev, dtype=tf.float32)

    def summary(self, var):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', var)

    def get_activations(self, input, filter_width, filter_height, filters_count, stride_x, stride_y, padding='VALID',
               init_biases_with_the_constant_1=False, name='conv'):
        with tf.name_scope(name):
            input_channels = input.get_shape()[-1].value
            filters = tf.Variable(
                self.random_values(shape=[filter_height, filter_width, input_channels, filters_count]),
                name='filters')
            convs = tf.nn.conv2d(input=input, filter=filters, strides=[1, stride_y, stride_x, 1], padding=padding,
                                 name='convs')
            if init_biases_with_the_constant_1:
                biases = tf.Variable(tf.ones(shape=[filters_count], dtype=tf.float32), name='biases')
            else:
                biases = tf.Variable(tf.zeros(shape=[filters_count], dtype=tf.float32), name='biases')
            preactivations = tf.nn.bias_add(convs, biases, name='preactivations')
            activations = tf.nn.relu(preactivations, name='activations')

            with tf.name_scope('filter_summaries'):
                self.summary(filters)

            with tf.name_scope('bias_summaries'):
                self.summary(biases)

            with tf.name_scope('preactivations_histogram'):
                tf.summary.histogram('preactivations', preactivations)

            with tf.name_scope('activations_histogram'):
                tf.summary.histogram('activations', activations)

            return activations

    def normalized_value(self, input, name='lrn'):
        # From article: Local Response Normalization: we used k=2, n=5, α=10^−4, and β=0.75.
        with tf.name_scope(name):
            lrn = tf.nn.local_response_normalization(input=input, depth_radius=2, alpha=10 ** -4,
                                                     beta=0.75, name='local_response_normalization')
            return lrn

    def get_max_pool(self, input, filter_width, filter_height, stride_x, stride_y, padding='VALID', name='pool'):
        with tf.name_scope(name):
            pool = tf.nn.max_pool(input, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1],
                                  padding=padding, name='pool')
            return pool

    def fully_connected(self, input, inputs_count, outputs_count, relu=True, init_biases_with_the_constant_1=False,
                          name='fully_connected'):
        with tf.name_scope(name):
            wights = tf.Variable(self.random_values(shape=[inputs_count, outputs_count]), name='wights')
            if init_biases_with_the_constant_1:
                biases = tf.Variable(tf.ones(shape=[outputs_count], dtype=tf.float32), name='biases')
            else:
                biases = tf.Variable(tf.zeros(shape=[outputs_count], dtype=tf.float32), name='biases')
            preactivations = tf.nn.bias_add(tf.matmul(input, wights), biases, name='preactivations')
            if relu:
                activations = tf.nn.relu(preactivations, name='activations')

            with tf.name_scope('wight_summaries'):
                self.summary(wights)

            with tf.name_scope('bias_summaries'):
                self.summary(biases)

            with tf.name_scope('preactivations_histogram'):
                tf.summary.histogram('preactivations', preactivations)

            if relu:
                with tf.name_scope('activations_histogram'):
                    tf.summary.histogram('activations', activations)

            if relu:
                return activations
            else:
                return preactivations

    def dropout(self, input, name='dropout'):
        with tf.name_scope(name):
            return tf.nn.dropout(input, keep_prob=self.dropout_keep_prob, name='dropout')