import tensorflow as tf

from model.alexnetmodel import AlexNetModel
from utils import fetch_data

INPUT_WIDTH = 70
INPUT_HEIGHT = 70
INPUT_CHANNELS = 3

NUM_CLASSES = 10

LEARNING_RATE = 0.001   # Original value: 0.01
MOMENTUM = 0.9
KEEP_PROB = 0.5

EPOCHS = 100
BATCH_SIZE = 128

print('Reading CIFAR-10...')
X_train, Y_train, X_test, Y_test = fetch_data.read_cifar_10(image_width=INPUT_WIDTH, image_height=INPUT_HEIGHT)

alexnet = AlexNet(input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, input_channels=INPUT_CHANNELS,
                  num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE, momentum=MOMENTUM, keep_prob=KEEP_PROB)

with tf.Session() as tf_session:
    print('Training dataset...')
    print()

    file_writer = tf.summary.FileWriter(logdir='./log', graph=tf_session.graph)

    summary_operation = tf.summary.merge_all()

    tf_session.run(tf.global_variables_initializer())

    for i in range(EPOCHS):

        print('Calculating accuracies...')

        train_accuracy = alexnet.evaluate(tf_session, X_train, Y_train, BATCH_SIZE)
        test_accuracy = alexnet.evaluate(tf_session, X_test, Y_test, BATCH_SIZE)

        print('Train Accuracy = {:.3f}'.format(train_accuracy))
        print('Test Accuracy = {:.3f}'.format(test_accuracy))
        print()

        print('Training epoch', i + 1, '...')
        alexnet.sgd(tf_session, X_train, Y_train, BATCH_SIZE, file_writer, summary_operation, i)
        print()

    final_train_accuracy = alexnet.evaluate(tf_session, X_train, Y_train, BATCH_SIZE)
    final_test_accuracy = alexnet.evaluate(tf_session, X_test, Y_test, BATCH_SIZE)

    print('Final Train Accuracy = {:.3f}'.format(final_train_accuracy))
    print('Final Test Accuracy = {:.3f}'.format(final_test_accuracy))
    print()

    alexnet.save(tf_session, './model/alexnet')
    print('Model saved.')
    print()

print('Training done successfully.')