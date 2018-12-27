import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import predictor
import matplotlib.pyplot as plt


def plot_image(i, predictions_array, true_label_array, img_array, class_names):
    predictions_array, true_label, img = predictions_array[i], true_label_array[i], img_array[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_predictions(predictions_array, test_labels, test_images, class_names, wrong_predictions=True):
    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    k = 0
    for i in range(num_images):
        if wrong_predictions:
            while np.argmax(predictions_array[k]) == test_labels[k]:
                k = int(np.random.rand(1)[0]*len(predictions_array))
        else:
            while np.argmax(predictions_array[k]) != test_labels[k]:
                k = int(np.random.rand(1)[0]*len(predictions_array))
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(k, predictions_array, test_labels, test_images, class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(k, predictions_array, test_labels)
        k = int(np.random.rand(1)[0] * len(predictions_array))
    plt.show()

def test():
    # load predictor
    full_model_dir = r"/tmp/mnist_convnet_vgg_16_model\\1545608795"
    #full_model_dir = r"/tmp/mnist_convnet_model\\1545510320"
    mnist_predictor = predictor.from_saved_model(full_model_dir)

    # load data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # predict
    input_to_predictor = {"x": eval_data.reshape(-1, 28, 28, 1)}
    output_dict = mnist_predictor(input_to_predictor)

    results_dict= {"x": eval_data.reshape(-1, 28, 28),
                   "true_labels": eval_labels,
                   "probabilities": output_dict['probabilities'],
                   "predicted class": output_dict['classes']}

    plot_predictions(results_dict['probabilities'],
                     results_dict['true_labels'],
                     results_dict['x'],
                     ['0', '1', '2', '3', '4',
                         '5', '6', '7', '8', '9'])

def test_context():
    # load predictor
    #full_model_dir = r"/tmp/mnist_convnet_vgg_16_model\\1545608795"
    #full_model_dir = r"/tmp/mnist_convnet_model\\1545510320"
    #full_model_dir = r"/tmp/mnist_convnet_context_model\\1545690570"
    #full_model_dir = r"/tmp/mnist_convnet_context_model\\1545692299"
    full_model_dir = r"/tmp/mnist_convnet_vgg_16_context_model\\1545774538"
    full_model_dir = r"/tmp/mnist_convnet_vgg_16_context_model\\1545775232"
    full_model_dir = r"/tmp/mnist_convnet_vgg_16_context_model\\1545852820"
    #full_model_dir = r"\tmp\mnist_convnet_context_model\1545692299"
    #full_model_dir = '/tmp/mnist_convnet_context_model\\1545934936'
    mnist_predictor = predictor.from_saved_model(full_model_dir)

    # load data
    # Load training and eval data
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_mnist_data = mnist.train.images  # Returns np.array
    train_mnist_data_context = np.zeros((len(train_mnist_data), 1), dtype=np.int32)
    train_mnist_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    eval_mnist_data = mnist.test.images.astype('float64')  # Returns np.array
    eval_mnist_data_context = np.zeros((len(eval_mnist_data), 1), dtype=np.int32)
    eval_mnist_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_fmnist_data, train_fmnist_labels), (eval_fmnist_data, eval_fmnist_labels) = fashion_mnist.load_data()
    train_fmnist_data = train_fmnist_data / 255.0
    train_fmnist_data = train_fmnist_data.reshape(len(train_fmnist_data), 28 * 28)
    train_fmnist_labels = train_fmnist_labels + 10
    eval_fmnist_data = eval_fmnist_data / 255.0
    eval_fmnist_data = eval_fmnist_data.reshape(len(eval_fmnist_data), 28 * 28)
    eval_fmnist_labels = eval_fmnist_labels + 10
    train_fmnist_data_context = np.ones((len(train_fmnist_data), 1), dtype=np.int32)
    eval_fmnist_data_context = np.ones((len(eval_fmnist_data), 1), dtype=np.int32)

    train_data = np.concatenate([train_mnist_data[:3000], train_fmnist_data[:3000]]).astype('float32')
    train_data_context = np.concatenate([train_mnist_data_context[:3000], train_fmnist_data_context[:3000]]).astype('int32')
    train_labels = np.concatenate([train_mnist_labels[:3000], train_fmnist_labels[:3000]]).astype('int32')

    eval_data = np.concatenate([eval_mnist_data[:3000], eval_fmnist_data[:3000]]).astype('float32')
    eval_data_context = np.concatenate([eval_mnist_data_context[:3000], eval_fmnist_data_context[:3000]]).astype('int32')
    eval_labels = np.concatenate([eval_mnist_labels[:3000], eval_fmnist_labels[:3000]]).astype('int32')


    # for i in range(len(train_labels)):
    #     i = int(np.random.rand(1)[0]*len(train_labels))
    #     plot_image(i, np.zeros(len(train_labels)), train_labels, train_data.reshape(-1, 28, 28), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    #                   'T-shirt/top', 'Trouser', 'Pullover', 'Dress',
    #                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
    #     plt.show()
    # predict
    input_to_predictor = {"x": eval_data.reshape(-1, 28, 28, 1),
                          "context_index": eval_data_context.reshape(-1, 1)}
    output_dict = mnist_predictor(input_to_predictor)

    results_dict= {"x": eval_data.reshape(-1, 28, 28),
                   "true_labels": eval_labels,
                   "probabilities": output_dict['probabilities'],
                   "predicted class": output_dict['classes']}

    conf_matrix(results_dict)

    plot_predictions(results_dict['probabilities'],
                     results_dict['true_labels'],
                     results_dict['x'],
                     ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      'T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                      'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
                     wrong_predictions=True)

    plot_predictions(results_dict['probabilities'],
                     results_dict['true_labels'],
                     results_dict['x'],
                     ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      'T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                      'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
                     wrong_predictions=False)


def conf_matrix(results_dict):
    conf_matrix = np.zeros((20, 20))
    for i, j in zip(results_dict["true_labels"], results_dict["predicted class"]):
        conf_matrix[i, j] = conf_matrix[i, j] + 1

    conf_matrix = (100.0 * conf_matrix) / len(results_dict["predicted class"])
    with np.printoptions(precision=2, linewidth=200):
        print(conf_matrix)
        print('accuracy: ' + str(np.trace(conf_matrix)))
    return conf_matrix


#     predictor = predictor.from_saved_model(r"/tmp/mnist_convnet_model\\1545510320")
#
# predictions = list(classifier.predict(input_fn=predict_input_fn))
# predicted_classes = [p["classes"] for p in predictions]


if __name__ == "__main__":
    #test()
    test_context()


# predict_fn = predictor.from_saved_model("/tmp/mnist_convnet_model")
# # predictions = predict_fn(
# #     {"x": [[6.4, 3.2, 4.5, 1.5],
# #            [5.8, 3.1, 5.0, 1.7]]})
# # print(predictions['scores'])


# def main():
#     # ...
#     # preprocess-> features_test_set
#     # ...
#     full_model_dir = r"/tmp/mnist_convnet_model\\1545510320"
#     with tf.Session() as sess:
#         tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], full_model_dir)
#         predictor   = predictor.from_saved_model(full_model_dir)
#         model_input = tf.train.Example(features=tf.train.Features( feature={"words": tf.train.Feature(int64_list=tf.train.Int64List(value=features_test_set)) }))
#         model_input = model_input.SerializeToString()
#         output_dict = predictor({"predictor_inputs":[model_input]})
#         y_predicted = output_dict["pred_output_classes"][0]
