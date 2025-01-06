import tensorflow as tf

node_editor_id = 0
training_window_id = 0
input_attributes = {} # input_of_node: [output_of_another_node, connection]
output_connections = {} # output_of_node: [input_of_another_node, connection]

layers = []
model = []

nn_model = tf.keras.Sequential([])

your_image_id = 0
your_image_data = None

global_stop_training = False

user_datasets = []
current_user_dataset_classes = []

training_settings_counter = 1

values = {
	"dataset": "mnist",
	"dataset_type": "system", # system or user
	"loss": "BinaryCrossentropy",
	"optimizer": "Adam",
	"lr": 0.001,
	"epochs": 50,
	"batch_size": 32,
}

elements = {
	"layers": [
		["Dense", "dense"],
		["Conv1D", "conv"],
		["Conv2D", "conv"],
		["Conv3D", "conv"],
		["MaxPooling1D", "pool1D"],
		["MaxPooling2D", "pool2D"],
		["MaxPooling3D", "pool3D"],
		['Flatten', 'flatten'],
		['Dropout', 'dropout'],
	],
	"activations": [
		"linear",
		"sigmoid",
		"tanh",
		"relu",
		"selu",
		"elu",
		"exponential",
		"leaky_relu",
		"relu6",
		"selu",
		"softmax",
		"softplus",

	],
	"datasets": [
		["mnist", "shape is (28, 28)", "system"],
		["cifar10", "shape is (32, 32, 3)", "system"],
		#["cifar100", "shape is (..., )", "system"],
	],
	"dataset_names": ["mnist", "cifar10", "cifar100"],
	"optimizers": [
		["SGD"],
		["RMSprop"],
		["Adam"],
	],
	"losses": [
		["BinaryCrossentropy"],
		["CategoricalCrossentropy"],
		["SparseCategoricalCrossentropy"],
		["MeanSquaredError"],
	]
}