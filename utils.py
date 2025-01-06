import dearpygui.dearpygui as dpg
import numpy as np
import h5py
from config import *
from callbacks import ProgressBarCallback
import tensorflow as tf
from PIL import Image
from functions import *
import pickle
def link_callback(sender, app_data):
	input_attr = app_data[1]
	output_attr = app_data[0]
	if input_attributes.get(input_attr) is None:
		if output_connections.get(output_attr) is None:
			dpg.add_node_link(output_attr, input_attr, parent=sender)
			input_attributes[input_attr] = [output_attr, dpg.last_item()]
			# adding layer from model list
			for layer in layers:
				if layer.input_id == input_attr:
					model.append(layer)
			output_connections[output_attr] = [input_attr, dpg.last_item()]
		else:
			print(f"Output attribute {output_attr} is already connected to {output_connections[output_attr][0]}")
	else:
		print(f"Input attribute {input_attr} already connected to {input_attributes[input_attr]}")
def delink_callback(sender, app_data):
	dpg.delete_item(app_data)
	for k, v in input_attributes.items():
		try:
			if input_attributes[k][1] == app_data:
				input_attributes[k] = None
				# deleting layer from model list
				for layer in layers:
					if layer.input_id == k:
						model.pop(layers.index(layer))
		except:
			pass
	for k, v in output_connections.items():
		try:
			if output_connections[k][1] == app_data:
				output_connections[k] = None
		except:
			pass

### Work with datasets and images ###

def pre_load_image():
	dpg.show_item("file_dialog")
	dpg.configure_item("file_dialog", user_data=[dpg.get_value("image_w"), dpg.get_value("image_h"), dpg.get_value("channels"), current_user_dataset_classes])
def choose_dataset(s, a):
	dpg.set_value(s, a["name"])
	for element in elements['datasets']:
		print(a['name'])
		print(element[0])
		if a['name'] != element[0]:
			dpg.show_item("user_dataset_settings")
		else:
			dpg.hide_item("user_dataset_settings")
			break
def load_your_image(sender, app_data, user_data):
	file_path = app_data['file_path_name']
	channels = 1
	image_pil = Image.open(file_path)
	image_resized = image_pil.resize((user_data[0], user_data[1]))
	image_array = np.array(image_resized)
	#image_flatten = image_array.reshape((1, num_px * num_px * channels))
	if user_data[2] > 1:
		image_flatten = image_array.reshape((1, user_data[0], user_data[1], user_data[2]))
	else:
		image_flatten = image_array.reshape((1, user_data[0], user_data[1]))
	my_image = image_flatten / 255.0
	prediction = nn_model.predict(my_image)
	ticks = []
	dpg.delete_item("predict_y", children_only=True)
	dpg.show_item("prediction")
	answer_num = int(np.where(prediction[0] == max(prediction[0]))[0])
	if len(user_data[3]) == 0:
		user_data[3] = range(len(prediction[0]))
	answer = str(user_data[3][answer_num])
	if len(prediction[0]) == 1 and len(user_data[3]) > 1:
		if prediction[0] > 0.5:
			answer = str(user_data[3][1])
		if prediction[0] < 0.5:
			answer = str(user_data[3][0])
	dpg.set_value(
		"answer",
		"File name: " + app_data['file_name'] + "\nAnswer: " + answer + ".\n" + "Prediction: " + str(
			prediction[0]))
	for i, p in enumerate(prediction[0]):
		dpg.add_bar_series([i], [p], parent="predict_y", label=str(i), weight=1)
		ticks.append((str(i), i))

	dpg.set_axis_limits(axis="predict_x", ymin=0, ymax=len(prediction[0]))
	dpg.set_axis_limits(axis="predict_y", ymin=0, ymax=max(prediction[0]))
def load_dataset(sender, app_data):
	with dpg.group(horizontal=True, parent="datasets"):
		dpg.add_button(label=app_data['file_name'])
		dpg.add_drag_payload(
			parent=dpg.last_item(),
			drag_data={"parent": training_window_id, "name": app_data['file_name']},
			payload_type="dataset_choice"
		)
		dpg.add_button(label="Explore dataset", user_data=app_data['file_path_name'], callback=explore_h5_file)

	user_datasets.append([app_data['file_name'], app_data['file_path_name']])
def start_training(s, a, user_data):
	global nn_model
	global current_user_dataset_classes
	nn_model = tf.keras.Sequential([])
	for layer in model:
		if layer not in layers:
			model.remove(layer)
		layer.save_values()
		### CREATING TF MODEL
		eval(layer.generate_layer_text())
		### END CREATING TF MODEL
	nn_model.summary()
	x_train, x_test, y_train, y_test = None, None, None, None
	loss = None
	optimizer = None
	counter = 0
	u = user_data.copy()
	for k, v in user_data.items():
		if dpg.get_value(user_data[k]) != None:
			u[k] = dpg.get_value(user_data[k])
	current_user_dataset_classes = []
	if u['dataset_type'] == "system":
		match u['dataset']:
			case "mnist":
				(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
				x_train, x_test = x_train / 255.0, x_test / 255.0
				counter += 1
			case "cifar10":
				(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
				x_train = x_train.astype("float32") / 255.0
				x_test = x_test.astype("float32") / 255.0
				counter += 1
	if counter == 0:
		for ud in user_datasets:
			if ud[0] == u['dataset']:
				dataset = h5py.File(ud[1], "r")
				if dpg.get_value("yes_train"):
					current_user_dataset_classes = []
					x_train = np.array(dataset[dpg.get_value("x_train_name")][:])
					y_train = np.array(dataset[dpg.get_value("y_train_name")][:])
					x_train = x_train.astype("float32") / 255.0
					if dpg.get_value("yes_classes"):
						current_user_dataset_classes = dataset[dpg.get_value("classes_column")][:]
				if dpg.get_value("yes_test"):
					x_test = np.array(dataset[dpg.get_value("x_test_name")][:])
					y_test = np.array(dataset[dpg.get_value("y_test_name")][:])
					x_test = x_test.astype("float32") / 255.0

	match u['loss']:
		case "BinaryCrossentropy":
			loss = tf.keras.losses.BinaryCrossentropy()
		case "CategoricalCrossentropy":
			loss = tf.keras.losses.CategoricalCrossentropy()
		case "SparseCategoricalCrossentropy":
			loss = tf.keras.losses.SparseCategoricalCrossentropy()
		case "MeanSquaredError":
			loss = tf.keras.losses.MeanSquaredError()

	match u['optimizer']:
		case "SGD":
			optimizer = tf.keras.optimizers.SGD(learning_rate=u['lr'])
		case "RMSprop":
			optimizer = tf.keras.optimizers.RMSprop(learning_rate=u['lr'])
		case "Adam":
			optimizer = tf.keras.optimizers.Adam(learning_rate = u['lr'])

	nn_model.compile(
		loss=loss,
		optimizer=optimizer,
		metrics=["accuracy"]
	)

	try:
		dpg.delete_item("training_window")
		dpg.delete_item("file_dialog")
		dpg.delete_item("export_model_fd")
	except:
		pass
	with dpg.window(label="Training Progress", tag="training_window", width=765, height=500, pos=(200, 100), no_close=True, no_resize=True):
		progress_bar = dpg.add_progress_bar(label="Training Progress", width=450, height=25)
		epoch_text = dpg.add_text(default_value="Epoch: 0")
		dpg.add_text(default_value="", tag="epoch_progress")
		dpg.add_separator()
		dpg.add_separator()

		with dpg.group(horizontal=True):
			with dpg.group():
				with dpg.plot(label="Loss graph", width=370, height=300):
					dpg.add_plot_legend()
					dpg.add_plot_axis(dpg.mvXAxis, label="Epochs", tag="x_axis")
					dpg.add_plot_axis(dpg.mvYAxis, label="Loss", tag="y_axis")
					dpg.add_line_series([], [], label="Losses", parent="y_axis", tag="loss_graph")
				with dpg.tree_node(label="Losses"):
					with dpg.group(horizontal=True, width=300, xoffset=200):
						with dpg.group(tag="losses_list_group"):
							pass

			with dpg.group():
				with dpg.plot(label="Accuracy graph", width=370, height=300):
					dpg.add_plot_legend()
					dpg.add_plot_axis(dpg.mvXAxis, label="Epochs", tag="x_axis_acc")
					dpg.add_plot_axis(dpg.mvYAxis, label="Accuracy", tag="y_axis_acc")
					dpg.add_line_series([], [], label="Accuracy", parent="y_axis_acc", tag="acc_graph")

				with dpg.tree_node(label="Accuracy"):
					with dpg.group(horizontal=True, width=300, xoffset=200):
						with dpg.group(tag="acc_list_group"):
							pass
		try:
			training_history = nn_model.fit(x_train, y_train, batch_size=u['batch_size'], epochs=u['epochs'], verbose=1, callbacks=[ProgressBarCallback(progress_bar, epoch_text, u)])
			evaluate_history = None
			try:
				if x_test.any() != None:
					evaluate_history = nn_model.evaluate(x_test, y_test, batch_size=u['batch_size'], verbose=1)
			except:
				pass
			dpg.add_separator()
			dpg.add_separator()

			with dpg.group():
				dpg.configure_item("training_window", no_close=False)
				dpg.add_text("TEST ON YOUR IMAGE")
				dpg.add_input_int(label="Width of image", tag="image_w", default_value=28, min_value=1, min_clamped=True)
				dpg.add_input_int(label="Height of image", tag="image_h", default_value=28, min_value=1, min_clamped=True)
				dpg.add_input_int(label="Number of channels", tag="channels", default_value=1, min_value=1, min_clamped=True)
				dpg.add_text("when image is transparent -> +1 to num of channels")
				with dpg.handler_registry():
					dpg.add_file_dialog(callback=load_your_image,tag="file_dialog", height=400, show=False, directory_selector=False, modal=True)
					dpg.add_file_extension(".jpg,.jpeg,.png", parent="file_dialog", color=(255, 0, 0))
				dpg.add_button(label="Load Image", callback=pre_load_image)
				with dpg.plot(label="Prediction", show=False, tag="prediction", height=300):
					dpg.add_plot_legend()
					predict_x = dpg.add_plot_axis(dpg.mvXAxis, label="Variants", no_gridlines=True, tag="predict_x")
					predict_y = dpg.add_plot_axis(dpg.mvYAxis, label="Procentage", tag="predict_y")
				dpg.add_text(label="Answer", tag="answer")
			with dpg.group():
				dpg.add_separator()
				dpg.add_separator()
				dpg.add_text(f"Result on a test set(loss, accuracy): {evaluate_history}")
				with dpg.file_dialog(directory_selector=True, show=False, callback=export_callback, height=400, id="export_model_fd"):
					pass
				dpg.add_input_text(id="model_name", default_value="model_name", width=740, height=40)
				dpg.add_button(label="Export model", width=740, height=40, callback=lambda: dpg.show_item("export_model_fd"))
		except Exception as e:
			dpg.delete_item("training_window")
			print(str(e))
			create_error_window(str(e))
### EXPORTING KERAS MODEL

def export_callback(sender, app_data):
	file_name = dpg.get_value("model_name").replace(" ", "")
	exporint_path = app_data['file_path_name'] + "/" + file_name + ".keras"
	nn_model.save(exporint_path)



### SAVING SYSTEM ###

def save_project(sender, app_data, node_editor):
	with open('saving_file.pkl', 'wb') as file:
		pickle.dump(layers, file)
def open_project(sender, app_data, user_data):
	with open('saving_file.pkl', 'rb') as file:
		layers = pickle.load(file)

	for layer in layers:
		print(layer.name, layer.pos)
		# layer.create(node_editor_id)