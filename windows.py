import dearpygui.dearpygui as dpg

from utils import *
from config import *
from layer import *

def elements_window():
	with dpg.window(label="Elements", width=280, height=720, pos=(0, 0), no_move=True, no_resize=True, no_close=True):
		### all layers start
		with dpg.tree_node(label="Layers"):
			with dpg.group(horizontal=True, xoffset=200):
				with dpg.group():
					for element in elements["layers"]:
						dpg.add_button(label=element[0], width=200)
						dpg.add_drag_payload(
							parent=dpg.last_item(),
							drag_data={"parent": node_editor_id, "name": dpg.get_item_label(dpg.last_item()),
									   "type": element[1]},
							payload_type="nn_editor"
						)
		### all layers end

		### all activations start
		with dpg.tree_node(label="Activations"):
			with dpg.group(horizontal=True, xoffset=200):
				with dpg.group():
					for element in elements["activations"]:
						dpg.add_button(label=element, width=200)
						dpg.add_drag_payload(parent=dpg.last_item(), drag_data=element, payload_type="activation")
		### all activations end

		dpg.add_button(label="Create model", width=240, height=50, pos=(20, 640), callback=start_training_window)
def nn_model_editor():
	with dpg.window(label="NN model editor", width=1000, height=720, pos=(280, 0), no_move=True, no_resize=True,
					no_close=True, no_scrollbar=True, no_scroll_with_mouse=True):
		with dpg.group(horizontal=True, width=980, height=670, payload_type="nn_editor",
					   drop_callback=lambda s, a: adding_layer(a['parent'], a['name'], a['type'])):
			with dpg.node_editor(callback=link_callback, delink_callback=delink_callback, minimap=True,
								 minimap_location=dpg.mvNodeMiniMap_Location_BottomRight) as node_editor:
				node_editor_id = dpg.last_item()
				adding_layer(node_editor_id, "Input", "input", pos=(20, 20))
				model.insert(0, layers[0])

# def training_window():
# 	with dpg.window(label="Training settings", tag="training_settings", width=500, height=500, pos=(440, 50),
# 					no_close=False, show=False):
# 		training_window_id = dpg.last_item()
# 		values["dataset"] = dpg.add_input_text(label="Choose dataset", default_value="mnist",
# 											   payload_type="dataset_choice", readonly=True,
# 											   drop_callback=choose_dataset)
# 		### all datasets start
# 		with dpg.tree_node(label="Datasets"):
# 			with dpg.group(horizontal=True, xoffset=200):
# 				with dpg.group():
# 					for element in elements["datasets"]:
# 						dpg.add_button(label=f"{element[0]} ({element[1]})")
# 						dpg.add_drag_payload(
# 							parent=dpg.last_item(),
# 							drag_data={"parent": training_window_id, "name": element[0]},
# 							payload_type="dataset_choice"
# 						)
# 					with dpg.group(tag="datasets"):
# 						pass
#
# 					with dpg.handler_registry():
# 						dpg.add_file_dialog(callback=load_dataset, tag="dataset_file_dialog", height=400, show=False,
# 											modal=True)
# 						dpg.add_file_extension(".h5", parent="dataset_file_dialog", color=(0, 255, 0))
# 					dpg.add_separator()
# 					dpg.add_button(label=f"Load your dataset", callback=lambda: dpg.show_item("dataset_file_dialog"))
# 		with dpg.group(label="Your dataset settings", tag="user_dataset_settings", show=False):
# 			dpg.add_checkbox(label="My dataset has training X and Y data", default_value=True, enabled=False,
# 							 tag="yes_train", callback=lambda s, a: dpg.configure_item("train_columns", show=a))
# 			with dpg.group(tag="train_columns"):
# 				dpg.add_input_text(label="X train column name", tag="x_train_name")
# 				dpg.add_input_text(label="Y train column name", tag="y_train_name")
# 				dpg.add_checkbox(label="My dataset has names of classes", default_value=False, enabled=True,
# 								 tag="yes_classes", callback=lambda s, a: dpg.configure_item("classes_column", show=a))
# 				dpg.add_input_text(label="Classes column", show=False, tag="classes_column")
# 				dpg.add_checkbox(label="My dataset has test X and Y data", tag="yes_test",
# 								 callback=lambda s, a: dpg.configure_item("test_columns", show=a))
# 				with dpg.group(tag="test_columns", show=False):
# 					dpg.add_input_text(label="X test column name", tag="x_test_name")
# 					dpg.add_input_text(label="Y test column name", tag="y_test_name")
# 		### all datasets end
# 		dpg.add_separator()
# 		dpg.add_separator()
# 		values["loss"] = dpg.add_input_text(label="Choose loss function", default_value="BinaryCrossentropy",
# 											payload_type="loss_choice", readonly=True,
# 											drop_callback=lambda s, a: dpg.set_value(s, a["name"]))
# 		### all losses start
# 		with dpg.tree_node(label="Losses"):
# 			with dpg.group(horizontal=True, xoffset=200):
# 				with dpg.group():
# 					for element in elements["losses"]:
# 						dpg.add_button(label=f"{element[0]}", width=200)
# 						dpg.add_drag_payload(
# 							parent=dpg.last_item(),
# 							drag_data={"parent": training_window_id, "name": element[0]},
# 							payload_type="loss_choice"
# 						)
# 		### all lossess end
# 		dpg.add_separator()
# 		dpg.add_separator()
# 		values["optimizer"] = dpg.add_input_text(label="Choose optimizer", default_value="Adam",
# 												 payload_type="opt_choice", readonly=True,
# 												 drop_callback=lambda s, a: dpg.set_value(s, a["name"]))
# 		### all optimizers start
# 		with dpg.tree_node(label="Optimizers"):
# 			with dpg.group(horizontal=True, xoffset=200):
# 				with dpg.group():
# 					for element in elements["optimizers"]:
# 						dpg.add_button(label=f"{element[0]}", width=200)
# 						dpg.add_drag_payload(
# 							parent=dpg.last_item(),
# 							drag_data={"parent": training_window_id, "name": element[0]},
# 							payload_type="opt_choice"
# 						)
# 		### all optimizers end
# 		values["lr"] = dpg.add_input_float(label="Learning rate", default_value=0.001, min_value=0, min_clamped=True,
# 										   max_value=1, max_clamped=True)
# 		dpg.add_separator()
# 		dpg.add_separator()
# 		values["epochs"] = dpg.add_input_int(label="Epochs", default_value=50, min_value=2, min_clamped=True)
# 		values["batch_size"] = dpg.add_input_int(label="Batch size", default_value=32, min_value=1, min_clamped=True)
# 		dpg.add_separator()
# 		dpg.add_separator()
# 		dpg.add_button(label="Start training", width=485, height=25, user_data=values,
# 					   callback=lambda s, a, u: start_training(s, a, u))