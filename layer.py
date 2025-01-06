import dearpygui.dearpygui as dpg
from utils import *
from config import *

### LAYER CLASS START
class Layer():
	def __init__(self, name, pos, type):
		self.name = name
		self.pos = pos
		self.type = type
		self.id = 0
		### VALUES FROM FIELDS
		self.values = {
			"nodes": 0,
			"activation": "linear",
			"padding": "valid",
		}
		### ID'S FOR FIELDS
		self.nodes = 0
		self.input_layer_type = 0
		self.shape = 0
		self.activation = 0
		self.padding = 0
		self.filters = 0
		self.filter_size = 0
		self.pool_size = 0
		self.stride = 0
		self.dropout_rate = 0

		### ID'S FOR INPUT AND OUTPUT
		self.input_id = 0
		self.output_id = 0

		### TEXT START FOR LAYERS
		self.text_start = f"nn_model.add(tf.keras.layers.{self.name}("
		self.text_end = f"))"

	def info(self, s, a, u):
		if self.id == u:
			print(f"Info about {self.name}")

	def generate_layer_text(self):
		match self.type:
			case "input":
				return f"nn_model.add(tf.keras.{self.name}(shape=({self.values['shape']})))"
			case "dense":
				return self.text_start + f"{self.values['nodes']}," + f"activation = '{self.values['activation']}'" + self.text_end
			case "conv":
				return self.text_start + f"{self.values['filters']}, {self.values['filter_size']}, strides={self.values['stride']}, activation='{self.values['activation']}', padding='{self.values['padding']}'" + self.text_end
			case "pool1D":
				return self.text_start + f"pool_size={self.values['pool_size']}, strides={self.values['stride']}" + self.text_end
			case "pool2D":
				return self.text_start + f"pool_size=[{self.values['pool_size'][0]}, {self.values['pool_size'][1]}], strides={self.values['stride']}" + self.text_end
			case "pool3D":
				return self.text_start + f"pool_size=[{self.values['pool_size'][0]}, {self.values['pool_size'][1]}, {self.values['pool_size'][2]}], strides={self.values['stride']}" + self.text_end
			case "flatten":
				return self.text_start + f"" + self.text_end
			case "dropout":
				return self.text_start + f"{self.values['dropout_rate']}" + self.text_end
	### GET VALUES FROM FIELDS
	def save_values(self):
		self.values["nodes"] = dpg.get_value(self.nodes)
		self.values["activation"] = dpg.get_value(self.activation)
		self.values["padding"] = dpg.get_value(self.padding)
		self.values["filters"] = dpg.get_value(self.filters)
		self.values['filter_size'] = dpg.get_value(self.filter_size)
		self.values['pool_size'] = dpg.get_value(self.pool_size)
		self.values['stride'] = dpg.get_value(self.stride)
		self.values['shape'] = dpg.get_value(self.shape)
		self.values['dropout_rate'] = dpg.get_value(self.dropout_rate)
		self.values['input_layer_type'] = dpg.get_value(self.input_layer_type)

	def create(self, parent):
		with dpg.node(label=self.name, parent=parent, pos=self.pos):
			self.id = dpg.last_item()
			### POPUP MENU
			with dpg.popup(self.id, tag=f"settings_{self.id}", no_move=True):
				if self.type != "input":
					dpg.add_selectable(label="Delete", user_data=self.id, callback=lambda s, a, u: delete_layer(s, a, u))

				dpg.add_selectable(label="Info", user_data=self.id, callback=lambda s, a, u: self.info(s, a, u))
			### END POPUP MENU

			### INPUT
			if self.type != "input":
				with dpg.node_attribute(label="Input layer", attribute_type=dpg.mvNode_Attr_Input) as input_attr:
					self.input_id = input_attr
					dpg.add_input_text(default_value="Input layer", readonly=True, width=150)
					input_attributes[input_attr] = None
			### END INPUT

			match self.type:
				case "input":
				### INPUT TYPE LAYER: nodes
					with dpg.node_attribute(label="Nodes in layer", attribute_type=dpg.mvNode_Attr_Static):
						self.shape = dpg.add_input_text(label="Shape", default_value="28,28", callback=filter_shape_input, width=150)
				### END INPUT TYPE LAYER

				case "dense":
				### DENSE TYPE LAYER: nodes, activation
					with dpg.node_attribute(label="Nodes in layer", attribute_type=dpg.mvNode_Attr_Static):
						self.nodes = dpg.add_input_int(label="Nodes", min_value=1, min_clamped=True, default_value=10, width=150)
					with dpg.node_attribute(label="Activation function", attribute_type=dpg.mvNode_Attr_Static):
						self.activation = dpg.add_input_text(label="Activation", default_value="linear", readonly=True, width=150, payload_type="activation", drop_callback=lambda s, a: dpg.set_value(s, a))
				### END DENSE TYPE LAYER

				case "conv":
				### CONV TYPE LAYER: filters, filter_size, activation, padding, stride
					with dpg.node_attribute(label="Num of filters", attribute_type=dpg.mvNode_Attr_Static):
						self.filters = dpg.add_input_int(label="Num of filters", min_value=1, min_clamped=True, default_value=10, width=150)
					with dpg.node_attribute(label="Filter size", attribute_type=dpg.mvNode_Attr_Static):
						self.filter_size = dpg.add_input_int(label="Filter size", min_value=1, min_clamped=True, default_value=3, width=150)
					with dpg.node_attribute(label="Stride", attribute_type=dpg.mvNode_Attr_Static):
						self.stride = dpg.add_input_int(label="stride", min_value=1, min_clamped=True, default_value=1, width=150)
					with dpg.node_attribute(label="Activation function", attribute_type=dpg.mvNode_Attr_Static):
						self.activation = dpg.add_input_text(label="Activation", default_value="linear", readonly=True, width=150, payload_type="activation", drop_callback=lambda s, a: dpg.set_value(s, a))
					with dpg.node_attribute(label="Padding", attribute_type=dpg.mvNode_Attr_Static):
						items = ["valid", "same"]
						self.padding = dpg.add_combo(items, default_value=items[0], label="Padding", width=150, height_mode=dpg.mvComboHeight_Small)
				### END CONV TYPE LAYER
				case "pool1D":
					### POOL TYPE LAYER: pool_size, stride, padding
					with dpg.node_attribute(label="Pool size", attribute_type=dpg.mvNode_Attr_Static):
						self.pool_size = dpg.add_input_int(label="Pool size", min_value=1, min_clamped=True, default_value=2, width=150)
					with dpg.node_attribute(label="Stride", attribute_type=dpg.mvNode_Attr_Static):
						self.stride = dpg.add_input_int(label="stride", min_value=1, min_clamped=True, default_value=dpg.get_value(self.pool_size), width=150)
					### END POOL TYPE LAYER
				case "pool2D":
					### POOL TYPE LAYER: pool_size, stride, padding
					with dpg.node_attribute(label="Pool size", attribute_type=dpg.mvNode_Attr_Static):
						self.pool_size = dpg.add_input_intx(label="Pool size", min_value=1, min_clamped=True, default_value=[2, 2], size=2, width=150)
					with dpg.node_attribute(label="Stride", attribute_type=dpg.mvNode_Attr_Static):
						self.stride = dpg.add_input_int(label="stride", min_value=1, min_clamped=True, default_value=dpg.get_value(self.pool_size)[0], width=150)
					### END POOL TYPE LAYER
				case "pool3D":
					### POOL TYPE LAYER: pool_size, stride, padding
					with dpg.node_attribute(label="Pool size", attribute_type=dpg.mvNode_Attr_Static):
						self.pool_size = dpg.add_input_intx(label="Pool size", min_value=1, min_clamped=True, default_value=[2, 2, 2], size=3, width=150)
					with dpg.node_attribute(label="Stride", attribute_type=dpg.mvNode_Attr_Static):
						self.stride = dpg.add_input_int(label="stride", min_value=1, min_clamped=True, default_value=dpg.get_value(self.pool_size)[0], width=150)
					### END POOL TYPE LAYER
				case "dropout":
					### DROPOUT TYPE LAYER: pool_size, stride, padding
					with dpg.node_attribute(label="Dropout rate", attribute_type=dpg.mvNode_Attr_Static):
						self.dropout_rate = dpg.add_input_float(label="Dropout rate", min_value=0, min_clamped=True, max_value=0, max_clamped=True, default_value=0.2, width=150)
					### END DROPOUT TYPE LAYER

			### OUTPUT
			with dpg.node_attribute(label="Next layer", attribute_type=dpg.mvNode_Attr_Output) as output_attr:
				self.output_id = output_attr
				dpg.add_input_text(default_value="Next layer", readonly=True, width=150)
				output_connections[output_attr] = None
			### END OUTPUT
### LAYER CLASS END

def adding_layer(parent, name, type, pos = (0, 0)):
	mouse_pos = dpg.get_mouse_pos()
	mouse_pos[0] -= 280
	if type != "input":
		pos = mouse_pos
	new_layer = Layer(name, pos, type)
	new_layer.create(parent)
	layers.append(new_layer)
def delete_layer(s, a, u):
	for layer in layers:
		if layer.id == u:
			try:
				delink_callback("nothing", input_attributes[layer.input_id][1])
			except:
				pass
			try:
				delink_callback("nothing", output_connections[layer.output_id][1])
			except:
				pass
			layers.pop(layers.index(layer))
			dpg.delete_item(u)