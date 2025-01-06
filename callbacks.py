import dearpygui.dearpygui as dpg
import tensorflow as tf
from functions import *

class ProgressBarCallback(tf.keras.callbacks.Callback):
	def __init__(self, progress_bar, epoch_text, values):
		self.progress_bar = progress_bar
		self.epoch_text = epoch_text
		self.values = values

		self.x_axis = [0]
		self.loss_axis = []
		self.acc_axis = []

	def on_epoch_begin(self, epoch, logs=None):
		dpg.set_value("epoch_progress", value="Training is in progress")

	def on_epoch_end(self, epoch, logs=None):
		self.x_axis.append(epoch+1)
		self.loss_axis.append(logs.get('loss'))
		self.acc_axis.append(logs.get('accuracy'))
		current_value = dpg.get_value(self.progress_bar)
		dpg.set_value(self.progress_bar, normilize(epoch+1, 1, self.values['epochs']))
		dpg.set_value(self.epoch_text, f"Epoch: {epoch + 1}")

		dpg.set_value('loss_graph', [self.x_axis, self.loss_axis])
		dpg.set_axis_limits("x_axis", 0, max(self.x_axis))
		dpg.set_axis_limits("y_axis", min(self.loss_axis), max(self.loss_axis))

		dpg.set_value('acc_graph', [self.x_axis, self.acc_axis])
		dpg.set_axis_limits("x_axis_acc", 0, max(self.x_axis))
		dpg.set_axis_limits("y_axis_acc", min(self.acc_axis), max(self.acc_axis))

		dpg.add_button(label=logs.get('loss'), width=200, parent="losses_list_group")
		dpg.add_button(label=logs.get('accuracy'), width=200, parent="acc_list_group")

	def on_train_begin(self, logs=None):
		dpg.set_value(self.progress_bar, 0)
		dpg.set_value('loss_graph', [[], []])
		dpg.set_axis_limits("x_axis", 0, 1)
		dpg.set_axis_limits("y_axis", 0, 1)

	def on_train_end(self, logs=None):
		dpg.set_value("epoch_progress", value="Training finished")