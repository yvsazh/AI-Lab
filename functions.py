import dearpygui.dearpygui as dpg
import h5py

def normilize(num, min, max):
	return (num-min)/(max - min)
def explore_h5_file(s, a, file_path):
	with dpg.window(label="Explore your h5 file", width=400, height=400, pos=(200, 100), no_close=False, no_resize=True):
		with h5py.File(file_path, "r") as file:
			def print_structure(name, obj):
				dpg.add_text(f"Field: {name}")
				if isinstance(obj, h5py.Dataset):
					dpg.add_text(f"  - shape: {obj.shape}")
					dpg.add_text(f"  - dtype: {obj.dtype}")
			file.visititems(print_structure)
def create_error_window(error):
	with dpg.window(label="Errors!!!", width=400, height=200, pos=(200, 100)):
		dpg.add_text("HERE IS SOME ERRORS!!!")
		dpg.add_separator()
		dpg.add_text(error, wrap=0)
def filter_shape_input(sender, app_data, user_data):
	filtered_text = ''.join(c for c in app_data if c.isdigit() or c == ',')
	if filtered_text != app_data:
		dpg.set_value(sender, filtered_text)