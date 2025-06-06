import h5py

h5_path = "BirdNET-CoreML/coreml_export/input/audio-model.h5"
with h5py.File(h5_path, "r") as f:
    def print_attrs(name, obj):
        print(name, dict(obj.attrs))
    f.visititems(print_attrs)
