import numpy
import scipy
import sklearn
import matplotlib
import pandas
import torch

print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pandas:", pandas.__version__)


print("Version de PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("GPU utilizada:", torch.cuda.get_device_name(0))

