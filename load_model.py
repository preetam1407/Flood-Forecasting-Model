# When you get "thresholding_model.pkl" run this script manually using "python load_model.py" this will generate "loaded_model.pkl" that was used to predict the inundation maps

import pickle
from thresholding_model_class import ThresholdingModel

with open('thresholding_model.pkl', 'rb') as model_file:
    loaded_tm = pickle.load(model_file)

# Save the loaded model to a separate file
with open('loaded_model.pkl', 'wb') as loaded_model_file:
    pickle.dump(loaded_tm, loaded_model_file)
