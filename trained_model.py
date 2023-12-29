import pickle
# from typing import Optional, Sequence, Tuple
# from thresholding_model_class import GroundTruthMeasurement

def pre_trained_model(input_gauge):
    with open('loaded_model.pkl', 'rb') as loaded_model_file:
        loaded_tm = pickle.load(loaded_model_file)

    # Use the loaded model to make predictions
    predicted_result = loaded_tm.infer(input_gauge)

    return predicted_result


# import pickle
# import numpy as np
# from typing import Optional, Sequence, Tuple
# from thresholding_model_class import ThresholdingModel, GroundTruthMeasurement


# def pre_trained_model(input_gauge):
#     with open('thresholding_model.pkl', 'rb') as model_file:
#         loaded_tm = pickle.load(model_file)

#     # Use the loaded model to make predictions
#     predicted_result = loaded_tm.infer(input_gauge)

#     return predicted_result



# # result=pre_trained_model(12)
# # print(result)




