"""
Evaluation methods for training data, GP, etc
"""

from ODE_models.DJIMatrice100 import DJIMatrice100
from gp import GPManager



def evaluate_gp():
    pass

def main():
    model = DJIMatrice100()

    gp_manager = GPManager("DJIMatrice")

    print(model.get_LHS_data_ranges_safe())

    model_list = gp_manager.create_ModelList(model)

    GPManager.train_models(model_list)

    res_s = model.step(DJIMatrice100.MatriceState.zero(), DJIMatrice100.MatriceInput.zero(), gp_model_list=model_list)

    #import torch
    #GPManager.make_predictions(model_list, torch.tensor([1,2,3,4,5], dtype=torch.float64).unsqueeze(0))



    #GPManager.make_predictions()

    #res_s = model.differentiate(res_s, DJIMatrice100.MatriceInput.zero())
    #print(res_s)

    print(type(model_list))

    print("This code runs")


if __name__ == "__main__":
    main()

