"""
Evaluation methods for training data, GP, etc
"""

from ODE_models.DIJMatrice100 import DJIMatrice100
from gp import GPManager



def evaluate_gp():
    pass

def main():
    model = DJIMatrice100()

    gp_manager = GPManager("DJIMatrice", model.gp_spec())

    print(model.get_LHS_data_ranges_safe(model.default_params))

    model_list = gp_manager.create_ModelList()

    print(type(model_list))

    print("This code runs")
    


if __name__ == "__main__":
    main()

