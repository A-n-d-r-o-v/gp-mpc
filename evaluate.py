"""
Evaluation methods for training data, GP, etc
"""

from ODE_models.DIJMatrice100 import DJIMatrice100




def evaluate_gp():
    pass

def main():
    model = DJIMatrice100()

    print(model.get_LHS_data_ranges_safe(model.default_params))

    print("This code runs")
    


if __name__ == "__main__":
    main()

