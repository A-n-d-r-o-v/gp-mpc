"""
Evaluation methods for training data, GP, etc
"""

from ODE_models.DJIMatrice100 import DJIMatrice100
from gp import GPManager

from gp_data import DataGenerator


def evaluate_gp():
    pass

def main():
    model = DJIMatrice100()

    gp_manager = GPManager("DJIMatrice")

    model_list = gp_manager.create_ModelList(model)

    GPManager.train_models(model_list)

    #DataGenerator.plot_dcor_heatmap(model)

    #res_s = model.step(DJIMatrice100.MatriceState.zero(), DJIMatrice100.MatriceInput.zero(), gp_model_list=model_list)

    GPManager.plot_GP_learning_curve(ode_model=model, model_list=model_list, training_sizes=range(10, 760, 150))

    print("This code runs")


if __name__ == "__main__":
    main()

