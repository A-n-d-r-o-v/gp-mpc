# Mastering the Unpredictable


This branch contains a soft-merge of first-working-example, gp-evaluation, multiGP, sparseGP, and a restructure of the ODE and GP code.

The ACADOS side is currently unsupported since it has not yet been updated to reflect the changes.

ODEModel: Centralized to easily create new models. To create a new ODE model, all you need to do is create a new class that inherits from ODEModel and specify the bare minimum, like ODE dataclasses (State, Input, Params), ODEs, and GP setup. The rest of the pipeline is handled separately.

String-based variable id (e.g. “roll_ref”), automatically cross-checked against the ODE dataclasses.

More configuration for GPs: use different types (sparse, multi, single, … ) for different state variables (the appropriate type is selected automatically). Specify different kernels and likelihood modules (unsupported on the ACADOS side of things right now, since I think only RBF and Gaussian are implemented there).

In the ODEs (“analytical dot”), the GPs automatically replace the corresponding variables. For example, if there is a GP for v3_dot, this GP’s prediction will be used in place of the ODE for v3_dot.

ODEModel: Base logic for ODE models.

ODEModels/DJIMatrice100.py: A model for the Matrice100 drone.

data.py: A generic base class for dataclasses like State, Input, etc, to inherit from.

gp_data.py: New data generation code and some data analysis (dcor heatmap).

gp_models.py: Contains the different GP models (exact single task GP, multi task GP, sparse single task GP) and some evaluation code.

gp.py: GPSettings and GPManager that automatically builds GPs from settings. From here, learning curves can be plotted as well, but note that this is currently broken. Please refer to the branch gp-evaluation for a working learning curve example (just run main.py there).
 

## Installation instruction

TODO


## Usage

Run evaluate.py for a usage example.


## License
The project is licensed under the MIT license (https://opensource.org/license/mit).

