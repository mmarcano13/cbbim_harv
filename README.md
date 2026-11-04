# Computer simulations of a coffee berry borer infestation model
The coffee berry borer (CBB) is a small beetle that flies to the coffee fruit, bores a hole to the seed, and penetrates it to shelter and reproduce in its interior. In this work, we study the level of CBB infestation as a function of the percentage of the coffee harvest. For each coffee season, we model CBB infestation using a population dynamics approach to examine the interaction between coffee berries and CBB. We use a one-dimensional map to track infestation levels over multiple coffee seasons. For details see Mariano Marcano, Amitabha Bose, and Paul Bayman. A one-dimensional map to study multi-seasonal coffee infestation by the coffee berry borer. *Mathematical Biosciences*, 333, March 2021. [DOI: 10.1016/j.mbs.2020.108530](https://doi.org/10.1016/j.mbs.2020.108530).

## Installation  
Clone the repository:  

```bash  
git clone https://github.com/mmarcano13/cbbim_harv  
cd cbbim_harv  
``` 
### Explanation of directories and files 
- data: data files
    - data_infestation.csv: field infestation data for model calibration.
    - params_descriptions.json: parameter descriptions.
    - params_constrepro_fixed.csv: model parameters held fixed during model calibration for constant CBB reproduction rate.
    - params_varrepro_fixed.csv: model parameters held fixed during model calibration for variable CBB reproduction rate.
    - params_constrepro.csv: all model parameters including the optimum for constant CBB reproduction rate.
    - params_varrepro.csv: all model parameters including the optimum for variable CBB reproduction rate.
- pyscripts: python files
    - calibration_cls.py: class for parameter estimation, reads model parameters held fixed and variable parameter bounds, computes a solution to a least-squares problem and writes out the file params_constrepro.csv or params_varrepro.csv depending on the case.
    - models_cls.py: five classes: initiate, reads the parameters; model: solve the model equations for a single coffee season; multiseason: solves the model equations for multiple seasons; map_1d: a one-dimensional map that takes the free CBB population at the beginning of a season and predicts what the population will be at the start of the next season for a prescribed percentage of harvesting; and bpoints: computes a bifurcation diagram of the one-dimesional map fixed points as a function of percentage of coffee harvesting percentages.
    - fitting.py: fits the CBB infestation model to the field data.
    - simulations.py: for a prescribed percentage of coffee harvest, it computes the one-dimensional map to find an interval of free CBB initial values for which the CBB infestation can be controlled by maintaining this level of coffee harvesting, and shows model simulations for a single and multiple seasons.
    - bifdiagram.py: generates the bifurcation diagram.
- notebooks: Jupyter notebooks
    - fitting.ipynb: same as fitting.py.
    - simulations.ipynb: same as simulations.py and bifdiagram.py together.   
### Required packages
- numpy
- scipy
- pandas
- matplotlib
- IPython
## Usage  
To reproduce results similar to those in Marcano et al 2021, run pyscripts/simulations.py and pyscripts/bifdiagram.py or the Jupyter notebook notebooks/simulations.ipynb.

If a different infestation dataset is available, replace the file data/data_infestation.csv with the new file (following the same format), if deeemed necessary, adjust the parameter bounds in the dictionary varrate_bnds (line 65) in the function bounds of pyscripts/calibration_cls.py, then run pyscripts/fitting.py or the Jupyter notebook notebooks/fitting.ipynb, several times and choose the "best" solution.
## Contributing  

Contributions are currently not accepted for this project.

## License  

This project is licensed under the BSD 3-Clause License.  
See the `LICENSE.txt` file for details.  

## Contact  

Created by Mariano Marcano (mariano.marcano@gmail.com)  
Feel free to reach out with questions or suggestions. 