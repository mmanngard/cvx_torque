# torque_estimation_masters

minimalistic version of the original repo 

# set up 

just run:

    conda env create -f anaconda_environment.yml

you'll then have an environment name "opc01" which has all of the nessesarry packages to run the 

file description:

    notebooks/ACC2025 material => initial provided notebook
    notebooks/ACC2025 material_cleanup => initial provided notebook with funcitonality moved to functions.py
    notebooks/ACC2025 material_cleanup => initial provided notebook with funcitonality moved to functions.py
    
NOTES:

in cvxpygen if you see the following it means that your machine is out of RAM, solution is to make the input matrix smaller:

    The Kernel crashed while executing code in the current cell or a previous cell. 
    Please review the code in the cell(s) to identify a possible cause of the failure. 
    Click here for more info. 
    View Jupyter log for further details.

in every cleaned up notebook theres a section called "# Batch size and overlap" you'll be able to modify batch sizes from there.