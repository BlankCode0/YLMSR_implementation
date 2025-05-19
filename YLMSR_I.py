!git clone https://github.com/BlankCode0/YLMSR_implementation.git

%cd YLMSR_implementation

# remove existing model if you want to retrain them from start
!rm -rf models/policy_model
!rm -rf models/ref_model

# train policy model
! python train_dpo.py

# check if model is saved properly
!ls models/policy_model/

# run test to check the responce between ref model and policy model
!python test_generate.py
