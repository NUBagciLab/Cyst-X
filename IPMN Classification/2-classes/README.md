FedProx0.3 sets mu = 0.3

FedProx sets mu = 0.1

To train model with cross validation:
On Linux: 

    chmod +x ./train.sh
    CUDA_VISIBLE_DEVICES=0 ./train.sh

On Windows:

    ./train.ps1

To test model with cross validation:

    python ./fold_test.py
    python ./fold_test.py --t 2
It will also output the results for LaTeX.
Please revise the --data-path to your data path.

Trained models and logs are available at

    https://drive.google.com/drive/folders/1pSgF97OU8CchUoHdFXZumcIb5Ay9Rkvu?usp=sharing
