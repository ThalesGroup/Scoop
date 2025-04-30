# Scoop: An Optimization Algorithm for Profiling Attacks against Higher-Order Masking
Authors: Nathan Rousselot [1,2], Karine Heydemann [1], Loïc Masure [2] and Vincent Migairou [1]

[1] Thales, France
[2] LIRMM, Univ. Montpellier, CNRS

This repository contains the implementation of the Scoop algorithm for profiling attacks against higher-order masking, as well as supplementary materials for additional figures and reproducing the results of the paper.

## Getting started

To avoid any conflicts, we recommend using a virtual environment. We recommend using `uv`, and you can get started by running the following command:

```bash
git clone https://github.com/ThalesGroup/Scoop
cd Scoop
```

Then, create a new virtual environment using `uv`:

```bash
uv venv scoop_ches25 --python 3.12.2
```
Activate the virtual environment:

```bash 
source scoop_ches25/bin/activate
```

Then, install the required packages. We provide a `requirements.txt` file that contains all the necessary packages to run the code. You can install them using `pip`:

```bash
uv pip install -r requirements.txt
```

WARNING: The `requirements.txt` file has been generated on a Linux platform. If you are using a different platform, you may need to manually install the required packages.

## Requirements

Here is a non-exhaustive list of the required packages with their default versions provided in the `requirements.txt` file:
- `torch` (2.6.0)
- `numpy` (2.2.4)
- `optuna` (4.2.1)
- `sympy` (1.13.1)

Additionally, the CUDA version on our platform is 12.0. If you have CUDA 11.8 or CUDA 12.6 (or any prior or subsequent version), you may need to install the corresponding version of `torch` and `torchvision`. You can find the appropriate step to install `torch` for your CUDA version [here](https://pytorch.org/get-started/locally/). 

## Usage

At this time, Scoop has only been implemented in PyTorch. To use Scoop, one needs to import `scoop` in his project:

```python
    from scoop import Scoop
```

`Scoop` is a class that inherits from `torch.optim.Optimizer`. Hence, one can use it as any other optimizer in PyTorch, we detail its hyperparameters later. The main difference with standard optimizer is that Scoop relies on a Hessian estimator. Hence, `Scoop` has a `hutchinson_hessian` method that update the Hessian estimation in-place.

The main contribution is located in `scoop.py`. To use Scoop, create an instance of the `Scoop` class as you would do with any other optimizer:

```python
optimizer = Scoop(model.parameters(), lr=lr)
```

The training loop is then slightly modified to include the Hessian computation and the Scoop update:

```python
...
        loss = F.nll_loss(Y_pred, Y_batch)/math.log(2)
        loss.backward(create_graph=True)
        if iter % hessian_update == hessian_update - 1:
            optimizer.hutchinson_hessian() # SCOOP SPECIFIC LINE
        optimizer.step()
        train_loss += loss.item()
...
```

In case the update is too costly, one can decide to update the Hessian estimation every $k$ iterations (meaning $k$ mini-batches). This should not hinder the performance of the algorithm too much and is actually used in some second-order optimization algorithms. The different hyperparameters of **Scoop** are:

### Hyperparameters of **Scoop**

| **Hyperparameter**     | **Description**                      | **Default**   | **Suggested Range**                   |
|-------------------------|--------------------------------------|---------------|----------------------------------------|
| `lr`                   | Learning rate                       | 1e-4          | [1e-5, 1e-2]                          |
| `betas`                | Momentum parameters                 | (0.965, 0.99) | [0.9, 0.999]                          |
| `weight_decay`         | $\ell_2$ regularization             | 0             | [0, 0.3]                              |
| `estimator`            | Hessian estimator                   | "biased_hutchinson" | ["classic", "biased_hutchinson"] |
| `hessian_iter`         | # of iterations for Hessian estimator | 5            | As much as you can afford[^1]         |

[^1]: One iteration is already much better than **Adam**.

While default values are given, we suggest adding those hyperparameters to the fine-tuning search grid. Additional hyperparameters can be added to the optimizer, for example $\epsilon$ where $\psi(x) = \|x\|_{1+\epsilon}$. It is by default set to 0.1, but in case you face a problem where sparsity in $\mathbf{F}$ is not desired, you can set $\epsilon \geq 1$. $\epsilon = 1$ would be Newton's method approximation of **Scoop**, and would behave similarly to the Hutchinson variant of Liu *et al.* work~\cite{liu2023sophia}.

## Examples

You can explore the different notebooks in this repository for more detailed examples and additional figures.

**Pre-trained model**: The pre-trained model is heavy (>900MB) and is not included in this repository. You can download it from [here](https://drive.google.com/file/d/14OCmebP356B9RkdsmhUW89tGpU2_WZDy/view?usp=drive_link), which is google drive link.

To run the examples scripts, you can use the following command:

```bash
python -m examples.<example_name>
```

For the jupyter notebooks, you can run them directly in a Jupyter notebook environment. For example:

```bash
cd examples
jupyter notebook
```

Then, open the desired notebook and run the cells.

## Citation

If you use Scoop, or this code, in your research please cite the following paper:

```bibtex
@misc{cryptoeprint:2025/498,
      author = {Nathan Rousselot and Karine Heydemann and Loïc Masure and Vincent Migairou},
      title = {Scoop: An Optimizer for Profiling Attacks against Higher-Order Masking},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/498},
      year = {2025},
      url = {https://eprint.iacr.org/2025/498}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.