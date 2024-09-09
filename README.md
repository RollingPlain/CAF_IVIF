# CAF_IVIF
Official Code for: Jinyuan Liu, Guanyao Wu, Zhu Liu, Long Ma, Risheng Liu, Xin Fan*, **"Where Elegance Meets Precision: Towards a Compact, Automatic, and Flexible Framework for Multi-modality Image Fusion and Applications"**,  in Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI), 2024.

## Set Up on Your Own Machine

### Virtual Environment

We strongly recommend that you use Conda as a package manager.

```shell
# create virtual environment
conda create -n CAF python=3.8
conda activate CAF
# select and install pytorch version yourself (Necessary & Important)
# install requirements package
pip install -r requirements.txt
```

### Test
This code natively supports the same naming for ir/vi image pairs. An naming example can be found in **./data** folder.
```shell
# Test: use given example and save fused color images to ./result.
# If you want to test the custom data, please modify the data path in 'eval.py'.
# The default is to use Det_final to pre-train the model, If you want to change the model, please modify the model path in 'eval.py'.
python eval.py
```


## Citation

If this work has been helpful to you, we would appreciate it if you could cite our paper! 

```
@inproceedings{CAF,
  title={Where Elegance Meets Precision: Towards a Compact, Automatic, and Flexible Framework for Multi-modality Image Fusion and Applications},
  author={Liu, Jinyuan and Wu, Guanyao and Liu, Zhu and Ma, Long and Liu, Risheng and Fan, Xin},
  booktitle={Proceedings of the 33rd International Joint Conference on Artificial Intelligence},
  year={2024}
}
```




