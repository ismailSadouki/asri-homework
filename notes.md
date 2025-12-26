**with LightGBM Try:**

![alt text](image.png)


- class_weight handling in lightgbm
- StratifiedKFold
- Encoding with target statistics
- Miss-ing continuous values are imputed with their corresponding median feature value,
while missing categorical features are imputed as a new category called 'missing'
prior to encoding.
- clean the columens check the unique types of each column and check for anomalies or missp
- metrics: accuracy, balanced accuracy, and F1 score.
<mark>balanced accuracy explicitly
accounts for class imbalance, it provides a better estimate of model performance than
accuracy. This illustrates the importance of using the right metric to evaluate our
models.</mark>