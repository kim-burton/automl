# automl
automlを作成しています。
まだ作成中

## 実行手順
### 2値分類
```python
from automl import BinaryClassifier

model = BinaryClassifier()
model.fit(X, y, categorical_features = categorical_features, scoring = "auc_roc")
```
