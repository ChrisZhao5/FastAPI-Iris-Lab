import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier  # 核心改动：导入随机森林

# 1. 加载数据 (保持不变，因为我们还需要预测 Iris)
iris = load_iris()
X, y = iris.data, iris.target

# 2. 定义模型 (核心改动)
# 原来是: clf = DecisionTreeClassifier()
# 现在升级为: RandomForestClassifier
# n_estimators=100 表示我们需要 100 棵树来进行集体投票，这比单棵树稳健得多
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 3. 训练模型
print("Training Random Forest model...")
model.fit(X, y)

# 4. 保存模型
# 这一步会直接覆盖掉你原来那个旧的 iris_model.pkl
joblib.dump(model, "../model/iris_model.pkl")

print("Success! A new Random Forest model has been saved to model/iris_model.pkl")
