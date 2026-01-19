# Titanic 数据处理方法

1. 去掉 "PassengerId", "Name", "Ticket", "Cabin" 四列
2. 去掉 "Age" 为空的行
3. 对 "Sex", "Embarked" 进行独热编码
4. 对 "Pclass", "Age", "SibSp", "Parch", "Fare" 进行标准化处理