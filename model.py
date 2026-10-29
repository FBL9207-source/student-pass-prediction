import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

student_data = pd.read_csv("student-data.csv")

# These fields are not collected by the prediction form.
removed_columns = [
    'address', 'famsize', 'reason', 'studytime', 'traveltime',
    'freetime', 'famrel', 'Walc', 'romantic'
]

x = student_data.drop(columns=['passed'] + removed_columns)
y = student_data['passed']

#Train and split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2, random_state=42)

cate_cols = x.select_dtypes(include=['object', 'str']).columns.tolist()
num_cols = x.select_dtypes(include=['number']).columns.tolist()

# Create the preprocessing pipeline using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cate_cols),
        ('num', StandardScaler(), num_cols)
    ]
)



# Fit preprocessing and classification together so inference accepts raw features.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
model.fit(x_train, y_train)
y_pred_model = model.predict(x_test)

print('Accuracy is:', accuracy_score(y_test, y_pred_model))

#Save the model
pickle.dump(model,open('model.pkl','wb'))

print('Model training completed')
