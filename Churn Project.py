# Import libraries for data handling, model building, and visualization
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the churn dataset from the CSV file
churn = r'C:\Users\27Jay\PycharmProjects\PythonProject\Kaggle ML\Churn.csv'
df = pd.read_csv(churn)

# Convert TotalCharges to numeric, coercing errors into NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop any rows where TotalCharges could not be converted
df = df.dropna(subset=['TotalCharges'])


# Display the shape of the dataset (rows, columns)
print("Dataset shape:", df.shape)

# Show the first few rows to understand what the data looks like
print(df.head())

# Check for any missing values
print(df.Churn.isnull().sum())

# Encode 'Yes'/'No' as 1/0 in the target column
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0}).astype(int)

# Select your features
selected_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Create X (features) and y (target)
X = df[selected_features]
y = df['Churn']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize a logistic regression model
model = LogisticRegression(solver='liblinear')

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_preds = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_preds)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_preds))

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_preds))


# Create the confusion matrix again
cm = confusion_matrix(y_test, y_preds)

# Set up the plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Customer Churn Prediction - Confusion Matrix')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Churn Distribution')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Number of Customers')
plt.xticks([0, 1], ['No Churn', 'Churn'])
plt.tight_layout()
plt.show()
