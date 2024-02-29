# Define file paths
file_path = '' #path to csv with all continuous features and string target (x0,y0,z0,p0,v0,...x20,y20,z20,p20,v20,target)

load_df = pd.read_csv(file_path).iloc[:, 1:]
labelencoder = LabelEncoder()
load_df['target'] = labelencoder.fit_transform(load_df['target'])
X = load_df.iloc[:, :-1]
y = load_df.loc[:,['target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y_train)
y_train = ohe.transform(y_train)
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y_test)
y_test = ohe.transform(y_test)

# Build the ANN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(50, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=4)

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


import dice_ml
from dice_ml.utils import helpers

cont_features = []

for i in range(21):
  cont_features.append('x'+str(i))
  cont_features.append('y'+str(i))
  cont_features.append('z'+str(i))
  cont_features.append('v'+str(i))
  cont_features.append('p'+str(i))

d = dice_ml.Data(dataframe=load_df, continuous_features=cont_features, outcome_name="target")

m = dice_ml.Model(model=model, backend="TF2", model_type='classifier')

exp = dice_ml.Dice(d, m, method="gradient")

print(X_test[1:2].shape)

dice_exp = exp.generate_counterfactuals(X_test[1:2], total_CFs=1, desired_class=10)

dice_exp.visualize_as_dataframe(show_only_changes=True)