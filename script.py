import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from split_train_test import split_data
import gensim.downloader as api
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import gensim.downloader as api
from sklearn.utils import resample
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


word2vec_model = api.load("glove-wiki-gigaword-50") 
input_file = "C:/Users/Luki/Desktop/AI/catology_database_modified.xlsx"

output_train_file = "train.xlsx"
output_test_file = "test.xlsx"

split_data(input_file, output_train_file, output_test_file)

train_data = pd.read_excel(output_train_file)
test_data = pd.read_excel(output_test_file)

numeric_columns = ['Breed','Number','Housing','Area','Outdoor','Obs','Shy','Calm','Fearfull','Intelligent','Vigilant',
                  'Persevering','Affection','Friendly','Solitary','Brutal','Dominant','Agressive','Impulsive',	
                   'Predictable','Distracte','Abundance','Birds','Mammal','More','HasHair','CatSize','FurColor',	
                  'HairType','EarType']


breed_mapping = {breed: idx for idx, breed in enumerate(train_data['Breed'].unique())}
train_data['Breed'] = train_data['Breed'].map(breed_mapping)
test_data['Breed'] = test_data['Breed'].map(breed_mapping)

for col in numeric_columns:
    train_data[col] = pd.to_numeric(train_data[col], errors='coerce').fillna(0)
    test_data[col] = pd.to_numeric(test_data[col], errors='coerce').fillna(0)


X_train = np.array(train_data[numeric_columns].values, dtype=float)
X_test = np.array(test_data[numeric_columns].values, dtype=float)

y_train = train_data['Breed'].values
y_test = test_data['Breed'].values


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_size = len(numeric_columns)
hidden_size = 500
output_size = len(breed_mapping)
learning_rate = 0.01
epochs = 300
dropout_rate = 0.5
lambda_reg = 0.03

np.random.seed(28)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / (input_size + hidden_size))
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / (hidden_size + output_size))
b2 = np.zeros((1, output_size))

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def apply_dropout(A, rate):
    mask = (np.random.rand(*A.shape) < (1 - rate)).astype(float)
    return A * mask


def forward_propagation(X, rate=0):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    A1_dropout = apply_dropout(A1, rate)
    Z2 = np.dot(A1_dropout, W2) + b2
    A2 = softmax(Z2)
    return A1, A2


def backward_propagation(X, A1, A2, y, lambda_reg):
    m = X.shape[0]

    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m + lambda_reg * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m + lambda_reg * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


def one_hot_encoding(y, num_classes):
    return np.eye(num_classes)[y]


def count_misclassifications(predictions, true_labels):
    return np.sum(predictions != true_labels)


def create_batches(X, y, batch_size):
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    num_batches = len(X) // batch_size
    for i in range(num_batches):
        X_batch = X_shuffled[i * batch_size:(i + 1) * batch_size]
        y_batch = y_shuffled[i * batch_size:(i + 1) * batch_size]
        yield X_batch, y_batch

    if len(X) % batch_size != 0:
        X_batch = X_shuffled[num_batches * batch_size:]
        y_batch = y_shuffled[num_batches * batch_size:]
        yield X_batch, y_batch


A1_test, A2_test = forward_propagation(X_test, dropout_rate)
predictions_before = np.argmax(A2_test, axis=1)
misclassifications_before = count_misclassifications(predictions_before, y_test)
accuracy_before = accuracy_score(y_test, predictions_before)

print(f"Accuracy înainte de antrenament: {accuracy_before * 100:.2f}%")
print(f"Număr greșeli înainte de antrenament: {misclassifications_before}")

losses = []
accuracies = []

batch_size = 32
for epoch in range(epochs):
    epoch_loss = 0
    for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
        A1, A2 = forward_propagation(X_batch, dropout_rate)

        y_one_hot = one_hot_encoding(y_batch, output_size)

        dW1, db1, dW2, db2 = backward_propagation(X_batch, A1, A2, y_one_hot, lambda_reg)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        batch_loss = -np.mean(np.log(A2[np.arange(len(y_batch)), y_batch])) + lambda_reg * (
                np.sum(W1 ** 2) + np.sum(W2 ** 2))
        epoch_loss += batch_loss

    losses.append(epoch_loss / len(X_train))

    A1_test, A2_test = forward_propagation(X_test, dropout_rate)
    predictions_after = np.argmax(A2_test, axis=1)
    accuracy_after = accuracy_score(y_test, predictions_after)
    accuracies.append(accuracy_after)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(X_train)}")

A1_test, A2_test = forward_propagation(X_test, dropout_rate)
predictions_after = np.argmax(A2_test, axis=1)

accuracy_after = accuracy_score(y_test, predictions_after)
misclassifications_after = count_misclassifications(predictions_after, y_test)

print(f"\nAccuracy după antrenament: {accuracy_after * 100:.2f}%")
print(f"Număr greșeli după antrenament: {misclassifications_after}")

#plt.figure(figsize=(12, 5))

# # Loss Plot
# plt.subplot(1, 2, 1)
# plt.plot(range(epochs), losses, label='Loss', color='blue')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss over Epochs')
# plt.grid(True)

# # Accuracy Plot
# plt.subplot(1, 2, 2)
# plt.plot(range(epochs), accuracies, label='Accuracy', color='green')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Accuracy over Epochs')
# plt.grid(True)

# plt.tight_layout()
# plt.show()
def normalize_word2vec_vectors(vectors):
    scaler = StandardScaler()
    return scaler.fit_transform(vectors)

def adjust_vector_dimension(vector, target_dim):
    if len(vector) > target_dim:
        return vector[:target_dim]
    elif len(vector) < target_dim:
        return np.pad(vector, (0, target_dim - len(vector)), mode='constant')
    return vector


def get_user_input(numeric_columns, word2vec_model, input_size):
    user_input = input("Descriere pisica:  ").strip().lower()
    
    tokens = [word for word in user_input.split() if word in word2vec_model.key_to_index]

    if not tokens:
        print("Nu s-au găsit cuvinte relevante în model. Folosesc un vector implicit.")
        input_vector = np.zeros((word2vec_model.vector_size,))
    else:
        #print(f"Cuvinte relevante găsite: {tokens}")
        input_vector = np.mean([word2vec_model[word] for word in tokens], axis=0)

    adjusted_vector = adjust_vector_dimension(input_vector, input_size)
    return adjusted_vector.reshape(1, -1)


inverse_breed_mapping = {v: k for k, v in breed_mapping.items()}

def predict_breed(user_input_normalized):
    _, A2 = forward_propagation(user_input_normalized)
    predicted_index = np.argmax(A2, axis=1)[0]
    predicted_breed = inverse_breed_mapping[predicted_index]
    return predicted_breed, A2


def main():
    print("\n*****************************\n")
    while True:
        user_input_vector = get_user_input([], word2vec_model, input_size)

        prediction, probabilities = predict_breed(user_input_vector)
        print(f"Distribuția probabilitatilor: {probabilities}")
        print(f"Rasa prezisa este: {prediction}")

        retry = input("Doriti să faceti o altă predictie? (da/nu): ").strip().lower()
        if retry != 'da':
            print("\nLa revedere!")
            break

if __name__ == "__main__":
    main()

#This is a small cat, very shy and calm. It has a long fur, with a white color and a fluffy tail.
#she is silent
#she is noisy