import numpy as np

def to_one_hot(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size),y]=1
    return one_hot

def evaluate(model, X, y_true):
    y_pred_logits = model.forward(X, training=False)  # shape: (batch_size, num_classes)
    acc = accuracy(y_pred_logits, y_true)
    return acc

def accuracy(y_pred_logits, y_true):
    y_pred = np.argmax(y_pred_logits, axis=1)
    acc = np.mean(y_pred == y_true)
    return acc

def test_model(model, dataset, batch_size=32):
    num_samples = len(dataset)
    num_correct = 0

    for i in range(0, num_samples, batch_size):
        batch_images = []
        batch_labels = []

        for j in range(i, min(i + batch_size, num_samples)):
            img, label = dataset[j]  # This works: dataset[j] returns a (img, label) tuple
            batch_images.append(img)
            batch_labels.append(label)

        X_batch = np.stack(batch_images)  # shape: (batch_size, 1, 28, 28)
        y_batch = np.array(batch_labels)  # shape: (batch_size,)

        logits = model.forward(X_batch)
        preds = np.argmax(logits, axis=1)
        num_correct += np.sum(preds == y_batch)

    accuracy = num_correct / num_samples
    return accuracy

def cross_entropy_loss(predictions, targets, epsilon=1e-12):
    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    #encoding targets as one hot encoded values
    loss = -np.sum(targets * np.log(predictions), axis=1)  # shape (batch_size,)
    return np.mean(loss)

def train(model, dataloader, epochs, lr):
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x= x_batch.numpy()
            y= y_batch.numpy()
        
            out = model.forward(x)
            y_one_hot = to_one_hot(y, out.shape[1])
            loss = cross_entropy_loss(out, y_one_hot)
            #prit loss and accuracy each batch
            print(loss, f"accuracy: {accuracy(out, y)}")
            model.backward(y)
            model.update(lr)


        print(f"epoch {epoch}", f"Loss: {loss:.4f}")
