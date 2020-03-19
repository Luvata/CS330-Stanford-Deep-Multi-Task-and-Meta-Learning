import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import DataGeneratorPreFetch
from tensorflow.python.platform import flags


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch')


class MANN(nn.Module):
    def __init__(self, num_classes, sample_per_class, embed_size=784):
        super().__init__()
        self.num_classes = num_classes
        self.sample_per_classes = sample_per_class
        self.embed_size = embed_size
        self.lstm1 = nn.LSTM(embed_size + num_classes, 128, bidirectional=True)
        self.lstm2 = nn.LSTM(128 * 2, num_classes)

    def forward(self, input_images, input_labels):
        # B, K+1, N, 784: input_images
        # B, K+1, N, N: input_labels
        B, k_plus_one, N, N = input_labels.shape
        input_images = input_images.reshape(B, -1, self.embed_size)
        input_labels = input_labels.reshape(B, -1, N)
        input_labels[:, -self.num_classes:] = 0.
        x = torch.tensor(np.dstack((input_images, input_labels))).transpose(0, 1)  # (K+1)*N, B, N
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x  # ((K + 1)*N, B, N)


def main():
    data_generator = DataGeneratorPreFetch(
        FLAGS.num_classes, FLAGS.num_samples + 1)

    model = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_step = 2000
    test_accs = []

    for step in range(n_step):
        images, labels = data_generator.sample_batch('train', FLAGS.meta_batch_size)
        last_n_step_labels = labels[:, -1:]
        last_n_step_labels = last_n_step_labels.squeeze(1).reshape(-1, FLAGS.num_classes)  # (B * N, N)
        target = torch.tensor(last_n_step_labels.argmax(axis=1))
        logits = model(images, labels)
        last_n_step_logits = logits[-FLAGS.num_classes:].\
            transpose(0, 1).contiguous().view(-1, FLAGS.num_classes)
        optimizer.zero_grad()
        loss = criterion(last_n_step_logits, target)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            with torch.no_grad():
                print("*" * 5 + "Iter " + str(step) + "*" * 5)
                images, labels = data_generator.sample_batch('test', 100)
                last_n_step_labels = labels[:, -1:]
                last_n_step_labels = last_n_step_labels.squeeze(1).reshape(-1, FLAGS.num_classes)  # (B * N, N)
                target = torch.tensor(last_n_step_labels.argmax(axis=1))
                logits = model(images, labels)
                last_n_step_logits = logits[-FLAGS.num_classes:].\
                    transpose(0, 1).contiguous().view(-1, FLAGS.num_classes)
                pred = last_n_step_logits.argmax(axis=1)
                test_loss = criterion(last_n_step_logits, target)

                print("Train Loss:", loss.item(), "Test Loss:", test_loss.item())
                test_accuracy = (1.0 * (pred == target)).mean().item()
                print("Test Accuracy", test_accuracy)
                test_accs.append(test_accuracy)

    import matplotlib.pyplot as plt
    plt.plot(range(len(test_accs)), test_accs)
    plt.xlabel("Step (x 100)")
    plt.ylabel("Test accuracy")
    plt.show()


if __name__ == '__main__':
    main()
