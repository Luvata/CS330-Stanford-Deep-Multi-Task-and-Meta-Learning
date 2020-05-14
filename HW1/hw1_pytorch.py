import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import DataGeneratorPreFetch
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch')
flags.DEFINE_integer('n_step', 20000, 'Number of training step')


class MANN(nn.Module):
    def __init__(self, num_classes, sample_per_class, embed_size=784):
        super().__init__()
        self.num_classes = num_classes
        self.sample_per_classes = sample_per_class
        self.embed_size = embed_size
        self.lstm1 = nn.LSTM(embed_size + num_classes, 128)
        self.lstm2 = nn.LSTM(128, num_classes)

    def forward(self, inputs):
        x, _ = self.lstm1(inputs)
        x, _ = self.lstm2(x)
        return x  # K*N,B,N


def prep_data(input_images, input_labels, device):
    # Handle reshape and data device
    B, K, N, D = input_images.shape
    input_images = input_images.reshape(B, -1, D)
    test_labels = input_labels[:, -1:]  # B,1,N,N
    train_labels = np.concatenate((input_labels[:, :-1], np.zeros_like(test_labels)), axis=1).reshape((B, -1, N))
    inputs = torch.tensor(np.dstack((input_images, train_labels))).transpose(0, 1).to(device)
    targets = torch.tensor(test_labels.squeeze(1).reshape(-1, N).argmax(axis=1)).to(device)
    return inputs, targets


def train(model, data_generator, n_step, optimizer, loss_fn, device, save=True):
    exp_name = f"_Exp_N={FLAGS.num_classes}_K={FLAGS.num_samples}_B={FLAGS.meta_batch_size}"
    writer = SummaryWriter(comment=exp_name)
    test_accs = []

    for step in range(n_step):
        model.train()
        images, labels = data_generator.sample_batch('train', FLAGS.meta_batch_size)
        inputs, targets = prep_data(images, labels, device)
        logits = model(inputs)
        last_n_step_logits = logits[-FLAGS.num_classes:].transpose(0, 1).contiguous().view(-1, FLAGS.num_classes)
        optimizer.zero_grad()
        loss = loss_fn(last_n_step_logits, targets)
        loss.backward()

        writer.add_scalar("Loss/Train", loss, step)

        optimizer.step()
        if step % 100 == 0:
            test_loss, test_accuracy = evaluate(model, data_generator, step, loss_fn, device)
            print("Train Loss:", loss.item(), "Test Loss:", test_loss.item())
            print("Test Accuracy", test_accuracy)
            test_accs.append(test_accuracy)
            writer.add_scalar("Loss/Test", test_loss, step)
            writer.add_scalar("Accuracy/Test", test_accuracy, step)

    plt.plot(range(len(test_accs)), test_accs)
    plt.xlabel("Step (x 100)")
    plt.ylabel("Test accuracy")
    plt.savefig(f"models/train_{exp_name}.png")
    if save:
        torch.save(model.state_dict(), f"models/model_{exp_name}")
    return model


def evaluate(model, data_generator, step, loss_fn, device):
    model.eval()
    with torch.no_grad():
        print("*" * 5 + "Iter " + str(step) + "*" * 5)
        images, labels = data_generator.sample_batch('test', 100)
        inputs, targets = prep_data(images, labels, device)
        logits = model(inputs)
        last_n_step_logits = logits[-FLAGS.num_classes:]. \
            transpose(0, 1).contiguous().view(-1, FLAGS.num_classes)

        pred = last_n_step_logits.argmax(axis=1)
        test_loss = loss_fn(last_n_step_logits, targets)
        test_accuracy = (1.0 * (pred == targets)).mean().cpu().item()
        return test_loss, test_accuracy


def main():
    data_generator = DataGeneratorPreFetch(
        FLAGS.num_classes, FLAGS.num_samples + 1)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using cuda")

    model = MANN(FLAGS.num_classes, FLAGS.num_samples + 1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, data_generator, FLAGS.n_step, optimizer, criterion, device)


if __name__ == '__main__':
    main()

