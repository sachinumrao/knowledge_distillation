import torch
import torchvision
import torchvision.transforms as transforms
from cnn_model import CNNModel

# transform dataset images into tensors
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# load train dataset
trainset = torchvision.datasets.CIFAR10(root='~/.data',
            train=True,
            download=True,
            transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                batch_size=32,
                shuffle=True,
                num_workers=4)


# load test dataset
testset = torchvision.datasets.CIFAR10(root='~/.data',
            train=False,
            download=True,
            transform=transform)

testloader = torch.utils.data.DataLoader(testset,
                batch_size=32,
                shuffle=True,
                num_workers=4)

# instantiate the model
model = CNNModel()

num_epochs=100

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        model.optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.loss_fn(outputs, labels)
        loss.backward()
        model.optimizer.step()
        if  (i+1)%100 == 0:
            running_loss = loss.item()
            print("Epoch : ", epoch+1, " , Step : ", i+1, " , Loss : ",running_loss)

model_path = '~/Models/CIFAR10/CNN/'
torch.save(model.state_dict(), model_path)

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        imgs, labels = data
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Test Accuracy: ", correct/total)
