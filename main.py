

# -------------------------Libraries-----------------------------------
import time
import torch
import numpy as np
from PIL import Image
from glob import glob
from vit_pytorch import ViT
from torch.optim import Adam
from termcolor import cprint
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, random_split
# ------------------------Libraries--------------------------------------



f = open("output.txt", "a")

# ------------------------- Training Constants-----------------------------

BATCH_SIZE = 32
EPOCHS = 20

print(f"Batch Size : {BATCH_SIZE}", file=f)
print(f"Number of epochs : {EPOCHS}", file=f)
print(f"Batch Size : {BATCH_SIZE}")
print(f"Number of epochs : {EPOCHS}")

# ------------------------- Training Constants-----------------------------





# ---------------------------Dataset & DataLoader ---------------------------
label_dict = {f"C{index}": index for index in range(261)}


class UbirisDataset(Dataset):

    def __init__(self, data_dir, h=256, w=256):
        self.dir = glob(f"{data_dir}/*tiff")
        self.h, self.w = h, w

    def __len__(self):
        return len(self.dir)

    def __getitem__(self, index):
        path = self.dir[index]
        img = self.load_img(path)
        img = ToTensor()(img)
        label = label_dict[path.split("\\")[-1].split("_")[0]]

        return img, label

    def load_img(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.h, self.w))
        return img


def accuracy_func(pred, true):
    pred = torch.argmax(pred, dim=1)
    accuracy = sum(true == pred)
    return accuracy


ds = UbirisDataset("real")
tr, val, ts = 0.75, 0.10, 0.15
train_size = int(len(ds)*tr)
val_size = int(len(ds)*val)
test_size = len(ds) - train_size - val_size


train_ds, val_ds, test_ds = random_split(ds, [train_size, val_size, test_size])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
print(f"Size of training dataset : {len(train_ds)}", file=f)
print(f"Size of validation dataset : {len(val_ds)}", file=f)
print(f"Size of testing dataset : {len(test_ds)}", file=f)
print(f"Size of training dataset : {len(train_ds)}")
print(f"Size of validation dataset : {len(val_ds)}")
print(f"Size of testing dataset : {len(test_ds)}")


# ---------------------------Dataset & DataLoader ---------------------------





# ------------------------------- Model ------------------------------------
model = ViT(
    image_size=256,
    patch_size=32,
    num_classes=len(label_dict.keys()),
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)


# ------------------------------- Model ------------------------------------






# -------------------------------Training Hyper parameters-------------------
optim = Adam(model.parameters(), lr=1e-4)
criterion = CrossEntropyLoss()
# -------------------------------Training Hyper parameters-------------------






# ------------------------------- Training ---------------------------------

train_init = time.time()
print("Started training...", file=f)
cprint("Started training...", "blue")
best_loss = np.inf
best_acc = 0.0
print("Model Loaded on GPU...", file=f)
print("Model Loaded on GPU...")
model = model.cuda()
update = 0
TL, VL, TSL, TA, VA, TSA = [], [], [], [], [], []
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1} :", file=f)
    print(f"Epoch {epoch + 1} :")
    epoch_init = time.time()
    train_loss = val_loss = test_loss = 0.0
    train_acc = val_acc = test_acc = 0
    model.train()


# ---------------------- train patch --------------------------
    for train_index, (patch, labels) in enumerate(train_dl):
        optim.zero_grad()
        dev_patch = patch.cuda()
        dev_labels = labels.cuda()
        output = model(dev_patch)
        acc = accuracy_func(output, dev_labels)
        train_acc += acc
        loss = criterion(output, dev_labels)
        train_loss += loss.item()
        for _ in range(2):
            TL.append(loss.item())
            TA.append(acc / dev_patch.shape[0])
        if train_index % 10 == 9:
            print(f"      [Step {train_index + 1}] Loss : {'%.6f' % loss.item()}", file=f)
            print(f"      [Step {train_index + 1}] Loss : {'%.6f' % loss.item()}")
        loss.backward()
        optim.step()

# ---------------------- train patch --------------------------

    model.eval()
    with torch.no_grad():


# --------------------- validation patch ------------------------
        for patch, labels in val_dl:
            dev_patch = patch.cuda()
            dev_labels = labels.cuda()
            output = model(dev_patch)
            acc = accuracy_func(output, dev_labels)
            val_acc += acc
            loss = criterion(output, dev_labels)
            val_loss += loss.item()
            for _ in range(15):
                VL.append(loss.item())
                VA.append(acc / dev_patch.shape[0])

# --------------------- validation patch ------------------------


# ----------------------test patch -----------------------------

        for patch, labels in test_dl:
            dev_patch = patch.cuda()
            dev_labels = labels.cuda()
            output = model(dev_patch)
            acc = accuracy_func(output, dev_labels)
            test_acc += acc
            loss = criterion(output, dev_labels)
            test_loss += loss.item()
            for _ in range(10):
                TSL.append(loss.item())
                TSA.append(acc / dev_patch.shape[0])

# ----------------------test patch -----------------------------


# --------------------- Updating metric values ------------------
    TRAIN_ACC = train_acc / len(train_ds)
    VAL_ACC = val_acc / len(val_ds)
    TEST_ACC = test_acc / len(test_ds)
    print(f"   Train Loss : {'%.6f' % train_loss} | Train accuracy : {'%.6f' % TRAIN_ACC}", file=f)
    print(f"   Validation Loss : {'%.6f' % val_loss} | Validation Accuracy : {'%.6f' % VAL_ACC}", file=f)
    print(f"   Test Loss : {'%.6f' % test_loss} | Test Accuracy : {'%.6f' % TEST_ACC}", file=f)
    print(f"   Train Loss : {'%.6f' % train_loss} | Train accuracy : {'%.6f' % TRAIN_ACC}")
    print(f"   Validation Loss : {'%.6f' % val_loss} | Validation Accuracy : {'%.6f' % VAL_ACC}")
    print(f"   Test Loss : {'%.6f' % test_loss} | Test Accuracy : {'%.6f' % TEST_ACC}")
    update_flag = False
    if val_loss < best_loss:
        update = 0
        update_flag = True
        best_loss = val_loss
        print("Loss Update : Positive", file=f)
        cprint("Loss Update : Positive", "green")
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch + 1
        }, "best_loss_model.pt")
    if VAL_ACC > best_acc:
        update = 0
        update_flag = True
        best_acc = VAL_ACC
        print("Accuracy Update : Positive", file=f)
        cprint("Accuracy Update : Positive", "green")
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch + 1
        }, "best_accuracy_model.pt")
    if not update_flag:
        print("Model Update : Negative\n", file=f)
        cprint("Model Update : Negative\n", "red")
        update += 1
    print(f"   Execution Time : {'%.3f' % (time.time() - epoch_init)} seconds\n", file=f)
    print(f"   Execution Time : {'%.3f' % (time.time() - epoch_init)} seconds\n")
    if update >= 5:
        print("Model Stopped due to continuous model learning degradation\n", file=f)
        print("Model Stopped due to continuous model learning degradation\n")
        break
# --------------------- Updating metric values ------------------
print("Training finished...", file=f)
print(f"Execution Time : {'%.3f' % (time.time() - train_init)} seconds", file=f)
cprint("Training finished...", "blue")
cprint(f"Execution Time : {'%.3f' % (time.time() - train_init)} seconds", "blue")
f.close()

# ------------------------------- Training ---------------------------------





# ------------------------------- Accuracy Data Processing ---------------------

for index in range(len(TA)):
    TA[index] = float(TA[index].cpu().detach())
for index in range(len(VA)):
    VA[index] = float(VA[index].cpu().detach())
for index in range(len(TSA)):
    TSA[index] = float(TSA[index].cpu().detach())


# ------------------------------- Accuracy Data Processing ---------------------





# ------------------------------Metric Plots-----------------------------------

plt.figure(figsize=(20, 6))
plt.plot(TL, label="training loss")
plt.plot(VL, label="validation loss")
plt.plot(TSL, label="test loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Loss Curves", size=20)
plt.legend()
plt.savefig("Loss Curve.png")

plt.figure(figsize=(20, 6))
plt.plot(TA, label="training accuracy score")
plt.plot(VA, label="validation accuracy score")
plt.plot(TSA, label="test accuracy score")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves", size=20)
plt.legend()
plt.savefig("Accuracy Curve.png")

# ------------------------------Metric Plots-----------------------------------
