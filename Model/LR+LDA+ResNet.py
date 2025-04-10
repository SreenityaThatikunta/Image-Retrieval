import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ----------------------------- Data Handling ----------------------------- #

class DataHandler:
    def __init__(self, batch_size=128):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.batch_size = batch_size
        self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)

    def get_loaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def get_datasets(self):
        return self.train_dataset, self.test_dataset

# -------------------------- Feature Extraction -------------------------- #

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else self.device)
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.model.to(self.device)
        self.model.eval()

    def extract(self, dataloader):
        all_feats = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels, *_ in tqdm(dataloader):
                imgs = imgs.to(self.device)
                feats = self.model(imgs)
                feats = feats / feats.norm(dim=1, keepdim=True)
                all_feats.append(feats.cpu().numpy())
                all_labels.extend(labels.numpy())
        return np.vstack(all_feats), np.array(all_labels)

# ---------------------- Dimensionality Reduction ------------------------ #

class LDAReducer:
    def __init__(self, n_components=9):
        self.lda = LDA(n_components=n_components)

    def fit_transform(self, X, y):
        return self.lda.fit_transform(X, y)

    def transform(self, X):
        return self.lda.transform(X)

# --------------------------- Classifier ---------------------------------- #

class Classifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y, class_names):
        preds = self.model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"Test Accuracy: {acc:.4f}")
        print(classification_report(y, preds, target_names=class_names))

# -------------------------- Prediction + Visualization -------------------------- #

class PredictorVisualizer:
    def __init__(self, extractor, lda, clf, transform, test_dataset):
        self.extractor = extractor
        self.lda = lda
        self.clf = clf
        self.transform = transform
        self.test_dataset = test_dataset
        self.device = extractor.device

    def predict_and_show(self, img_path):
        # Step 1: Load and preprocess query image
        img = Image.open(img_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Step 2: Extract features
        with torch.no_grad():
            feat = self.extractor.model(input_tensor)
            feat = feat / feat.norm()

        # Step 3: LDA Transform
        lda_feat = self.lda.transform(feat.cpu().numpy())

        # Step 4: Predict class
        pred = self.clf.model.predict(lda_feat)[0]
        pred_class = self.test_dataset.classes[pred]

        print(f"\nPredicted class for query image: {pred_class}")

        # Step 5: Find 5 matching predictions from test set
        matching_images = []
        for i in tqdm(range(len(self.test_dataset))):
            test_img, _ = self.test_dataset[i]
            input_tensor = test_img.unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.extractor.model(input_tensor)
                feat = feat / feat.norm()
            lda_feat = self.lda.transform(feat.cpu().numpy())
            test_pred = self.clf.model.predict(lda_feat)[0]

            if test_pred == pred:
                matching_images.append(test_img)
                if len(matching_images) == 5:
                    break

        # Step 6: Plot query + matching
        fig, axes = plt.subplots(1, 6, figsize=(15, 4))
        axes[0].imshow(img)
        axes[0].set_title("Query Image")
        axes[0].axis('off')

        for i, example_img in enumerate(matching_images):
            example_img = example_img.permute(1, 2, 0).numpy()
            example_img = example_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            example_img = np.clip(example_img, 0, 1)

            axes[i+1].imshow(example_img)
            axes[i+1].axis('off')
            axes[i+1].set_title(f"{pred_class}")

        plt.tight_layout()
        plt.show()

# ------------------------------ Main Script ------------------------------ #

def main():
    # Step 1: Load data
    data_handler = DataHandler()
    train_loader, test_loader = data_handler.get_loaders()
    train_dataset, test_dataset = data_handler.get_datasets()

    # Step 2: Extract features
    extractor = FeatureExtractor()
    train_feats, train_labels = extractor.extract(train_loader)
    test_feats, test_labels = extractor.extract(test_loader)

    # Step 3: Apply LDA
    lda = LDAReducer(n_components=9)
    X_train = lda.fit_transform(train_feats, train_labels)
    X_test = lda.transform(test_feats)

    # Step 4: Train classifier
    clf = Classifier()
    clf.train(X_train, train_labels)
    clf.evaluate(X_test, test_labels, train_dataset.classes)

    # Step 5: Save everything
    with open("full_pipeline.pkl", "wb") as f:
        pickle.dump({
            'train_feats': train_feats,
            'test_feats': test_feats,
            'train_labels': train_labels,
            'test_labels': test_labels,
            'lda': lda.lda,
            'classifier': clf.model,
            'classes': train_dataset.classes
        }, f)
    print("✅ Saved all components to full_pipeline.pkl")

    # Step 6: Predict + show
    visualizer = PredictorVisualizer(extractor, lda, clf, data_handler.transform, test_dataset)
    visualizer.predict_and_show("test_images/plane1.jpg")

if __name__ == '__main__':
    main()
