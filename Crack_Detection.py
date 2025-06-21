import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier # <<<<<<<<<<<<---------------------- Fix your code here, figure out the library name from sklearn
from sklearn.linear_model import LogisticRegression # <<<<<<<<<<<<---------------------- Fix your code here, figure out the library name from sklearn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import datetime
import time

# ==================== SETTINGS =====================
SELECTED_MODEL = "LogisticRegression"  # Options: 'SVM', 'RandomForest', 'LogisticRegression' <<<<<<<<<<<<<<<<<<<<<<-------------- Change your Code here
USE_GRID_SEARCH = True
SAVE_REPORT = True

# ========== AUGMENTATION FLAGS ==========
USE_H_FLIP = True
USE_V_FLIP = True
USE_BRIGHTNESS = False #since the dataset brightness have not a big differneces
USE_ROTATION = True
USE_NOISE = True
USE_CONTRAST = True
USE_ZOOM = True
USE_TRANSLATION = True
USE_ELASTIC = True
USE_PERSPECTIVE = True
USE_CLAHE = True

# Complete the rest of the augmentation toggle here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<------------------------ Your Code here

# ========================================

def load_images_from_folder(folder):
    images, labels = [], []
    print("[INFO] Loading images...")
    for label in ['Negative', 'Positive']:
        path = os.path.join(folder, label)
        for filename in tqdm(os.listdir(path), desc=f"Reading {label} images"):
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(0 if label == 'Negative' else 1)
    return images, labels

def preprocess_image(img):
    return cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), 50, 150)

def extract_features(img):
    return cv2.resize(img, (64, 64)).flatten() # <<<<<<<<<<<<<<<<<<<<<<-------------- Change your Code here, change the size of the training image

def elastic_transform(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
    return cv2.remap(image, indices[1].astype(np.float32), indices[0].astype(np.float32), interpolation=cv2.INTER_LINEAR)

def perspective_transform(img):
    h, w = img.shape
    pts1 = np.float32([[5, 5], [w-5, 5], [5, h-5], [w-5, h-5]])
    pts2 = np.float32([[0, 10], [w-10, 0], [10, h], [w, h-10]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_clahe(img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)

def augment_image(img):
    augmented = []
    if USE_H_FLIP: augmented.append(cv2.flip(img, 1)) # <<<<<<<<<<<<---------------------- Fix your code here
    if USE_V_FLIP: augmented.append(cv2.flip(img, 0)) # <<<<<<<<<<<<---------------------- Fix your code here
    if USE_BRIGHTNESS:
        augmented.append(cv2.convertScaleAbs(img, alpha=1.2, beta=30))
        augmented.append(cv2.convertScaleAbs(img, alpha=0.8, beta=-30))
    if USE_ROTATION:
        for angle in [15, -15]:
            M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1.0)
            augmented.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT))
    if USE_NOISE:
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        augmented.append(cv2.add(img, noise))
    if USE_CONTRAST:
        augmented.append(cv2.convertScaleAbs(img, alpha=1.5, beta=0)) # <<<<<<<<<<<<<<----------- fix your code here
        augmented.append(cv2.convertScaleAbs(img, alpha=0.7, beta=0)) # <<<<<<<<<<<<<<----------- fix your code here
    if USE_ZOOM:
        zoomed = cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 0, 1.2), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
        augmented.append(zoomed)
    if USE_TRANSLATION:
        M = np.float32([[1, 0, 5], [0, 1, 5]]) # <<<<<<<-------------------- Change your code here by translate the image up by 10 pixels and left by 2 pixels
        augmented.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT))
    if USE_ELASTIC:
        augmented.append(elastic_transform(img, 34, 4).astype(np.uint8))
    if USE_PERSPECTIVE:
        augmented.append(perspective_transform(img))
    if USE_CLAHE:
        augmented.append(apply_clahe(img))
    return augmented

def save_classification_report(report, model_name, settings):
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/{model_name}_report_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Settings: {settings}\n\n")
        f.write(report)
    print(f"[INFO] Classification report saved to {filename}")

if __name__ == '__main__':
    dataset_folder = 'Dataset - Small' # <<<<<<<-------------------- Change your dataset here
    images, labels = load_images_from_folder(dataset_folder)

    print("[INFO] Preprocessing images...") # <<<<<<<-------------- Measure the time needed for this process
    processed_images = [preprocess_image(img) for img in tqdm(images)]
    features = [extract_features(img) for img in processed_images]
    X, y = np.array(features), np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # <<<<<------- Change your test_size here

    print("[INFO] Augmenting training data...")
    for i in tqdm(range(len(X_train))):
        aug_imgs = augment_image(X_train[i].reshape(64, 64))
        for aug in aug_imgs:
            X_train = np.vstack([X_train, extract_features(preprocess_image(aug))])
            y_train = np.append(y_train, y_train[i])

    print(f"[INFO] Training model: {SELECTED_MODEL}")
    if SELECTED_MODEL == "SVM":
        model = SVC()
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    elif SELECTED_MODEL == "RandomForest":
        model = RandomForestClassifier()
        param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
    elif SELECTED_MODEL == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
        param_grid = {'C': [0.1, 1, 10]}

    if USE_GRID_SEARCH:
        grid = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"[INFO] Best parameters: {grid.best_params_}")
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    print(report)

    if SAVE_REPORT:
        save_classification_report(report, SELECTED_MODEL, str(grid.best_params_ if USE_GRID_SEARCH else "Default"))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive']).plot()
    plt.title("Confusion Matrix")
    plt.savefig(f"confusion_matrix_{SELECTED_MODEL}.png")
    plt.show() # <<<<<<<<<<<<<<----------- fix your code here

    # Visualization of CV scores if grid search is used
    if USE_GRID_SEARCH:
        results = grid.cv_results_
        means = results['mean_test_score']
        params = results['params']
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(means)), means, tick_label=[str(p) for p in params])
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Cross-Validation Accuracy Scores ({SELECTED_MODEL})")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(f"cv_scores_{SELECTED_MODEL}.png")
        plt.show()

    # Save model
    model_filename = f"trained_model_{SELECTED_MODEL}.pkl"
    joblib.dump(model, model_filename)
    print(f"[INFO] Trained model saved as '{model_filename}'")