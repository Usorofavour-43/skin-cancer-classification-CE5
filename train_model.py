"""
train_model.py

Standalone training script extracted from Lab_10.ipynb
(GET 324 Mini-Project — Skin Lesion Classifier: Benign vs Malignant)

Trains and compares two architectures:
  1. A custom CNN (three conv blocks, trained from scratch)
  2. MobileNetV3Small via transfer learning (frozen -> fine-tuned)

The best-performing model (lowest false-negative rate) is saved for deployment.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

# ---------------------------------------------------------------------------
# STEP 1: Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

print(f'TensorFlow version : {tf.__version__}')
print(f'Random seed set to : {SEED}')

# ---------------------------------------------------------------------------
# STEP 2: Results / model directories
# ---------------------------------------------------------------------------
results_dir = "results/mini_project/"

if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 3: GPU check
# ---------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'\nGPUs available: {len(gpus)}')
    for gpu in gpus:
        print(f' - {gpu}')
else:
    print('\nNo GPU found — training will run on CPU (slower).')

# ---------------------------------------------------------------------------
# STEP 4: Dataset paths — update these to match your environment
# ---------------------------------------------------------------------------
TRAIN_DIR = r"C:\Users\HP\Desktop\GET_324\Notebooks\Mini-Project Dataset\train"
TEST_DIR = r"C:\Users\HP\Desktop\GET_324\Notebooks\Mini-Project Dataset\test"

for split, d in [('train', TRAIN_DIR), ('test', TEST_DIR)]:
    exists = os.path.exists(d)
    count = 0
    if exists:
        for cls in os.listdir(d):
            cls_path = os.path.join(d, cls)
            if os.path.isdir(cls_path):
                count += len(os.listdir(cls_path))
    print(f'{split:>5} dir exists={exists} images≈{count}')

# ---------------------------------------------------------------------------
# STEP 5: Global hyperparameters
# ---------------------------------------------------------------------------
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

print(f'Image size  : {IMAGE_HEIGHT} x {IMAGE_WIDTH}')
print(f'Batch size  : {BATCH_SIZE}')
print(f'Max epochs  : {EPOCHS}')
print(f'Learning rate: {LR}')

# ---------------------------------------------------------------------------
# STEP 6: Datasets
# ---------------------------------------------------------------------------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="training",
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE,
    seed=SEED, shuffle=True,
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="validation",
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE,
    seed=SEED, shuffle=False,
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE, seed=SEED, shuffle=False,
)

class_names = train_dataset.class_names
num_classes = len(class_names)
print(f'Classes ({num_classes}): {class_names}')

# ---------------------------------------------------------------------------
# STEP 7: Optimise the data pipeline
# ---------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
print('Pipeline optimised with cache + shuffle + prefetch.')

# ---------------------------------------------------------------------------
# STEP 8: Data augmentation
# ---------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
], name='data_augmentation')

# ---------------------------------------------------------------------------
# STEP 9: Shared callbacks + evaluation utilities
# ---------------------------------------------------------------------------
def make_callbacks(name):
    """Return a fresh set of callbacks for each model."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'models/{name}_best.keras',
            monitor='val_accuracy', save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.3,
            patience=4, min_lr=1e-7, verbose=1,
        ),
    ]


def plot_learning_curves(history, title='Learning Curves'):
    """Plot accuracy and loss curves side-by-side."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2, linestyle='--')
    plt.legend()
    plt.title(f'{title} — Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2, linestyle='--')
    plt.legend()
    plt.title(f'{title} — Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir + f'{title.lower().replace(" ", "_")}_curves.png',
                dpi=120, bbox_inches='tight')
    plt.show()


def evaluate_model(model, dataset, class_names, title='Model', results_dir=results_dir):
    """Evaluate a binary skin-lesion classification model."""
    y_true, y_pred, y_prob = [], [], []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_prob.append(probs)
        y_pred.append(preds)
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    print(f'\n{title}')
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{title} — Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(results_dir + f'{title.lower().replace(" ", "_")}_confusion_matrix.png',
                dpi=120, bbox_inches='tight')
    plt.show()

    malignant_idx = class_names.index('malignant') if 'malignant' in class_names else 1
    benign_idx = 1 - malignant_idx

    fn = cm[malignant_idx][benign_idx]       # malignant misclassified as benign
    tp = cm[malignant_idx][malignant_idx]
    fp = cm[benign_idx][malignant_idx]       # benign misclassified as malignant
    tn = cm[benign_idx][benign_idx]

    fnr = fn / (fn + tp) if (fn + tp) > 0 else float('nan')
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')

    print(f'False Negative Rate (FNR): {fnr:.4f}')
    print(f'False Positive Rate (FPR): {fpr:.4f}')

    return accuracy_score(y_true, y_pred), fnr


print('Callbacks and utility functions defined.')

# ---------------------------------------------------------------------------
# STEP 10: Custom CNN — build, compile, train, evaluate
# ---------------------------------------------------------------------------
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
OUTPUTS = num_classes


def build_custom_cnn(input_shape, augmentation):
    """
    A three-block CNN built entirely from scratch.
    Filters double each block (32 -> 64 -> 128) following a standard design
    pattern. Augmentation is embedded inside the model so it runs
    automatically during training.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(OUTPUTS, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='custom_cnn')


cnn_model = build_custom_cnn(INPUT_SHAPE, data_augmentation)
cnn_model.summary()

cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

cnn_history = cnn_model.fit(
    train_dataset, validation_data=val_dataset, epochs=EPOCHS,
    callbacks=make_callbacks('custom_cnn'), verbose=1,
)

cnn_val_loss, cnn_val_acc = cnn_model.evaluate(val_dataset, verbose=0)
print(f'Custom CNN -> val_loss: {cnn_val_loss:.4f}')
print(f'val_accuracy: {cnn_val_acc:.4f}')

plot_learning_curves(cnn_history, 'Custom CNN')

cnn_test_acc, cnn_fnr = evaluate_model(cnn_model, test_dataset, class_names, 'Custom CNN')

# ---------------------------------------------------------------------------
# STEP 11: Transfer learning model — build, compile, train (feature extraction)
# ---------------------------------------------------------------------------
def build_transfer_model(input_shape, augmentation, backbone='mobilenetv3small'):
    """
    Feature extraction model:
      1. Augment input (training only)
      2. Preprocess for MobileNetV3 (scales to [-1, 1])
      3. Frozen MobileNetV3Small backbone (no top)
      4. GlobalAveragePooling2D
      5. Dense classification head
    """
    if backbone == 'mobilenetv3small':
        base_model = tf.keras.applications.MobileNetV3Small(
            weights='imagenet', input_shape=input_shape, include_top=False,
        )
        preprocess_fn = tf.keras.applications.mobilenet_v3.preprocess_input
        model_name = 'mobilenetv3small_transfer'
    else:
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', input_shape=input_shape, include_top=False,
        )
        preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
        model_name = 'mobilenetv2_transfer'

    base_model.trainable = False
    print(f'Base model      : {base_model.name}')
    print(f'Trainable layers: {sum(1 for l in base_model.layers if l.trainable)} / {len(base_model.layers)}')

    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation(inputs)
    x = preprocess_fn(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(OUTPUTS, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name=model_name), base_model


tl_model, base_model = build_transfer_model(INPUT_SHAPE, data_augmentation, backbone='mobilenetv3small')
tl_model.summary()

TL_LR = 1e-4   # Lower LR for transfer learning head
tl_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=TL_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

tl_history = tl_model.fit(
    train_dataset, validation_data=val_dataset, epochs=EPOCHS,
    callbacks=make_callbacks('tl_feature_extraction'), verbose=1,
)

tl_val_loss, tl_val_acc = tl_model.evaluate(val_dataset, verbose=0)
print(f'MobileNetV3 TL -> val_loss: {tl_val_loss:.4f}')
print(f'MobileNetV3 TL -> val_accuracy: {tl_val_acc:.4f}')
plot_learning_curves(tl_history, 'MobileNetV3 Transfer Learning')

tl_test_acc, tl_fnr = evaluate_model(tl_model, test_dataset, class_names, 'MobileNetV3 Transfer')

# ---------------------------------------------------------------------------
# STEP 12: Fine-tuning — unfreeze top layers and continue training
# ---------------------------------------------------------------------------
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f'Trainable base layers after unfreezing: {trainable_count} / {len(base_model.layers)}')

FINETUNE_LR = TL_LR / 10
tl_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

ft_history = tl_model.fit(
    train_dataset, validation_data=val_dataset, epochs=15,
    callbacks=make_callbacks('tl_finetuned'), verbose=1,
)

plot_learning_curves(ft_history, 'MobileNetV3 Fine-Tuned')

ft_test_acc, ft_fnr = evaluate_model(tl_model, test_dataset, class_names, 'MobileNetV3 Fine-Tuned')

# ---------------------------------------------------------------------------
# STEP 13: Compare models and save the best one for deployment
# ---------------------------------------------------------------------------
results = {
    'Custom CNN': {'acc': cnn_test_acc, 'fnr': cnn_fnr, 'params': cnn_model.count_params()},
    'MobileNetV3 (TL)': {'acc': tl_test_acc, 'fnr': tl_fnr, 'params': tl_model.count_params()},
    'MobileNetV3 (FT)': {'acc': ft_test_acc, 'fnr': ft_fnr, 'params': tl_model.count_params()},
}

print(f'{"Model":<22} {"Test Acc":>10} {"FNR":>10} {"Params":>14}')
print('-' * 60)
for name, vals in results.items():
    print(f'{name:<22} {vals["acc"]:>10.4f} {vals["fnr"]:>10.4f} {vals["params"]:>14,}')

labels = list(results.keys())
accs = [v['acc'] * 100 for v in results.values()]
colors = ['#e06c75', '#61afef', '#98c379']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, accs, color=colors, edgecolor='black', width=0.5)
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Skin Cancer Pattern Recognition: Model Comparison')
ax.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f'{acc:.2f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(results_dir + 'model_comparison.png', dpi=120, bbox_inches='tight')
plt.show()

best_name = min(results, key=lambda k: (results[k]['fnr'], -results[k]['acc']))
print(f'Best model for deployment: {best_name}')

best_model = tl_model if 'MobileNetV3' in best_name else cnn_model
best_model.save('models/skin_lesion_classifier.keras')
print('Saved deployment model to models/skin_lesion_classifier.keras')

cnn_model.save('models/custom_cnn.keras')
tl_model.save('models/mobilenetv3_final.keras')
print('Also saved: models/custom_cnn.keras and models/mobilenetv3_final.keras')
