# ============================================================
# ID3 DECISION TREE CLASSIFIER - PRICERUNNER DATASET
# Supervised Classification: Product Category Prediction
# ============================================================
# Problem Statement:
# - Goal: Classify products into categories using machine learning
# - Input: Product Title (text) + Merchant ID (categorical)
# - Target: Category Label (10 classes)
# - Dataset: PriceRunner from UCI
# ============================================================

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

def get_user_input():
    """Collect user configuration for model training"""
    print("\n" + "=" * 70)
    print("ID3 DECISION TREE CLASSIFIER - CONFIGURATION")
    print("=" * 70)
    
    # Training/Test split
    print("\n[TRAINING/TEST SPLIT]")
    print("  Suggested: 80% (standard practice)")
    print("  Range: 1-99 (must be between 0 and 100)")
    while True:
        try:
            user_input = input("  Enter training set percentage (or press Enter for 80): ").strip()
            if user_input == "":
                train_size = 80
            else:
                train_size = float(user_input)
            if 0 < train_size < 100:
                test_size = (100 - train_size) / 100
                train_size = train_size / 100
                print(f"  -> Selected: {train_size*100:.0f}% training, {test_size*100:.0f}% testing")
                break
            else:
                print("  ERROR: Please enter a value between 0 and 100")
        except ValueError:
            print("  ERROR: Please enter a valid number")
    
    # Max depth
    print("\n[MAX TREE DEPTH]")
    print("  Suggested: 5 (industry standard, good generalization)")
    print("  Range: 1-20 (higher = more complex tree)")
    while True:
        try:
            user_input = input("  Enter max_depth (or press Enter for 5): ").strip()
            if user_input == "":
                max_depth = 5
            else:
                max_depth = int(user_input)
            if 1 <= max_depth <= 20:
                print(f"  -> Selected: depth {max_depth}")
                break
            else:
                print("  ERROR: Please enter a value between 1 and 20")
        except ValueError:
            print("  ERROR: Please enter a valid integer")
    
    # Cross-validation folds
    print("\n[CROSS-VALIDATION FOLDS]")
    print("  Suggested: 5 (standard k-fold)")
    print("  Range: 2-10 (more folds = more robust)")
    while True:
        try:
            user_input = input("  Enter number of CV folds (or press Enter for 5): ").strip()
            if user_input == "":
                cv_folds = 5
            else:
                cv_folds = int(user_input)
            if 2 <= cv_folds <= 10:
                print(f"  -> Selected: {cv_folds}-fold cross-validation")
                break
            else:
                print("  ERROR: Please enter a value between 2 and 10")
        except ValueError:
            print("  ERROR: Please enter a valid integer")
    
    # Min samples split
    print("\n[MIN SAMPLES SPLIT]")
    print("  Suggested: 2 (allows fine-grained splitting)")
    print("  Range: 1-20 (higher = simpler tree, less overfitting)")
    while True:
        try:
            user_input = input("  Enter min_samples_split (or press Enter for 2): ").strip()
            if user_input == "":
                min_samples = 2
            else:
                min_samples = int(user_input)
            if 1 <= min_samples <= 20:
                print(f"  -> Selected: {min_samples}")
                break
            else:
                print("  ERROR: Please enter a value between 1 and 20")
        except ValueError:
            print("  ERROR: Please enter a valid integer")
    
    # Feature selection
    print("\n[FEATURE SELECTION]")
    print("  Suggested: Option 1 (recommended)")
    print("  Available options:")
    print("    1. Product Title + Merchant ID")
    print("    2. Product Title only")
    
    feature_choice = input("  Select option (or press Enter for 1): ").strip() or "1"
    if feature_choice == "1":
        print("  -> Selected: Product Title + Merchant ID")
    else:
        print("  -> Selected: Product Title only")
    
    return {
        'train_size': train_size,
        'test_size': test_size,
        'max_depth': max_depth,
        'cv_folds': cv_folds,
        'min_samples_split': min_samples,
        'feature_choice': feature_choice
    }

def load_and_prepare_data(config):
    """Load data and prepare features based on user config"""
    csv_file = 'pricerunner_aggregate.csv'
    
    print("\n" + "=" * 70)
    print("DATA LOADING AND PREPARATION")
    print("=" * 70)
    
    # Read data
    df = pd.read_csv(csv_file)
    print(f"\n[OK] Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} columns")
    
    # Data verification
    print("\nData Verification:")
    print(f"  - Missing values: {df.isnull().sum().sum()} total")
    
    # Extract text and categorical features
    product_titles = df['Product Title'].astype(str)
    merchant_ids = df[' Merchant ID']
    y = df[' Category Label']
    
    print(f"\n[OK] Data extracted:")
    print(f"  - Product titles: {len(product_titles)} samples")
    print(f"  - Merchant IDs: {len(merchant_ids)} unique values")
    print(f"  - Categories: {len(y.unique())} classes")
    
    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Vectorize Product Title using CountVectorizer
    print(f"\n[OK] Vectorizing Product Title text...")
    vectorizer = CountVectorizer(max_features=100, lowercase=True, stop_words='english')
    X_title = vectorizer.fit_transform(product_titles)
    print(f"  - Generated {X_title.shape[1]} text features (word counts)")
    
    # Encode Merchant ID
    le_merchant = LabelEncoder()
    X_merchant = le_merchant.fit_transform(merchant_ids).reshape(-1, 1)
    print(f"  - Merchant ID encoded: {X_merchant.shape[1]} feature")
    
    # Combine features based on choice
    if config['feature_choice'] == '1':
        # Combine Product Title + Merchant ID
        X = hstack([X_title, X_merchant]).toarray()  # Convert to dense for Decision Tree
        features_used = "Product Title + Merchant ID"
    else:
        # Product Title only
        X = X_title.toarray()  # Convert to dense for Decision Tree
        features_used = "Product Title"
    
    print(f"\n[OK] Using {config['feature_choice']} feature(s): {features_used}")
    print(f"  - Total features: {X.shape[1]}")
    
    return X, y_encoded, features_used, le_target, df


def display_data_summary(X, y, features_used):
    """Display summary statistics about the data"""
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset shape: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"\nFeatures used: {features_used}")
    print(f"Target variable: Category Label")
    
    print("\nClass distribution:")
    unique_classes, counts = np.unique(y, return_counts=True)
    for class_label, count in zip(unique_classes, counts):
        percentage = (count / len(y)) * 100
        print(f"  Class {class_label}: {count:6,} samples ({percentage:5.2f}%)")

def train_and_evaluate(X, y, config):
    """Train model and perform comprehensive evaluation"""
    print("\n" + "=" * 70)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=config['train_size'],
        random_state=42
    )
    
    print(f"\nTrain-Test Split ({config['train_size']:.0%}/{config['test_size']:.0%}):")
    print(f"  - Training samples: {len(X_train):,}")
    print(f"  - Test samples: {len(X_test):,}")
    
    # Initialize model
    clf = tree.DecisionTreeClassifier(
        criterion='entropy',
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        random_state=42
    )
    
    print(f"\nModel Configuration:")
    print(f"  - Criterion: entropy (ID3)")
    print(f"  - Max depth: {config['max_depth']}")
    print(f"  - Min samples split: {config['min_samples_split']}")
    
    # Train model
    clf.fit(X_train, y_train)
    print("\n[OK] Model trained successfully")
    
    # Cross-validation
    print(f"\nCross-Validation ({config['cv_folds']}-Fold):")
    cv_scores = cross_val_score(clf, X, y, cv=config['cv_folds'], scoring='accuracy')
    print(f"  - Fold scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"  - Mean accuracy: {cv_scores.mean():.4f}")
    print(f"  - Std deviation: {cv_scores.std():.4f}")
    
    # Test predictions
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Set Results:")
    print(f"  - Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Generate visualizations
    print("\n[OK] Generating visualizations...")
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_id3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] confusion_matrix_id3.png saved")
    
    # Cross-validation scores visualization
    plt.figure(figsize=(10, 6))
    folds = list(range(1, len(cv_scores) + 1))
    plt.plot(folds, cv_scores, marker='o', linewidth=2, markersize=8, label='Fold Accuracy', color='steelblue')
    plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean():.4f}')
    plt.fill_between(folds, cv_scores.mean() - cv_scores.std(), cv_scores.mean() + cv_scores.std(), 
                     alpha=0.2, color='red', label=f'±1 Std Dev: {cv_scores.std():.4f}')
    plt.xlabel('Fold Number', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('ID3 Decision Tree - Cross-Validation Scores (5-Fold)', fontsize=14, fontweight='bold')
    plt.xticks(folds)
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('id3_cv_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] id3_cv_scores.png saved")
    
    # Classification report visualization
    from sklearn.metrics import precision_recall_fscore_support
    
    # Get metrics for all classes
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    x = np.arange(len(precision))
    width = 0.25
    
    # Plot bars
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='orange', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='green', alpha=0.8)
    
    # Labels and formatting
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('ID3 Decision Tree - Classification Report by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(range(10))
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('id3_classification_report.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] id3_classification_report.png saved")
    
    return clf, X_train, X_test, y_train, y_test, y_pred, test_accuracy, cv_scores

def display_detailed_results(y_test, y_pred, test_accuracy):
    """Display comprehensive classification results"""
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print(f"\n[METRIC EXPLANATIONS]")
    print(f"  Precision: Of predictions I made for class X, how many were correct?")
    print(f"  Recall: Of actual class X samples, how many did I correctly identify?")
    print(f"  F1-Score: Harmonic mean of precision and recall (balanced metric)")
    print(f"  Support: Number of actual samples for each class in test set")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\n[INTERPRETATION GUIDE]")
    print(f"  High Precision (>0.90): Avoid false positives (accurate when predicting)")
    print(f"  High Recall (>0.90): Avoid false negatives (catches most actual samples)")
    print(f"  High F1-Score (>0.90): Good balance between precision and recall")
    print(f"  Confusion Matrix diagonal: Correct predictions")
    print(f"  Confusion Matrix off-diagonal: Misclassifications")

def display_summary(config, test_accuracy, cv_scores):
    """Display final summary"""
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"\nConfiguration Summary:")
    print(f"  - Train/Test split: {config['train_size']:.0%}/{config['test_size']:.0%}")
    print(f"  - Max depth: {config['max_depth']}")
    print(f"  - Min samples split: {config['min_samples_split']}")
    print(f"  - Cross-validation folds: {config['cv_folds']}")
    
    print(f"\nPerformance Summary:")
    print(f"  - Test accuracy: {test_accuracy:.4f}")
    print(f"  - Mean CV accuracy: {cv_scores.mean():.4f}")
    print(f"  - CV std deviation: {cv_scores.std():.4f}")
    
    if abs(test_accuracy - cv_scores.mean()) < 0.1:
        print(f"\n[OK] Model shows good generalization (test ~= CV)")
    else:
        print(f"\n[WARNING] Model may be overfitting (test >> CV)")
    
    print("\n" + "=" * 70)
    print("Output files saved:")
    print("  - confusion_matrix_id3.png (predictions vs actual)")
    print("  - id3_cv_scores.png (cross-validation performance)")
    print("  - id3_classification_report.png (precision/recall/f1 by class)")
    print("=" * 70 + "\n")

def main():
    """Main execution flow"""
    print("\n")
    print("=" * 70)
    print("  ID3 DECISION TREE CLASSIFIER - PRICERUNNER DATASET".center(70))
    print("  Classification Analysis and Evaluation Tool".center(70))
    print("=" * 70)
    
    run_again = True
    while run_again:
        # Get user configuration
        config = get_user_input()
        
        # Load and prepare data
        X, y, features_used, le_target, df = load_and_prepare_data(config)
        
        # Display data summary
        display_data_summary(X, y, features_used)
        
        # Train and evaluate
        clf, X_train, X_test, y_train, y_test, y_pred, test_accuracy, cv_scores = train_and_evaluate(X, y, config)
        
        # Display detailed results
        display_detailed_results(y_test, y_pred, test_accuracy)
        
        # Display summary
        display_summary(config, test_accuracy, cv_scores)
        
        # Ask if user wants to run again with different parameters
        print("\n" + "=" * 70)
        while True:
            response = input("Would you like to run with different parameters? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                run_again = True
                print("\n" + "-" * 70 + "\n")
                break
            elif response in ['no', 'n']:
                run_again = False
                print("\n[OK] Analysis complete.")
                print("=" * 70 + "\n")
                break
            else:
                print("Please enter 'yes' or 'no'")

if __name__ == "__main__":
    main()
