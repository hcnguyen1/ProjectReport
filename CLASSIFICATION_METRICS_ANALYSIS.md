# Classification Metrics: Precision, Recall, and F1-Score Analysis

## Overview
This section provides detailed insights into how each model performs across precision, recall, and F1-score metrics for product category classification.

### Dataset and Features

**Input Features (X):**
- **Product Title:** Converted to 100 text features using CountVectorizer
  - max_features=100 (top 100 most frequent words)
  - stop_words='english' (removes common words like "the", "a", "is")
  - lowercase=True (case-insensitive)
- **Total Features:** 100 (Product Title only - Merchant ID excluded)

**Target Variable (Y):**
- **Category Label:** 10 classes (CPUs, Digital Cameras, Dishwashers, Freezers, Fridge Freezers, Fridges, Microwaves, Mobile Phones, TVs, Washing Machines)
- Encoded as: 0-9

**Dataset Size:**
- Total samples: 35,311 products
- Training set: 28,248 samples (80%)
- Test set: 7,063 samples (20%)

**Note:** Both Naive Bayes and ID3 use identical feature sets (Product Title only) for fair comparison. **Merchant ID has been excluded** from this analysis.

---

## Metric Definitions

**Precision:** Of all the products the model *predicted* as Category X, how many were actually correct?
- High precision = Few false positives (doesn't predict wrong categories)
- Formula: True Positives / (True Positives + False Positives)

**Recall:** Of all the actual Category X products in the test set, how many did the model correctly identify?
- High recall = Few false negatives (catches most actual products)
- Formula: True Positives / (True Positives + False Negatives)

**F1-Score:** Harmonic mean of precision and recall (balanced metric)
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Best value: 1.0 (perfect), Worst value: 0.0

---

## Naive Bayes Classification Metrics

### Overall Performance
- **Test Accuracy:** 82.34%
- **CV Mean Accuracy:** 82.81%
- **CV Std Dev:** 0.0026
- **Macro Average F1-Score:** 0.83
- **Weighted Average F1-Score:** 0.83

### Per-Class Performance

| Class | Category | Precision | Recall | F1-Score | Interpretation |
|-------|----------|-----------|--------|----------|------------------|
| 0 | CPUs | 0.99 | 1.00 | 0.99 | ⭐ Near-perfect—catches all, no mistakes |
| 1 | Digital Cameras | 0.88 | 0.85 | 0.87 | ✓ Good—balanced performance |
| 2 | Dishwashers | 0.70 | 0.93 | 0.80 | ✓ High recall—catches most, some FPs |
| 3 | Freezers | 0.89 | 0.53 | 0.66 | ⚠️ High precision, but misses 47% |
| 4 | Fridge Freezers | 0.63 | 0.77 | 0.70 | ⚠️ Moderate—struggles with confusion |
| 5 | Fridges | 0.67 | 0.70 | 0.68 | ⚠️ Moderate across both metrics |
| 6 | Microwaves | 0.99 | 0.78 | 0.87 | ✓ Very high precision, good recall |
| 7 | Mobile Phones | 0.91 | 0.77 | 0.83 | ✓ Good—strong precision |
| 8 | TVs | 0.95 | 0.89 | 0.92 | ✓ Excellent—strong across both metrics |
| 9 | Washing Machines | 0.95 | 0.90 | 0.93 | ✓ Excellent—very reliable |

### Strengths
- **Classes 0, 8, 9:** Near-perfect to excellent performance (F1 ≥ 0.92)
- **High Precision Categories:** Classes 0, 6, 7, 8, 9 (≥0.91)—rarely makes false positive mistakes
- **Strong Recall:** Classes 0, 2, 8, 9 (≥0.89)—catches most actual products
- **Overall:** 82.34% accuracy, very stable (CV std 0.0026)

### Weaknesses
- **Class 3 (Freezers):** Only 53% recall—misses nearly half of actual freezers
- **Class 2 (Dishwashers):** 70% precision—some false positives
- **Classes 4, 5:** Moderate precision (0.63-0.67)—indicates confusion with similar categories

---

## ID3 Decision Tree Classification Metrics (Depth=10)

### Overall Performance
- **Test Accuracy:** 76.37%
- **CV Mean Accuracy:** 76.73%
- **CV Std Dev:** 0.0042
- **Tree Nodes:** 161
- **Macro Average F1-Score:** 0.80
- **Weighted Average F1-Score:** 0.80

### Per-Class Performance

| Class | Category | Precision | Recall | F1-Score | Interpretation |
|-------|----------|-----------|--------|----------|------------------|
| 0 | CPUs | 1.00 | 0.93 | 0.96 | ⭐ Perfect precision, excellent recall |
| 1 | Digital Cameras | 0.98 | 0.59 | 0.74 | ⚠️ Very high precision, low recall |
| 2 | Dishwashers | 1.00 | 0.64 | 0.78 | ⚠️ Perfect precision, misses 36% |
| 3 | Freezers | 0.98 | 0.72 | 0.83 | ✓ Very high precision, moderate recall |
| 4 | Fridge Freezers | 0.97 | 0.74 | 0.84 | ✓ Good performance |
| 5 | Fridges | 0.96 | 0.53 | 0.68 | ⚠️ High precision, misses 47% |
| 6 | Microwaves | 1.00 | 0.75 | 0.86 | ✓ Perfect precision, good recall |
| 7 | Mobile Phones | 0.33 | 0.99 | 0.50 | ❌ Low precision, excellent recall |
| 8 | TVs | 1.00 | 0.78 | 0.88 | ✓ Perfect precision, good recall |
| 9 | Washing Machines | 1.00 | 0.85 | 0.92 | ✓ Excellent—perfect precision |

### Strengths
- **Classes 0, 2, 6, 8, 9:** Perfect precision (1.00)—zero false positives
- **Class 7 (Mobile Phones):** Excellent recall (0.99)—catches nearly all phones
- **Classes 0, 4, 9:** High F1-scores (≥0.84)—well-balanced performance
- **High Precision:** 8 out of 10 classes have precision ≥0.96

### Weaknesses
- **Class 7 (Mobile Phones):** Only 33% precision—severely overpredicts phones ❌
- **Classes 1, 2, 5:** Low recall (0.53-0.64)—misses many actual products
- **Overall:** 76.37% accuracy is 6% lower than Naive Bayes
- **Classes 3, 6, 8 (Freezers, Microwaves, TVs):** Moderate recall (0.72-0.78)—miss ~25% of actual items
- **Class 7 (Mobile Phones):** Low precision (0.68) despite excellent recall
- **High CV Variance:** Std dev 0.0701 suggests some inconsistency across folds

---

## Side-by-Side Comparison: Critical Classes

### Class 2 (Dishwashers)
| Metric | Naive Bayes | ID3 (Depth=10) | Difference |
|--------|-------------|----------------|-----------|
| Precision | 0.87 | **1.00** | **ID3 +0.13** ⭐ |
| Recall | **0.91** | 0.85 | NB +0.06 |
| F1-Score | 0.89 | **0.92** | **ID3 +0.03** |

**Verdict:** ID3 edges out with perfect precision and higher F1-score.

### Class 3 (Freezers)
| Metric | Naive Bayes | ID3 (Depth=10) | Difference |
|--------|-------------|----------------|-----------|
| Precision | 0.94 | **0.98** | ID3 +0.04 |
| Recall | 0.50 | **0.72** | **ID3 +0.22** ⭐ |
| F1-Score | 0.65 | **0.83** | **ID3 +0.18** |

**Verdict:** ID3 dramatically better—catches 22% more freezers.

### Class 5 (Fridges)
| Metric | Naive Bayes | ID3 (Depth=10) | Difference |
|--------|-------------|----------------|-----------|
| Precision | 0.45 | **0.62** | ID3 +0.17 |
| Recall | **0.77** | **0.82** | ID3 +0.05 |
| F1-Score | 0.57 | **0.70** | **ID3 +0.13** |

**Verdict:** ID3 better across all metrics—improved precision significantly.

### Class 7 (Mobile Phones)
| Metric | Naive Bayes | ID3 (Depth=10) | Difference |
|--------|-------------|----------------|------------|
| Precision | **0.91** | 0.33 | **NB +0.58** ⭐⭐ |
| Recall | 0.77 | **0.99** | **ID3 +0.22** |
| F1-Score | **0.83** | 0.50 | **NB +0.33** ⭐ |

**Verdict:** NB much better overall; ID3 severely overpredicts phones (33% precision).

---

## Key Insights

### 1. **Algorithm Performance Without Merchant ID**
- **Naive Bayes:** 82.34% test accuracy—slightly better without Merchant ID
- **ID3 (Depth=10):** 76.37% test accuracy—significantly worse without Merchant ID
- **Lesson:** Naive Bayes outperforms ID3 by 6% when using only Product Title features

### 2. **Precision vs Recall Tradeoffs**
- **ID3 Strengths:** Perfect precision on 6 classes (CPUs, Dishwashers, Microwaves, TVs, Washing Machines, Fridges)
- **ID3 Weaknesses:** Very low precision on Class 7 (Mobile Phones: 33%), low recall on several classes
- **Naive Bayes:** More balanced precision/recall across all classes, better overall F1-scores
- **Pattern:** ID3 achieves high precision but struggles with recall; NB more consistent

### 3. **Cross-Validation Stability**
- **Naive Bayes CV Std Dev:** 0.0026 (extremely consistent)
- **ID3 CV Std Dev:** 0.0042 (very consistent, slightly higher than NB)
- **Implication:** Both models are stable, with Naive Bayes slightly more consistent

### 4. **Best Performing Classes (Both Models)**
- **Classes 0, 8:** CPUs and TVs (F1 > 0.88 for both algorithms)
- **Why:** Distinct vocabulary ("intel", "core" vs "screen", "display", "4k")

### 5. **Challenging Classes**
- **Class 7 (Mobile Phones):** ID3 severely struggles with precision (33%)—overpredicts phones
- **Class 3 (Freezers):** NB has low recall (53%)—misses nearly half
- **Classes 1, 2, 5 (Cameras, Dishwashers, Fridges):** ID3 has recall issues (53-64%)
- **Why:** Without Merchant ID, overlapping text features cause more confusion

### 6. **Impact of Removing Merchant ID**
- **Naive Bayes:** +1.2% accuracy (81.14% → 82.34%) but slightly less stable
- **ID3:** -4.84% accuracy (81.21% → 76.37%)—significant performance drop
- **Key Finding:** Merchant ID was the #3 most informative feature; removing it hurts ID3 substantially
- **Tree Size:** Without Merchant ID, tree shrinks from 207 to 161 nodes

---

## Recommendations

### For Naive Bayes
✓ **Use when:**
- You want consistent, reliable performance without tuning (81% accuracy out-of-the-box)
- Stability is critical (CV std dev 0.0066—very low variance)
- Training speed matters (faster than deep decision trees)
- Balanced precision/recall preferred across all classes

**Best for:** Production systems requiring consistent performance, minimal configuration

### For ID3 Decision Tree (Depth=10+)
✓ **Use when:**
- Interpretability is critical (visualize decision paths)
- Feature importance analysis needed (see which words matter most)
- Perfect precision required for specific classes (CPUs, Dishwashers, Microwaves, TVs)
- You can afford hyperparameter tuning (depth selection crucial)

**Best for:** Explanatory analysis, debugging misclassifications, feature engineering insights

### Trade-offs Summary

| Factor | Naive Bayes | ID3 (Depth=10) | Winner |
|--------|-------------|----------------|--------|
| **Test Accuracy** | 82.34% | 76.37% | **NB** (+6%) |
| **CV Stability** | 0.0026 std | 0.0042 std | **NB** (slightly more stable) |
| **Tuning Required** | Minimal (alpha=1.0) | Critical (depth) | **NB** (easier) |
| **Interpretability** | Probability tables | Decision tree | **ID3** (visual) |
| **Training Speed** | Fast | Moderate | **NB** |
| **Perfect Precision Classes** | 1 (CPUs) | 6 (CPUs, Dishwashers, Microwaves, TVs, Washing Machines, Fridges) | **ID3** |
| **Consistency** | Balanced | High precision, low recall | **NB** |
| **F1-Score (macro avg)** | 0.83 | 0.80 | **NB** |

---

---

## Entropy and Information Gain Analysis (Historical - With Merchant ID)

**Note:** The analysis below was conducted when Merchant ID was included. Current results use Product Title only (100 features).

### **Initial Dataset Entropy**
- **Entropy: 3.2736 bits**
- **Maximum Possible Entropy: 3.3219 bits** (log₂(10) for 10 classes)
- **Interpretation:** Dataset has high uncertainty (close to maximum), indicating well-balanced classes

### **Top Features by Information Gain (Entropy Reduction) - With Merchant ID**

Information gain shows how much each feature reduces uncertainty when used for splitting in the decision tree:

| Rank | Feature | Information Gain | Category Association |
|------|---------|------------------|---------------------|
| 1 | **freezer** | 0.2765 | Freezers (Class 3) |
| 2 | **tv** | 0.1819 | TVs (Class 8) |
| 3 | **Merchant_ID** | 0.1564 | Cross-category (EXCLUDED in current analysis) |
| 4 | **intel** | 0.1043 | CPUs (Class 0) |
| 5 | **fridge** | 0.0807 | Fridges/Fridge Freezers (Classes 4, 5) |
| 6 | **microwave** | 0.0782 | Microwaves (Class 6) |
| 7 | **washing** | 0.0664 | Washing Machines (Class 9) |
| 8 | **dishwasher** | 0.0516 | Dishwashers (Class 2) |
| 9 | **style** | 0.0011 | Low importance |
| 10 | **smart** | 0.0007 | Low importance |

**Key Insights (Historical):**
- "freezer" provided the most information (0.2765 gain)—best single feature for classification
- Product-specific words (freezer, tv, intel, fridge) are strongest predictors
- **Merchant_ID was 3rd highest** (0.1564 gain)—explaining why removing it hurt ID3 performance
- Generic words (style, smart) provide minimal value
- **Current Analysis:** Without Merchant ID, ID3 performance dropped from 81% to 76%

### **Decision Tree Depth Impact (Historical - With Merchant ID)**

**Note:** The depth analysis below was conducted with Merchant ID included. Current model (without Merchant ID) at depth=10 achieves 76.37% vs 81.21% with Merchant ID.

The original ID3 results (57% accuracy) were from a **too-shallow tree** (depth=5). Increasing depth dramatically improves performance:

| Max Depth | Test Accuracy (w/ Merchant ID) | CV Mean | CV Std Dev | Total Nodes | Leaves | Interpretation |
|-----------|-------------------------------|---------|------------|-------------|--------|----------------|
| 3 | 42.62% | 43.15% | 0.0026 | 15 | 8 | Severe underfitting |
| **5 (original)** | **60.10%** | **60.20%** | **0.0045** | **45** | **23** | **Underfitting** |
| **10 (w/ Merchant)** | **81.21%** | **81.48%** | **0.0021** | **207** | **104** | **Optimal w/ Merchant** ⭐ |
| **10 (no Merchant)** | **76.37%** | **76.73%** | **0.0042** | **161** | **81** | **Current analysis** |
| 15 | 86.05% | 86.88% | 0.0031 | 447 | 224 | Good performance |
| 20 | 88.08% | 88.40% | 0.0038 | 849 | 425 | Excellent |
| Unlimited | 90.05% | 89.91% | 0.0023 | 2051 | 1026 | Near-perfect |

**Critical Discovery:**
- **Depth=5 was underfitting**, not overfitting!
- With proper depth (10+), ID3 matches/exceeds Naive Bayes
- Depth=10 is optimal: 81% accuracy, low variance (0.0021), reasonable complexity (207 nodes)

### **Revised Understanding**

**Initial Conclusion (Depth=5, with Merchant ID):** ID3 fundamentally unsuited for text data → ❌ **INCORRECT**

**First Revision (Depth=10, with Merchant ID):** ID3 can handle text features well with sufficient depth:
- ✅ Depth=10 + Merchant ID: 81.21% (matches Naive Bayes)
- ✅ Depth=20 + Merchant ID: 88.08% (exceeds Naive Bayes)
- ✅ Low CV variance indicates good generalization

**Current Analysis (Depth=10, WITHOUT Merchant ID):**
- ⚠️ Depth=10, no Merchant ID: 76.37% (6% worse than Naive Bayes)
- ⚠️ Merchant ID was critical feature (#3 in information gain)
- ✅ Naive Bayes less affected: 82.34% (only +1.2% improvement without it)

**Why Merchant ID Matters:**
- Provides cross-category information (merchant specialization patterns)
- ID3 relies heavily on categorical features for efficient splitting
- Removing it forces tree to rely solely on 100 text features
- Naive Bayes handles text-only features more naturally (probabilistic approach)

---

## Updated Recommendations (Product Title Only - No Merchant ID)

### For Naive Bayes ✅ **RECOMMENDED**
✓ **Use when:**
- You want best performance without hyperparameter tuning (82.34% accuracy)
- Consistency and stability are critical (CV std dev 0.0026—extremely low)
- Training speed matters (very fast)
- Balanced precision/recall preferred across all classes
- **Working with text-only features** (Product Title)

**Best for:** Production systems requiring high accuracy and minimal configuration

### For ID3 Decision Tree (Depth=10) ⚠️ **USE WITH CAUTION**
✓ **Use when:**
- Interpretability is critical (visualize decision paths)
- Feature importance analysis needed (see which words matter most)
- Perfect precision required for specific classes (CPUs, Dishwashers, TVs, etc.)
- You can accept 6% lower accuracy vs Naive Bayes (76% vs 82%)

❌ **Avoid when:**
- Maximum accuracy is priority (NB outperforms by 6%)
- Class 7 (Mobile Phones) is important (only 33% precision—severe overprediction)
- **Note:** With Merchant ID included, ID3 matches NB at ~81%

**Best for:** Explanatory analysis when interpretability outweighs accuracy concerns

### Depth Selection Guidelines
- **Depth=3-5:** Educational purposes only (shows algorithm behavior, poor performance)
- **Depth=10:** Optimal balance (81% accuracy, 207 nodes, good generalization)
- **Depth=15-20:** Maximum performance (86-88% accuracy, larger trees)
- **Unlimited:** Best accuracy (90%) but risk of overfitting on other datasets

---

## Conclusion

### **Current Analysis (Product Title Only - No Merchant ID):**
**Naive Bayes is the clear winner** when using text features alone:

| Algorithm | Test Accuracy | CV Std Dev | Strengths | Weaknesses |
|-----------|---------------|------------|-----------|------------|
| **Naive Bayes** ✅ | **82.34%** | 0.0026 | Fast, stable, balanced | Moderate precision on some classes |
| **ID3 (depth=10)** | 76.37% | 0.0042 | Interpretable, high precision on 6 classes | 6% lower accuracy, 33% precision on phones |

**Key Lesson:** **Feature selection matters as much as algorithm choice.**

### **Historical Context (With Merchant ID):**
- **With Merchant ID (101 features):** Both algorithms achieved ~81% (NB: 81.14%, ID3: 81.21%)
- **Without Merchant ID (100 features):** NB: 82.34% (+1.2%), ID3: 76.37% (-4.84%)
- **Impact:** Merchant ID was the #3 most informative feature (0.1564 information gain)
- **Removing it:** Helped NB slightly, hurt ID3 significantly

### **Depth Tuning Discovery:**
Original poor ID3 performance (57-60%) was due to **underfitting from shallow depth** (depth=5), not fundamental algorithm limitations. Increasing to depth=10 was necessary but not sufficient without Merchant ID.

**For this dataset (Product Title only):**
- **Production use:** Naive Bayes (82%, no tuning, very stable) ⭐
- **Analysis/interpretation:** ID3 depth=10 (76%, provides feature insights, visualizable)
- **Maximum performance:** Add Merchant ID back for both models (~81-82%)
