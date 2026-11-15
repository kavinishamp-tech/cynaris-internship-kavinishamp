
"""
W1D1: Python for ML - NumPy Fundamentals

Author: Kavi Nisha
Purpose:
    Demonstrate NumPy arrays, broadcasting, vectorization,
    matrix multiplication, and statistical analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. NUMPY ARRAY FUNDAMENTALS
# ============================================================

def demonstrate_arrays():
    """Create 1D, 2D, and 3D arrays and verify their shapes."""

    # 1D Array
    arr_1d = np.array([10, 20, 30, 40])

    # 2D Array
    arr_2d = np.array([
        [10, 20],
        [30, 40]
    ])

    # 3D Array
    arr_3d = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])

    print("\n========== ARRAY FUNDAMENTALS ==========")

    print("1D Array:", arr_1d)
    print("1D Shape:", arr_1d.shape)

    print("\n2D Array:\n", arr_2d)
    print("2D Shape:", arr_2d.shape)

    print("\n3D Array:\n", arr_3d)
    print("3D Shape:", arr_3d.shape)


# ============================================================
# 2. BROADCASTING
# ============================================================

def demonstrate_broadcasting():
    """Perform operations using NumPy broadcasting."""

    marks = np.array([70, 80, 90])

    bonus = 5

    result = marks + bonus

    print("\n========== BROADCASTING ==========")
    print("Original Marks:", marks)
    print("Bonus:", bonus)
    print("Updated Marks:", result)


# ============================================================
# 3. VECTORIZED OPERATIONS
# ============================================================

def demonstrate_vectorization():
    """Perform array operations without Python loops."""

    numbers = np.array([1, 2, 3, 4, 5])

    doubled = numbers * 2
    squared = numbers ** 2

    print("\n========== VECTORIZATION ==========")
    print("Original:", numbers)
    print("Doubled:", doubled)
    print("Squared:", squared)


# ============================================================
# 4. MATRIX MULTIPLICATION
# ============================================================

def demonstrate_matrix_multiplication():
    """Perform matrix multiplication using NumPy."""

    matrix_a = np.array([
        [1, 2],
        [3, 4]
    ])

    matrix_b = np.array([
        [5, 6],
        [7, 8]
    ])

    result = matrix_a @ matrix_b

    print("\n========== MATRIX MULTIPLICATION ==========")
    print("Matrix A:\n", matrix_a)
    print("Matrix B:\n", matrix_b)
    print("A @ B:\n", result)


# ============================================================
# 5. LOAD CSV DATASET
# ============================================================

def load_dataset():
    """Load student scores from the CSV file."""

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "student_scores.csv"

    df = pd.read_csv(dataset_path)

    scores = df[["Math", "Science", "English"]].to_numpy()

    print("\n========== DATASET ==========")
    print(df)

    print("\nNumPy Scores Array:")
    print(scores)

    print("Dataset Shape:", scores.shape)

    return df, scores


# ============================================================
# 6. STATISTICAL ANALYSIS
# ============================================================

def perform_statistics(df, scores):
    """Calculate mean, standard deviation, and correlation."""

    print("\n========== STATISTICAL ANALYSIS ==========")

    # Mean marks for each subject
    subject_mean = np.mean(scores, axis=0)

    # Standard deviation for each subject
    subject_std = np.std(scores, axis=0)

    # Correlation between subjects
    correlation = np.corrcoef(scores, rowvar=False)

    print("\nSubject Mean:")
    print("Math:", subject_mean[0])
    print("Science:", subject_mean[1])
    print("English:", subject_mean[2])

    print("\nSubject Standard Deviation:")
    print("Math:", subject_std[0])
    print("Science:", subject_std[1])
    print("English:", subject_std[2])

    print("\nSubject Correlation Matrix:")
    print(correlation)

    print("\nSubject Names:")
    print(df[["Math", "Science", "English"]].columns.tolist())


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main():
    """Execute all NumPy fundamentals demonstrations."""

    demonstrate_arrays()

    demonstrate_broadcasting()

    demonstrate_vectorization()

    demonstrate_matrix_multiplication()

    df, scores = load_dataset()

    perform_statistics(df, scores)


if __name__ == "__main__":
    main()
