import os
from src.data_preprocessing import run_preprocessing
from src.model_training import run_model_training

def main():
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)

    print("--- Starting Data Preprocessing ---")
    run_preprocessing()
    print("--- Preprocessing Complete ---")

    print("\n--- Starting Model Training ---")
    run_model_training()
    print("--- Model Training Complete and Comparison Plotted ---")

    print("\n--- Starting Model Evaluation ---")

if __name__ == "__main__":
    main()
