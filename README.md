![decision_tree](https://github.com/user-attachments/assets/40bea5dd-91f1-477f-a6c8-204cd9830e88)

How to Run:

1. Clone the Repository:
```
   git clone <repository_url>
   cd <repository_directory>
```
2. Install Dependencies:
   Ensure you have Python installed (preferably 3.7 or later). Then install the required packages using:
```
   pip install -r requirements.txt
```
3. Prepare the Dataset:
   Place your customer_booking.csv (the British Airways dataset) in the repository's root folder.

4. Run the Application:
   Launch the application by running:
```
   python main.py
```
   This will open the GUI where you can select the model you want to train.

5. Using the GUI:
   - Decision Tree:
     Select "Decision Tree" to train the model, view performance metrics, and generate visualizations (SVG and PNG files).
   - Random Forest:
     Select "Random Forest" to train the model and view performance metrics (no visualization is provided).

6. Review Outputs:
   The console will display the confusion matrix and classification metrics (precision, recall, accuracy, F1 score). 
