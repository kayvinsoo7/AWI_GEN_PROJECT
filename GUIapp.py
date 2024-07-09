import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

# Function to browse and select the dataset file
def browse_dataset():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    dataset_entry.delete(0, tk.END)
    dataset_entry.insert(tk.END, file_path)

# Function to run the machine learning model
def run_model():
    # Retrieve input values from the user interface
    dataset = dataset_entry.get()
    parameter = parameter_combobox.get()
    user_input = user_input_entry.get()

    # Perform necessary data preprocessing and model training/validation/prediction
    # ...
    # Your machine learning model code here

    # Display the results to the user
    messagebox.showinfo("Results", "Model execution completed.")

# Create the main window
window = tk.Tk()
window.title("Machine Learning Model Dashboard")

# Styling
style = ttk.Style(window)
style.theme_use("clam")

# Dataset File Entry
dataset_frame = ttk.Frame(window)
dataset_frame.pack(pady=10)

dataset_label = ttk.Label(dataset_frame, text="Dataset:")
dataset_label.pack(side=tk.LEFT, padx=(0, 10))

dataset_entry = ttk.Entry(dataset_frame, width=50)
dataset_entry.pack(side=tk.LEFT)

browse_button = ttk.Button(dataset_frame, text="Browse", command=browse_dataset)
browse_button.pack(side=tk.LEFT)

# Parameter Combobox
parameter_frame = ttk.Frame(window)
parameter_frame.pack(pady=10)

parameter_label = ttk.Label(parameter_frame, text="Parameter:")
parameter_label.pack(side=tk.LEFT, padx=(0, 10))

parameters = ["Age", "Sex", "BMI", "Alcohol", "Smoker"]  # Update with your parameters
parameter_combobox = ttk.Combobox(parameter_frame, values=parameters, state="readonly")
parameter_combobox.pack(side=tk.LEFT)

# User Input Entry
user_input_frame = ttk.Frame(window)
user_input_frame.pack(pady=10)

user_input_label = ttk.Label(user_input_frame, text="User Input:")
user_input_label.pack(side=tk.LEFT, padx=(0, 10))

user_input_entry = ttk.Entry(user_input_frame, width=50)
user_input_entry.pack(side=tk.LEFT)

# Run Button
run_button = ttk.Button(window, text="Run Model", command=run_model)
run_button.pack(pady=20)

# # Run Button
# selectData_button = ttk.Button(window, text="Upload data", command=run_model)
# selectData_button.pack(pady=20)

# Results Treeview
results_frame = ttk.Frame(window)
results_frame.pack(pady=10)

results_label = ttk.Label(results_frame, text="Results:")
results_label.pack()

results_treeview = ttk.Treeview(results_frame, columns=("Result"), show="headings")
results_treeview.heading("Result", text="Result")
results_treeview.pack()

# Configure treeview style
style.configure("Treeview", rowheight=25)

# Add sample result for demonstration purposes
results_treeview.insert("", "end", values=("Sample Result",))

# Start the GUI event loop
window.mainloop()
