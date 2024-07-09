import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

class MultiMorbidityRiskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-based Multimorbidity Risk Stratification")
        self.root.geometry("800x600")
        self.root.configure(bg='#f7f7f7')

        # Variable to hold the selected input method
        self.input_method = tk.StringVar(value="file")

        # Creating the layout
        self.create_widgets()

    def create_widgets(self):
        # Left side buttons
        button_frame = tk.Frame(self.root, bg='#87CEEB', padx=10, pady=10)
        button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        patient_btn = tk.Button(button_frame, text="Patient", command=self.load_patient, bg='#00BFFF', fg='white', font=('Arial', 12, 'bold'))
        patient_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        load_btn = tk.Button(button_frame, text="Load", command=self.load_patient, bg='#1E90FF', fg='white', font=('Arial', 12, 'bold'))
        load_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        model_btn = tk.Button(button_frame, text="Trained Model", command=self.load_model, bg='#00BFFF', fg='white', font=('Arial', 12, 'bold'))
        model_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        predict_btn = tk.Button(button_frame, text="Predict", command=self.predict, bg='#1E90FF', fg='white', font=('Arial', 12, 'bold'))
        predict_btn.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        display_btn = tk.Button(button_frame, text="Display", command=self.toggle_display, bg='#00BFFF', fg='white', font=('Arial', 12, 'bold'))
        display_btn.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        close_btn = tk.Button(button_frame, text="Close", command=self.root.quit, bg='#1E90FF', fg='white', font=('Arial', 12, 'bold'))
        close_btn.grid(row=5, column=0, padx=5, pady=5, sticky="ew")

        # Top section for input method selection
        top_frame = tk.Frame(self.root, bg='#f7f7f7', padx=10, pady=10)
        top_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        input_method_label = tk.Label(top_frame, text="Select Input Method:", bg='#f7f7f7', font=('Arial', 12))
        input_method_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        input_file_rb = tk.Radiobutton(top_frame, text="Load from file", variable=self.input_method, value="file", bg='#f7f7f7', font=('Arial', 12))
        input_file_rb.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        input_manual_rb = tk.Radiobutton(top_frame, text="Input manually", variable=self.input_method, value="manual", bg='#f7f7f7', font=('Arial', 12))
        input_manual_rb.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Main section
        main_frame = tk.Frame(self.root, bg='#f7f7f7', padx=20, pady=10)
        main_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Model Section
        model_frame = tk.LabelFrame(main_frame, text="Model Selection", bg='#87CEFA', fg='white', padx=20, pady=10, font=('Arial', 12, 'bold'))
        model_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        model_label = tk.Label(model_frame, text="Select Model:", bg='#87CEFA', fg='white', font=('Arial', 12))
        model_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.model_var = tk.StringVar()
        model_combobox = ttk.Combobox(model_frame, textvariable=self.model_var, font=('Arial', 12))
        model_combobox['values'] = ('LSTM', 'Random Forest', 'SVM', 'KNN')
        model_combobox.grid(row=0, column=1, padx=5, pady=5)

        # Variable Section
        variable_frame = tk.LabelFrame(main_frame, text="Variables", bg='#FFA07A', fg='white', padx=20, pady=10, font=('Arial', 12, 'bold'))
        variable_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.var_file_label = tk.Label(variable_frame, text="Input Data File:", bg='#FFA07A', fg='white', font=('Arial', 12))
        self.var_file_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.var_file_entry = tk.Entry(variable_frame, width=50, font=('Arial', 12))
        self.var_file_entry.grid(row=1, column=1, padx=5, pady=5)

        browse_var_btn = tk.Button(variable_frame, text="Browse", command=self.browse_var_file, bg='#FF6347', fg='white', font=('Arial', 12, 'bold'))
        browse_var_btn.grid(row=1, column=2, padx=5, pady=5)

        # Input Variable Values Section
        self.input_frame = tk.LabelFrame(main_frame, text="Input Variable Values", bg='#FFA07A', fg='white', padx=20, pady=10, font=('Arial', 12, 'bold'))
        self.input_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        input_label = tk.Label(self.input_frame, text="Variable Values:", bg='#FFA07A', fg='white', font=('Arial', 12))
        input_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.variables = ['Sex', 'Age', 'BMI', 'Cholesterol']
        self.var_value_dict = {}

        for i, var in enumerate(self.variables):
            var_label = tk.Label(self.input_frame, text=f"{var}:", bg='#FFA07A', fg='white', font=('Arial', 12))
            var_label.grid(row=i+1, column=0, padx=5, pady=5, sticky="w")

            if var == 'Sex':
                var_value = ttk.Combobox(self.input_frame, values=["Male", "Female"], font=('Arial', 12))
            else:
                var_value = tk.Entry(self.input_frame, width=20, font=('Arial', 12))

            var_value.grid(row=i+1, column=1, padx=5, pady=5)
            self.var_value_dict[var] = var_value

        # Prediction Section
        predict_frame = tk.LabelFrame(main_frame, text="Risk Stratification", bg='#FFA07A', fg='white', padx=20, pady=10, font=('Arial', 12, 'bold'))
        predict_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.predicted_class_label = tk.Label(predict_frame, text="Predicted Class:", bg='#FFA07A', fg='white', font=('Arial', 12))
        self.predicted_class_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.risk_score_label = tk.Label(predict_frame, text="Risk Score:", bg='#FFA07A', fg='white', font=('Arial', 12))
        self.risk_score_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.risk_type_label = tk.Label(predict_frame, text="Risk Type:", bg='#FFA07A', fg='white', font=('Arial', 12))
        self.risk_type_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        # Feature Contribution Section
        feature_frame = tk.LabelFrame(main_frame, text="Top 3 Features Contributing to Prediction", bg='#FFA07A', fg='white', padx=20, pady=10, font=('Arial', 12, 'bold'))
        feature_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

        self.feature1_label = tk.Label(feature_frame, text="1. Feature 1", bg='#FFA07A', fg='white', font=('Arial', 12))
        self.feature1_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.feature2_label = tk.Label(feature_frame, text="2. Feature 2", bg='#FFA07A', fg='white', font=('Arial', 12))
        self.feature2_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.feature3_label = tk.Label(feature_frame, text="3. Feature 3", bg='#FFA07A', fg='white', font=('Arial', 12))
        self.feature3_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

    def load_patient(self):
        # Load patient data functionality
        messagebox.showinfo("Load Patient", "Loading patient data...")

    def browse_var_file(self):
        # Browse for variable file functionality
        file_path = filedialog.askopenfilename()
        self.var_file_entry.insert(0, file_path)

    def load_model(self):
        # Load trained model functionality
        messagebox.showinfo("Load Model", "Loading trained model...")

    def predict(self):
        # Predict functionality
        predicted_class = "1"  # Dummy prediction
        risk_score = "25%"     # Dummy risk score
        risk_type = "Low-Risk"  # Dummy risk type
        features = ["Age", "BMI", "Cholesterol"]  # Dummy features

        self.predicted_class_label.config(text=f"Predicted Class: {predicted_class}")
        self.risk_score_label.config(text=f"Risk Score: {risk_score}")
        self.risk_type_label.config(text=f"Risk Type: {risk_type}")

        self.feature1_label.config(text=f"1. {features[0]}")
        self.feature2_label.config(text=f"2. {features[1]}")
        self.feature3_label.config(text=f"3. {features[2]}")

    def toggle_display(self):
        # Toggle display functionality
        messagebox.showinfo("Display", "Toggling display...")

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiMorbidityRiskApp(root)
    root.mainloop()
