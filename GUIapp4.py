import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter import StringVar

class MultiMorbidityRiskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimorbidity Risk Prediction App")
        self.root.geometry("1200x800")
        self.root.configure(bg='#e0e0e0')

        # Variable to hold the selected input method
        self.input_method = tk.StringVar(value="file")

        # Creating the layout
        self.create_widgets()

    def create_widgets(self):
        # Left side buttons
        button_frame = tk.Frame(self.root, bg='#d0d0d0', padx=10, pady=10)
        button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        patient_btn = tk.Button(button_frame, text="Patient", command=self.load_patient, bg='#C1B2AB', fg='#407899')
        patient_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        load_btn = tk.Button(button_frame, text="Load", command=self.load_patient, bg='#C1B2AB', fg='#407899')
        load_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        model_btn = tk.Button(button_frame, text="Trained Model", command=self.load_model, bg='#b0c4de', fg='#407899')
        model_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        predict_btn = tk.Button(button_frame, text="Predict", command=self.predict, bg='#add8e6', fg='#407899')
        predict_btn.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        display_btn = tk.Button(button_frame, text="Display", command=self.toggle_display, bg='#b0c4de', fg='#407899')
        display_btn.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        close_btn = tk.Button(button_frame, text="Close", command=self.root.quit, bg='#add8e6', fg='#92374D')
        close_btn.grid(row=5, column=0, padx=5, pady=5, sticky="ew")

        # Main section
        main_frame = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=10)
        main_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Model Section
        model_frame = tk.LabelFrame(main_frame, text="Model Selection", bg='#e0e0e0', padx=20, pady=10)
        model_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        model_label = tk.Label(model_frame, text="Select Model:", bg='#e0e0e0')
        model_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.model_var = tk.StringVar()
        model_combobox = ttk.Combobox(model_frame, textvariable=self.model_var)
        model_combobox['values'] = ('XGBoost', 'Random Forest', 'SVM', 'DNN')
        model_combobox.grid(row=0, column=1, padx=5, pady=5)

        # Variable Section
        variable_frame = tk.LabelFrame(main_frame, text="Variables", bg='#e0e0e0', padx=20, pady=10)
        variable_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Radio buttons to choose input method
        input_file_rb = tk.Radiobutton(variable_frame, text="Load from file", variable=self.input_method, value="file", bg='#e0e0e0')
        input_file_rb.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        input_manual_rb = tk.Radiobutton(variable_frame, text="Input manually", variable=self.input_method, value="manual", bg='#e0e0e0')
        input_manual_rb.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.var_file_label = tk.Label(variable_frame, text="Input Data File:", bg='#e0e0e0')
        self.var_file_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.var_file_entry = tk.Entry(variable_frame, width=50)
        self.var_file_entry.grid(row=1, column=1, padx=5, pady=5)

        browse_var_btn = tk.Button(variable_frame, text="Browse", command=self.browse_var_file, bg='#add8e6', fg='black')
        browse_var_btn.grid(row=1, column=2, padx=5, pady=5)

        # Input Variable Values Section
        self.input_frame = tk.LabelFrame(main_frame, text="Input Variable Values", bg='#e0e0e0', padx=20, pady=10)
        self.input_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        input_label = tk.Label(self.input_frame, text="Variable Values:", bg='#e0e0e0')
        input_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.variables = ['Sex', 'Age', 'BMI', 'Cholesterol']
        self.var_value_dict = {}

        for i, var in enumerate(self.variables):
            var_label = tk.Label(self.input_frame, text=f"{var}:", bg='#e0e0e0')
            var_label.grid(row=i+1, column=0, padx=5, pady=5, sticky="w")

            if var == 'Sex':
                var_value = ttk.Combobox(self.input_frame, values=["Male", "Female"])
            else:
                var_value = tk.Entry(self.input_frame, width=20)

            var_value.grid(row=i+1, column=1, padx=5, pady=5)
            self.var_value_dict[var] = var_value

        # Prediction Section
        predict_frame = tk.LabelFrame(main_frame, text="Risk Stratification", bg='#e0e0e0', padx=20, pady=10)
        predict_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.predicted_class_label = tk.Label(predict_frame, text="Predicted Class:", bg='#e0e0e0')
        self.predicted_class_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.risk_score_label = tk.Label(predict_frame, text="Risk Score:", bg='#e0e0e0')
        self.risk_score_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.risk_type_label = tk.Label(predict_frame, text="Risk Type:", bg='#e0e0e0')
        self.risk_type_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        # Feature Contribution Section
        feature_frame = tk.LabelFrame(main_frame, text="Top 3 Features Contributing to Prediction", bg='#e0e0e0', padx=20, pady=10)
        feature_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

        self.feature1_label = tk.Label(feature_frame, text="1. Feature 1", bg='#e0e0e0')
        self.feature1_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.feature2_label = tk.Label(feature_frame, text="2. Feature 2", bg='#e0e0e0')
        self.feature2_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.feature3_label = tk.Label(feature_frame, text="3. Feature 3", bg='#e0e0e0')
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
