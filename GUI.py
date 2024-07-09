import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

class MultiMorbidityRiskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-based Multimorbidity Risk Stratification")
        self.root.geometry("800x600")
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # Variable to hold the selected input method
        self.input_method = tk.StringVar(value="file")

        # Creating the layout
        self.create_widgets()

    def create_widgets(self):
        # Left side buttons
        button_frame = ctk.CTkFrame(self.root, corner_radius=10)
        button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        patient_btn = ctk.CTkButton(button_frame, text="Patient", command=self.load_patient)
        patient_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        load_btn = ctk.CTkButton(button_frame, text="Load", command=self.load_patient)
        load_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        model_btn = ctk.CTkButton(button_frame, text="Trained Model", command=self.load_model)
        model_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        predict_btn = ctk.CTkButton(button_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        display_btn = ctk.CTkButton(button_frame, text="Display", command=self.toggle_display)
        display_btn.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        close_btn = ctk.CTkButton(button_frame, text="Close", command=self.root.quit)
        close_btn.grid(row=5, column=0, padx=5, pady=5, sticky="ew")

        # Top section for input method selection
        top_frame = ctk.CTkFrame(self.root, corner_radius=10)
        top_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        input_method_label = ctk.CTkLabel(top_frame, text="Select Input Method:")
        input_method_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        input_file_rb = ctk.CTkRadioButton(top_frame, text="Load from file", variable=self.input_method, value="file")
        input_file_rb.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        input_manual_rb = ctk.CTkRadioButton(top_frame, text="Input manually", variable=self.input_method, value="manual")
        input_manual_rb.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Main section
        main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        main_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Model Section
        model_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        model_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        model_label = ctk.CTkLabel(model_frame, text="Select Model:")
        model_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.model_var = tk.StringVar()
        model_combobox = ctk.CTkComboBox(model_frame, variable=self.model_var, values=['LSTM', 'Random Forest', 'SVM', 'KNN'])
        model_combobox.grid(row=0, column=1, padx=5, pady=5)

        # Variable Section
        variable_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        variable_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.var_file_label = ctk.CTkLabel(variable_frame, text="Input Data File:")
        self.var_file_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.var_file_entry = ctk.CTkEntry(variable_frame, width=250)
        self.var_file_entry.grid(row=1, column=1, padx=5, pady=5)

        browse_var_btn = ctk.CTkButton(variable_frame, text="Browse", command=self.browse_var_file)
        browse_var_btn.grid(row=1, column=2, padx=5, pady=5)

        # Input Variable Values Section
        self.input_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        self.input_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        input_label = ctk.CTkLabel(self.input_frame, text="Variable Values:")
        input_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.variables = ['Sex', 'Age', 'BMI', 'Cholesterol']
        self.var_value_dict = {}

        for i, var in enumerate(self.variables):
            var_label = ctk.CTkLabel(self.input_frame, text=f"{var}:")
            var_label.grid(row=i+1, column=0, padx=5, pady=5, sticky="w")

            if var == 'Sex':
                var_value = ctk.CTkComboBox(self.input_frame, values=["Male", "Female"])
            else:
                var_value = ctk.CTkEntry(self.input_frame, width=100)

            var_value.grid(row=i+1, column=1, padx=5, pady=5)
            self.var_value_dict[var] = var_value

        # Prediction Section
        predict_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        predict_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.predicted_class_label = ctk.CTkLabel(predict_frame, text="Predicted Class:")
        self.predicted_class_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.risk_score_label = ctk.CTkLabel(predict_frame, text="Risk Score:")
        self.risk_score_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.risk_type_label = ctk.CTkLabel(predict_frame, text="Risk Type:")
        self.risk_type_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        # Feature Contribution Section
        feature_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        feature_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

        self.feature1_label = ctk.CTkLabel(feature_frame, text="1. Feature 1")
        self.feature1_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.feature2_label = ctk.CTkLabel(feature_frame, text="2. Feature 2")
        self.feature2_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.feature3_label = ctk.CTkLabel(feature_frame, text="3. Feature 3")
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
    root = ctk.CTk()
    app = MultiMorbidityRiskApp(root)
    root.mainloop()

