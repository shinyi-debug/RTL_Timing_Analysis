import cv2
import numpy as np
import networkx as nx
import re
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model

class CircuitImageProcessor:
    def __init__(self, image_path, model_path="circuit_detection_model.h5"):
        self.image_path = image_path
        self.model = load_model(model_path)
        self.graph = nx.DiGraph()
        self.rtl_code = ""

    def process_image(self):
        """Processes the circuit image to detect components and connections."""
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image, 50, 150)
        return edges

    def detect_components(self):
        """Uses a pre-trained deep learning model to detect logic gates."""
        image = cv2.imread(self.image_path)
        image_resized = cv2.resize(image, (128, 128)) / 255.0
        image_expanded = np.expand_dims(image_resized, axis=0)
        predictions = self.model.predict(image_expanded)
        detected_gates = ["AND" if pred[0] > 0.5 else "OR" if pred[1] > 0.5 else "FLIPFLOP" for pred in predictions]
        return detected_gates

    def generate_rtl(self):
        """Converts detected circuit components into RTL code dynamically."""
        detected_gates = self.detect_components()
        rtl_code = """
module generated_circuit (
    input wire A, B,
    output wire Q
);
    wire C, D, D_int;
        """
        if "AND" in detected_gates:
            rtl_code += "\n    assign C = A & B;"
        if "OR" in detected_gates:
            rtl_code += "\n    assign D = C | A;"
        if "FLIPFLOP" in detected_gates:
            rtl_code += "\n    always @(posedge clk) Q <= D;"
        rtl_code += "\nendmodule"
        self.rtl_code = rtl_code
        return rtl_code

class RTLParser:
    def __init__(self, rtl_code, model_path="timing_regression_model.pkl", timing_data_path="timing_report.csv"):
        self.rtl_code = rtl_code
        self.graph = nx.DiGraph()
        self.model_path = model_path
        self.timing_data_path = timing_data_path
        self.model = self.load_or_train_model()
        self.extract_netlist()
        self.create_graph()
        self.load_real_timing_data()

    def extract_netlist(self):
        lines = self.rtl_code.split("\n")
        for line in lines:
            line = line.strip()
            match = re.match(r'assign\s+(\w+)\s*=\s*([\w\s&|~^()]+);', line)
            if match:
                output_signal = match.group(1)
                inputs = re.findall(r'\w+', match.group(2))
                for input_signal in inputs:
                    self.graph.add_edge(input_signal, output_signal)
    
    def get_longest_path_depth(self):
        try:
            return nx.dag_longest_path_length(self.graph)
        except nx.NetworkXError:
            return -1
    
    def load_real_timing_data(self):
        """Loads timing analysis data from an AMD Vivado synthesis report."""
        try:
            timing_data = pd.read_csv(self.timing_data_path)
            print("Loaded real timing data for improved accuracy.")
            return timing_data
        except FileNotFoundError:
            print("Warning: Timing report not found. Using synthetic data instead.")
            return None
    
    def train_model(self):
        num_samples = 5000  # Increased dataset for better accuracy
        fan_in = np.random.randint(1, 100, num_samples)
        fan_out = np.random.randint(1, 80, num_samples)
        depth = np.random.randint(1, 150, num_samples)
        gate_count = np.random.randint(10, 5000, num_samples)
        wire_length = np.random.randint(5, 5000, num_samples)
        clock_skew = np.random.uniform(0.1, 5.0, num_samples)
        logic_utilization = np.random.uniform(10, 90, num_samples)
        wns = np.random.uniform(-3.0, 2.0, num_samples)  # Worst Negative Slack
        tns = np.random.uniform(-100, 10, num_samples)  # Total Negative Slack
        failing_endpoints = np.random.randint(0, 50, num_samples)
        setup_violations = np.random.randint(0, 30, num_samples)
        hold_violations = np.random.randint(0, 20, num_samples)
        path_delay = np.random.uniform(0.5, 10.0, num_samples)
        
        data = pd.DataFrame({"fan_in": fan_in, "fan_out": fan_out, "depth": depth, "gate_count": gate_count,
                             "wire_length": wire_length, "clock_skew": clock_skew, "logic_utilization": logic_utilization,
                             "wns": wns, "tns": tns, "failing_endpoints": failing_endpoints,
                             "setup_violations": setup_violations, "hold_violations": hold_violations,
                             "path_delay": path_delay})
        
        real_data = self.load_real_timing_data()
        if real_data is not None:
            data = pd.concat([data, real_data], ignore_index=True)
        
        X = data.drop(columns=["wns"])  # Predicting slack (WNS)
        y = data["wns"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.02, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, self.model_path)
        return model

# Processing the image to generate RTL
image_processor = CircuitImageProcessor("circuit.png")
rtl_generated = image_processor.generate_rtl()
parser = RTLParser(rtl_generated)
analysis = parser.get_longest_path_depth()
print(f"Longest Logic Depth: {analysis}")



