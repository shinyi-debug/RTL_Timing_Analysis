import networkx as nx
import re
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class RTLParser:
    def __init__(self, rtl_code, model_path="timing_violation_model.pkl"):
        self.rtl_code = rtl_code
        self.graph = nx.DiGraph()
        self.model_path = model_path
        self.model = self.load_or_train_model()
        self.extract_netlist()
        self.create_graph()

    def extract_netlist(self):
        """Parses RTL code to extract signals and logic operations."""
        lines = self.rtl_code.split("\n")
        for line in lines:
            line = line.strip()
            match = re.match(r'assign\s+(\w+)\s*=\s*([\w\s&|~^()]+);', line)
            if match:
                output_signal = match.group(1)
                inputs = re.findall(r'\w+', match.group(2))
                for input_signal in inputs:
                    self.graph.add_edge(input_signal, output_signal)
    
    def create_graph(self):
        """Automatically builds a graph representation of the RTL design with attributes."""
        for node in self.graph.nodes:
            fan_io = self.get_fan_in_out(node)
            nx.set_node_attributes(self.graph, {node: {"fan_in": fan_io["fan_in"], "fan_out": fan_io["fan_out"]}})
    
    def get_fan_in_out(self, signal):
        """Computes fan-in and fan-out of a given signal."""
        if signal not in self.graph:
            return {"fan_in": 0, "fan_out": 0}
        return {"fan_in": self.graph.in_degree(signal), "fan_out": self.graph.out_degree(signal)}

    def get_longest_path_depth(self):
        """Finds the longest logic depth in the design graph."""
        try:
            return nx.dag_longest_path_length(self.graph)
        except nx.NetworkXError:
            return -1

    def train_model(self):
        """Trains a more accurate Gradient Boosting model for timing prediction with diverse metadata."""
        np.random.seed(42)
        num_samples = 500
        fan_in = np.random.randint(1, 50, num_samples)
        fan_out = np.random.randint(1, 40, num_samples)
        depth = np.random.randint(1, 100, num_samples)
        gate_count = np.random.randint(10, 1000, num_samples)
        wire_length = np.random.randint(5, 500, num_samples)
        timing_violation = np.where((depth > 20) & (fan_in > 10) & (fan_out > 15) & (gate_count > 200), 1, 0)
        
        data = pd.DataFrame({
            "fan_in": fan_in,
            "fan_out": fan_out,
            "depth": depth,
            "gate_count": gate_count,
            "wire_length": wire_length,
            "timing_violation": timing_violation
        })
        
        X = data.drop(columns=["timing_violation"])
        y = data["timing_violation"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        joblib.dump(model, self.model_path)
        return model

parser = RTLParser(rtl_example)
parser.visualize_netlist()
analysis = parser.analyze_signal("w1")
print(analysis)
