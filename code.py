import networkx as nx
import re
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class RTLParser:
    def __init__(self, rtl_code, timing_threshold=3, model_path="timing_violation_model.pkl"):
        """
        Initializes the RTL parser with the given RTL code.
        Constructs a directed graph representation of the RTL.
        """
        self.rtl_code = rtl_code
        self.graph = nx.DiGraph()
        self.signal_depths = {}
        self.timing_threshold = timing_threshold  # Threshold for timing violation
        self.model_path = model_path
        self.model = self.load_or_train_model()

    def extract_netlist(self):
        """Extracts netlist by identifying logic operations and builds a graph."""
        lines = self.rtl_code.split("\n")
        for line in lines:
            line = line.strip()
            match = re.match(r'assign\s+(\w+)\s*=\s*([\w\s&|~^()]+);', line)
            if match:
                output_signal = match.group(1)  # The signal being assigned a value
                inputs = re.findall(r'\w+', match.group(2))  # Extract input signals
                for input_signal in inputs:
                    self.graph.add_edge(input_signal, output_signal)  # Create edges in the graph

    def detect_cycles(self):
        """Detects cycles in the circuit to check for combinational loops."""
        try:
            cycle = nx.find_cycle(self.graph, orientation='original')
            return True, cycle  # Cycle detected
        except nx.NetworkXNoCycle:
            return False, []  # No cycle detected

    def get_fan_in_out(self, signal):
        """Computes fan-in (inputs) and fan-out (outputs) of a given signal."""
        if signal not in self.graph:
            return {"fan_in": 0, "fan_out": 0}
        return {"fan_in": self.graph.in_degree(signal), "fan_out": self.graph.out_degree(signal)}

    def get_longest_path_depth(self):
        """Finds the longest logic path depth in the entire RTL design."""
        try:
            return nx.dag_longest_path_length(self.graph)  # Compute longest path in DAG
        except nx.NetworkXError:
            return -1  # Return -1 if an error occurs

    def analyze_signal(self, signal):
        """Extracts metadata for a given signal and predicts timing violations."""
        if signal not in self.graph:
            return f"Signal {signal} not found."
        
        fan_io = self.get_fan_in_out(signal)
        depth = self.get_longest_path_depth()
        prediction = self.model.predict([[fan_io["fan_in"], fan_io["fan_out"], depth]])[0]
        timing_violation = "Yes" if prediction == 1 else "No"
        
        suggestions = []
        if timing_violation == "Yes":
            if fan_io["fan_in"] > 2:
                suggestions.append("Warning: High fan-in detected. Consider breaking down logic into smaller stages.")
            if fan_io["fan_out"] > 3:
                suggestions.append("Warning: High fan-out detected. Consider buffering signals to reduce load.")
            if depth > self.timing_threshold + 2:
                suggestions.append("Warning: Deep combinational path detected. Consider adding pipeline registers or restructuring logic.")
        
        analysis = {
            "signal": signal,
            "fan_in": fan_io["fan_in"],
            "fan_out": fan_io["fan_out"],
            "estimated_depth": depth,
            "timing_violation": timing_violation,
            "suggestions": suggestions if suggestions else "No major issues detected."
        }
        
        return analysis
    
    def visualize_netlist(self):
        """Displays a graphical representation of the RTL netlist."""
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        plt.title("RTL Netlist Visualization")
        plt.show()
    
    def load_or_train_model(self):
        """Loads a pre-trained model if available, otherwise trains a new one."""
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        else:
            return self.train_model()

    def train_model(self):
        """Trains a machine learning model for timing violation prediction."""
        data = pd.DataFrame({
            "fan_in": [1, 3, 2, 4, 5],
            "fan_out": [2, 1, 3, 2, 1],
            "depth": [2, 5, 3, 6, 7],
            "timing_violation": [0, 1, 0, 1, 1]  # Labels: 1 for violation, 0 for no violation
        })
        
        X = data.drop(columns=["timing_violation"])
        y = data["timing_violation"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        
        joblib.dump(model, self.model_path)
        return model

# Example RTL module
rtl_example = """
module example(a, b, c, d, out);
    input a, b, c, d;
    output out;
    wire w1, w2, w3;

    assign w1 = a & b;
    assign w2 = c | d;
    assign w3 = w1 ^ w2;
    assign out = ~w3;
endmodule
"""

# Run the parser
token_parser = RTLParser(rtl_example, timing_threshold=3)
token_parser.extract_netlist()

# Detect and report cycles
has_cycle, cycle_info = token_parser.detect_cycles()
if has_cycle:
    print("Critical Warning: Combinational loop detected! Cycle:", cycle_info)

timing_violations = []
for signal in ["w1", "w2", "w3", "out"]:
    analysis = token_parser.analyze_signal(signal)
    print(analysis)
    if analysis["timing_violation"] == "Yes":
        timing_violations.append(signal)

if timing_violations:
    print("Timing Violations Detected for Signals:", timing_violations)

# Visualize the RTL netlist
token_parser.visualize_netlist()
