import networkx as nx
import re
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    def rule_based_heuristics(self, fan_in, fan_out, depth):
        """Applies rule-based heuristics for timing analysis."""
        warnings = []
        if fan_in > 5:
            warnings.append("High fan-in detected, may cause delay issues. Consider buffering.")
        if fan_out > 3:
            warnings.append("High fan-out detected, may cause excessive load. Consider pipelining.")
        if depth > 10:
            warnings.append("Deep combinational path detected. Consider restructuring logic or adding pipeline registers.")
        return warnings

    def analyze_signal(self, signal):
        """Extracts metadata for a given signal and predicts timing violations."""
        if signal not in self.graph:
            return f"Signal {signal} not found."
        fan_io = self.get_fan_in_out(signal)
        depth = self.get_longest_path_depth()
        prediction = self.model.predict([[fan_io["fan_in"], fan_io["fan_out"], depth]])[0]
        timing_violation = "Yes" if prediction == 1 else "No"
        rule_based_warnings = self.rule_based_heuristics(fan_io["fan_in"], fan_io["fan_out"], depth)
        return {
            "signal": signal,
            "fan_in": fan_io["fan_in"],
            "fan_out": fan_io["fan_out"],
            "estimated_depth": depth,
            "timing_violation": timing_violation,
            "heuristic_warnings": rule_based_warnings
        }

    def visualize_netlist(self):
        """Displays a graphical representation of the RTL netlist."""
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        plt.title("RTL Netlist Visualization")
        plt.show()

    def load_or_train_model(self):
        """Loads or trains an ML model for timing violation prediction."""
        if joblib.os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        else:
            return self.train_model()

    def train_model(self):
        """Trains a more accurate Gradient Boosting model for timing prediction."""
        data = pd.DataFrame({
            "fan_in": [1, 3, 2, 4, 5, 6, 7],
            "fan_out": [2, 1, 3, 2, 1, 4, 5],
            "depth": [2, 5, 3, 6, 7, 9, 12],
            "timing_violation": [0, 1, 0, 1, 1, 1, 1]
        })
        X = data.drop(columns=["timing_violation"])
        y = data["timing_violation"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
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

parser = RTLParser(rtl_example)
parser.visualize_netlist()
analysis = parser.analyze_signal("w1")
print(analysis)
