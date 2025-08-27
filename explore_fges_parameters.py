#!/usr/bin/env python3
# explore_fges_parameters.py

import os, glob, numpy as np, pandas as pd, jpype, jpype.imports
from importlib.resources import files

def ensure_tetrad_jvm():
    if jpype.isJVMStarted():
        return
    jars = []
    try:
        jars.append(str(files("pytetrad.resources") / "tetrad-current.jar"))
    except Exception:
        pass
    if os.getenv("TETRAD_JAR"):
        jars.append(os.getenv("TETRAD_JAR"))
    jars += glob.glob(os.path.join("resources", "*tetrad*jar"))
    if not jars:
        raise RuntimeError("No Tetrad JAR found")
    jpype.startJVM(classpath=jars)

ensure_tetrad_jvm()

import edu.cmu.tetrad.search as search
import edu.cmu.tetrad.search.score as score
import pytetrad.tools.translate as ptt

# Create a simple test dataset
rng = np.random.default_rng(123)
n = 1000

D = rng.choice([0,1,2], size=n, p=[0.2, 0.5, 0.3])
C = rng.integers(0, 2, size=n)
X = (D - D.mean()) + rng.normal(0, 1, size=n)
Y = 1.2*X + 0.8*(D==1) + 1.6*(D==2) + 0.7*C + rng.normal(0, 1, size=n)
E = rng.normal(0, 1, size=n)

df = pd.DataFrame({
    "D": D.astype("int64"),
    "C": C.astype("int64"),
    "X": X.astype("float64"),
    "Y": Y.astype("float64"),
    "E": E.astype("float64"),
})

print("=== FGES Parameter Exploration ===\n")

# Convert to Tetrad
tetrad_data = ptt.pandas_data_to_tetrad(df)
if hasattr(tetrad_data, "getDataSet"):
    tetrad_data = tetrad_data.getDataSet()

# Create FGES instance
sc = score.ConditionalGaussianScore(tetrad_data, 2.0, False)
fges = search.Fges(sc)

print("1. FGES Class Information:")
print(f"   Class: {fges.getClass().getName()}")
print(f"   Superclass: {fges.getClass().getSuperclass().getName()}")

print("\n2. Available FGES Methods (excluding private methods):")
methods = [attr for attr in dir(fges) if not attr.startswith('_') and callable(getattr(fges, attr))]
for method in sorted(methods):
    try:
        method_obj = getattr(fges, method)
        if hasattr(method_obj, '__call__'):
            print(f"   {method}()")
    except:
        pass

print("\n3. FGES Configuration Methods:")
config_methods = [m for m in methods if any(keyword in m.lower() for keyword in 
                ['set', 'get', 'is', 'has', 'add', 'remove', 'clear'])]
for method in sorted(config_methods):
    print(f"   {method}")

print("\n4. Testing Key Parameters:")

# Test different parameter combinations
print("\n   Testing different parameter combinations...")

# Test 1: Default parameters
print("\n   Test 1: Default parameters")
fges1 = search.Fges(sc)
dag1 = fges1.search()
edges1 = list(dag1.getEdges())
print(f"     Edges found: {len(edges1)}")
for e in edges1:
    print(f"       {e.getNode1().getName()} -> {e.getNode2().getName()}")

# Test 2: Different penalty discount
print("\n   Test 2: Penalty discount = 1.0 (more aggressive)")
sc2 = score.ConditionalGaussianScore(tetrad_data, 1.0, False)
fges2 = search.Fges(sc2)
dag2 = fges2.search()
edges2 = list(dag2.getEdges())
print(f"     Edges found: {len(edges2)}")
for e in edges2:
    print(f"       {e.getNode1().getName()} -> {e.getNode2().getName()}")

# Test 3: Different penalty discount
print("\n   Test 3: Penalty discount = 3.0 (more conservative)")
sc3 = score.ConditionalGaussianScore(tetrad_data, 3.0, False)
fges3 = search.Fges(sc3)
dag3 = fges3.search()
edges3 = list(dag3.getEdges())
print(f"     Edges found: {len(edges3)}")
for e in edges3:
    print(f"       {e.getNode1().getName()} -> {e.getNode2().getName()}")

# Test 4: With max degree limit
print("\n   Test 4: Max degree = 2 (limit node connections)")
sc4 = score.ConditionalGaussianScore(tetrad_data, 2.0, False)
fges4 = search.Fges(sc4)
fges4.setMaxDegree(2)
dag4 = fges4.search()
edges4 = list(dag4.getEdges())
print(f"     Edges found: {len(edges4)}")
for e in edges4:
    print(f"       {e.getNode1().getName()} -> {e.getNode2().getName()}")

# Test 5: With max degree limit and different penalty
print("\n   Test 5: Max degree = 2, penalty = 1.0")
sc5 = score.ConditionalGaussianScore(tetrad_data, 1.0, False)
fges5 = search.Fges(sc5)
fges5.setMaxDegree(2)
dag5 = fges5.search()
edges5 = list(dag5.getEdges())
print(f"     Edges found: {len(edges5)}")
for e in edges5:
    print(f"       {e.getNode1().getName()} -> {e.getNode2().getName()}")

print("\n5. Score-Specific Parameters:")

# Test different score parameters
print("\n   Testing ConditionalGaussianScore parameters:")

# Test different penalty discounts
penalties = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
print(f"     Testing penalty discounts: {penalties}")

for penalty in penalties:
    try:
        sc_test = score.ConditionalGaussianScore(tetrad_data, penalty, False)
        fges_test = search.Fges(sc_test)
        dag_test = fges_test.search()
        edges_test = list(dag_test.getEdges())
        print(f"       Penalty {penalty}: {len(edges_test)} edges")
    except Exception as e:
        print(f"       Penalty {penalty}: Error - {e}")

print("\n6. FGES Algorithm Parameters Summary:")
print("   Key tunable parameters:")
print("   - Penalty discount in score (0.5-5.0, default ~2.0)")
print("   - Max degree per node (default: unlimited)")
print("   - Score type selection (ConditionalGaussian, BDeu, SemBic)")
print("   - Parallel execution flag in score constructor")

print("\n7. Recommendations for Better Results:")
print("   - Lower penalty discount (1.0-1.5) → more edges, higher recall")
print("   - Higher penalty discount (3.0-5.0) → fewer edges, higher precision")
print("   - Limit max degree (2-3) → prevent overfitting, more interpretable")
print("   - Use appropriate score for data type (mixed: ConditionalGaussian)")
print("   - Consider parallel execution for large datasets")
