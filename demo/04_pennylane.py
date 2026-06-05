#!/usr/bin/env python
#
# file: demo/04_pennylane.py
#
# revision history:
#  20260605 (am): initial version
#
# PennyLane QNN binary classifier + HilbertBench passive recording.
#
# Trains a 2-qubit QNN on the two-moons dataset (300 samples, 20
# epochs) using PennyLane's Adam optimizer with parameter-shift
# gradients.  HilbertPennyLaneDeviceProxy intercepts every device
# execution, stores the circuit QASM once per unique structure
# (content-addressed), and records parameter bindings and the
# expectation-value outcome inline.
#
# After the tape closes the trace is converted to Parquet and the
# built-in barren-plateau and optimization-convergence analyzers
# are run on the recorded spans.
#
# Prerequisites:
#   pip install hilbertbench[pennylane,storage] scikit-learn
#
# Usage:
#   python demo/04_pennylane.py
#------------------------------------------------------------------------------

# import system modules
#
import os

# import third-party modules
#
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# import hilbertbench modules
#
from hilbertbench import HilbertTrace
from hilbertbench.integrations.pennylane import HilbertPennyLaneDeviceProxy
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.recorder.storage.writer import convert_trace_to_parquet
from hilbertbench.analysis import (
    detect_barren_plateau,
    optimization_convergence,
)

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# output root — timestamped run directory is created inside
#
RUNS_DIR = "runs/pennylane_qnn"

# model architecture
#
N_QUBITS = 2
N_LAYERS = 2

# training hyper-parameters
#
N_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.05
SEED = 42

# dataset size
#
N_SAMPLES = 300

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def load_data() -> tuple:
    """
    function: load_data

    arguments:
     none

    return:
     (X_train, X_test, y_train, y_test) as numpy arrays

    description:
     Generates the two-moons dataset with N_SAMPLES points, re-labels
     classes to {-1, +1} for sign-based prediction, splits 75/25, and
     scales features to [0, π] to suit angle-embedding encoders.
    """

    # generate and split the dataset
    #
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.15, random_state=SEED)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )

    # scale features to [0, π]
    #
    scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # exit gracefully
    #
    return (
        pnp.array(X_train, requires_grad=False),
        pnp.array(X_test, requires_grad=False),
        pnp.array(y_train, requires_grad=False),
        pnp.array(y_test, requires_grad=False),
    )
#
# end of function


def make_qnode(device) -> qml.QNode:
    """
    function: make_qnode

    arguments:
     device: a PennyLane device (real or proxy)

    return:
     a qml.QNode — the circuit forward pass

    description:
     Two-qubit QNN circuit:
       AngleEmbedding  — encodes x as RY rotations
       StronglyEntanglingLayers — trainable ansatz
       ⟨Z₀⟩ expectation — output in [-1, +1]
     Uses parameter-shift for gradient computation.
    """

    @qml.qnode(device, diff_method="parameter-shift")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

    # exit gracefully
    #
    return circuit
#
# end of function


def train(
    circuit,
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights,
) -> tuple:
    """
    function: train

    arguments:
     circuit: the qml.QNode forward-pass function
     X_train: training feature array
     y_train: training label array ({-1, +1})
     weights: initial parameter tensor (requires_grad=True)

    return:
     (weights, loss_history) — final weights and per-epoch MSE list

    description:
     Mini-batch Adam gradient descent for N_EPOCHS. Shuffles the
     dataset at the start of each epoch. MSE loss between ⟨Z₀⟩ and
     the ±1 targets. Prints epoch loss every 5 epochs.
    """

    # initialise the optimizer
    #
    optimizer = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    rng = np.random.default_rng(SEED)
    loss_history: list[float] = []
    n_train = len(X_train)

    for epoch in range(1, N_EPOCHS + 1):

        # shuffle at the start of each epoch
        #
        idx = rng.permutation(n_train)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        epoch_loss = 0.0
        n_batches = max(1, n_train // BATCH_SIZE)

        for b in range(n_batches):
            X_b = X_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            y_b = y_shuf[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            # one Adam step; parameter-shift gradients fire here
            #
            def batch_loss(w):
                preds = pnp.array([circuit(x, w) for x in X_b])
                return pnp.mean((preds - y_b) ** 2)

            weights, loss = optimizer.step_and_cost(batch_loss, weights)
            epoch_loss += float(loss)

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d}/{N_EPOCHS}"
                f"  loss = {avg_loss:.4f}"
            )

    # exit gracefully
    #
    return weights, loss_history
#
# end of function


def evaluate(
    circuit,
    X: np.ndarray,
    y: np.ndarray,
    weights,
    split: str = "",
) -> float:
    """
    function: evaluate

    arguments:
     circuit: the qml.QNode forward-pass function
     X:       feature array
     y:       label array ({-1, +1})
     weights: trained parameter tensor
     split:   display label ('Train' or 'Test')

    return:
     classification accuracy as a float in [0, 1]

    description:
     Runs the circuit on every sample in X, takes the sign of ⟨Z₀⟩
     as the predicted class, and computes accuracy.
    """

    # run inference and compute accuracy
    #
    preds = np.sign(np.array([float(circuit(x, weights)) for x in X]))

    # handle zero-valued predictions (sign(0) = 0 → assign +1)
    #
    preds = np.where(preds == 0, 1, preds)
    accuracy = float(np.mean(preds == y))
    print(f"  {split:5s} accuracy : {accuracy * 100:.1f}%")

    # exit gracefully
    #
    return accuracy
#
# end of function


def main() -> None:
    """
    function: main

    arguments:
     none

    return:
     none

    description:
     Loads two-moons data, builds a 2-qubit StronglyEntanglingLayers
     QNN, and wraps default.qubit in HilbertPennyLaneDeviceProxy.

     Every device execution inside the training loop is intercepted:
     the circuit QASM is stored once per unique circuit structure
     (content-addressed in the file store), parameter bindings and
     the expval outcome are stored inline in the span record.

     After the tape closes the trace is sealed, converted to Parquet,
     and the barren-plateau and convergence analyzers produce a
     diagnostic report from the recorded spans.
    """

    # print run header
    #
    sep = "-" * 60
    print(f"\n[{__FILE__}]  HilbertBench — PennyLane QNN (two-moons)")
    print(sep)

    # load and inspect the dataset
    #
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Dataset    : two-moons  ({N_SAMPLES} samples)")
    print(f"  Train/Test : {len(X_train)} / {len(X_test)}")
    print(f"  QNN        : {N_QUBITS} qubits, {N_LAYERS} layers")
    print(
        f"  Epochs     : {N_EPOCHS}"
        f"  batch = {BATCH_SIZE}"
        f"  lr = {LEARNING_RATE}"
    )
    print(sep)

    # initialise weights
    #
    shape = qml.StronglyEntanglingLayers.shape(
        n_layers=N_LAYERS, n_wires=N_QUBITS
    )
    weights = pnp.array(
        np.random.default_rng(SEED).uniform(0.0, 2.0 * np.pi, shape),
        requires_grad=True,
    )

    # open tape and create the proxied device
    #
    with HilbertTape(
        RUNS_DIR,
        tags={
            "demo":     "pennylane_qnn",
            "dataset":  "two_moons",
            "n_qubits": str(N_QUBITS),
        },
    ) as tape:

        # wrap default.qubit — no other code changes required
        #
        real_dev = qml.device("default.qubit", wires=N_QUBITS)
        proxy_dev = HilbertPennyLaneDeviceProxy(real_dev, tape)
        circuit = make_qnode(proxy_dev)

        # training loop — every execution is intercepted and recorded
        #
        weights, loss_history = train(circuit, X_train, y_train, weights)

        # evaluation — also recorded (same proxy is active)
        #
        print(sep)
        evaluate(circuit, X_train, y_train, weights, "Train")
        evaluate(circuit, X_test, y_test, weights, "Test")

    # convert to Parquet
    #
    parquet_path = convert_trace_to_parquet(tape.dir_path)

    # load the trace and run built-in analyzers
    #
    trace = HilbertTrace(tape.dir_path)
    bp = detect_barren_plateau(trace)
    conv = optimization_convergence(trace)

    print(sep)
    print(f"  Trace dir     : {tape.dir_path}")
    print(f"  Parquet       : {parquet_path.name}")
    print(f"  Trace status  : {trace.status}")
    print(f"  Spans recorded: {len(trace)}")
    print(f"  Trainability  : {bp['status']}")
    print(f"  Variance      : {bp['variance']:.6f}")
    print(f"  Convergence   : {conv['status']}")
    print(sep)
#
# end of function

#------------------------------------------------------------------------------
#
# main entry point
#
#------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#
# end of file
