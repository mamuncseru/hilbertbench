# Circuit Structure

```python
from hilbertbench.analysis import circuit_structure

result = circuit_structure(trace)
```

---

## What it measures

`circuit_structure` parses the OpenQASM circuit stored in the trace and
reports the structural facts that determine hardware cost and expressibility:
qubit count, circuit depth, gate composition, entangling-gate fraction, and
trainable parameter count.

This is **structural evidence** — it reports what the circuit *is*, not
whether it is a good design. Use it to:

- Verify the circuit that actually ran matches your expectations
- Understand the entangling-gate fraction (related to expressibility and
  hardware noise sensitivity)
- Count trainable parameters without inspecting the original Python code

---

## Function signature

```python
circuit_structure(
    trace: HilbertTrace | str | Path,
) -> dict
```

---

## Output fields

```python
{
    'status':       'OK',
    'num_circuits': 1,
    'primary': {
        'num_qubits':          2,
        'depth':               3,
        'total_gates':         3,
        'single_qubit_gates':  2,
        'entangling_gates':    1,
        'entangling_fraction': 0.333,
        'gate_counts':         {'ry': 2, 'cx': 1},
        'num_parameters':      2,
        'num_measurements':    0,
    },
    'circuits': [ ... ],   # list of per-circuit dicts (one per unique QASM)
}
```

### Top-level fields

| Field | Meaning |
|---|---|
| `status` | `'OK'` or `'No QASM circuit recorded'` |
| `num_circuits` | Number of distinct circuit structures found in the trace |
| `primary` | Structure of the largest circuit by gate count — the dominant circuit for single-circuit traces |
| `circuits` | List of structure dicts, one per unique QASM artifact |

### Per-circuit fields

| Field | Meaning |
|---|---|
| `num_qubits` | Qubit count from the register declaration |
| `depth` | Circuit depth under greedy ASAP layer assignment |
| `total_gates` | `single_qubit_gates + entangling_gates` |
| `single_qubit_gates` | Gates acting on exactly one qubit |
| `entangling_gates` | Gates acting on two or more qubits (CX, CZ, ECR, …) |
| `entangling_fraction` | `entangling_gates / total_gates` (0 if no gates) |
| `gate_counts` | Per-gate-name counts, e.g. `{'ry': 4, 'cx': 2, 'rz': 4}` |
| `num_parameters` | Distinct trainable parameter count |
| `num_measurements` | Number of `measure` instructions |

---

## Verdicts

| `status` | Meaning |
|---|---|
| `OK` | At least one QASM circuit was found and parsed |
| `No QASM circuit recorded` | The integration did not record a circuit (e.g., the backend does not expose QASM) |

There is no threshold-based classification — this function reports evidence,
not a verdict. Interpretation is left to the user.

---

## Multiple circuits

When the trace contains circuits of different structures (e.g., the ansatz
was modified mid-run, or transpilation produced different ISA circuits),
`num_circuits` will be greater than 1. The `primary` field selects the
largest by gate count. Inspect `circuits` for all structures.

---

## Limitations

- Depth is computed by **greedy ASAP layering**: each gate is placed in the
  earliest layer where all its qubits are free. This matches the standard
  definition of circuit depth for un-scheduled circuits.
- After transpilation to ISA, the stored circuit is the transpiled version.
  If you transpile with `generate_preset_pass_manager`, the `num_qubits`
  and gate mix will reflect the hardware topology.
- Circuits stored as binary blobs (some backends) cannot be parsed. The
  function returns `'No QASM circuit recorded'` in that case.

---

## Example

```python
from hilbertbench.analysis import circuit_structure

result = circuit_structure(trace)
p = result['primary']

print(f"Qubits:             {p['num_qubits']}")
print(f"Depth:              {p['depth']}")
print(f"Parameters:         {p['num_parameters']}")
print(f"Entangling fraction:{p['entangling_fraction']:.1%}")
print(f"Gate breakdown:     {p['gate_counts']}")
```
