"""
Week 4 - Lab 1: Every layer, every shape                                run:  python lab1_shapes.py

Code is much easier to read with the tensor shapes written next to it. This lab runs GTCRN once on 4 s of
audio and records, for every layer that runs, what goes in, what comes out and how many weights it owns -
by attaching a "forward hook" to every module. A hook is a function PyTorch calls right after a module's
forward(), with the module, its inputs and its output.

The result, results/layer_table.md, is what you use when walking through the code with your mentor, and
the backbone of the master's students' Report 1 (block diagram with tensor shapes).

Uses your my_model.py from Week 3 (copy it into this folder first: README, Step 0).
"""
import sys
from pathlib import Path

import torch

import my_model

RESULTS = Path("results")
_passed = []


def check(name, fn, hint=""):
    try:
        ok = bool(fn())
    except NotImplementedError:
        ok, hint = False, "not written yet - look for the # TODO"
    except Exception as e:
        ok, hint = False, f"{type(e).__name__}: {e}  {hint}".strip()
    _passed.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok and hint:
        print(f"         hint: {hint}")
    return ok


# ---------------------------------------------------------------- Exercise 1.1
def shape_of(x):
    """A readable description of what a module received or returned:
         a tensor            -> its shape as a tuple, e.g. (1, 257, 251, 2)
         a tuple or a list   -> a list with shape_of() of each element (modules often take or return several)
         None                -> None
         anything else       -> the name of its type, e.g. "int"   (type(x).__name__)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------- Exercise 1.2
def make_hook(name, records):
    """Return a hook function for the module called `name`. When PyTorch calls hook(module, inputs, output),
    it appends one dict to `records`:
        {"name": name, "type": type(module).__name__, "in": shape_of(inputs), "out": shape_of(output),
         "params": the number of weights the module owns ITSELF - not those of its children:
                   sum(p.numel() for p in module.parameters(recurse=False))}
    (`inputs` is always a tuple: everything that was passed to forward().)
    """
    # TODO
    raise NotImplementedError


def trace(model, inputs):
    """(Given) Hook every module, run the model once, remove the hooks. Returns the records in call order."""
    records, handles = [], []
    for name, module in model.named_modules():
        if name:                                            # "" is the whole model itself
            handles.append(module.register_forward_hook(make_hook(name, records)))
    with torch.no_grad():
        out = model(*inputs)
    for h in handles:
        h.remove()
    return records, out


def write_table(records, model, inputs, out, path):
    """(Given) Two tables: the top-level blocks (for the block diagram) and every layer."""
    top = [r for r in records if "." not in r["name"]]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [f"# {my_model.MODEL_NAME}: layers and tensor shapes", "",
             f"Input {shape_of(inputs)} (4 s of audio) -> output {shape_of(out)}. "
             f"{total:,} parameters in total, {trainable:,} of them trainable (requires_grad).", "",
             "## Top-level blocks, in the order they run", "", "| block | type | input | output | weights inside |",
             "|---|---|---|---|---|"]
    for r in top:
        inside = sum(p.numel() for p in dict(model.named_modules())[r["name"]].parameters())
        lines.append(f"| {r['name']} | {r['type']} | {r['in']} | {r['out']} | {inside:,} |")
    lines += ["", "## Every layer that ran", "", "| layer | type | input | output | own weights |", "|---|---|---|---|---|"]
    for r in records:
        indent = "&nbsp;&nbsp;" * r["name"].count(".")
        lines.append(f"| {indent}{r['name']} | {r['type']} | {r['in']} | {r['out']} | {r['params']:,} |")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print("--- Exercise 1.1  shape_of ---")
    check("a tensor -> its shape", lambda: shape_of(torch.zeros(2, 3)) == (2, 3))
    check("a tuple of tensors -> a list of shapes", lambda: shape_of((torch.zeros(1), torch.zeros(4, 5))) == [(1,), (4, 5)])
    check("nested, None and other types", lambda: shape_of([torch.zeros(2), (None, 3)]) == [(2,), [None, "int"]])

    print("\n--- Exercise 1.2  a hook ---")
    lin = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
    recs = []
    try:
        handle = lin[0].register_forward_hook(make_hook("0", recs))
        lin(torch.zeros(2, 4))
        handle.remove()
    except NotImplementedError:
        pass
    check("the hook records name, type, shapes and own weights",
          lambda: recs == [{"name": "0", "type": "Linear", "in": [(2, 4)], "out": (2, 3), "params": 15}],
          "inputs is a tuple, so 'in' is a list; Linear(4, 3) owns 4*3 + 3 = 15 weights")
    seq = []
    try:
        handle = lin.register_forward_hook(make_hook("seq", seq))
        lin(torch.zeros(2, 4))
        handle.remove()
    except NotImplementedError:
        pass
    check("a container owns no weights itself (recurse=False)", lambda: seq and seq[0]["params"] == 0)

    print(f"\n--- {my_model.MODEL_NAME}, 4 s of audio ---")
    model = my_model.build_model("cpu")
    inputs = my_model.example_input(4.0, "cpu")
    try:
        records, out = trace(model, inputs)
    except NotImplementedError:
        records, out = [], None
    total = sum(p.numel() for p in model.parameters())
    counted = sum({r["name"]: r["params"] for r in records}.values())     # a module that runs twice counts once
    print(f"  {len(records)} layer calls recorded; {total:,} parameters in the model, {counted:,} counted in the layers that ran")
    fixed = [n for n, p in model.named_parameters() if not p.requires_grad]
    print(f"  trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}; fixed (requires_grad=False): {fixed} (question Q6)")
    check("layers were recorded", lambda: len(records) > 10)
    check("output has the shape of the input", lambda: shape_of(out) == shape_of(inputs[0]))
    never_ran = [n for n, m in model.named_modules() if n and any(True for _ in m.parameters(recurse=False))
                 and n not in {r["name"] for r in records}]
    if never_ran:
        print(f"  modules that own weights but never ran forward(): {never_ran} (question Q6)")
    check("every weight is accounted for: layers that ran + modules whose forward() is never called",
          lambda: counted + sum(sum(p.numel() for p in dict(model.named_modules())[n].parameters(recurse=False))
                                for n in never_ran) == total)
    if records:
        write_table(records, model, inputs, out, RESULTS / "layer_table.md")
        print("\n  top-level blocks:")
        for r in (r for r in records if "." not in r["name"]):
            print(f"    {r['name']:<10} {str(r['in']):<28} -> {r['out']}")
        print(f"\n  saved {RESULTS / 'layer_table.md'}")

    print("\n--- SEtrain's GTCRN: the same network? ---")
    setrain = Path.home() / "baselines" / "SEtrain"
    if (setrain / "models" / "gtcrn_end2end.py").exists():
        sys.path.insert(0, str(setrain))
        from models.gtcrn_end2end import GTCRN as SEtrainGTCRN
        a = {k: tuple(v.shape) for k, v in SEtrainGTCRN().state_dict().items()}
        b = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        check("SEtrain's GTCRN has exactly the official GTCRN's weights (names and shapes)", lambda: a == b)
    else:
        print("  ~/baselines/SEtrain not found yet - README, Step 2")
    n, p = len(_passed), sum(_passed)
    print(f"\n{p}/{n} checks passed. " + ("All done!" if p == n else "Fix the FAIL lines and run again."))
