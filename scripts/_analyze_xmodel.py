"""scripts/_analyze_xmodel.py — Subgraph analysis helper for compile_blip1.sh"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "compiled/blip1_vision.xmodel"
try:
    import xir
    graph = xir.Graph.deserialize(path)
    sgs   = graph.get_root_subgraph().toposort_child_subgraph()
    dpu_count = 0
    cpu_count = 0
    print(f"\n  Subgraphs in {path}:")
    for sg in sgs:
        device = sg.get_attr("device").upper() if sg.has_attr("device") else "UNKNOWN"
        n_ops  = len(list(sg.get_ops())) if hasattr(sg, "get_ops") else "?"
        print(f"    [{device:6s}]  {sg.get_name()[:60]}  ({n_ops} ops)")
        if device == "DPU":
            dpu_count += 1
        else:
            cpu_count += 1
    print(f"\n  Total subgraphs : {dpu_count + cpu_count}")
    print(f"  DPU subgraphs   : {dpu_count}")
    print(f"  CPU subgraphs   : {cpu_count}")
    if dpu_count == 0:
        print("  WARNING: No DPU subgraphs found!")
        sys.exit(1)
    else:
        print("  OK: DPU subgraphs found — ready to deploy on KV260")
except Exception as e:
    print(f"  Could not analyse xmodel: {e}")
    sys.exit(1)
