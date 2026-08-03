import graphviz

# ---- Diagram 1: End-to-end full-stack architecture (as built in this repo) ----
g1 = graphviz.Digraph("architecture", format="png")
g1.attr(rankdir="LR", bgcolor="white", fontname="Helvetica")
g1.attr("node", shape="box", style="rounded,filled", fillcolor="#eaf4ea", fontname="Helvetica", fontsize="11")

g1.node("input", "Input Layer\n(Image A + Image B\nupload / camera capture)")
g1.node("pre", "Preprocessing\n(resize, grayscale,\nORB feature alignment)")
g1.node("core", "Difference Engine\n(SSIM map -> Otsu threshold\n-> morphology -> contours)")
g1.node("post", "Post-processing\n(area filter, bbox extraction)")
g1.node("render", "Result Renderer\n(Matplotlib bbox overlay,\nJSON summary)")
g1.node("service", "Service Layer\n(Flask REST API\n+ browser UI)")
g1.node("client", "Client\n(Web browser /\nfactory-line integration)")

g1.edge("input", "pre")
g1.edge("pre", "core")
g1.edge("core", "post")
g1.edge("post", "render")
g1.edge("render", "service")
g1.edge("service", "client")
g1.edge("client", "input", label="  new pair", style="dashed")

g1.render("architecture", cleanup=True)

# ---- Diagram 2: Edge deployment (hardware + software) ----
g2 = graphviz.Digraph("edge_deployment", format="png")
g2.attr(rankdir="TB", bgcolor="white", fontname="Helvetica")

with g2.subgraph(name="cluster_hw") as hw:
    hw.attr(label="Hardware (Edge Device, e.g. NVIDIA Jetson Orin Nano / Raspberry Pi 5 + Coral TPU)",
            style="rounded", color="#2b6cb0", fontname="Helvetica", fontsize="11")
    hw.node("cam", "Industrial\nline-scan camera", shape="box", style="filled", fillcolor="#dbeafe")
    hw.node("lighting", "Controlled\nlighting rig", shape="box", style="filled", fillcolor="#dbeafe")
    hw.node("soc", "SoC / GPU\n(inference compute)", shape="box", style="filled", fillcolor="#dbeafe")
    hw.node("io", "Digital I/O\n(reject-arm actuator,\nPLC trigger)", shape="box", style="filled", fillcolor="#dbeafe")

with g2.subgraph(name="cluster_sw") as sw:
    sw.attr(label="Software Stack (runs on-device)", style="rounded", color="#2f855a", fontname="Helvetica", fontsize="11")
    sw.node("capture", "Frame capture\nservice (V4L2/GStreamer)", shape="box", style="filled", fillcolor="#eaf4ea")
    sw.node("infer", "Inference engine\n(ONNX Runtime / TensorRT\nquantized INT8 model)", shape="box", style="filled", fillcolor="#eaf4ea")
    sw.node("logic", "Decision logic\n(defect score threshold,\nbbox aggregation)", shape="box", style="filled", fillcolor="#eaf4ea")
    sw.node("mqtt", "Edge messaging\n(MQTT / local REST)\n-> MES / dashboard", shape="box", style="filled", fillcolor="#eaf4ea")

g2.edge("cam", "capture")
g2.edge("lighting", "cam", style="dashed")
g2.edge("capture", "soc")
g2.edge("soc", "infer")
g2.edge("infer", "logic")
g2.edge("logic", "io", label="reject signal")
g2.edge("logic", "mqtt")

g2.render("edge_deployment", cleanup=True)

print("Diagrams generated.")
