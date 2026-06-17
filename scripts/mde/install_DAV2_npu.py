import degirum as dg

zoo = dg.connect(
    dg.CLOUD, "https://cs.degirum.com/degirum/hailo", "<your cloud API access token>"
)

model = zoo.load_model("depth_anything_v2--224x224_quant_hailort_multidevice_1")

result = model("<file_path>")
print(result)
