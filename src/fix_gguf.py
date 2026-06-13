import struct
import shutil

src = "Qwen3_5_0_8B_UT_6750.gguf"
dst = "Qwen3_5_0_8B_UT_6750_fixed.gguf"

shutil.copy2(src, dst)

with open(src, "rb") as f:
    data = f.read()

patches = {
    "qwen35.block_count": 24,
    "qwen35.nextn_predict_layers": 0,  # no nextn layers
}

with open(dst, "r+b") as f:
    for key, new_value in patches.items():
        key_bytes = key.encode("utf-8")
        key_pos = data.find(key_bytes)
        if key_pos == -1:
            print(f"Key not found: {key}")
            continue
        value_offset = key_pos + len(key_bytes) + 4
        current = struct.unpack_from("<I", data, value_offset)[0]
        print(f"{key}: {current} -> {new_value} (offset {value_offset})")
        f.seek(value_offset)
        f.write(struct.pack("<I", new_value))

# Verify
from gguf import GGUFReader
r = GGUFReader(dst)
for key in patches:
    field = r.fields[key]
    print(f"{key} = {field.parts[field.data[0]][0]}")
