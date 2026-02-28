import os
train_file = "tools/train.py"
with open(train_file, "r", encoding="utf-8") as f:
    code = f.read()

# 将之前的 30000/60000 进一步压缩到 16000/32000
code = code.replace("(30000, 60000)", "(16000, 32000)").replace("[30000, 60000]", "[16000, 32000]")
code = code.replace("30000/60000", "16000/32000")

with open(train_file, "w", encoding="utf-8") as f:
    f.write(code)
print("✅ 成功将体素上限进一步压缩至 16000(训练)/32000(验证)！")
