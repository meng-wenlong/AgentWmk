from caw.watermarks import all_watermarks

print("Available watermarks:")
for wm in all_watermarks():
    print(f"- {wm}")