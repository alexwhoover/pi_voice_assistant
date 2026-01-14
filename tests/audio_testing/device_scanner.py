import sounddevice as sd

def list_devices():
    print("Available Audio Devices:")
    devices = sd.query_devices()

    for i, dev in enumerate(devices):
        default_in = "*" if i == sd.default.device[0] else " "
        default_out = "*" if i == sd.default.device[1] else " "

        print(f"{i}: {dev['name']} (In: {dev['max_input_channels']}, Out: {dev['max_output_channels']}) {default_in}In {default_out}Out")
    print("\nDefault Input/Output IDs:", sd.default.device)

if __name__ == "__main__":
    list_devices()

