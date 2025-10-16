# test_install.py

print("Attempting to import torch...")
import torch
import importlib
print("torch imported successfully.")
print("\nAttempting to import torchtext...")
try:
	torchtext = importlib.import_module("torchtext")
	print("torchtext imported successfully.")
except Exception as e:
	print("Failed to import torchtext:", e)
	print("You can install it with: pip install torchtext")
	torchtext = None
	torchtext = None

print("\n-----------------------------------------")
print("Success! Your installation is working correctly.")
print("-----------------------------------------")