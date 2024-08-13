# test_imports.py
try:
    from models.gan import build_generator
    from models.vae import build_encoder
    print("Imports are working correctly!")
except ModuleNotFoundError as e:
    print(f"Error: {e}")
