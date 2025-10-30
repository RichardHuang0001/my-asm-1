# Install gdown if not already installed
# If you encountered this error:
#    import gdown
# ModuleNotFoundError: No module named 'gdown'
# Just install gdown:
# pip install gdown

# download_drive.py
import gdown



file_url = "https://drive.google.com/uc?id=1yK_5g1bQWZMQoXw4ixqz4zj-Ue7Uq8Eh"
output = "sample_4k.pgm"

gdown.download(file_url, output, quiet=False)


file_url = "https://drive.google.com/uc?id=1U3S4h5Y-25qE5UHXT3B9fCvCLiae_tdI"
output = "sample_16k.pgm"

gdown.download(file_url, output, quiet=False)




