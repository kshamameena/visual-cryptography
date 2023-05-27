import numpy as np
import cv2
import time
import os
import secrets
import math
from Crypto.Cipher import DES, DES3
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# Define the block and key size for each algorithm
DES_BLOCK_SIZE = 64
DES_KEY_SIZE = 64

tripleDES_BLOCK_SIZE = 64
tripleDES_KEY_SIZE = 192

AES_BLOCK_SIZE = 128
AES_KEY_SIZE = 256




# Define a function to generate a random key
def generate_random_key(key_size):
    # Generate a random byte array of the given size
    key = os.urandom(key_size // 8)

    # Pad the key with zeros if necessary
    if len(key) < 8:
        key += b'\x00' * (8 - len(key))

    # Return the first 8 bytes of the key
    return key[:8]

def cipher_gen(key):
    cipher = PKCS1_OAEP.new(key)
    return cipher


# Define a function to encrypt data using DES
def encrypt_des(data, key):
    # Initialize the cipher
    cipher = DES.new(key, DES.MODE_ECB)

    # Encrypt the data
    encrypted_data = cipher.encrypt(data)

    return encrypted_data

# Define a function to encrypt data using 3DES
def encrypt_3des(data, key):
    # Initialize the cipher
    cipher = DES3.new(key, DES3.MODE_ECB)

    # Encrypt the data
    encrypted_data = cipher.encrypt(data)

    return encrypted_data

# Define a function to encrypt data using AES
def encrypt_aes(data, key):
    # Initialize the cipher
    cipher = AES.new(key, AES.MODE_ECB)

    # Encrypt the data
    encrypted_data = cipher.encrypt(data)

    return encrypted_data

# Define a function to encrypt data using RSA
def encrypt_arr(data, cipher):
    encrypted_chunks = []
    chunk_size = 446  # Maximum chunk size for RSA 4096-bit key

    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size]
        encrypted_chunk = cipher.encrypt(chunk)
        encrypted_chunks.append(encrypted_chunk)

    encrypted_data = b"".join(encrypted_chunks)
    return encrypted_data


# Read the image into an array
image = cv2.imread('target.jpg')

# Convert the image to a 2D array
image_array = np.array(image)

# # Write the image array to a text file
# np.savetxt('Targettext.txt', image_array, fmt='%d')

# If you read in color-code, you have to use three separate arrays to store each component of the image in a separate file
if image.ndim == 3:
    r_array = image_array[:, :, 0]
    g_array = image_array[:, :, 1]
    b_array = image_array[:, :, 2]

    np.savetxt('Targettext_r.txt', r_array, fmt='%d')
    np.savetxt('Targettext_g.txt', g_array, fmt='%d')
    np.savetxt('Targettext_b.txt', b_array, fmt='%d')


# Define the encryption key
DES_KEY = secrets.token_bytes(DES_KEY_SIZE//8)
tripleDES_KEY = secrets.token_bytes(tripleDES_KEY_SIZE//8)
AES_KEY = secrets.token_bytes(AES_KEY_SIZE//8)
#RSA_KEY = secrets.token_bytes(RSA_KEY_SIZE//8)


key = RSA.generate(4096)
cipher = PKCS1_OAEP.new(key)

print(DES_KEY)
print(tripleDES_KEY)
print(AES_KEY)
print(key)


# Start the timer
start_time = time.time()
# Encrypt the data using DES
encrypted_des_data = encrypt_des(image_array.tobytes(), DES_KEY)
# Stop the timer
end_time = time.time()
# Print the time taken to encrypt the data
print("Time taken to encrypt the data using DES:", end_time - start_time)

# Start the timer
start_time = time.time()
# Encrypt the data using 3DES
encrypted_3des_data = encrypt_3des(image_array.tobytes(), tripleDES_KEY)
# Stop the timer
end_time = time.time()
# Print the time taken to encrypt the data
print("Time taken to encrypt the data using 3DES:", end_time - start_time)

# Start the timer
start_time = time.time()
# Encrypt the data using AES
encrypted_aes_data = encrypt_aes(image_array.tobytes(), AES_KEY)
# Stop the timer
end_time = time.time()
# Print the time taken to encrypt the data
print("Time taken to encrypt the data using AES:", end_time - start_time)

# Start the timer
start_time = time.time()
# Encrypt the data using RSA
encrypted_rsa_data = encrypt_arr(image_array.tobytes(), cipher)
# Stop the timer
end_time = time.time()
# Print the time taken to encrypt the data
print("Time taken to encrypt the data using RSA:", end_time - start_time)


# Convert encrypted data to a NumPy array
encrypted_des_array = np.frombuffer(encrypted_des_data, dtype=np.uint8)
encrypted_3des_array = np.frombuffer(encrypted_3des_data, dtype=np.uint8)
encrypted_aes_array = np.frombuffer(encrypted_aes_data, dtype=np.uint8)
encrypted_rsa_array = np.frombuffer(encrypted_rsa_data, dtype=np.uint8)

# Save the arrays as text files
np.savetxt('encrypted_des_data.txt', encrypted_des_array)
np.savetxt('encrypted_3des_data.txt', encrypted_3des_array)
np.savetxt('encrypted_aes_data.txt', encrypted_aes_array)
np.savetxt('encrypted_rsa_data.txt', encrypted_rsa_array)

# Calculate the new shape based on the desired dimensions
width = 1855
height = 1855
new_size = width * height

# Determine the resizing factor
resizing_factor = int(len(encrypted_rsa_array) / new_size)

# Create images from the encrypted data
encrypted_des_image = np.reshape(encrypted_des_array, image_array.shape)
encrypted_3des_image = np.reshape(encrypted_3des_array, image_array.shape)
encrypted_aes_image = np.reshape(encrypted_aes_array, image_array.shape)
encrypted_rsa_image = np.reshape(encrypted_rsa_array[:new_size * resizing_factor], (width, height))  # Reshape without the color channel

# Save the images
cv2.imwrite('encrypted_des_image.png', encrypted_des_image)
cv2.imwrite('encrypted_3des_image.png', encrypted_3des_image)
cv2.imwrite('encrypted_aes_image.png', encrypted_aes_image)
cv2.imwrite('encrypted_rsa_image.png', encrypted_rsa_image)
