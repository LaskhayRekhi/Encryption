import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random

# --- Playfair Cipher Functions ---
def create_key_matrix(key):
    """Creates the 6x6 matrix used for encryption and decryption based on the provided key."""
    base_set = "abcdefghijklmnopqrstuvwxyz0123456789"
    matrix_key = []

    # Add characters from key to the matrix
    for char in key:
        if char not in matrix_key:
            matrix_key.append(char)

    # Add remaining characters from base_set to complete the matrix
    for char in base_set:
        if char not in matrix_key:
            matrix_key.append(char)

    # Create a 6x6 matrix from the list of characters
    matrix = [matrix_key[i:i+6] for i in range(0, 36, 6)]
    return matrix

def search(char, matrix):
    """Searches for a character in the matrix and returns its coordinates."""
    for i in range(6):
        for j in range(6):
            if matrix[i][j] == char:
                return i, j
    return None

def encrypt(text, matrix):
    """Encrypts the input text based on the matrix."""
    encrypted_text = ""
    m = 0
    while m < len(text):
        if m == len(text) - 1:
            encrypted_text += text[m]
        else:
            q, r = search(text[m], matrix)
            s, t = search(text[m + 1], matrix)
            if text[m] == text[m + 1]:
                encrypted_text += text[m] + "@"
            elif q == s:
                # Same row
                encrypted_text += matrix[q][(r + 1) % 6] + matrix[s][(t + 1) % 6]
            elif r == t:
                # Same column
                encrypted_text += matrix[(q + 1) % 6][r] + matrix[(s + 1) % 6][t]
            else:
                # Rectangle swap
                encrypted_text += matrix[q][t] + matrix[s][r]
        m += 2
    return encrypted_text

def decrypt(encrypted_text, matrix):
    """Decrypts the encrypted text based on the matrix."""
    decrypted_text = ""
    m = 0
    while m < len(encrypted_text):
        if m == len(encrypted_text) - 1:
            decrypted_text += encrypted_text[m]
        elif encrypted_text[m] == "@" or encrypted_text[m + 1] == "@":
            if encrypted_text[m] != "@":
                decrypted_text += encrypted_text[m] * 2
            else:
                decrypted_text += encrypted_text[m + 1] * 2
        else:
            q, r = search(encrypted_text[m], matrix)
            s, t = search(encrypted_text[m + 1], matrix)
            if q == s:
                # Same row
                decrypted_text += matrix[q][(r - 1) % 6] + matrix[s][(t - 1) % 6]
            elif r == t:
                # Same column
                decrypted_text += matrix[(q - 1) % 6][r] + matrix[(s - 1) % 6][t]
            else:
                # Rectangle swap
                decrypted_text += matrix[q][t] + matrix[s][r]
        m += 2
    return decrypted_text

# --- Image Encryption Functions ---
def encrypt_image(img, key, rows, cols):
    """Encrypts the image using the provided key."""
    encrypted_img = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            for k in range(3):  # For each channel (R, G, B)
                encrypted_img[i, j, k] = key[img[i, j, k]]
    return encrypted_img

def decrypt_image(encrypted_img, key, rows, cols):
    """Decrypts the image using the provided key."""
    decrypted_img = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            for k in range(3):  # For each channel (R, G, B)
                decrypted_img[i, j, k] = key.index(encrypted_img[i, j, k])
    return decrypted_img

# --- Streamlit UI ---
st.title("Playfair Cipher and Image Encryption")

# --- Text Encryption ---
st.header("Text Encryption/Decryption")

key = st.text_input("Enter key for encryption (max 36 characters):").lower()
if key:
    key_matrix = create_key_matrix(key)
    st.write("Generated Key Matrix:")
    for row in key_matrix:
        st.write(row)

    # Option to choose encrypt or decrypt
    operation = st.radio("Choose an operation", ("Encrypt", "Decrypt"))

    if operation == "Encrypt":
        text_to_encrypt = st.text_input("Enter the text (in lowercase) to encrypt:").lower()
        if text_to_encrypt:
            encrypted_message = encrypt(text_to_encrypt, key_matrix)
            st.success(f"Encrypted message: {encrypted_message}")

    elif operation == "Decrypt":
        text_to_decrypt = st.text_input("Enter the text to decrypt:").lower()
        if text_to_decrypt:
            decrypted_message = decrypt(text_to_decrypt, key_matrix)
            st.success(f"Decrypted message: {decrypted_message}")

# --- Image Encryption ---
st.header("Image Encryption/Decryption")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = plt.imread(uploaded_file)

    # Display the original image
    st.image(img, caption="Original Image", use_column_width=True)

    rows, cols, _ = img.shape
    key = random.sample(range(256), 256)  # Random key generation for image encryption

    # Encrypt image
    encrypted_img = encrypt_image(img, key, rows, cols)
    st.image(encrypted_img, caption="Encrypted Image", use_column_width=True)

    # Decrypt image
    decrypted_img = decrypt_image(encrypted_img, key, rows, cols)
    st.image(decrypted_img, caption="Decrypted Image", use_column_width=True)
