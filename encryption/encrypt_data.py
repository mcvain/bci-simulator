from cryptography.fernet import Fernet


def encrypt_data(filename, key):
    f = Fernet(key)

    with open(str(filename), 'rb') as original_file:
        original = original_file.read()

    encrypted = f.encrypt(original)

    with open(str(filename), 'wb') as encrypted_file:
        encrypted_file.write(encrypted)

    return
