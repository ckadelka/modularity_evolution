



import hashlib



def hash_params(**kwargs):
    """Generate a unique hash for a given set of python function parameters. It also sorts the params so that order passed doesn't matter."""
    print(f'Hashing parameters: {kwargs}')
    sig_str = ''.join(f'{value}' for _ , value in sorted(kwargs.items()))
    return hashlib.sha256(sig_str.encode()).hexdigest()[:10]