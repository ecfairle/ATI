# Mock torch module for testing without PyTorch
class MockTorch:
    def save(self, data, path):
        # For testing, just save as regular file
        with open(path, 'wb') as f:
            f.write(data)
    
    def load(self, path):
        # For testing, just load as regular file
        with open(path, 'rb') as f:
            return f.read()

# Create mock instance
torch = MockTorch()